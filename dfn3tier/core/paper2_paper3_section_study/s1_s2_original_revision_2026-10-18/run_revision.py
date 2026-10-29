"""Bounded original EVAE/Transformer/scorer revision, S1 fit -> S2 assessment.

Original class/function bodies are loaded without legacy import-time experiments.
No existing research artifact is edited. See PROTOCOL.json and source_changes.json.
"""
from __future__ import annotations
import copy, hashlib, importlib.util, itertools, json, math, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings('ignore',category=UserWarning,module='torch')
HERE=Path(__file__).resolve().parent; STUDY=HERE.parent; ROOT=STUDY.parent
sys.path.insert(0,str(STUDY/'original_pipeline_step1_audit_2026-10-18'))
from audit_original_pipeline import definitions, sha, EXP, GEN, LEGACY, REV
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS=[42001,42002,42003]
PROTOCOL=dict(seeds=SEEDS,train_faces=['W02_RL1039-1066','W03_RL1012-1039','W04_RL0985-1012','W05_RL0958-0985','W06_RL0938-0958'],
 validation_faces=['W00_RL1102-1120','W01_RL1066-1102'],test_section=2,
 test_label='additional cross-section validation; previously inspected in other diagnostics',
 canonical_layouts=['centre','5m'],sensitivity_layouts=['3m','7m','9m','11m'],strip_width_m=.1,
 epochs_evae=150,epochs_head=50,epochs_transformer=50,proposal_multiplier=8,
 realization_selection=1,scorer_subset=80,N_output=128,max_context=512,max_generation=128,
 count_rule='round(S1 training density * public target rectangle area), minimum one; never arm-specific continuation count',
 primary_metric='symmetric axial distribution Wasserstein on doubled-angle circle, degrees',
 secondary_metrics=['Hungarian centre-matched axial MAE','length W1 metres','matched length MAE/MAPE','count/P21','intersection density per m2','I/Y/X shares','largest component fraction','top-bottom spanning'],
 selection='S1 validation loss only; no hyperparameter search; S2 never used for fit/selection',
 arms={'A': ['stat','stat',False], 'B':['stat','stat',True], 'C':['evae','stat',True],
       'D':['stat','transformer_no_latent',True], 'E':['evae','transformer_latent',True],
       'F':['evae','transformer_latent',False], 'KDE':['stat','kde',True]})
CHECKS=[]
def check(name,condition,detail=''):
    CHECKS.append(dict(check=name,passed=bool(condition),detail=str(detail)))
    if not condition: raise AssertionError(name+': '+str(detail))
def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')
def seedall(seed):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

common=definitions(EXP/'exp_common.py')
legacy=definitions(LEGACY/'model.py')
trans=definitions(EXP/'transformer_v4.py',dict(MAX_CTX=512,MAX_GEN=128))
lossenv=definitions(EXP/'model_v4.py',dict(kl_loss=legacy['kl_loss']))
scorer=definitions(EXP/'scorer.py')
gen=definitions(GEN/'run_final_integrated.py',names=['EVAEContHead'])
assign=common['assign_family_ids'];split=common['split_panel'];intersect=common['segments_intersect']
# Isolate the original geological-reference builder's dynamic exp_common import.
import types
cm=types.ModuleType('exp_common');cm.__dict__.update(common);sys.modules['exp_common']=cm
geo=definitions(EXP/'run_ablation_v3.py',dict(assign_family_ids=assign),names=['_train_geology_stats','_calibrate_scorer_weights'])

def ends(arr):
    a=np.asarray(arr,dtype=float).reshape(-1,4);d=np.column_stack([np.cos(a[:,3]),np.sin(a[:,3])])*a[:,2,None]/2
    return a[:,:2]-d,a[:,:2]+d
def lines(p,q):
    p=np.asarray(p);q=np.asarray(q);d=q-p
    return np.column_stack([(p+q)/2,np.linalg.norm(d,axis=-1),np.arctan2(d[:,1],d[:,0])%np.pi]).astype(np.float32)
def strips(layout,w):
    if layout=='centre': centres=[w/2]
    else:
        sp=float(layout[:-1]); centres=np.arange(sp,w,sp)
    return [(max(0,(x-.05)/w),min(1,(x+.05)/w)) for x in centres]
def observe(arr,st):
    clips=[]; parents=[]
    for i,row in enumerate(arr):
        c,_=split(row[None],st)
        clips.extend(c);parents.extend([i]*len(c))
    return np.asarray(clips,dtype=np.float32).reshape(-1,4),np.asarray(parents,dtype=int)

def prepare():
    frozen=pd.read_csv(STUDY/'data_prep/manual_trace_manifest.csv')
    world=pd.read_csv(STUDY/'data_prep/manual_trace_segments_world_s1_s4.csv')
    frames=pd.read_csv(STUDY/'data_prep/face_frame.csv')
    prov=json.loads((STUDY/'data_prep/manifest_provenance.json').read_text())
    for fn,digest in prov['sha256_of_outputs'].items():check('frozen_manifest:'+fn,sha(STUDY/'data_prep'/fn)==digest)
    for fn,digest in prov['registered_s2_s4_transform']['sha256_of_outputs'].items():check('frozen_world:'+fn,sha(STUDY/'data_prep'/fn)==digest)
    selected=frozen[frozen.section_id.isin([1,2])].copy(); faces={};audit=[];inputs={}
    for fid,rows in selected.groupby('face_id',sort=True):
        p=Path(rows.source_json.iloc[0]);j=json.loads(p.read_text(encoding='utf-8-sig'));inputs[str(p)]=sha(p)
        check('label_hash:'+fid,inputs[str(p)]==rows.source_sha256.iloc[0])
        w=float(rows.image_width_px.iloc[0])*.02;h=float(rows.image_height_px.iloc[0])*.02
        p1=rows[['u1_m','v1_m']].to_numpy()+[0,h];p2=rows[['u2_m','v2_m']].to_numpy()+[0,h]
        check('rectangle_endpoints:'+fid,bool((p1[:,0]>=-.02).all() and (p2[:,0]>=-.02).all() and (p1[:,0]<=w+.02).all() and (p2[:,0]<=w+.02).all() and (p1[:,1]>=-.02).all() and (p2[:,1]>=-.02).all() and (p1[:,1]<=h+.02).all() and (p2[:,1]<=h+.02).all()))
        fr=frames[frames.face_id==fid];wr=world[world.face_id==fid]
        role='test' if int(rows.section_id.iloc[0])==2 else 'train' if fid in PROTOCOL['train_faces'] else 'validation'
        if len(fr):
            fp=Path(fr.metadata_json.iloc[0]);npz=Path(fr.geometry_npz.iloc[0]);inputs[str(fp)]=sha(fp);inputs[str(npz)]=sha(npz)
            check('frame_metadata_hash:'+fid,sha(fp)==fr.metadata_sha256.iloc[0]);check('geometry_hash:'+fid,sha(npz)==fr.geometry_sha256.iloc[0])
            meta=json.loads(fp.read_text());check('registered_scale:'+fid,abs(float(meta['pixel_size_m'])-.02)<1e-12)
            top=float(fr.RL_top.iloc[0]);bottom=float(fr.RL_bottom.iloc[0])
        else:
            # Archived full frames are absent for W01/W02, but all world label
            # endpoints have an exact snapshot/endpoint join. Use conservative
            # nominal RL envelope from the archived face name for separation.
            bottom,top=map(float,fid.split('RL')[1].split('-'))
        check('world_label_join:'+fid,len(wr)==len(rows) and wr[['E1','N1','RL1','E2','N2','RL2']].notna().all().all())
        faces[fid]=dict(id=fid,role=role,w=w,h=h,ym=h/w,arr=lines(p1/w,p2/w),rows=rows)
        audit.append(dict(face_id=fid,role=role,section=int(rows.section_id.iloc[0]),traces=len(rows),width_m=w,height_up_face_m=h,area_m2=w*h,RL_low=bottom,RL_high=top,full_frame_present=len(fr)>0,pixel_size_m=.02,
            scale_status='registered builder scale; independent survey accuracy not assessed',polyline_minus_end_to_end_m=float((rows.polyline_length_m-rows.end_to_end_length_m).sum())))
    audit=pd.DataFrame(audit);audit.to_csv(HERE/'face_split_geometry.csv',index=False)
    train_audit=audit[audit.role.isin(['train','validation'])];test_audit=audit[audit.role=='test']
    gap=float(train_audit.RL_low.min()-test_audit.RL_high.max());check('S1_S2_disjoint_registered_elevation_envelopes',gap>0,gap)
    valgap=float(audit[audit.role=='validation'].RL_low.min()-audit[audit.role=='train'].RL_high.max())
    check('S1_validation_block_above_training',valgap>=0,valgap)
    # Actual labelled world endpoints provide an independent lower bound on
    # endpoint-set separation, not a distance between all rock-surface pixels.
    wp=world[world.section_id==1];wq=world[world.section_id==2]
    v=lambda x:np.vstack([x[['E1','N1','RL1']].to_numpy(),x[['E2','N2','RL2']].to_numpy()])
    mind=float(cdist(v(wp),v(wq)).min())
    dump('spatial_provenance.json',dict(S1_S2_RL_envelope_gap_m=gap,minimum_label_endpoint_distance_m=mind,S1_inner_RL_envelope_gap_m=valgap,
         complete_wall_continuity=False,mapped_rectangles_only=True,site_independence=False,
         limitation='Same mine, separate mapped section/elevation; not an independent mine or proven distinct structural domain. Missing W01/W02 full frames disclosed.'))
    for p in [STUDY/'data_prep/manual_trace_manifest.csv',STUDY/'data_prep/manual_trace_segments_world_s1_s4.csv',STUDY/'data_prep/face_frame.csv',LEGACY/'model.py',EXP/'model_v4.py',EXP/'transformer_v4.py',EXP/'scorer.py',GEN/'run_final_integrated.py',EXP/'run_ablation_v3.py',EXP/'exp_common.py']:
        inputs[str(p)]=sha(p)
    dump('input_hashes.json',inputs)
    return faces

def fit_stats(faces):
    fs=[f for f in faces.values() if f['role']=='train'];arr=np.vstack([f['arr'] for f in fs])
    angles=arr[:,3];xy=np.column_stack([np.cos(2*angles),np.sin(2*angles)])
    km=KMeans(3,random_state=42001,n_init=20).fit(xy)
    fc=np.sort((np.arctan2(km.cluster_centers_[:,1],km.cluster_centers_[:,0])/2)%np.pi)
    fam=assign(arr,fc); gs=geo['_train_geology_stats']([f['arr'] for f in fs],fc)
    # Extend background stats explicitly instead of silently dropping random joints.
    phys=np.concatenate([f['arr'][:,2]*f['w'] for f in fs]);rate=sum(len(f['arr']) for f in fs)/sum(f['w']*f['h'] for f in fs)
    pools={}; logmu={};logsig={}
    for k in range(4):
        ids=fam==k;sub=arr[ids] if ids.any() else arr
        lens=phys[ids] if ids.any() else phys
        pools[k]=dict(arr=sub,lengths_m=lens,angles=sub[:,3],centres=sub[:,:2])
        logmu[k]=float(np.log(sub[:,2]).mean());logsig[k]=max(float(np.log(sub[:,2]).std()),.1)
        if k==3:
            gs['spacing_floor'][k]=.008;gs['count_std'][k]=1.
            gs['centre_mass_ref'][k]=dict(cx_mean=float(sub[:,0].mean()),cx_std=max(float(sub[:,0].std()),.02),cy_mean=float(sub[:,1].mean()),cy_std=max(float(sub[:,1].std()),.02))
            gs['fam_length_ref'][k]=dict(p90=float(np.quantile(sub[:,2],.9)),p95=float(np.quantile(sub[:,2],.95)),mean=float(sub[:,2].mean()),std=max(float(sub[:,2].std()),.01))
    stats=dict(fc=fc,gs=gs,pools=pools,shares=np.bincount(fam,minlength=4)/len(fam),rate=rate,logmu=logmu,logsig=logsig,yscale=max(f['ym'] for f in fs))
    dump('training_only_statistics.json',stats)
    return stats

def tensor_encode(evae,arr):
    arr=np.asarray(arr,dtype=np.float32).reshape(-1,4)
    if not len(arr):return torch.zeros(1,12,device=DEV),torch.zeros(1,20,device=DEV)
    t=torch.tensor(arr,device=DEV)[None];mask=torch.ones(1,len(arr),device=DEV,dtype=torch.bool)
    with torch.no_grad():sm,_,gm,_=evae.encode(t[:,:,:2],t[:,:,2:3],t[:,:,3:4],mask)
    return sm,gm
def augment(face,kind):
    a=face['arr'].copy()
    if kind=='mirror':a[:,0]=1-a[:,0];a[:,3]=(np.pi-a[:,3])%np.pi
    if kind=='rot180':a[:,0]=1-a[:,0];a[:,1]=face['ym']-a[:,1]
    return a
def examples(faces,role):
    out=[]
    for face in faces.values():
        if face['role']!=role:continue
        for layout in ['centre','3m','5m','7m','9m','11m']:
            for aug in (['orig','mirror','rot180'] if role=='train' else ['orig']):
                arr=augment(face,aug);st=strips(layout,face['w']);obs,parents=observe(arr,st)
                out.append(dict(face=face,layout=layout,arr=arr,obs=obs,parents=parents,strips=st))
    return out

def train_evae(train,val,stats,seed):
    mdl=legacy['DualLatentVAE'](sd=12,gd=20,N=128,yscale=stats['yscale']).to(DEV)
    # Correct train-derived family centres and the omitted von Mises normalizer;
    # otherwise the negative cosine term rewards arbitrarily large concentration.
    import ast
    tree=ast.parse((EXP/'model_v4.py').read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_loss')
    source=ast.unparse(node).replace('torch.tensor([0.623, 1.165, 2.221], device=pc.device)','torch.tensor(FAMILY_CENTRES, device=pc.device)')
    source=source.replace('-kpr[:, k] * torch.cos(2 * diff)','-kpr[:, k] * torch.cos(2 * diff) + torch.log(torch.special.i0e(kpr[:, k])) + kpr[:, k] + math.log(math.pi)')
    check('EVAE_loss_family_centres_replaced', 'FAMILY_CENTRES' in source)
    check('EVAE_loss_axial_NLL_normalizer_added','i0e' in source)
    env=dict(lossenv,FAMILY_CENTRES=stats['fc']);exec(source,env);lossfn=env['compute_loss']
    def pack(items):
        n=max(1,max(len(e['obs']) for e in items));B=len(items)
        arrays={k:np.zeros(shape,dtype=np.float32) for k,shape in [('C_in',(B,n,2)),('L_in',(B,n,1)),('A_in',(B,n,1)),('C_tgt',(B,128,2)),('L_tgt',(B,128,1)),('A_tgt',(B,128,1)),('E_tgt',(B,128))]}
        arrays['mask']=np.zeros((B,n),bool);arrays['F_tgt']=np.full((B,128),-1,np.int64)
        for b,e in enumerate(items):
            a=e['arr'];c=e['obs'];ni=len(c);no=len(a)
            check('no_training_target_truncation',no<=128)
            arrays['C_in'][b,:ni]=c[:,:2];arrays['L_in'][b,:ni]=c[:,2:3];arrays['A_in'][b,:ni]=c[:,3:4];arrays['mask'][b,:ni]=True
            arrays['C_tgt'][b,:no]=a[:,:2];arrays['L_tgt'][b,:no]=a[:,2:3];arrays['A_tgt'][b,:no]=a[:,3:4];arrays['F_tgt'][b,:no]=assign(a,stats['fc']);arrays['E_tgt'][b,:no]=1
        return {k:torch.as_tensor(v,device=DEV) for k,v in arrays.items()}
    opt=torch.optim.Adam(mdl.parameters(),lr=3e-4,weight_decay=1e-5);best=math.inf;state=None;logs=[]
    for ep in range(1,151):
        mdl.train();ls=[]
        for inds in np.array_split(np.random.permutation(len(train)),math.ceil(len(train)/8)):
            b=pack([train[i] for i in inds]);p=mdl(b['C_in'],b['L_in'],b['A_in'],b['mask'])
            loss,_=lossfn(p,b,min(1.,ep/50)*.001,torch.ones(4,device=DEV));opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(mdl.parameters(),1.);opt.step();ls.append(float(loss))
        mdl.eval();vl=[]
        with torch.no_grad(),torch.random.fork_rng(devices=[0] if DEV.type=='cuda' else []):
            torch.manual_seed(seed+777)
            for e in val:
                b=pack([e]);p=mdl(b['C_in'],b['L_in'],b['A_in'],b['mask']);l,_=lossfn(p,b,.001,torch.ones(4,device=DEV));vl.append(float(l))
        v=float(np.mean(vl));logs.append(dict(component='evae',epoch=ep,train=float(np.mean(ls)),validation=v))
        if v<best:best=v;state={k:v.detach().cpu().clone() for k,v in mdl.state_dict().items()};selected=ep
        if ep%50==0:print(f'seed {seed} EVAE {ep}/150 val={v:.4f}',flush=True)
    mdl.load_state_dict(state);mdl.eval()
    return mdl,logs,selected

def continuation_features(obs,idx,side,face):
    row=obs[idx];p,q=ends(row[None]);a=p[0] if side==0 else q[0];u=(q[0]-p[0])/max(row[2],1e-10);u=u*(-1 if side==0 else 1)
    # Canonical axial row endpoint ordering is valid for this direction vector.
    dist=[]
    for z,lim,d in zip(a,[1.,face['ym']],u):
        if abs(d)>1e-12:dist.append(((lim-z) if d>0 else -z)/d)
    bound=max(0,min(dist)) if dist else 0
    fam=int(assign(row[None],STATS['fc'])[0]);A=float(row[3]);L=float(row[2])
    feat=np.array([math.log(max(L,1e-8)),math.sin(A),math.cos(A),float(fam==0),float(fam==1),float(fam==2),-1. if side==0 else 1.,a[0],a[1],bound,float(np.sum(np.linalg.norm(obs[:,:2]-row[:2],axis=1)<.1)-1)],np.float32)
    return feat,a,u,bound

def train_head(train,val,evae):
    def make(exs):
        X=[];Y=[]
        for e in exs:
            lat=torch.cat(tensor_encode(evae,e['obs']),-1).cpu().numpy()[0]
            for i,parent in enumerate(e['parents']):
                tp,tq=ends(e['arr'][parent:parent+1])
                for side in [0,1]:
                    feat,a,u,bound=continuation_features(e['obs'],i,side,e['face'])
                    # Exact simulated parent used only for supervised S1 targets.
                    extension=max(float(np.dot(tp[0]-a,u)),float(np.dot(tq[0]-a,u)),0.)
                    if bound>1e-8:X.append(np.r_[lat,feat]);Y.append(math.log(max(extension,1e-6)))
        return torch.tensor(np.array(X),device=DEV),torch.tensor(Y,device=DEV)[:,None]
    X,Y=make(train);V,Z=make(val);mu=X.mean(0);sd=X.std(0).clamp_min(1e-3);X=(X-mu)/sd;V=(V-mu)/sd
    mdl=gen['EVAEContHead']().to(DEV);opt=torch.optim.Adam(mdl.parameters(),lr=1e-3,weight_decay=1e-5);best=math.inf;logs=[]
    for ep in range(1,51):
        mdl.train();ls=[]
        for inds in torch.randperm(len(X),device=DEV).split(256):
            m,l=mdl(X[inds]);loss=(.5*((Y[inds]-m)/l.exp())**2+l).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(mdl.parameters(),1.);opt.step();ls.append(float(loss))
        mdl.eval()
        with torch.no_grad():m,l=mdl(V);vl=float((.5*((Z-m)/l.exp())**2+l).mean())
        logs.append(dict(component='head',epoch=ep,train=float(np.mean(ls)),validation=vl))
        if vl<best:best=vl;state={k:v.detach().cpu().clone() for k,v in mdl.state_dict().items()};selected=ep
    mdl.load_state_dict(state);mdl.eval()
    return (mdl,mu,sd),logs,selected

def indicated(obs,face,st,mode,models,rng):
    if len(obs)==0:return np.zeros((0,4),np.float32)
    lat=torch.cat(tensor_encode(models['evae'],obs),-1)
    rows=[]
    for i,row in enumerate(obs):
        a,b=ends(row[None]);p=a[0].copy();q=b[0].copy();f=int(assign(row[None],STATS['fc'])[0])
        for side in [0,1]:
            feat,t,u,bound=continuation_features(obs,i,side,face)
            if not any(abs(t[0]-edge)<1e-6 for strip in st for edge in strip):continue
            if mode=='evae':
                head,mu,sd=models['head'];ft=torch.tensor(feat,device=DEV)[None]
                with torch.no_grad():m,l=head((torch.cat([lat,ft],-1)-mu)/sd)
                extension=math.exp(float(m.clamp(-15,2)))
            else:
                extension=float(np.median(STATS['pools'][f]['lengths_m']))/face['w']/2
            extension=min(extension,bound,float(np.quantile(STATS['pools'][f]['lengths_m'],.95))/face['w'])
            # No hypothesized extension crosses another observed strip.
            if abs(u[0])>1e-9:
                for strip in st:
                    for edge in strip:
                        distance=(edge-t[0])/u[0]
                        if distance>1e-6:extension=min(extension,max(0,distance-1e-6))
            if side==0:p=t+extension*u
            else:q=t+extension*u
        rows.append(lines(p[None],q[None])[0])
    # Merge co-linear overlapping reconstructed parents using observable geometry,
    # never the simulator's hidden parent IDs. Measured clips remain separately saved.
    merged=[]
    for row in rows:
        p,q=ends(row[None]);done=False
        for j,old in enumerate(merged):
            op,oq=ends(old[None]);u=(oq[0]-op[0])/old[2];d=abs(math.sin(float(row[3]-old[3])))
            cross=abs(u[0]*(p[0,1]-op[0,1])-u[1]*(p[0,0]-op[0,0]))
            vals=np.array([np.dot(p[0]-op[0],u),np.dot(q[0]-op[0],u)])
            if d<1e-5 and cross<1e-5 and vals.min()<=old[2]+1e-5 and vals.max()>=-1e-5:
                a=op[0]+min(0,vals.min())*u;b=op[0]+max(old[2],vals.max())*u;merged[j]=lines(a[None],b[None])[0];done=True;break
        if not done:merged.append(row)
    return np.asarray(merged,np.float32).reshape(-1,4)

def sequence_example(e,models,latent):
    context=indicated(e['obs'],e['face'],e['strips'],'evae' if latent else 'stat',models,np.random.default_rng(0))
    # Supervised target membership is known only for S1 examples.
    target=e['arr'][~np.isin(np.arange(len(e['arr'])),np.unique(e['parents']))]
    fam=assign(target,STATS['fc']);order=np.lexsort((target[:,0],fam));target=target[order];fam=fam[order]
    context=context[np.argsort(context[:,0])]
    check('no_sequence_target_truncation',len(target)<=128)
    check('no_context_truncation',len(context)<=512)
    ct=common['tok_array'](context)
    if len(ct)==0:ct=np.zeros((1,5),np.float32)
    sz,gz=tensor_encode(models['evae'],context)
    return dict(ctx=ct,tgt=common['tok_array'](target),fam=fam,sz=sz.detach(),gz=gz.detach())

def seq_batch(items):
    nc=max(len(e['ctx']) for e in items);nt=max(len(e['tgt']) for e in items);B=len(items)
    c=torch.zeros(B,nc,5,device=DEV);cm=torch.ones(B,nc,dtype=torch.bool,device=DEV)
    y=torch.zeros(B,nt,5,device=DEV);mask=torch.ones(B,nt,dtype=torch.bool,device=DEV);fam=torch.zeros(B,nt,dtype=torch.long,device=DEV)
    for i,e in enumerate(items):
        n=len(e['ctx']);t=len(e['tgt']);c[i,:n]=torch.as_tensor(e['ctx'],device=DEV);cm[i,:n]=False;y[i,:t]=torch.as_tensor(e['tgt'],device=DEV);mask[i,:t]=False;fam[i,:t]=torch.as_tensor(e['fam'],device=DEV)
    return c,cm,y,mask,fam,torch.cat([e['sz'] for e in items]),torch.cat([e['gz'] for e in items])
def seq_loss(mdl,b,latent):
    c,cm,y,mask,fam,sz,gz=b
    di=torch.cat([torch.zeros(len(c),1,5,device=DEV),y[:,:-1]],dim=1)
    dm=torch.cat([torch.zeros(len(c),1,dtype=torch.bool,device=DEV),mask[:,:-1]],dim=1)
    p,l,a,f=mdl(c,cm,di,dm,sz if latent else None,gz if latent else None)
    valid=~mask
    angle=torch.atan2(y[:,:,3],y[:,:,4]);pa=torch.atan2(a[:,:,0],a[:,:,1])
    # Teacher forcing now aligns input BOS/previous token with SAME target and family.
    loss=F.mse_loss(p[valid],y[:,:,:2][valid])+.5*F.mse_loss(l[valid],y[:,:,2:3][valid])+1.5*(1-torch.cos(2*(angle[valid]-pa[valid]))).mean()+F.cross_entropy(f[valid],fam[valid])
    probs=f[valid].softmax(-1).mean(0);loss=loss+.3*F.mse_loss(probs,torch.as_tensor(STATS['shares'],device=DEV,dtype=torch.float32))
    # Retain original crossing surrogate coefficient/functional form.
    ints=[];tau=max(float(np.std(np.vstack([v['arr'][:,:2] for v in FACES.values() if v['role']=='train']),axis=0).sum()),1e-3)
    for z in range(len(c)):
        pp=p[z][valid[z]];ll=l[z,:,0][valid[z]].exp();aa=pa[z][valid[z]]
        if len(pp)<2:continue
        dij=torch.cdist(pp,pp);rij=dij/((ll[:,None]+ll[None,:])/2+1e-6);aij=torch.sin(aa[:,None]-aa[None,:]).square();iu=torch.triu_indices(len(pp),len(pp),1,device=DEV)
        ints.append((aij*torch.exp(-rij.square()/(tau*tau+1e-6)))[iu[0],iu[1]].mean())
    if ints:loss=loss+.5*(torch.stack(ints).mean()-STATS['gs']['intersection_mean'])**2
    return loss
def train_transformer(train,val,models,latent):
    tr=[sequence_example(e,models,latent) for e in train];va=[sequence_example(e,models,latent) for e in val]
    tr=[e for e in tr if len(e['tgt'])];va=[e for e in va if len(e['tgt'])]
    mdl=trans['SeqFracTransformerV4']().to(DEV);opt=torch.optim.Adam(mdl.parameters(),lr=3e-4,weight_decay=1e-5);best=math.inf;logs=[];gradient=None
    for ep in range(1,51):
        mdl.train();ls=[]
        for inds in np.array_split(np.random.permutation(len(tr)),math.ceil(len(tr)/8)):
            loss=seq_loss(mdl,seq_batch([tr[i] for i in inds]),latent);opt.zero_grad();loss.backward()
            if gradient is None:
                gradient=mdl.latent_proj.weight.grad is not None
                check('latent_projection_receives_gradient' if latent else 'no_latent_ablation_disconnected',gradient if latent else not gradient)
            nn.utils.clip_grad_norm_(mdl.parameters(),1.);opt.step();ls.append(float(loss))
        mdl.eval()
        with torch.no_grad():vl=float(np.mean([float(seq_loss(mdl,seq_batch([e]),latent)) for e in va]))
        logs.append(dict(component='transformer_latent' if latent else 'transformer_no_latent',epoch=ep,train=float(np.mean(ls)),validation=vl))
        if vl<best:best=vl;state={k:v.detach().cpu().clone() for k,v in mdl.state_dict().items()};selected=ep
        if ep%25==0:print(f'Transformer latent={latent} {ep}/50 val={vl:.4f}',flush=True)
    mdl.load_state_dict(state);mdl.eval();return mdl,logs,selected

def neural_proposals(mdl,context,face,total,latent,evae,rng):
    ctx=context[np.argsort(context[:,0])];ct=common['tok_array'](ctx)
    if len(ct)==0:ct=np.zeros((1,5),np.float32)
    c=torch.tensor(ct,device=DEV)[None];cm=torch.zeros(1,len(ct),device=DEV,dtype=torch.bool)
    sz,gz=tensor_encode(evae,context);samples=[]
    # Eight autoregressive realizations share one fixed raw-attempt budget.
    batch=PROTOCOL['proposal_multiplier'];steps=math.ceil(total/batch)
    check('generation_capacity_sufficient',steps<=128,steps)
    seq=torch.zeros(batch,1,5,device=DEV)
    with torch.no_grad():
        for step in range(steps):
            mask=torch.zeros(batch,len(seq[0]),dtype=torch.bool,device=DEV)
            p,l,a,f=mdl(c.expand(batch,-1,-1),cm.expand(batch,-1),seq,mask,sz.expand(batch,-1) if latent else None,gz.expand(batch,-1) if latent else None)
            xy=p[:,-1].cpu().numpy();L=l[:,-1,0].clamp(-12,0).exp().cpu().numpy();A=torch.atan2(a[:,-1,0],a[:,-1,1]).cpu().numpy()%np.pi
            # Original sampling temperature represented by small train-scale jitter;
            # no statistical fallback is inserted into a neural-only arm.
            xy+=rng.normal(0,.006,xy.shape);A=(A+rng.normal(0,.025,len(A)))%np.pi
            row=np.column_stack([xy,L,A]);samples.extend(row)
            tok=common['tok_array'](row);seq=torch.cat([seq,torch.tensor(tok,device=DEV)[:,None]],dim=1)
    return np.asarray(samples,np.float32)[:total]

def statistical_proposals(face,total,mode,rng):
    fam=rng.choice(4,total,p=STATS['shares']);out=[]
    for k in fam:
        pool=STATS['pools'][int(k)];i=int(rng.integers(len(pool['angles'])))
        if mode=='kde':
            # Empirical Gaussian product-kernel mixture, bandwidth fixed using S1
            # sample size (Scott exponent); family conditioning shared with MC.
            n=len(pool['angles']);bw=n**(-1/5);A=(float(pool['angles'][i])+rng.normal(0,max(float(np.std(pool['angles'])),.01)*bw))%np.pi
            L=math.exp(math.log(float(pool['lengths_m'][i]))+rng.normal(0,max(float(np.std(np.log(pool['lengths_m']))),.1)*bw))
        else:
            A=float(pool['angles'][i]);logs=np.log(pool['lengths_m']);L=math.exp(float(rng.normal(logs.mean(),max(logs.std(),.1))))
        out.append([rng.uniform(),rng.uniform(0,face['ym']),L/face['w'],A])
    return np.asarray(out,np.float32)

def candidate_gate(row,face,st):
    if not np.isfinite(row).all() or row[2]<=1e-8:return 'invalid_numeric'
    p,q=ends(row[None]);z=np.vstack([p,q])
    if (z[:,0]<0).any() or (z[:,0]>1).any() or (z[:,1]<0).any() or (z[:,1]>face['ym']).any():return 'outside_rectangle'
    # No extra hypothesized crossings in completely observed strips.
    if len(split(row[None],st)[0]):return 'observation_conflict'
    return None

def reconstruct(face,obs,st,arm,models,seed):
    tier2,proposal,scoring=PROTOCOL['arms'][arm];rng=np.random.default_rng(seed)
    fixed=indicated(obs,face,st,tier2,models,rng)
    # Population target must be identical before arm-specific continuation/merging.
    # Retain all measured geometry even when it exceeds this density prediction;
    # record excess rather than allowing an arm to buy more proposal attempts.
    target=max(1,int(round(STATS['rate']*face['w']*face['h'])))
    missing=max(0,target-len(fixed));budget=max(1,target)*PROTOCOL['proposal_multiplier']
    if proposal=='mixed_original':
        # The historical integrated method combines statistical and Transformer
        # proposals. Retain one fixed mixture, with the SAME total attempt budget.
        n_neural=budget//8
        neural=neural_proposals(models['transformer_latent'],fixed,face,n_neural,True,models['evae'],np.random.default_rng(seed+100000))
        statistical=statistical_proposals(face,budget-n_neural,'stat',rng)
        raw=np.vstack([statistical,neural]);source_labels=['stat']*len(statistical)+['transformer_latent']*len(neural)
    elif proposal.startswith('transformer'):
        raw=neural_proposals(models[proposal],fixed,face,budget,proposal=='transformer_latent',models['evae'],rng)
    else:raw=statistical_proposals(face,budget,proposal,rng)
    fids=assign(raw,STATS['fc']);knownfam=assign(fixed,STATS['fc']);logs=[];available=[];selected=[];sfams=[]
    for i,row in enumerate(raw):
        why=candidate_gate(row,face,st)
        logs.append(dict(candidate_id=i,source=source_labels[i] if proposal=='mixed_original' else proposal,cx=float(row[0]),cy=float(row[1]),length=float(row[2]),angle=float(row[3]),family=int(fids[i]),disposition=why or 'budget_not_selected',score_evaluations=0))
        if why is None:available.append(i)
    gs=copy.deepcopy(STATS['gs']);family_targets=STATS['shares']*target
    needed={f:max(0,float(family_targets[f])-int((knownfam==f).sum())) for f in range(4)}
    placed={f:0 for f in range(4)}
    # Fix original HCScorer tail/length vector order explicitly.
    sw=gs['scorer_weights'];weights=np.array([sw[k] for k in ['w_spacing','w_xbin','w_fam_deficit','w_cm_dev','w_f2_tail','w_len_quality','w_intersection']])
    for step in range(missing):
        if not available:break
        ids=rng.choice(available,min(len(available),80),replace=False).tolist() if scoring else available[:1]
        best=None;bestscore=-math.inf
        placedlines=list(fixed)+selected;allf=list(knownfam)+sfams
        h=np.histogram(np.asarray(placedlines)[:,0] if len(placedlines) else [],bins=np.linspace(0,1,6))[0];props=h/max(h.sum(),1)
        for i in ids:
            r=raw[i];f=int(fids[i]);same=[e for e,ef in zip(placedlines,allf) if f==ef]
            if placedlines and min(np.linalg.norm(r[:2]-e[:2])+abs(r[2]-e[2]) for e in placedlines)<1e-5:
                logs[i]['disposition']='duplicate';available.remove(i);continue
            if scoring:
                feat=scorer['extract_features'](*map(float,r),f,same,placedlines,gs['spacing_floor'],gs['xbin_ref'],gs['xbin_std'],props,needed,placed,gs['count_std'],gs['centre_mass_ref'],list(fixed),selected,list(knownfam),sfams,STATS['logmu'],STATS['logsig'],gs['fam_length_ref'],gs['intersection_std'],segments_intersect_fn=intersect)
                score=float(weights@feat);logs[i]['score_evaluations']+=1
            else:score=0.
            if score>bestscore:best=i;bestscore=score
        if best is None:continue
        available.remove(best);selected.append(raw[best]);sfams.append(int(fids[best]));placed[int(fids[best])]+=1;logs[best]['disposition']='accepted';logs[best]['selection_score']=bestscore
    pred=np.vstack([fixed,np.asarray(selected).reshape(-1,4)]).astype(np.float32)
    return pred,np.asarray(selected,np.float32).reshape(-1,4),fixed,pd.DataFrame(logs),dict(target_count=target,indicated_parents=len(fixed),observed_clips=len(obs),raw_budget=budget,generated=len(pred),shortfall=max(0,target-len(pred)))

def axial_w1(a,b):
    # Minimum common-cut linear W1 over fixed angle cuts, approximating circular
    # axial W1. Identical cuts and 1-degree grid for every arm.
    if not len(a) or not len(b):return math.nan
    return min(wasserstein_distance((np.degrees(a)-cut)%180,(np.degrees(b)-cut)%180) for cut in np.arange(180))
def topology(arr,w,h):
    # Segment graph; contacts cluster only within 2 cm in physical face coordinates.
    p,q=ends(arr);p*=w;q*=w;n=len(arr);parent=list(range(n));contacts=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(n):
        for j in range(i):
            u=q[i]-p[i];v=q[j]-p[j];den=u[0]*v[1]-u[1]*v[0]
            if abs(den)<1e-10:continue
            d=p[j]-p[i];t=(d[0]*v[1]-d[1]*v[0])/den;s=(d[0]*u[1]-d[1]*u[0])/den
            if 0<=t<=1 and 0<=s<=1:
                parent[root(i)]=root(j);point=p[i]+t*u
                near_i=min(np.linalg.norm(point-p[i]),np.linalg.norm(point-q[i]))<=.02
                near_j=min(np.linalg.norm(point-p[j]),np.linalg.norm(point-q[j]))<=.02
                contacts.append(('Y' if near_i != near_j else 'I' if near_i and near_j else 'X',point))
    deg={'I':2*n,'Y':0,'X':0}
    for k,point in contacts:
        deg[k]+=1
        for i in range(n):
            deg['I']-=int(np.linalg.norm(point-p[i])<=.02)+int(np.linalg.norm(point-q[i])<=.02)
    deg['I']=max(0,deg['I']);tot=max(sum(deg.values()),1)
    groups={}
    for i in range(n):groups.setdefault(root(i),[]).append(i)
    span=False
    for g in groups.values():
        ys=np.r_[p[g,1],q[g,1]]
        if len(ys) and ys.min()<=.02 and ys.max()>=h-.02:span=True
    return dict(intersections_per_m2=len(contacts)/(w*h),I_share=deg['I']/tot,Y_share=deg['Y']/tot,X_share=deg['X']/tot,components=len(groups),largest_component_fraction=max([len(g) for g in groups.values()]+[0])/max(n,1),top_bottom_spanning=int(span))
def metrics(real,pred,w,h,obs):
    r=np.asarray(real);p=np.asarray(pred);out=dict(truth_count=len(r),predicted_count=len(p),count_error_pct=100*abs(len(p)-len(r))/max(len(r),1),length_w1_m=wasserstein_distance(r[:,2]*w,p[:,2]*w) if len(p) else math.nan,axial_w1_deg=axial_w1(r[:,3],p[:,3]),P21_truth=float(r[:,2].sum()*w/(w*h)),P21_pred=float(p[:,2].sum()*w/(w*h)))
    if len(p):
        i,j=linear_sum_assignment(cdist(r[:,:2],p[:,:2]));delta=abs(r[i,3]-p[j,3]);ad=np.minimum(delta,np.pi-delta)
        out.update(matched_angle_MAE_deg=float(np.degrees(ad).mean()),matched_length_MAE_m=float(np.abs(r[i,2]-p[j,2]).mean()*w),matched_length_MAPE_pct=float((np.abs(r[i,2]-p[j,2])/np.maximum(r[i,2],1e-8)).mean()*100))
    for label,a in [('truth',r),('pred',p)]:out.update({k+'_'+label:v for k,v in topology(a,w,h).items()})
    out['measured_preservation']=preservation(obs,p)
    return out
def preservation(obs,pred):
    if not len(obs):return 1.
    if not len(pred):return 0.
    op,oq=ends(obs);p,q=ends(pred);hit=0
    for a,b in zip(op,oq):
        for c,d in zip(p,q):
            v=d-c;ln=np.linalg.norm(v)
            if ln<1e-10:continue
            v/=ln;t=np.array([np.dot(a-c,v),np.dot(b-c,v)])
            cross=max(abs(v[0]*(a[1]-c[1])-v[1]*(a[0]-c[0])),abs(v[0]*(b[1]-c[1])-v[1]*(b[0]-c[0])))
            if cross<2e-5 and t.min()>=-2e-5 and t.max()<=ln+2e-5:hit+=1;break
    return hit/len(obs)

def evaluate(faces,models,seed):
    rows=[];obsrows=[];preds=[];ledgers=[];budgets=[]
    for face in faces.values():
        if face['role']!='test':continue
        for layout in PROTOCOL['canonical_layouts']+PROTOCOL['sensitivity_layouts']:
            st=strips(layout,face['w']);obs,parents=observe(face['arr'],st)
            obsrows.append(dict(seed=seed,face_id=face['id'],layout=layout,truth=len(face['arr']),unique_observed=len(set(parents)),clips=len(obs),trace_fraction=len(set(parents))/len(face['arr']),area_fraction=sum(b-a for a,b in st),clipped_length_fraction=float(obs[:,2].sum()/face['arr'][:,2].sum()),empty=int(len(obs)==0)))
            arms=list(PROTOCOL['arms']) if layout in PROTOCOL['canonical_layouts'] else ['B','E']
            for arm in arms:
                pred,inferred,fixed,ledger,budget=reconstruct(face,obs,st,arm,models,seed)
                meta=dict(seed=seed,face_id=face['id'],layout=layout,arm=arm)
                check('measurement_preservation:'+str(meta),preservation(obs,pred)==1.)
                check('complete_proposal_accounting:'+str(meta),len(ledger)==budget['raw_budget'])
                truth_inferred=face['arr'][~np.isin(np.arange(len(face['arr'])),np.unique(parents))]
                met=metrics(face['arr'],pred,face['w'],face['h'],obs)
                met['inferred_axial_w1_deg']=axial_w1(truth_inferred[:,3],inferred[:,3]);met['inferred_length_w1_m']=wasserstein_distance(truth_inferred[:,2]*face['w'],inferred[:,2]*face['w']) if len(inferred) and len(truth_inferred) else math.nan
                rows.append(dict(meta,**met));ledgers.append(ledger.assign(**meta));budgets.append(dict(meta,**budget))
                for tier,a in [('fixed_measured_indicated',fixed),('inferred',inferred),('measured_clips',obs)]:
                    df=pd.DataFrame(a,columns=['cx_normalized','cy_normalized','length_normalized','angle_rad']);preds.append(df.assign(tier=tier,width_m=face['w'],**meta))
        print(f'seed {seed}: evaluated {face["id"]}',flush=True)
    return pd.DataFrame(rows),pd.DataFrame(obsrows),pd.concat(preds),pd.concat(ledgers),pd.DataFrame(budgets)

def tests():
    # Independent known geometries: crossing pair, disjoint pair, axial wrap,
    # metric scaling, exact clipped observation containment and label alignment.
    cross=lines(np.array([[0,.5],[.5,0]]),np.array([[1,.5],[.5,1]]))
    t=topology(cross,10,10);check('known_crossing_density',abs(t['intersections_per_m2']-.01)<1e-10);check('known_crossing_component',t['largest_component_fraction']==1)
    check('known_crossing_X_node',abs(t['X_share']-.2)<1e-10)
    check('axial_180_wrap',axial_w1(np.radians([1]),np.radians([179]))<2.0001)
    a=np.array([[.5,.5,.1,.2]]);b=a.copy();b[:,2]=.2
    check('length_units_scale',abs(metrics(a,b,20,20,np.zeros((0,4)))['matched_length_MAE_m']-2)<1e-6)
    obs,_=observe(a,[(.49,.51)]);check('known_clip_preservation',preservation(obs,a)==1)
    check('teacher_forcing_alignment',np.array_equal(np.r_[[-1],np.arange(4)[:-1]],[-1,0,1,2]))
    check('scorer_feature_weight_order',scorer['FEATURE_NAMES'][4]=='tail_persistence')

def main():
    global STATS,FACES
    if (HERE/'RUN_COMPLETE.json').exists():raise RuntimeError('Completed experiment exists; preserve it')
    torch.set_num_threads(2);start=time.time();dump('PROTOCOL.json',PROTOCOL)
    FACES=prepare();STATS=fit_stats(FACES);tests()
    train=examples(FACES,'train');val=examples(FACES,'validation')
    dump('training_dataset_summary.json',dict(training_examples=len(train),validation_examples=len(val),training_trace_entries=sum(len(f['arr']) for f in FACES.values() if f['role']=='train'),validation_trace_entries=sum(len(f['arr']) for f in FACES.values() if f['role']=='validation'),maximum_observed_clips=max(len(e['obs']) for e in train+val),test_used_for_selection=False))
    dump('source_changes.json',dict(original_classes=['DualLatentVAE 12/20, dm64, original attention encoders/decoder heads','SeqFracTransformerV4 d128,h4,encoder3,decoder3','EVAEContHead 43->64->32->mean/logsigma'],
        corrections=['N output 120->128 to retain all S1 targets; context capacity 80->512, generation80->128, no target truncation','S1-only family clustering/statistics/scale/counts and validation; S2 no fit/model selection','EVAE loss uses train-derived centres and normalized axial von Mises component','Teacher forcing target/family alignment; axial cosine angular loss; train and inference latent contract matched','Exact simulated-parent continuation targets on S1 only; both observed strip endpoints treated explicitly','Original seven-feature extractor with corrected tail/length weight indexing; background stats explicit','Equal raw proposal attempts, one realization, shared gates/scorer and geometry-only count budget; complete proposal ledger','Variable actual rectangular bounds; isotropic width normalization and explicit metric conversion to metres'],
        deviations=['Source class bodies retained but this is a documented corrected adapter, not exact historical reproduction','Training uses current implemented objectives with listed corrections, not a newly added Sinkhorn physics objective','Full neural arm is neural-only inferred proposals; no statistical fallback or unequal extra pool','KDE control is a family-conditioned Gaussian product-kernel mixture; not licensed ADFNE','Only five original spacings for B/E sensitivity; no broad architecture/loss search']))
    allmet=[];allobs=[];allpred=[];allledger=[];allbudget=[]
    for seed in SEEDS:
        seedall(seed);sd=HERE/f'seed_{seed}';sd.mkdir(exist_ok=True)
        models={};logs=[];selected={}
        if '--evaluate-only' in sys.argv or ('--resume' in sys.argv and all((sd/fn).exists() for fn in ['evae.pt','continuation_head.pt','transformer_latent.pt','transformer_no_latent.pt'])):
            models['evae']=legacy['DualLatentVAE'](sd=12,gd=20,N=128,yscale=STATS['yscale']).to(DEV)
            for key in ['transformer_latent','transformer_no_latent']:models[key]=trans['SeqFracTransformerV4']().to(DEV)
            for key in ['evae','transformer_latent','transformer_no_latent']:
                ck=torch.load(sd/(key+'.pt'),map_location=DEV,weights_only=True)
                check('reused_checkpoint_training_split:'+key,ck['protocol']['train_faces']==PROTOCOL['train_faces'])
                models[key].load_state_dict(ck['state_dict'],strict=True);models[key].eval()
            ck=torch.load(sd/'continuation_head.pt',map_location=DEV,weights_only=True);head=gen['EVAEContHead']().to(DEV);head.load_state_dict(ck['state_dict'],strict=True);head.eval()
            models['head']=(head,ck['X_mean'],ck['X_std'])
            gradient_probe=[]
            for latent in [False,True]:
                key='transformer_latent' if latent else 'transformer_no_latent'
                probe=sequence_example(train[0],models,latent)
                mdl=models[key];mdl.zero_grad(set_to_none=True)
                loss=seq_loss(mdl,seq_batch([probe]),latent);loss.backward()
                g=mdl.latent_proj.weight.grad;norm=float(g.norm()) if g is not None else None
                check('saved_model_latent_gradient_nonzero' if latent else 'saved_no_latent_model_disconnected',norm is not None and norm>1e-10 if latent else norm is None,norm)
                gradient_probe.append(dict(component=key,projection_gradient_norm=norm,loss=float(loss)))
                mdl.zero_grad(set_to_none=True)
            dump(f'seed_{seed}/gradient_probe.json',gradient_probe)
            # Preserve the first execution's predictions; correction is solely an
            # arm-parity bug fix, not fitting/tuning or selection on S2 metrics.
            import shutil
            archive=sd/'superseded_arm_budget_bug';archive.mkdir(exist_ok=True)
            for fn in ['metrics.csv','observations.csv','predictions.csv','proposal_ledger.csv.gz','count_budgets.csv']:
                src=sd/fn
                if src.exists() and not (archive/fn).exists():shutil.copy2(src,archive/fn)
        else:
            models['evae'],ll,selected['evae']=train_evae(train,val,STATS,seed);logs+=ll
            models['head'],ll,selected['head']=train_head(train,val,models['evae']);logs+=ll
            for latent in [False,True]:
                key='transformer_latent' if latent else 'transformer_no_latent'
                models[key],ll,selected[key]=train_transformer(train,val,models,latent);logs+=ll
            for key in ['evae','transformer_latent','transformer_no_latent']:
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in models[key].state_dict().items()},selected_epoch=selected[key],protocol=PROTOCOL),sd/(key+'.pt'))
            head,mu,ss=models['head'];torch.save(dict(state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},X_mean=mu.cpu(),X_std=ss.cpu(),selected_epoch=selected['head'],protocol=PROTOCOL),sd/'continuation_head.pt')
            pd.DataFrame(logs).to_csv(sd/'training_log.csv',index=False);dump(f'seed_{seed}/selected_epochs.json',selected)
        met,ob,pr,le,bu=evaluate(FACES,models,seed)
        for name,df in [('metrics',met),('observations',ob),('predictions',pr),('proposal_ledger',le),('count_budgets',bu)]:df.to_csv(sd/(name+'.csv.gz' if name=='proposal_ledger' else name+'.csv'),index=False)
        allmet.append(met);allobs.append(ob);allpred.append(pr);allledger.append(le);allbudget.append(bu)
        del models;torch.cuda.empty_cache()
    met=pd.concat(allmet);met.to_csv(HERE/'all_face_seed_metrics.csv',index=False)
    pd.concat(allobs).to_csv(HERE/'observation_audit.csv',index=False);pd.concat(allbudget).to_csv(HERE/'count_budget_audit.csv',index=False)
    le=pd.concat(allledger);le.groupby(['arm','layout','source','disposition']).size().rename('count').to_csv(HERE/'proposal_disposition_summary.csv')
    met.groupby(['layout','arm']).mean(numeric_only=True).to_csv(HERE/'section_arm_summary.csv')
    # Keep face as the spatial unit; seeds are repeated realizations, not walls.
    met.groupby(['face_id','layout','arm']).mean(numeric_only=True).to_csv(HERE/'face_mean_over_seeds.csv')
    for p,digest in json.loads((HERE/'input_hashes.json').read_text()).items():check('input_unchanged:'+p,sha(Path(p))==digest)
    protected=json.loads((STUDY/'FOCUSED_REVISION_PLAN_MANIFEST_2026-10-18.json').read_text())['protected_source_hashes']
    for p,digest in protected.items():check('manuscript_reference_unchanged:'+p,sha(Path(p))==digest)
    # One count budget per face/layout/seed, independent of model/scorer arm.
    bu=pd.concat(allbudget);check('arm_count_budget_parity',bool((bu.groupby(['face_id','layout','seed']).target_count.nunique()==1).all()))
    check('arm_raw_budget_parity',bool((bu.groupby(['face_id','layout','seed']).raw_budget.nunique()==1).all()))
    pd.DataFrame(CHECKS).to_csv(HERE/'structured_checks.csv',index=False)
    dump('RUN_COMPLETE.json',dict(status='completed',seeds=SEEDS,metric_rows=len(met),checks=len(CHECKS),checks_passed=all(r['passed'] for r in CHECKS),device=str(DEV),torch=torch.__version__,elapsed_seconds=time.time()-start,original_files_modified=False,manuscript_modified=False))
    print((HERE/'RUN_COMPLETE.json').read_text(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        pd.DataFrame(CHECKS).to_csv(HERE/'structured_checks.csv',index=False)
        dump('RUN_FAILURE.json',dict(error=repr(exc),time=time.time()))
        raise
