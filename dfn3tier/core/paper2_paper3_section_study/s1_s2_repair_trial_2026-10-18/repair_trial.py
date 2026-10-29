"""Versioned repair trial: original neural classes and three continuity stages.

The failed adapter is imported only for explicitly reused data/training/metric
utilities. Repaired behavior is declared below; no prior result is overwritten.
"""
from __future__ import annotations
import ast,contextlib,copy,hashlib,importlib.util,io,json,math,sys,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;OLD=STUDY/'s1_s2_original_revision_2026-10-18'
spec=importlib.util.spec_from_file_location('reused_failed_adapter',OLD/'run_revision.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.HERE=HERE
r.PROTOCOL=copy.deepcopy(r.PROTOCOL)
r.PROTOCOL.update(repair_trial=True,augmentation=['original','rot180'],continuity_reference_m=36.,anchor_tolerance_m=.002,
    anchor_angle_tolerance_deg=.25,raw_budget_rule='24 * min(128,ceil(1.5*target)) + 10*target, shared by all arms',
    sampler='original family/posterior/multi-candidate contract; axial angle correction; complete draw ledger',
    continuity='anchor association then original pre-count/post-selection/final merge; S1-only calibration and anchor veto',
    status='corrected trial; not exact numerical reproduction of historical submission')
r.PROTOCOL['arms']['G_mixed']=['evae','mixed_original',True]
warnings.filterwarnings('ignore',category=UserWarning)
STAGE_LOG=[];PROBES=[];LOADED_HASHES={}

class MaskedRectangularEVAE(r.legacy['DualLatentVAE']):
    """Same parameters/heads; masked attention and public per-example bounds."""
    def encode(self,C,L,A,mask):
        pm=~mask;attention_pm=pm.clone()
        empty=attention_pm.all(1)
        if empty.any():attention_pm[empty,0]=False
        sf=torch.nan_to_num(self.se(C),0.);se=torch.nan_to_num(self.st(sf,src_key_padding_mask=attention_pm),0.)
        sp=self._pool(se,self.sp,pm)
        gf=torch.nan_to_num(self.ge(torch.cat([L,A],-1)),0.);ge=torch.nan_to_num(self.gt(gf,src_key_padding_mask=attention_pm),0.)
        gp=self._pool(ge,self.gp,pm)
        return self.smu(sp),self.slv(sp).clamp(-6,6),self.gmu(gp),self.glv(gp).clamp(-6,6)
    def decode_centers_B(self,sz):
        B=sz.size(0);raw=self.cd(sz).view(B,self.N,2)
        ys=getattr(self,'batch_yscale',None)
        if ys is None:ys=torch.full((B,),self.yscale,device=sz.device)
        ys=torch.as_tensor(ys,device=sz.device).reshape(B,1,1)
        scale=torch.cat([torch.full_like(ys,.5),ys/2],dim=-1)
        base=torch.tanh(raw)*scale+scale
        shift=torch.tanh(self.cd_shift(sz)).view(B,1,2)*torch.cat([torch.full_like(ys,.35),ys*.4],dim=-1)
        c=base+shift
        return torch.cat([c[:,:,:1].clamp(0,1),torch.minimum(c[:,:,1:2].clamp_min(0),ys)],dim=-1)

r.legacy['DualLatentVAE']=MaskedRectangularEVAE
def forward_evae(model,b):
    model.batch_yscale=b['aspect']
    return model(b['C_in'],b['L_in'],b['A_in'],b['mask'])
r.forward_evae=forward_evae

# Reuse the existing explicit training loop, changing only its geometry contract.
tree=ast.parse((OLD/'run_revision.py').read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_evae')
class GeometryTrainingPatch(ast.NodeTransformer):
    def visit_FunctionDef(self,n):
        self.generic_visit(n)
        if n.name=='pack':
            i=next(i for i,x in enumerate(n.body) if isinstance(x,ast.Return))
            n.body.insert(i,ast.parse("arrays['aspect']=np.array([e['face']['ym'] for e in items],dtype=np.float32)").body[0])
        return n
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Name) and n.func.id=='mdl' and len(n.args)==4:
            return ast.copy_location(ast.Call(func=ast.Name(id='forward_evae',ctx=ast.Load()),args=[ast.Name(id='mdl',ctx=ast.Load()),ast.Name(id='b',ctx=ast.Load())],keywords=[]),n)
        return n
node=GeometryTrainingPatch().visit(node);source=ast.unparse(ast.fix_missing_locations(node))
(HERE/'geometry_aware_training_definition.py').write_text(source,encoding='utf-8')
exec(compile(source,str(HERE/'geometry_aware_training_definition.py'),'exec'),r.__dict__)

def examples(faces,role):
    out=[]
    for face in faces.values():
        if face['role']!=role:continue
        for layout in ['centre','3m','5m','7m','9m','11m']:
            for aug in ['orig','rot180'] if role=='train' else ['orig']:
                a=r.augment(face,aug);st=r.strips(layout,face['w']);obs,parents=r.observe(a,st)
                out.append(dict(face=face,layout=layout,arr=a,obs=obs,parents=parents,strips=st))
    return out
r.examples=examples

def contains(clip,line,w,tol=.002):
    a,b=r.ends(np.asarray(clip)[None]);p,q=r.ends(np.asarray(line)[None]);v=q[0]-p[0];ln=float(np.linalg.norm(v))
    if ln<1e-12:return False
    u=v/ln;d=np.vstack([a[0]-p[0],b[0]-p[0]])
    perp=np.max(abs(u[0]*d[:,1]-u[1]*d[:,0]))*w;t=d@u
    return perp<=tol and t.min()*w>=-tol and (t.max()-ln)*w<=tol
def preservation(obs,arr,w):
    if not len(obs):return 1.
    return sum(any(contains(c,p,w) for p in arr) for c in obs)/len(obs)
def hull(rows,w):
    rows=np.asarray(rows);p,q=r.ends(rows)
    ref=rows[np.argmax(rows[:,2])];u=np.array([np.cos(ref[3]),np.sin(ref[3])]);v=np.array([-u[1],u[0]])
    points=np.vstack([p,q]);along=points@u;perp=float((points@v).mean())
    a=u*along.min()+v*perp;b=u*along.max()+v*perp
    line=r.lines(a[None],b[None])[0]
    return line if all(contains(c,line,w) for c in rows) else None

def anchor_groups(obs,face,st):
    """Link compatible hits on consecutive scanlines using observed geometry only."""
    n=len(obs)
    if not n:return []
    w=face['w'];p,q=r.ends(obs);centres=np.array([(a+b)/2 for a,b in st])
    strip_ids=np.argmin(abs(obs[:,0,None]-centres[None,:]),axis=1)
    groups=[[i] for i in range(n)];angle_tol=np.radians(.25)
    pairs=[]
    for i in range(n):
        for j in range(i):
            if abs(int(strip_ids[i])-int(strip_ids[j]))!=1:continue
            da=abs(float(obs[i,3]-obs[j,3]));da=min(da,np.pi-da)
            if da>angle_tol:continue
            u=np.array([np.cos(obs[j,3]),np.sin(obs[j,3])]);perp=abs(np.cross(u,obs[i,:2]-obs[j,:2]))*w
            if perp<=.02:pairs.append((perp,i,j))
    for _,i,j in sorted(pairs):
        gi=next(k for k,g in enumerate(groups) if i in g);gj=next(k for k,g in enumerate(groups) if j in g)
        if gi==gj:continue
        joined=groups[gi]+groups[gj]
        if len(set(strip_ids[joined]))!=len(joined):continue
        h=hull(obs[joined],w)
        if h is None:continue
        fam=int(r.assign(h[None],r.STATS['fc'])[0]);cap=float(np.quantile(r.STATS['pools'][fam]['lengths_m'],.995))
        if h[2]*w>cap:continue
        groups[gi]=joined;groups.pop(gj)
    return groups

old_features=r.continuation_features
def continuation_features(obs,idx,side,face):
    feat,a,u,bound=old_features(obs,idx,side,face)
    feat[6]=float(np.sign(u[0]));feat[-1]=float(np.sum(np.linalg.norm(obs[:,:2]-obs[idx,:2],axis=1)*face['w']<3.6)-1)
    return feat,a,u,bound
r.continuation_features=continuation_features

def group_outer_observations(obs,members):
    ref=obs[members[np.argmax(obs[members,2])]];u=np.array([np.cos(ref[3]),np.sin(ref[3])]);p,q=r.ends(obs)
    out=[]
    for side in [0,1]:
        coords=np.array([np.dot((p if side==0 else q)[i],u) for i in members])
        out.append((members[int(np.argmin(coords) if side==0 else np.argmax(coords))],side))
    return out
def train_head(train,val,evae):
    def make(exs):
        X=[];Y=[]
        for e in exs:
            lat=torch.cat(r.tensor_encode(evae,e['obs']),-1).cpu().numpy()[0]
            for members in anchor_groups(e['obs'],e['face'],e['strips']):
                for i,side in group_outer_observations(e['obs'],members):
                    feat,a,u,bound=continuation_features(e['obs'],i,side,e['face'])
                    if not any(abs(a[0]-edge)<1e-6 for strip in e['strips'] for edge in strip):continue
                    tp,tq=r.ends(e['arr'][e['parents'][i]:e['parents'][i]+1]);extension=max(float(np.dot(tp[0]-a,u)),float(np.dot(tq[0]-a,u)),0.)
                    if bound>1e-8:X.append(np.r_[lat,feat]);Y.append(math.log(max(extension,1e-6)))
        return torch.tensor(np.asarray(X),device=r.DEV),torch.tensor(Y,device=r.DEV)[:,None]
    X,Y=make(train);V,Z=make(val);mu=X.mean(0);sd=X.std(0).clamp_min(1e-3);X=(X-mu)/sd;V=(V-mu)/sd
    mdl=r.gen['EVAEContHead']().to(r.DEV);opt=torch.optim.Adam(mdl.parameters(),lr=1e-3,weight_decay=1e-5);best=math.inf;logs=[]
    for ep in range(1,51):
        mdl.train();ls=[]
        for ii in torch.randperm(len(X),device=r.DEV).split(256):
            m,l=mdl(X[ii]);loss=(.5*((Y[ii]-m)/l.exp())**2+l).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(mdl.parameters(),1.);opt.step();ls.append(float(loss))
        mdl.eval()
        with torch.no_grad():m,l=mdl(V);vl=float((.5*((Z-m)/l.exp())**2+l).mean())
        logs.append(dict(component='head',epoch=ep,train=float(np.mean(ls)),validation=vl))
        if vl<best:best=vl;state={k:v.detach().cpu().clone() for k,v in mdl.state_dict().items()};selected=ep
    mdl.load_state_dict(state);mdl.eval();return (mdl,mu,sd),logs,selected
r.train_head=train_head

def continuity(arr,face,obs,stage):
    arr=np.asarray(arr,np.float32).reshape(-1,4)
    if len(arr)<2:return arr
    factor=face['w']/36.;canonical=arr.copy();canonical[:,:3]*=factor
    fam=r.assign(arr,r.STATS['fc']);tiers=np.array([2 if any(contains(o,a,face['w']) for o in obs) else 3 for a in arr])
    cap={f:float(np.quantile(r.STATS['pools'][f]['lengths_m'],.95))/36 for f in range(3)}
    with contextlib.redirect_stdout(io.StringIO()):
        result,_,_,stats=CONT['apply_continuity_rules'](canonical,fam,angle_tol_deg=8,perp_tol=.01,gap_frac=.6,gap_floor=.04,check_t_junctions=True,max_len_p95=cap,tier_ids=tiers)
    result[:,:3]/=factor
    veto=preservation(obs,result,face['w'])<1.
    if veto:result=arr.copy()
    STAGE_LOG.append(dict(face_id=face['id'],stage=stage,before=len(arr),after=len(result),anchor_veto=veto,raw_chains=stats.get('n_chains',0),narrow_merges=stats.get('n_narrow_merged',0)))
    return result

def indicated(obs,face,st,mode,models,rng):
    if not len(obs):return np.zeros((0,4),np.float32)
    lat=torch.cat(r.tensor_encode(models['evae'],obs),-1);out=[]
    for members in anchor_groups(obs,face,st):
        support=hull(obs[members],face['w'])
        if support is None:
            out.extend(obs[members]);continue
        p,q=r.ends(support[None]);p=p[0].copy();q=q[0].copy()
        for i,side in group_outer_observations(obs,members):
            feat,t,u,bound=continuation_features(obs,i,side,face)
            if not any(abs(t[0]-edge)<1e-6 for strip in st for edge in strip):continue
            fam=int(r.assign(obs[i:i+1],r.STATS['fc'])[0]);pool=r.STATS['pools'][fam]['lengths_m']
            if mode=='evae':
                head,mu,sd=models['head'];ft=torch.tensor(feat,device=r.DEV)[None]
                with torch.no_grad():m,l=head((torch.cat([lat,ft],-1)-mu)/sd)
                extension=math.exp(min(float(m)+.67*max(float(l.exp()),.05),2.))
                extension=max(extension,1.08/face['w'])
            else:extension=float(pool[int(rng.integers(len(pool)))])/face['w']
            extension=min(extension,bound,float(np.quantile(pool,.95))/face['w'])
            if abs(u[0])>1e-9:
                for strip in st:
                    for edge in strip:
                        distance=(edge-t[0])/u[0]
                        if distance>1e-6:extension=min(extension,max(0,distance-.002/face['w']))
            if side==0:p=t+extension*u
            else:q=t+extension*u
        candidate=r.lines(p[None],q[None])[0]
        out.append(candidate if all(contains(o,candidate,face['w']) for o in obs[members]) else support)
    out=continuity(np.asarray(out,np.float32),face,obs,'pre_count')
    r.check('continuation_preserves_all_clips',preservation(obs,out,face['w'])==1.)
    return out
r.indicated=indicated

def axial_kappa(angles):
    R=abs(np.exp(2j*np.asarray(angles)).mean());return float(np.clip(R*(2-R*R)/max(1-R*R,1e-5),.1,100))
def posterior_draws(position,loglength,angle,family_logits,face,rng,n):
    probs=np.exp(family_logits-np.max(family_logits));probs/=probs.sum()
    families=rng.choice(4,n,p=probs);rows=[];scores=[]
    cxstd=max(TRAIN_X_STD*.2,.01);cystd=max(TRAIN_Y_STD*.2,.003)
    for fam in families:
        pool=r.STATS['pools'][int(fam)];prior_logs=np.log(pool['lengths_m']/face['w']);prior_mu=float(prior_logs.mean());prior_sd=max(float(prior_logs.std()),.08)
        model_sd=max(prior_sd*.7,.06);pmodel=1/model_sd**2;pprior=1/prior_sd**2
        mu=(loglength*pmodel+prior_mu*pprior)/(pmodel+pprior);sd=math.sqrt(1/(pmodel+pprior));L=math.exp(float(rng.normal(mu,sd)))
        if fam<3:
            kp=axial_kappa(pool['angles']);vec=.5*kp*np.exp(2j*angle)+kp*np.exp(2j*r.STATS['fc'][fam]);A=(rng.vonmises(np.angle(vec),abs(vec))/2)%np.pi
        else:A=(rng.vonmises(2*angle,.3)/2)%np.pi
        cx=float(np.clip(rng.normal(position[0],cxstd),0,1));cy=float(np.clip(rng.normal(position[1],cystd),0,face['ym']))
        d=abs(float(A-angle));d=min(d,np.pi-d)
        score=((cx-position[0])/cxstd)**2+((cy-position[1])/cystd)**2+((math.log(L)-mu)/max(sd,.08))**2+(d/np.radians(12))**2-math.log(max(probs[fam],1e-8))
        rows.append([cx,cy,L,A]);scores.append(score)
    return np.asarray(rows,np.float32),np.asarray(scores),families

def sample_neural(mdl,context,face,budget,latent,evae,rng,st):
    ctx=context[np.argsort(context[:,0])] if len(context) else context;ct=r.common['tok_array'](ctx)
    if not len(ct):ct=np.zeros((1,5),np.float32)
    c=torch.tensor(ct,device=r.DEV)[None];cm=torch.zeros(1,len(ct),device=r.DEV,dtype=torch.bool)
    sz,gz=r.tensor_encode(evae,context);seq=torch.zeros(1,1,5,device=r.DEV);rows=[];eligible=[];sampled=[];steps=[]
    with torch.no_grad():
        step=0
        while len(rows)<budget:
            if seq.size(1)>128:seq=torch.zeros(1,1,5,device=r.DEV)
            p,l,a,f=mdl(c,cm,seq,torch.zeros(1,seq.size(1),device=r.DEV,dtype=torch.bool),sz if latent else None,gz if latent else None)
            pos=p[0,-1].cpu().numpy();logL=float(l[0,-1,0]);angle=float(torch.atan2(a[0,-1,0],a[0,-1,1]))%np.pi;fl=f[0,-1].cpu().numpy()
            candidates,scores,fids=posterior_draws(pos,logL,angle,fl,face,rng,min(24,budget-len(rows)))
            valid=[i for i,row in enumerate(candidates) if r.candidate_gate(row,face,st) is None]
            best=min(valid,key=lambda i:scores[i]) if valid else None
            rows.extend(candidates);eligible.extend([i==best for i in range(len(candidates))]);sampled.extend(fids);steps.extend([step]*len(candidates))
            if best is not None:new=candidates[best]
            else:new=np.array([np.clip(pos[0],0,1),np.clip(pos[1],0,face['ym']),math.exp(float(np.clip(logL,-12,0))),angle],np.float32)
            seq=torch.cat([seq,torch.tensor(r.common['tok_array'](new[None]),device=r.DEV)[None]],dim=1);step+=1
    return np.asarray(rows,np.float32),np.asarray(eligible),np.asarray(sampled),np.asarray(steps)

def quota_allocation(total,weights):
    if total<=0:return np.zeros(4,int)
    x=np.asarray(weights,float);x=x/x.sum() if x.sum()>0 else np.ones(4)/4
    values=x*total;ints=np.floor(values).astype(int)
    for j in np.argsort(-(values-ints))[:total-ints.sum()]:ints[j]+=1
    return ints
def duplicate_or_spacing(row,placed,fam,placed_fam,face):
    if not len(placed):return None
    w=face['w']
    for old in placed:
        da=abs(float(row[3]-old[3]));da=min(da,np.pi-da)
        if da<np.radians(.5) and (contains(row,old,w,tol=.02) or contains(old,row,w,tol=.02)):return 'duplicate_or_contained'
    same=[old for old,f in zip(placed,placed_fam) if f==fam]
    if same:
        distance=min(float(np.linalg.norm(row[:2]-old[:2])*w) for old in same)
        if distance<SPACING_METRES[fam]*.25:return 'same_family_spacing'
    return None

def reconstruct(face,obs,st,arm,models,seed):
    mode,proposal,scoring=r.PROTOCOL['arms'][arm];rng=np.random.default_rng(seed)
    before=len(STAGE_LOG);fixed=indicated(obs,face,st,mode,models,rng)
    target=max(1,int(round(r.STATS['rate']*face['w']*face['h'])))
    nsteps=min(128,math.ceil(1.5*target));neural_draws=24*nsteps;stat_draws=10*target;budget=neural_draws+stat_draws
    if proposal in ['mixed_original','transformer_latent','transformer_no_latent']:
        use=proposal!='transformer_no_latent';key='transformer_latent' if use else proposal
        n=neural_draws if proposal=='mixed_original' else budget
        raw,eligible,sampled,step=sample_neural(models[key],fixed,face,n,use,models['evae'],np.random.default_rng(seed+100000),st)
        source=[key]*len(raw)
        if proposal=='mixed_original':
            stat=r.statistical_proposals(face,stat_draws,'stat',rng);raw=np.vstack([raw,stat]);eligible=np.r_[eligible,np.ones(stat_draws,bool)];sampled=np.r_[sampled,r.assign(stat,r.STATS['fc'])];step=np.r_[step,np.full(stat_draws,-1)];source+=['stat']*stat_draws
    else:
        raw=r.statistical_proposals(face,budget,proposal,rng);eligible=np.ones(budget,bool);sampled=r.assign(raw,r.STATS['fc']);step=np.full(budget,-1);source=[proposal]*budget
    fids=r.assign(raw,r.STATS['fc']);knownfam=r.assign(fixed,r.STATS['fc']);missing=max(0,target-len(fixed))
    deficits=np.maximum(0,r.STATS['shares']*max(target,len(fixed))-np.bincount(knownfam,minlength=4));need=quota_allocation(missing,deficits);pc=np.zeros(4,int)
    logs=[];available=[]
    for i,row in enumerate(raw):
        reason=r.candidate_gate(row,face,st)
        if reason is None and not eligible[i]:reason='posterior_not_selected'
        logs.append(dict(candidate_id=i,source=source[i],cx=float(row[0]),cy=float(row[1]),length=float(row[2]),angle=float(row[3]),family=int(fids[i]),sampled_family=int(sampled[i]),decoder_step=int(step[i]),disposition=reason or 'budget_not_selected',score_evaluations=0))
        if reason is None:available.append(i)
    selected=[];sfams=[];gs=copy.deepcopy(r.STATS['gs']);gs['spacing_floor']={f:SPACING_METRES[f]/face['w'] for f in range(4)}
    weights=np.array([gs['scorer_weights'][k] for k in ['w_spacing','w_xbin','w_fam_deficit','w_cm_dev','w_f2_tail','w_len_quality','w_intersection']])
    for _ in range(missing):
        if not available:break
        ids=rng.choice(available,min(len(available),200),replace=False).tolist() if scoring else list(available)
        best=None;value=-math.inf;placed=list(fixed)+selected;pf=list(knownfam)+sfams
        h=np.histogram(np.asarray(placed)[:,0] if len(placed) else [],bins=np.linspace(0,1,6))[0];props=h/max(h.sum(),1)
        for i in ids:
            row=raw[i];f=int(fids[i])
            reason='family_quota' if pc[f]>=need[f] else duplicate_or_spacing(row,placed,f,pf,face)
            if reason:logs[i]['disposition']=reason;available.remove(i);continue
            if scoring:
                same=[a for a,b in zip(placed,pf) if b==f]
                feat=r.scorer['extract_features'](*map(float,row),f,same,placed,gs['spacing_floor'],gs['xbin_ref'],gs['xbin_std'],props,dict(enumerate(need)),dict(enumerate(pc)),gs['count_std'],gs['centre_mass_ref'],list(fixed),selected,list(knownfam),sfams,r.STATS['logmu'],r.STATS['logsig'],gs['fam_length_ref'],gs['intersection_std'],segments_intersect_fn=r.intersect)
                score=float(weights@feat);logs[i]['score_evaluations']+=1
            else:score=0.
            if score>value:best=i;value=score
            if not scoring:break
        if best is None:
            if not scoring and not available:break
            continue
        available.remove(best);selected.append(raw[best]);sfams.append(int(fids[best]));pc[int(fids[best])]+=1;logs[best]['disposition']='accepted';logs[best]['selection_score']=value
    before_merge=np.vstack([fixed,np.asarray(selected).reshape(-1,4)])
    after=continuity(before_merge,face,obs,'post_selection');final=continuity(after,face,obs,'final_safety')
    anchored=np.array([any(contains(o,a,face['w']) for o in obs) for a in final],bool)
    fixedfinal=final[anchored];inferred=final[~anchored]
    r.check('final_preserves_measured_clips',preservation(obs,final,face['w'])==1.)
    r.check('complete_raw_attempt_budget',len(raw)==budget)
    for item in STAGE_LOG[before:]:item.update(seed=seed,arm=arm)
    return final,inferred,fixedfinal,pd.DataFrame(logs),dict(target_count=target,indicated_parents=len(fixed),final_anchored_parents=len(fixedfinal),observed_clips=len(obs),raw_budget=budget,generated=len(final),shortfall=max(0,target-len(final)),accepted_before_continuity=len(selected),continuity_count_change=len(before_merge)-len(final),posterior_selected=int(eligible.sum()))
r.reconstruct=reconstruct

def gates_and_contract_tests(faces):
    model=MaskedRectangularEVAE(N=128,yscale=.2).to(r.DEV);model.eval()
    f=next(f for f in faces.values() if f['role']=='train');obs,_=r.observe(f['arr'],r.strips('centre',f['w']));obs=obs[:max(1,min(3,len(obs)))]
    vectors=[]
    for n in [len(obs),128,230]:
        x=torch.zeros(1,n,4,device=r.DEV,requires_grad=True)
        with torch.no_grad():x[0,:len(obs)]=torch.tensor(obs,device=r.DEV)
        mask=torch.zeros(1,n,device=r.DEV,dtype=torch.bool);mask[:,:len(obs)]=True
        sm,_,gm,_=model.encode(x[:,:,:2],x[:,:,2:3],x[:,:,3:4],mask);z=torch.cat([sm,gm],-1);vectors.append(z.detach());z.sum().backward()
        error=float(x.grad[:,len(obs):].abs().max()) if n>len(obs) else 0.
        r.check('padding_has_zero_gradient:'+str(n),error<1e-7,error)
    delta=max(float((v-vectors[0]).abs().max()) for v in vectors)
    r.check('padding_invariance_before_training',delta<1e-5,delta)
    model.batch_yscale=torch.tensor([.08,.22],device=r.DEV);centres=model.decode_centers_B(torch.zeros(2,12,device=r.DEV))
    r.check('per_rectangle_decoder_support',bool((centres[:,:,1]>=0).all() and (centres[0,:,1]<=.08).all() and (centres[1,:,1]<=.22).all()))
    # Controlled logits must causally change sampled family, not just labels.
    a,_,fa=posterior_draws([.5,.06],-3,.9,np.array([50.,-50.,-50.,-50.]),f,np.random.default_rng(3),96)
    b,_,fb=posterior_draws([.5,.06],-3,.9,np.array([-50.,50.,-50.,-50.]),f,np.random.default_rng(3),96)
    r.check('family_head_controls_sampling',bool((fa==0).all() and (fb==1).all()))
    r.check('posterior_has_length_uncertainty',float(np.std(a[:,2]))>1e-6)
    # Audit association using S1 simulator IDs only as test assertions, never inputs.
    association=[]
    for ff in faces.values():
        if ff['role']=='test':continue
        for layout in ['centre','3m','5m','7m','9m','11m']:
            st=r.strips(layout,ff['w']);o,p=r.observe(ff['arr'],st);groups=anchor_groups(o,ff,st)
            wrong=sum(len(set(p[g]))>1 for g in groups)
            association.append(dict(face_id=ff['id'],layout=layout,clips=len(o),observed_truth_parents=len(set(p)),groups=len(groups),groups_combining_different_manual_parents=wrong))
            r.check('S1_anchor_association_no_cross_parent:'+ff['id']+layout,wrong==0,wrong)
    pd.DataFrame(association).to_csv(HERE/'S1_anchor_association_checks.csv',index=False)
    # Exercise all three original continuity stages on existing S1 observations.
    st=r.strips('5m',f['w']);o,_=r.observe(f['arr'],st);groups=anchor_groups(o,f,st)
    base=np.asarray([hull(o[g],f['w']) for g in groups],np.float32)
    for stage in ['pre_count','post_selection','final_safety']:
        base=continuity(base,f,o,stage);r.check('stage_anchor_invariance:'+stage,preservation(o,base,f['w'])==1.)
    # Physical gate: a nearby same-family parallel segment must be rejected.
    old=f['arr'][0];candidate=old.copy();candidate[0]+=.00001
    family=int(r.assign(old[None],r.STATS['fc'])[0])
    r.check('near_duplicate_rejected',duplicate_or_spacing(candidate,[old],family,[family],f) is not None)
    r.dump('PRETRAIN_CONTRACT_CHECKS.json',dict(padding_max_absolute_difference=delta,checks=len(r.CHECKS),all_pass=all(x['passed'] for x in r.CHECKS),new_neural_parameters=0,independent_geometry_tests=True))

def evaluate(faces,models,seed,role='test'):
    metrics=[];obsrows=[];preds=[];ledgers=[];budgets=[]
    for face in faces.values():
        if face['role']!=role:continue
        for layout in ['centre','3m','5m','7m','9m','11m'] if role=='test' else ['centre','5m']:
            st=r.strips(layout,face['w']);obs,parents=r.observe(face['arr'],st)
            groups=anchor_groups(obs,face,st)
            obsrows.append(dict(seed=seed,face_id=face['id'],role=role,layout=layout,truth=len(face['arr']),observed_unique=len(set(parents)),observed_fraction=len(set(parents))/len(face['arr']),clips=len(obs),observed_anchor_groups=len(groups),cross_parent_links=sum(len(set(parents[g]))>1 for g in groups)))
            arms=list(r.PROTOCOL['arms']) if role=='test' and layout in ['centre','5m'] else ['B','E','G_mixed']
            for arm in arms:
                final,inf,fixed,ledger,budget=reconstruct(face,obs,st,arm,models,seed)
                meta=dict(seed=seed,face_id=face['id'],layout=layout,arm=arm,role=role)
                met=r.metrics(face['arr'],final,face['w'],face['h'],obs);met['measured_preservation_physical_2mm']=preservation(obs,final,face['w'])
                target=face['arr'][~np.isin(np.arange(len(face['arr'])),np.unique(parents))]
                met['inferred_axial_w1_deg']=r.axial_w1(target[:,3],inf[:,3]);met['inferred_length_w1_m']=r.wasserstein_distance(target[:,2]*face['w'],inf[:,2]*face['w']) if len(target) and len(inf) else math.nan
                metrics.append(dict(meta,**met));budgets.append(dict(meta,**budget));ledgers.append(ledger.assign(**meta))
                for tier,a in [('fixed_measured_indicated',fixed),('inferred',inf),('measured_clips',obs)]:preds.append(pd.DataFrame(a,columns=['cx_normalized','cy_normalized','length_normalized','angle_rad']).assign(tier=tier,width_m=face['w'],**meta))
        print(f'Seed {seed} {role} checked: {face["id"]}',flush=True)
    return pd.DataFrame(metrics),pd.DataFrame(obsrows),pd.concat(preds),pd.concat(ledgers),pd.DataFrame(budgets)

def main():
    global CONT,SPACING_METRES,TRAIN_X_STD,TRAIN_Y_STD
    if (HERE/'RUN_COMPLETE.json').exists():raise RuntimeError('Preserve completed repair trial')
    torch.set_num_threads(2);start=time.time();r.dump('PROTOCOL.json',r.PROTOCOL)
    for p in [OLD/'run_revision.py',Path(__file__),r.LEGACY/'model.py',r.EXP/'hybrid_v6.py',r.EXP/'transformer_v4.py',r.GEN/'run_final_integrated.py']:LOADED_HASHES[str(p)]=r.sha(p)
    r.FACES=r.prepare();r.STATS=r.fit_stats(r.FACES)
    trainingfaces=[f for f in r.FACES.values() if f['role']=='train'];training=np.vstack([f['arr'] for f in trainingfaces]);TRAIN_X_STD=float(training[:,0].std());TRAIN_Y_STD=float(training[:,1].std())
    canonical=[]
    for f in trainingfaces:
        a=f['arr'].copy();a[:,:3]*=f['w']/36.;canonical.append(a)
    CONT=r.definitions(r.EXP/'hybrid_v6.py',dict(load_panels_raw=lambda:canonical,get_physics_stats=lambda:{'fam_centers_rad':r.STATS['fc']},assign_family_ids=r.assign,segments_intersect=r.intersect,FAM_ASSIGN_THRESHOLD_DEG=20.),names=['apply_continuity_rules'])
    # Same source statistic in fixed physical reference units.
    with contextlib.redirect_stdout(io.StringIO()):physical_gs=r.geo['_train_geology_stats'](canonical,r.STATS['fc'])
    SPACING_METRES={f:float(physical_gs['spacing_floor'][f])*36 for f in range(3)};SPACING_METRES[3]=.008*36
    r.dump('PHYSICAL_CALIBRATION.json',dict(spacing_floor_metres=SPACING_METRES,continuity_reference_m=36.,reference_perp_tolerance_m=.36,reference_gap_floor_m=1.44,minimum_neural_extension_m=1.08,anchor_preservation_tolerance_m=.002,all_statistics_fit_on_S1_training=True))
    gates_and_contract_tests(r.FACES)
    train=examples(r.FACES,'train');val=examples(r.FACES,'validation')
    r.dump('TRAINING_DATASET.json',dict(training_examples=len(train),validation_examples=len(val),training_faces=[f['id'] for f in trainingfaces],validation_faces=r.PROTOCOL['validation_faces'],test_used_to_tune=False))
    allmetrics=[];allobs=[];allbudgets=[]
    for seed in r.SEEDS:
        r.seedall(seed);sd=HERE/f'seed_{seed}';sd.mkdir(exist_ok=True);models={};logs=[];epochs={}
        files=['evae.pt','continuation_head.pt','transformer_latent.pt','transformer_no_latent.pt']
        if '--resume' in sys.argv and all((sd/fn).exists() for fn in files):
            models['evae']=MaskedRectangularEVAE(N=128,yscale=r.STATS['yscale']).to(r.DEV)
            for key in ['transformer_latent','transformer_no_latent']:models[key]=r.trans['SeqFracTransformerV4']().to(r.DEV)
            for key in ['evae','transformer_latent','transformer_no_latent']:
                ck=torch.load(sd/(key+'.pt'),map_location=r.DEV,weights_only=True);models[key].load_state_dict(ck['state_dict'],strict=True);models[key].eval()
            ck=torch.load(sd/'continuation_head.pt',map_location=r.DEV,weights_only=True);head=r.gen['EVAEContHead']().to(r.DEV);head.load_state_dict(ck['state_dict'],strict=True);head.eval();models['head']=(head,ck['X_mean'],ck['X_std'])
            print('Reused S1-only trained weights after validation-time sampler tensor-shape correction:',seed,flush=True)
        else:
            models['evae'],ll,epochs['evae']=r.train_evae(train,val,r.STATS,seed);logs+=ll
            models['head'],ll,epochs['head']=train_head(train,val,models['evae']);logs+=ll
            for latent in [False,True]:
                key='transformer_latent' if latent else 'transformer_no_latent';models[key],ll,epochs[key]=r.train_transformer(train,val,models,latent);logs+=ll
            for key in ['evae','transformer_latent','transformer_no_latent']:torch.save(dict(state_dict={k:v.detach().cpu() for k,v in models[key].state_dict().items()},protocol=r.PROTOCOL,selected_epoch=epochs[key]),sd/(key+'.pt'))
            head,mu,ss=models['head'];torch.save(dict(state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},X_mean=mu.cpu(),X_std=ss.cpu(),protocol=r.PROTOCOL,selected_epoch=epochs['head']),sd/'continuation_head.pt')
            pd.DataFrame(logs).to_csv(sd/'training_log.csv',index=False);r.dump(f'seed_{seed}/selected_epochs.json',epochs)
        # Check a trained encoder as well, not only random initialization.
        o=train[0]['obs'];z=[]
        for n in [len(o),max(230,len(o))]:
            x=torch.zeros(1,n,4,device=r.DEV);x[0,:len(o)]=torch.tensor(o,device=r.DEV);mk=torch.zeros(1,n,device=r.DEV,dtype=torch.bool);mk[:,:len(o)]=True
            with torch.no_grad():sm,_,gm,_=models['evae'].encode(x[:,:,:2],x[:,:,2:3],x[:,:,3:4],mk)
            z.append(torch.cat([sm,gm],-1))
        delta=float((z[0]-z[1]).abs().max());r.check('trained_encoder_padding_invariant:'+str(seed),delta<1e-5,delta);PROBES.append(dict(seed=seed,padding_max_difference=delta))
        for role in ['validation','test']:
            met,obs,pred,ledger,budget=evaluate(r.FACES,models,seed,role)
            for name,df in [('metrics',met),('observations',obs),('predictions',pred),('proposal_ledger',ledger),('count_budgets',budget)]:df.to_csv(sd/(role+'_'+name+('.csv.gz' if name=='proposal_ledger' else '.csv')),index=False)
            if role=='test':allmetrics.append(met);allobs.append(obs);allbudgets.append(budget)
        del models;torch.cuda.empty_cache()
    met=pd.concat(allmetrics);obs=pd.concat(allobs);bu=pd.concat(allbudgets)
    met.to_csv(HERE/'all_face_seed_metrics.csv',index=False);obs.to_csv(HERE/'observation_audit.csv',index=False);bu.to_csv(HERE/'count_budget_audit.csv',index=False)
    r.check('shared_raw_draw_ceiling',bu.groupby(['seed','face_id','layout']).raw_budget.nunique().eq(1).all())
    r.check('all_test_measurements_preserved_2mm',met.measured_preservation_physical_2mm.eq(1).all())
    r.check('test_association_no_cross_parent_links',obs.cross_parent_links.eq(0).all())
    met.groupby(['layout','arm']).mean(numeric_only=True).to_csv(HERE/'section_arm_summary.csv')
    pd.DataFrame(STAGE_LOG).to_csv(HERE/'continuity_stage_log.csv',index=False);pd.DataFrame(PROBES).to_csv(HERE/'trained_padding_probes.csv',index=False)
    for p,digest in LOADED_HASHES.items():r.check('source_unchanged:'+p,r.sha(Path(p))==digest)
    for p,digest in json.loads((HERE/'input_hashes.json').read_text()).items():r.check('input_unchanged:'+p,r.sha(Path(p))==digest)
    r.dump('source_hashes.json',LOADED_HASHES);pd.DataFrame(r.CHECKS).to_csv(HERE/'structured_checks.csv',index=False)
    r.dump('RUN_COMPLETE.json',dict(status='repair_trial_complete',metric_rows=len(met),seeds=r.SEEDS,checks=len(r.CHECKS),checks_passed=all(x['passed'] for x in r.CHECKS),elapsed_seconds=time.time()-start,original_artifacts_modified=False,exact_historical_reproduction=False))
    print((HERE/'RUN_COMPLETE.json').read_text(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        pd.DataFrame(r.CHECKS).to_csv(HERE/'structured_checks.csv',index=False);r.dump('RUN_FAILURE.json',dict(error=repr(exc),elapsed_time=time.time()))
        raise
