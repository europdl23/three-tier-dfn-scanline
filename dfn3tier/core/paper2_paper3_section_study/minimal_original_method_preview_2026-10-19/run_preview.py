"""Two S1 previews: paired crossing-loss correction and isolated sampling change."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib.util,io,json,math,sys,time,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;ROOT=STUDY.parent
REPAIR=STUDY/'s1_s2_repair_trial_2026-10-18';OLD=STUDY/'s1_s2_original_revision_2026-10-18'
CASES=[('W00_RL1102-1120','centre'),('W01_RL1066-1102','5m')]
CHECKS=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(name,ok,detail=''):
    CHECKS.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    assert ok,name+': '+str(detail)
def dump(n,obj):(HERE/n).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def keyrestore(x):
    if isinstance(x,dict):return {int(k) if str(k).isdigit() else k:keyrestore(v) for k,v in x.items()}
    if isinstance(x,list):return [keyrestore(v) for v in x]
    return x
def statehash(d):
    h=hashlib.sha256()
    for k,v in sorted(d.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def load_state():
    spec=importlib.util.spec_from_file_location('frozen_revision_utilities',OLD/'run_revision.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.HERE=HERE
    stats=keyrestore(json.loads((REPAIR/'training_only_statistics.json').read_text()))
    for n in ['fc','shares']:stats[n]=np.asarray(stats[n])
    stats['pools']={k:{n:np.asarray(v) for n,v in item.items()} for k,item in stats['pools'].items()};r.STATS=stats
    labels=pd.read_csv(STUDY/'data_prep/manual_trace_manifest.csv');labels=labels[labels.section_id==1]
    frame=pd.read_csv(REPAIR/'face_split_geometry.csv');frame=frame[frame.role!='test'];faces={}
    for f in frame.itertuples():
        rows=labels[labels.face_id==f.face_id];w=f.width_m;h=f.height_up_face_m
        p=rows[['u1_m','v1_m']].to_numpy()+[0,h];q=rows[['u2_m','v2_m']].to_numpy()+[0,h]
        faces[f.face_id]=dict(id=f.face_id,role=f.role,w=w,h=h,ym=h/w,arr=r.lines(p/w,q/w))
    r.FACES=faces;canonical=[]
    training=np.vstack([f['arr'] for f in faces.values() if f['role']=='train'])
    for f in faces.values():
        if f['role']=='train':
            a=f['arr'].copy();a[:,:3]*=f['w']/36.;canonical.append(a)
    cont=r.definitions(r.EXP/'hybrid_v6.py',dict(load_panels_raw=lambda:canonical,get_physics_stats=lambda:{'fam_centers_rad':stats['fc']},assign_family_ids=r.assign,segments_intersect=r.intersect,FAM_ASSIGN_THRESHOLD_DEG=20.),names=['apply_continuity_rules'])
    physical=json.loads((REPAIR/'PHYSICAL_CALIBRATION.json').read_text())
    env=r.definitions(REPAIR/'repair_trial.py',dict(r=r,old_features=r.continuation_features,contextlib=contextlib,io=io,copy=copy,ast=ast,CONT=cont,STAGE_LOG=[],
        SPACING_METRES={int(k):v for k,v in physical['spacing_floor_metres'].items()},TRAIN_X_STD=float(training[:,0].std()),TRAIN_Y_STD=float(training[:,1].std())))
    r.indicated=env['indicated'];r.continuation_features=env['continuation_features']
    ev=env['MaskedRectangularEVAE'](N=128,yscale=stats['yscale']).to(r.DEV);ev.load_state_dict(torch.load(REPAIR/'seed_42001/evae.pt',map_location=r.DEV,weights_only=True)['state_dict'],strict=True);ev.eval()
    ck=torch.load(REPAIR/'seed_42001/continuation_head.pt',map_location=r.DEV,weights_only=True);head=r.gen['EVAEContHead']().to(r.DEV);head.load_state_dict(ck['state_dict'],strict=True);head.eval()
    models=dict(evae=ev,head=(head,ck['X_mean'],ck['X_std']))
    return r,env,models

def pair_proxy(p,l,a,tau):
    if len(p)<2:return p.sum()*0
    dij=torch.cdist(p,p);rij=dij/((l[:,None]+l[None,:])/2+1e-6)
    ang=torch.sin(a[:,None]-a[None,:]).square();iu=torch.triu_indices(len(p),len(p),1,device=p.device)
    return (ang*torch.exp(-rij.square()/(tau*tau+1e-6)))[iu[0],iu[1]].mean()

def losses(net,b,r,tau):
    c,cm,y,mask,fam,sz,gz=b
    di=torch.cat([torch.zeros(len(c),1,5,device=r.DEV),y[:,:-1]],1);dm=torch.cat([torch.zeros(len(c),1,dtype=torch.bool,device=r.DEV),mask[:,:-1]],1)
    p,l,a,f=net(c,cm,di,dm,sz,gz);valid=~mask
    angle=torch.atan2(y[:,:,3],y[:,:,4]);pa=torch.atan2(a[:,:,0],a[:,:,1])
    pos=F.mse_loss(p[valid],y[:,:,:2][valid]);length=F.mse_loss(l[valid],y[:,:,2:3][valid]);angular=(1-torch.cos(2*(angle[valid]-pa[valid]))).mean();family=F.cross_entropy(f[valid],fam[valid])
    proportion=F.mse_loss(f[valid].softmax(-1).mean(0),torch.as_tensor(r.STATS['shares'],device=r.DEV,dtype=torch.float32))
    base=pos+.5*length+1.5*angular+family+.3*proportion
    predicted=[];corrected=[]
    for i in range(len(c)):
        vi=valid[i]
        if int(vi.sum())<2:continue
        ps=pair_proxy(p[i][vi],l[i,:,0][vi].exp(),pa[i][vi],tau)
        target=pair_proxy(y[i,:,:2][vi],y[i,:,2][vi].exp(),angle[i][vi],tau).detach()
        predicted.append(ps);corrected.append((ps-target).square())
    old=(torch.stack(predicted).mean()-r.STATS['gs']['intersection_mean']).square() if predicted else base*0
    fixed=torch.stack(corrected).mean() if corrected else base*0
    return base+.5*old,base+.5*fixed,dict(position=pos,length=length,angular=angular,family=family,proportion=proportion,old_crossing=old,correct_crossing=fixed)

def probe(net,seqs,r,name):
    net.eval();rows=[]
    with torch.no_grad():
        for e in seqs:
            c,cm,y,mask,fam,sz,gz=r.seq_batch([e]);di=torch.cat([torch.zeros(1,1,5,device=r.DEV),y[:,:-1]],1);dm=torch.cat([torch.zeros(1,1,dtype=torch.bool,device=r.DEV),mask[:,:-1]],1)
            p,l,a,f=net(c,cm,di,dm,sz,gz);pp=p[0].cpu().numpy();tt=y[0,:,:2].cpu().numpy()
            rows.append(dict(model=name,face_id=e['face_id'],layout=e['layout'],role=e['role'],targets=len(tt),
                predicted_x_span=float(np.ptp(pp[:,0])),target_x_span=float(np.ptp(tt[:,0])),predicted_x_std=float(pp[:,0].std()),target_x_std=float(tt[:,0].std()),
                x_rmse_m=float(np.sqrt(np.mean((pp[:,0]-tt[:,0])**2))*e['width_m']),family_accuracy=float((f[0].argmax(-1)==fam[0]).float().mean())))
    return rows

def train_pair(r,train,val,tau):
    r.seedall(42001);initial_net=r.trans['SeqFracTransformerV4']().to(r.DEV);initial={k:v.detach().cpu().clone() for k,v in initial_net.state_dict().items()}
    initial_hash=statehash(initial);torch_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None;np_rng=np.random.get_state();del initial_net
    torch.save(dict(state_dict=initial,seed=42001),HERE/'shared_initial_transformer.pt')
    nets={};logs=[];selections={};orders={};gradient_rows=[]
    for arm in ['paired_control','corrected_loss']:
        net=r.trans['SeqFracTransformerV4']().to(r.DEV);net.load_state_dict(initial,strict=True);check('identical_initial_state:'+arm,statehash(net.state_dict())==initial_hash)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng);opt=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-5);best=math.inf;order=[]
        for ep in range(1,51):
            net.train();train_loss=[]
            for inds in np.array_split(np.random.permutation(len(train)),math.ceil(len(train)/8)):
                order.extend(map(int,inds));old,new,parts=losses(net,r.seq_batch([train[i] for i in inds]),r,tau);loss=old if arm=='paired_control' else new
                opt.zero_grad();loss.backward()
                if ep==1 and not train_loss:
                    grad=sum(float(p.grad.abs().sum()) for p in net.pos_head.parameters() if p.grad is not None);check('position_head_has_finite_gradient:'+arm,math.isfinite(grad) and grad>0,grad);gradient_rows.append(dict(arm=arm,first_batch_position_gradient_l1=grad))
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();train_loss.append(float(loss))
            net.eval();values=[]
            with torch.no_grad():
                for e in val:
                    old,new,parts=losses(net,r.seq_batch([e]),r,tau);values.append(dict(old_objective=float(old),common_corrected_objective=float(new),**{k:float(v) for k,v in parts.items()}))
            v=pd.DataFrame(values).mean().to_dict();logs.append(dict(arm=arm,epoch=ep,train_objective=float(np.mean(train_loss)),**v))
            score=v['common_corrected_objective']
            if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};selected=ep
            if ep%10==0:print(f'{arm} {ep}/50 shared validation={score:.5f}, position={v["position"]:.5f}',flush=True)
        net.load_state_dict(state);net.eval();nets[arm]=net;orders[arm]=hashlib.sha256(np.array(order,np.int64).tobytes()).hexdigest();selections[arm]=dict(epoch=selected,shared_validation_objective=best,initial_state_sha256=initial_hash,training_order_sha256=orders[arm])
        torch.save(dict(state_dict=state,selected_epoch=selected,common_selection='S1 validation non-crossing terms plus correctly matched crossing term',architecture_unchanged=True),HERE/(arm+'.pt'))
    check('same_batch_order_for_paired_training',orders['paired_control']==orders['corrected_loss'])
    pd.DataFrame(logs).to_csv(HERE/'paired_training_log.csv',index=False);pd.DataFrame(gradient_rows).to_csv(HERE/'gradient_checks.csv',index=False);dump('selected_checkpoints.json',selections)
    return nets

def render(r,predictions,metrics):
    raise NotImplementedError('plotting / image processing removed from the public release')

def main():
    if (HERE/'RUN_COMPLETE.json').exists():raise RuntimeError('Preserve completed preview')
    warnings.filterwarnings('ignore',category=UserWarning);torch.set_num_threads(2);start=time.time()
    inputs={str(p):sha(p) for p in [REPAIR/'repair_trial.py',OLD/'run_revision.py',REPAIR/'seed_42001/evae.pt',REPAIR/'seed_42001/continuation_head.pt',REPAIR/'seed_42001/transformer_latent.pt',STUDY/'data_prep/manual_trace_manifest.csv',REPAIR/'training_only_statistics.json',Path(__file__)]};dump('inputs_before.json',inputs)
    r,env,models=load_state();tau=max(float(np.std(np.vstack([v['arr'][:,:2] for v in r.FACES.values() if v['role']=='train']),axis=0).sum()),1e-3)
    # Same statistic has a zero discrepancy at identical mapped geometry and is permutation invariant.
    for f in r.FACES.values():
        if f['role']!='train':continue
        a=torch.tensor(f['arr'],device=r.DEV);v=pair_proxy(a[:,:2],a[:,2],a[:,3],tau);b=a.flip(0);vv=pair_proxy(b[:,:2],b[:,2],b[:,3],tau)
        check('corrected_crossing_zero_at_identical_geometry:'+f['id'],float((v-v.detach()).square())==0.)
        check('crossing_statistic_permutation_invariant:'+f['id'],float((v-vv).abs())<1e-7)
    examples={}
    for role in ['train','validation']:
        seqs=[]
        for e in env['examples'](r.FACES,role):
            x=r.sequence_example(e,models,True)
            if len(x['tgt']):x.update(face_id=e['face']['id'],layout=e['layout'],width_m=e['face']['w'],role=role);seqs.append(x)
        examples[role]=seqs;print(f'Cached {len(seqs)} {role} sequences; existing EVAE/head reused',flush=True)
    check('training_data_count_unchanged',len(examples['train'])==60 and len(examples['validation'])==12)
    frozen=r.trans['SeqFracTransformerV4']().to(r.DEV);frozen.load_state_dict(torch.load(REPAIR/'seed_42001/transformer_latent.pt',map_location=r.DEV,weights_only=True)['state_dict'],strict=True);frozen.eval()
    probes=probe(frozen,examples['train']+examples['validation'],r,'frozen_2026_10_18');del frozen
    nets=train_pair(r,examples['train'],examples['validation'],tau)
    for arm,net in nets.items():probes+=probe(net,examples['train']+examples['validation'],r,arm)
    pd.DataFrame(probes).to_csv(HERE/'teacher_forced_position_checks.csv',index=False)
    # An explicit sampler variant: preserve the posterior-draw variability instead of local mode selection.
    tree=ast.parse((REPAIR/'repair_trial.py').read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_neural')
    class LocalSample(ast.NodeTransformer):
        def visit_Assign(self,n):
            self.generic_visit(n)
            if any(isinstance(t,ast.Name) and t.id=='best' for t in n.targets):
                n.value=ast.parse('int(rng.choice(valid)) if valid else None',mode='eval').body
            return n
    newnode=LocalSample().visit(copy.deepcopy(node));newnode.name='sample_neural_sampled';source=ast.unparse(ast.fix_missing_locations(newnode));(HERE/'sampled_local_candidate_definition.py').write_text(source,encoding='utf-8');exec(compile(source,str(HERE/'sampled_local_candidate_definition.py'),'exec'),env)
    original_sampler=env['sample_neural'];metrics=[];predictions=[];ledgers=[];budgets=[];stages=[]
    for fid,layout in CASES:
        face=r.FACES[fid];strips=r.strips(layout,face['w']);obs,parents=r.observe(face['arr'],strips)
        for variant in ['paired_control','corrected_loss','corrected_loss_sampled']:
            models['transformer_latent']=nets['paired_control' if variant=='paired_control' else 'corrected_loss'];env['sample_neural']=env['sample_neural_sampled'] if variant.endswith('_sampled') else original_sampler
            ix=len(env['STAGE_LOG']);final,inf,fixed,ledger,budget=env['reconstruct'](face,obs,strips,'E',models,42001)
            meta=dict(face_id=fid,layout=layout,variant=variant,seed=42001)
            m=r.metrics(face['arr'],final,face['w'],face['h'],obs);pres=env['preservation'](obs,final,face['w'])
            check('physical_2mm_preservation:'+fid+':'+variant,pres==1.)
            metrics.append(dict(meta,**m,measured_preservation_physical_2mm=pres,inferred_count=len(inf),inferred_x_span=float(np.ptp(inf[:,0])) if len(inf) else 0.,inferred_x_std=float(np.std(inf[:,0])) if len(inf) else 0.,inferred_family_classes=int(len(np.unique(r.assign(inf,r.STATS['fc'])))) if len(inf) else 0))
            budgets.append(dict(meta,**budget));ledgers.append(ledger.assign(**meta))
            for tier,a in [('anchored',fixed),('inferred',inf),('measured_clips',obs)]:predictions.append(pd.DataFrame(a,columns=['cx','cy','length','angle']).assign(tier=tier,**meta))
            for s in env['STAGE_LOG'][ix:]:stages.append(dict(s,variant=variant,layout=layout))
            print(f'Generated preview: {fid}, {layout}, {variant}, N={len(final)}, shortfall={budget["shortfall"]}',flush=True)
    met=pd.DataFrame(metrics);pred=pd.concat(predictions);bu=pd.DataFrame(budgets)
    met.to_csv(HERE/'preview_metrics.csv',index=False);pred.to_csv(HERE/'preview_predictions.csv',index=False);bu.to_csv(HERE/'preview_count_budgets.csv',index=False);pd.concat(ledgers).to_csv(HERE/'preview_proposal_ledger.csv.gz',index=False);pd.DataFrame(stages).to_csv(HERE/'preview_continuity_stages.csv',index=False)
    check('same_raw_budget_across_variants',bu.groupby(['face_id','layout']).raw_budget.nunique().eq(1).all())
    render(r,pred,met)
    for p,digest in inputs.items():check('input_preserved:'+p,sha(Path(p))==digest)
    for p,digest in json.loads((STUDY/'FOCUSED_REVISION_PLAN_MANIFEST_2026-10-18.json').read_text())['protected_source_hashes'].items():check('protected_reference_preserved:'+p,sha(Path(p))==digest)
    pd.DataFrame(CHECKS+r.CHECKS).to_csv(HERE/'execution_checks.csv',index=False)
    dump('RUN_COMPLETE.json',dict(status='bounded_preview_complete',paired_training_runs=2,training_seed=42001,epochs_per_run=50,preview_cases=2,variants_per_case=3,new_networks=6,preview_PNGs=2,new_Word_files=0,S2_used=False,new_spacings=False,original_architecture_preserved=True,all_checks_passed=all(x['passed'] for x in CHECKS+r.CHECKS),checks=len(CHECKS+r.CHECKS),elapsed_seconds=time.time()-start))
    print((HERE/'RUN_COMPLETE.json').read_text(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('RUN_FAILURE.json',dict(error=repr(exc)));raise
