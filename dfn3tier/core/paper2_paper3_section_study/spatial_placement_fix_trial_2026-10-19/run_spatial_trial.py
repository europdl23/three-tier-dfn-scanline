"""Finite original-component placement repairs; S1 development only."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,math,sys,time,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from scipy.stats import truncnorm,gamma

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent
PREV=STUDY/'minimal_original_method_preview_2026-10-19';REPAIR=STUDY/'s1_s2_repair_trial_2026-10-18'
CASES=[('W00_RL1102-1120','centre'),('W01_RL1066-1102','5m')]
CONFIGS={'A_height':dict(model='height',residual=False,count='density',supported=False),
         'B_order':dict(model='x_order',residual=False,count='density',supported=True),
         'C_uncertainty':dict(model='x_order',residual=True,count='density',supported=True),
         'D_hits':dict(model='x_order',residual=True,count='hits',supported=True)}
CHECKS=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,obj):(HERE/n).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def check(n,v,detail=''):
    CHECKS.append(dict(check=n,passed=bool(v),detail=str(detail)));assert v,n+': '+str(detail)
def import_path(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def clip_outer(r,rows,face):
    """Clip once, 1 mm inside public bounds to avoid float32 endpoint overshoot."""
    rows=np.asarray(rows).reshape(-1,4);p,q=r.ends(rows);v=q-p;t0=np.zeros(len(rows));t1=np.ones(len(rows));eps=.001/face['w']
    for axis,upper in [(0,1.),(1,face['ym'])]:
        den=v[:,axis];safe=np.where(abs(den)>1e-12,den,1.);a=(eps-p[:,axis])/safe;b=(upper-eps-p[:,axis])/safe
        parallel=abs(den)<=1e-12;lo=np.minimum(a,b);hi=np.maximum(a,b);lo[parallel]=-np.inf;hi[parallel]=np.inf
        bad=parallel&((p[:,axis]<eps)|(p[:,axis]>upper-eps));lo[bad]=np.inf
        t0=np.maximum(t0,lo);t1=np.minimum(t1,hi)
    good=np.isfinite(t0)&(t1>t0+1e-10);out=np.zeros_like(rows,dtype=np.float32)
    if good.any():
        a=p[good]+t0[good,None]*v[good];b=p[good]+t1[good,None]*v[good];out[good]=r.lines(a,b)
    return out

def training_references(r):
    refs={};whole=[];all_y=[]
    for f in r.FACES.values():
        if f['role']!='train':continue
        a=f['arr'];fam=r.assign(a,r.STATS['fc']);whole.append(np.column_stack([a[:,2]*f['w'],a[:,3]]));all_y.extend(a[:,1]/f['ym'])
        for k in range(4):
            sub=a[fam==k]
            if len(sub):refs.setdefault(k,[]).append(float((sub[:,1]/f['ym']).mean()))
    cm={k:dict(mean=float(np.mean(refs.get(k,[.5]))),std=max(float(np.std(refs.get(k,[.5]))),.02)) for k in range(4)}
    return cm,np.vstack(whole),float(np.std(all_y))

def corrected_references(r,base,face,cm):
    stats=copy.deepcopy(base)
    for k in range(4):
        stats['gs']['centre_mass_ref'][k]['cy_mean']=cm[k]['mean']*face['ym'];stats['gs']['centre_mass_ref'][k]['cy_std']=cm[k]['std']*face['ym']
        lens=stats['pools'][k]['lengths_m']/face['w'];logs=np.log(lens)
        stats['logmu'][k]=float(logs.mean());stats['logsig'][k]=max(float(logs.std()),.1)
        stats['gs']['fam_length_ref'][k]={**stats['gs']['fam_length_ref'].get(k,{}),**{f'p{q}':float(np.percentile(lens,q)) for q in [75,90,95]},'mean':float(lens.mean()),'std':max(float(lens.std()),.001)}
    r.STATS=stats

def cache_examples(r,env,models):
    out={}
    for role in ['train','validation']:
        seq=[]
        for e in env['examples'](r.FACES,role):
            x=r.sequence_example(e,models,True)
            if not len(x['tgt']):continue
            x.update(face_id=e['face']['id'],layout=e['layout'],role=role,aspect=e['face']['ym'],width_m=e['face']['w']);seq.append(x)
        out[role]=seq;print(f'Cached {len(seq)} {role} examples',flush=True)
    return out

def set_aspect(net,items,device):net.aspect=torch.tensor([x['aspect'] for x in items],device=device,dtype=torch.float32)

def get_losses(m,net,b,r,tau):
    old,correct,parts=m.losses(net,b,r,tau);y=b[2];valid=~b[3]
    scale=torch.stack([torch.ones_like(net.aspect),net.aspect],-1)[:,None,:]
    position=torch.nn.functional.mse_loss((net.last_position/scale)[valid],(y[:,:,:2]/scale)[valid])
    return correct-parts['position']+position,dict(**parts,balanced_position=position)

def train_models(m,r,examples,tau):
    Base=r.trans['SeqFracTransformerV4']
    class HeightTransformer(Base):
        def forward(self,ctx,ctx_mask,tgt,tgt_mask,sz=None,gz=None):
            p,l,a,f=super().forward(ctx,ctx_mask,tgt,tgt_mask,sz,gz)
            aspect=torch.as_tensor(self.aspect,device=p.device,dtype=p.dtype).reshape(-1,1,1)
            p=torch.cat([p[:,:,:1],p[:,:,1:]*aspect],-1);self.last_position=p
            return p,l,a,f
    initial=torch.load(PREV/'shared_initial_transformer.pt',map_location='cpu',weights_only=True)['state_dict'];initial_hash=m.statehash(initial)
    nets={};logs=[];selected={};residuals={};probes=[];orders={}
    for name in ['height','x_order']:
        r.seedall(42001);net=HeightTransformer().to(r.DEV);net.load_state_dict(initial,strict=True);check('same_original_parameter_shapes:'+name,all(net.state_dict()[k].shape==v.shape for k,v in initial.items()));check('same_initial_weights:'+name,m.statehash(net.state_dict())==initial_hash)
        seqs=copy.copy(examples)
        if name=='x_order':
            seqs={role:[] for role in examples}
            for role,items in examples.items():
                for x in items:
                    z=dict(x);ix=np.argsort(x['tgt'][:,0],kind='stable');z['tgt']=x['tgt'][ix];z['fam']=x['fam'][ix];seqs[role].append(z)
        opt=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-5);best=math.inf;order=[]
        for ep in range(1,51):
            net.train();ls=[]
            for ix in np.array_split(np.random.permutation(len(seqs['train'])),math.ceil(len(seqs['train'])/8)):
                order.extend(map(int,ix));items=[seqs['train'][i] for i in ix];set_aspect(net,items,r.DEV);loss,parts=get_losses(m,net,r.seq_batch(items),r,tau)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();ls.append(float(loss))
            net.eval();vals=[]
            with torch.no_grad():
                for x in seqs['validation']:
                    set_aspect(net,[x],r.DEV);loss,parts=get_losses(m,net,r.seq_batch([x]),r,tau);vals.append(dict(total=float(loss),**{k:float(v) for k,v in parts.items()}))
            v=pd.DataFrame(vals).mean().to_dict();logs.append(dict(model=name,epoch=ep,train_loss=float(np.mean(ls)),**v))
            if v['total']<best:best=v['total'];state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};bestep=ep
            if ep%10==0:print(f'{name} {ep}/50 validation={v["total"]:.5f}, balanced position={v["balanced_position"]:.5f}',flush=True)
        net.load_state_dict(state);net.eval();nets[name]=net;orders[name]=hashlib.sha256(np.array(order,np.int64).tobytes()).hexdigest()
        selected[name]=dict(epoch=bestep,validation_loss=best,initial_sha256=initial_hash,training_order_sha256=orders[name],new_parameters=0)
        torch.save(dict(state_dict=state,selected_epoch=bestep,forward='original Transformer parameters; y output times public aspect',position_loss='per-axis public fraction',target_order=name,architecture_parameters_unchanged=True),HERE/(name+'.pt'))
        residuals[name]={};bylayout={}
        with torch.no_grad():
            for role,items in seqs.items():
                for x in items:
                    set_aspect(net,[x],r.DEV);b=r.seq_batch([x]);get_losses(m,net,b,r,tau);pp=net.last_position[0].cpu().numpy();tt=b[2][0,:,:2].cpu().numpy();scale=np.array([1,x['aspect']]);pf=pp/scale;tf=tt/scale
                    probes.append(dict(model=name,role=role,face_id=x['face_id'],layout=x['layout'],targets=len(tt),x_span=float(np.ptp(pf[:,0])),y_span=float(np.ptp(pf[:,1])),target_x_span=float(np.ptp(tf[:,0])),target_y_span=float(np.ptp(tf[:,1])),x_rmse_m=float(np.sqrt(np.mean((pf[:,0]-tf[:,0])**2))*x['width_m']),y_rmse_m=float(np.sqrt(np.mean((pf[:,1]-tf[:,1])**2))*x['width_m']*x['aspect'])))
                    if role=='train':bylayout.setdefault(x['layout'],[]).append(tf-pf)
        for layout,arrs in bylayout.items():
            e=np.vstack(arrs);bias=e.mean(0);sd=np.sqrt(np.mean((e-bias)**2,0));residuals[name][layout]=dict(bias=bias.tolist(),sigma=np.maximum(sd,.001).tolist(),source='S1 fitting teacher-forced residuals only',n=len(e))
    check('paired_training_orders_equal',orders['height']==orders['x_order'])
    pd.DataFrame(logs).to_csv(HERE/'training_log.csv',index=False);pd.DataFrame(probes).to_csv(HERE/'teacher_forced_spatial_probes.csv',index=False);dump('selected_checkpoints.json',selected);dump('training_residual_calibration.json',residuals)
    return nets,residuals

def count_from_hits(r,geom,strips,nhits,prior,training):
    assert 'arr' not in geom
    rng=np.random.default_rng(7319);n=32768;ix=rng.integers(len(training),size=n);L=training[ix,0]/geom['w'];A=training[ix,1]
    arr=np.column_stack([rng.uniform(0,1,n),rng.uniform(0,geom['ym'],n),L,A]);arr=clip_outer(r,arr,geom);p,q=r.ends(arr);lo=np.minimum(p[:,0],q[:,0]);hi=np.maximum(p[:,0],q[:,0]);hits=np.zeros(n,int)
    for a,b in strips:hits+=(hi>=a)&(lo<=b)&(arr[:,2]>1e-8)
    rate=float(hits.mean());shape=nhits+2.;denom=rate+2./prior;estimate=shape/denom
    return dict(estimated_count=estimate,rounded_count=max(1,int(round(estimate))),original_density_prior=prior,observed_clipped_hits=nhits,simulated_expected_hits_per_trace=rate,simulated_unique_hit_probability=float(np.mean(hits>0)),samples=n,simulation_seed=7319,prior_hit_weight=2.,approximate_interval_95=[float(gamma.ppf(.025,shape,scale=1/denom)),float(gamma.ppf(.975,shape,scale=1/denom))])

def spatial_values(points):
    if not len(points):return dict(x_span=0.,y_span=0.,occupied_cells=0,max_cell_share=1.)
    x=np.minimum((np.clip(points[:,0],0,1)*5).astype(int),4);y=np.minimum((np.clip(points[:,1],0,1)*4).astype(int),3);counts=np.bincount(x*4+y,minlength=20)
    return dict(x_span=float(np.ptp(points[:,0])),y_span=float(np.ptp(points[:,1])),occupied_cells=int(np.sum(counts>0)),max_cell_share=float(counts.max()/len(points)))

def hidden_reference_cloud(r,layout):
    pts=[]
    for f in r.FACES.values():
        if f['role']!='train':continue
        _,parents=r.observe(f['arr'],r.strips(layout,f['w']));a=f['arr'][~np.isin(np.arange(len(f['arr'])),np.unique(parents))];z=a[:,:2]/[1,f['ym']];pts.extend([z,1-z])
    return np.vstack(pts)

def envelope(cloud,n):
    if n<4:return dict(assessable=False)
    rng=np.random.default_rng(991);rows=[spatial_values(cloud[rng.integers(len(cloud),size=n)]) for _ in range(500)];d=pd.DataFrame(rows)
    return dict(assessable=True,x_span_min=float(d.x_span.quantile(.05)),y_span_min=float(d.y_span.quantile(.05)),occupied_cells_min=float(d.occupied_cells.quantile(.05)),max_cell_share_max=float(d.max_cell_share.quantile(.95)),reference_n=len(cloud),resamples=500,seed=991)

def main():
    raise NotImplementedError('plotting / image processing removed from the public release')

if __name__=='__main__':
    try:main()
    except Exception as e:dump('RUN_FAILURE.json',dict(error=repr(e)));raise
