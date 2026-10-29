"""Versioned original-component experiment utilities. No experiment on import."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib.util,io,json,math,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent
LOCAL=STUDY/'local_window_retrain_2026-10-19'
REPAIR=STUDY/'s1_s2_repair_trial_2026-10-18'
PREV=STUDY/'minimal_original_method_preview_2026-10-19'
SPATIAL=STUDY/'spatial_placement_fix_trial_2026-10-19'
BASELINE=STUDY/'local_window_baseline_check_2026-10-19'
SEEDS=[42001,42002,42003]
REFERENCE=dict(position=1.,length=.5,angular=1.5,family=1.,proportion=.3,crossing=.5)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item() if isinstance(a,np.generic) else str(a)),encoding='utf-8')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def segment_graph(segments,height,tol=.02):
    """Connected components of physical segments, with endpoint/contact tolerance."""
    a=np.asarray(segments,dtype=float).reshape(-1,2,2);n=len(a);parent=list(range(n));crossings=0
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def cross(u,v):return u[0]*v[1]-u[1]*v[0]
    def distance(p,line):
        v=line[1]-line[0];t=np.clip(np.dot(p-line[0],v)/max(np.dot(v,v),1e-16),0,1)
        return np.linalg.norm(p-line[0]-t*v)
    for i in range(n):
        for j in range(i):
            u=a[i,1]-a[i,0];v=a[j,1]-a[j,0];den=cross(u,v);hit=False
            if abs(den)>1e-12:
                d=a[j,0]-a[i,0];s=cross(d,v)/den;t=cross(d,u)/den
                if -1e-10<=s<=1+1e-10 and -1e-10<=t<=1+1e-10:hit=True;crossings+=1
            if hit or min(distance(a[i,0],a[j]),distance(a[i,1],a[j]),distance(a[j,0],a[i]),distance(a[j,1],a[i]))<=tol:
                parent[root(i)]=root(j)
    groups={}
    for i in range(n):groups.setdefault(root(i),[]).append(i)
    spanning=any(a[g,:,1].min()<=tol and a[g,:,1].max()>=height-tol for g in groups.values())
    return dict(graph_components=len(groups),graph_largest_fraction=max([len(v) for v in groups.values()]+[0])/max(n,1),graph_top_bottom=int(spanning),graph_crossings=crossings)

def noisy_aperture_clips(r,obs,parents,strips,aspect,noise):
    """Perturb reported direction through each observed centre, reclip aperture.

    Ends originally censored by aperture/frame stay censored. Interior observed
    terminations keep their along-line distance; hidden parent ends are unused.
    """
    result=[]
    for row,parent in zip(obs,parents):
        p,q=r.ends(row[None]);p=p[0];q=q[0];c=row[:2].astype(float)
        strip=min(strips,key=lambda s:abs(c[0]-(s[0]+s[1])/2));lo,hi=strip
        old=np.array([math.cos(row[3]),math.sin(row[3])]);angle=float(row[3])+noise[int(parent)]
        u=np.array([math.cos(angle),math.sin(angle)])
        if u@old<0:u=-u
        def censored(point):return min(abs(point[0]-lo),abs(point[0]-hi),abs(point[1]),abs(point[1]-aspect))<2e-6
        t0=-math.inf if censored(p) else -float(row[2])/2
        t1=math.inf if censored(q) else float(row[2])/2
        for axis,low,high in [(0,lo,hi),(1,0.,aspect)]:
            if abs(u[axis])<1e-12:continue
            aa=(low-c[axis])/u[axis];bb=(high-c[axis])/u[axis]
            t0=max(t0,min(aa,bb));t1=min(t1,max(aa,bb))
        if not math.isfinite(t0+t1) or t1<=t0:raise RuntimeError('Noisy observed clip has no aperture support')
        result.append(r.lines((c+t0*u)[None],(c+t1*u)[None])[0])
    return np.asarray(result,np.float32).reshape(-1,4)

class Engine:
    def __init__(self):
        torch.set_num_threads(2)
        self.m=load_module('stepwise_preview',PREV/'run_preview.py')
        self.s=load_module('stepwise_geometry',SPATIAL/'run_spatial_trial.py')
        self.b=load_module('stepwise_baseline',BASELINE/'compare_local_baselines.py')
        r,env,unused=self.m.load_state();del unused
        self.r=r;self.env=env;r.HERE=HERE
        stats=self.m.keyrestore(json.loads((LOCAL/'training_only_statistics.json').read_text()))
        for k in ['fc','shares']:stats[k]=np.asarray(stats[k])
        stats['pools']={k:{n:np.asarray(v) for n,v in p.items()} for k,p in stats['pools'].items()}
        self.base=copy.deepcopy(stats);r.STATS=stats
        self.inventory=pd.read_csv(LOCAL/'window_inventory.csv');self.trace=pd.read_csv(LOCAL/'window_trace_manifest.csv')
        self.faces={}
        for row in self.inventory.itertuples():
            arr=self.trace[self.trace.window_id.eq(row.window_id)][['cx','cy','length','angle']].to_numpy(np.float32)
            self.faces[row.window_id]=dict(id=row.window_id,parent_face=row.parent_face,role=row.role,w=20.,h=float(row.height_m),ym=float(row.height_m)/20,arr=arr,x0=row.x0_m)
        r.FACES={k:f for k,f in self.faces.items() if f['role']!='test'}
        fit=np.vstack([f['arr'] for f in r.FACES.values() if f['role']=='train'])
        self.tau=max(float(np.std(fit[:,:2],axis=0).sum()),1e-3)
        canonical=[]
        for f in r.FACES.values():
            if f['role']=='train':a=f['arr'].copy();a[:,:3]*=20/36;canonical.append(a)
        env['CONT']=r.definitions(r.EXP/'hybrid_v6.py',dict(load_panels_raw=lambda:canonical,get_physics_stats=lambda:{'fam_centers_rad':stats['fc']},assign_family_ids=r.assign,segments_intersect=r.intersect,FAM_ASSIGN_THRESHOLD_DEG=20.),names=['apply_continuity_rules'])
        with contextlib.redirect_stdout(io.StringIO()):gs=r.geo['_train_geology_stats'](canonical,stats['fc'])
        env['SPACING_METRES']={k:float(gs['spacing_floor'][k])*36 for k in range(3)};env['SPACING_METRES'][3]=.288
        env['TRAIN_X_STD']=float(fit[:,0].std());env['TRAIN_Y_STD']=float(fit[:,1].std())
        self.cm,_,_=self.s.training_references(r)
        self.original_posterior=env['posterior_draws'];self.original_gate=r.candidate_gate
        original_stat=r.statistical_proposals
        def propose(face,total,mode,rng):
            if mode!='adfne_proxy':a=original_stat(face,total,mode,rng)
            else:
                families=rng.choice(4,total,p=r.STATS['shares']);rows=[]
                for fam in families:
                    pool=r.STATS['pools'][int(fam)];hist,edges=np.histogram(pool['angles'],bins=12,range=(0,np.pi));k=int(rng.choice(12,p=(hist+1)/(hist.sum()+12)));angle=rng.uniform(edges[k],edges[k+1]);logs=np.log(pool['lengths_m'])
                    length=math.exp(float(rng.normal(logs.mean(),max(logs.std(),.08))))
                    rows.append([rng.uniform(),rng.uniform(0,face['ym']),length/face['w'],angle])
                a=np.asarray(rows,np.float32)
            return self.s.clip_outer(r,a,face)
        r.statistical_proposals=propose
        r.PROTOCOL['arms'].update(ADFNE=['stat','adfne_proxy',True],KDE=['stat','kde',True])
        self.current={}
        def posterior(position,loglength,angle,family_logits,face,rng,n):
            a,_,fids=self.original_posterior(position,loglength,angle,family_logits,face,rng,n)
            pos=np.asarray(position)/[1,face['ym']];e=self.current['errors'][self.current['layout']]
            eligible=np.flatnonzero(np.all((pos+e>=0)&(pos+e<=1),axis=1))
            if not len(eligible):a[:]=0;return a,np.zeros(n),fids
            a[:,:2]=(pos+e[rng.choice(eligible,n,replace=True)])*[1,face['ym']]
            return self.s.clip_outer(r,a,face),np.zeros(n),fids
        env['posterior_draws']=posterior
        exec(compile((SPATIAL/'supported_sampled_generator.py').read_text(),str(SPATIAL/'supported_sampled_generator.py'),'exec'),env)
        env['sample_neural']=env['sample_neural_sampled']
        Base=r.trans['SeqFracTransformerV4']
        class HeightTransformer(Base):
            def forward(net,ctx,ctx_mask,tgt,tgt_mask,sz=None,gz=None):
                if not getattr(net,'latent',True):sz=gz=None
                p,l,a,f=super().forward(ctx,ctx_mask,tgt,tgt_mask,sz,gz)
                aspect=torch.as_tensor(net.aspect,device=p.device,dtype=p.dtype).reshape(-1,1,1)
                p=torch.cat([p[:,:,:1],p[:,:,1:]*aspect],-1);net.last_position=p
                return p,l,a,f
        self.Net=HeightTransformer
        self.train=load_module('stepwise_local_source',LOCAL/'run_local_preview.py').make_examples(r,r.FACES,'train')
        self.valid=load_module('stepwise_local_source_valid',LOCAL/'run_local_preview.py').make_examples(r,r.FACES,'validation')
        self.bundles={};self.sequence_cache={}

    def load_bundle(self,seed):
        if seed in self.bundles:return self.bundles[seed]
        dest=LOCAL if seed==42001 else HERE/f'models/seed_{seed}'
        if not (dest/'evae.pt').exists():return None
        r=self.r;ev=self.env['MaskedRectangularEVAE'](N=128,yscale=self.base['yscale']).to(r.DEV)
        ev.load_state_dict(torch.load(dest/'evae.pt',map_location=r.DEV,weights_only=True)['state_dict']);ev.eval()
        ck=torch.load(dest/'continuation_head.pt',map_location=r.DEV,weights_only=True);head=r.gen['EVAEContHead']().to(r.DEV);head.load_state_dict(ck['state_dict']);head.eval()
        models=dict(evae=ev,head=(head,ck['X_mean'],ck['X_std']))
        if (dest/'transformer_latent.pt').exists():models['transformer_latent']=self.load_net(dest/'transformer_latent.pt',True)
        if (dest/'transformer_no_latent.pt').exists():models['transformer_no_latent']=self.load_net(dest/'transformer_no_latent.pt',False)
        bundle=dict(models=models,dest=dest,seed=seed)
        self.bundles[seed]=bundle;return bundle

    def load_net(self,path,latent):
        net=self.Net().to(self.r.DEV);net.latent=latent
        net.load_state_dict(torch.load(path,map_location=self.r.DEV,weights_only=True)['state_dict']);net.eval();return net

    def sequences(self,bundle,latent):
        key=(bundle['seed'],latent)
        if key in self.sequence_cache:return self.sequence_cache[key]
        r=self.r;r.STATS=copy.deepcopy(self.base);out={}
        for role,items in [('train',self.train),('validation',self.valid)]:
            seq=[]
            for e in items:
                x=r.sequence_example(e,bundle['models'],latent)
                if not len(x['tgt']):continue
                ix=np.argsort(x['tgt'][:,0],kind='stable');x['tgt']=x['tgt'][ix];x['fam']=x['fam'][ix]
                x.update(aspect=e['face']['ym'],layout=e['layout'],face_id=e['face']['id']);seq.append(x)
            out[role]=seq
        self.sequence_cache[key]=out;return out

    def loss(self,net,b,weights):
        ref,parts=self.s.get_losses(self.m,net,b,self.r,self.tau)
        result=ref
        for name,key in [('position','balanced_position'),('length','length'),('angular','angular'),('family','family'),('proportion','proportion'),('crossing','correct_crossing')]:result=result+(weights[name]-REFERENCE[name])*parts[key]
        return result,ref

    def train_net(self,bundle,latent,path,weights=REFERENCE,seed=42001):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():return self.load_net(path,latent)
        seq=self.sequences(bundle,latent);r=self.r;r.STATS=copy.deepcopy(self.base);r.seedall(seed)
        net=self.Net().to(r.DEV);net.latent=latent
        initial=self.m.statehash(net.state_dict());opt=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-5);best=math.inf;logs=[]
        for ep in range(1,51):
            net.train();values=[]
            for ix in np.array_split(np.random.permutation(len(seq['train'])),math.ceil(len(seq['train'])/8)):
                items=[seq['train'][i] for i in ix];self.s.set_aspect(net,items,r.DEV)
                loss,_=self.loss(net,r.seq_batch(items),weights)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                opt.zero_grad();loss.backward()
                if not latent and ep==1 and not values:assert net.latent_proj.weight.grad is None,'D must not receive latent gradient'
                nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();values.append(float(loss))
            net.eval();validation=[]
            with torch.no_grad():
                for x in seq['validation']:
                    self.s.set_aspect(net,[x],r.DEV);_,ref=self.loss(net,r.seq_batch([x]),weights);validation.append(float(ref))
            score=float(np.mean(validation));logs.append(dict(epoch=ep,train=float(np.mean(values)),common_validation_loss=score))
            if score<best:best=score;selected=ep;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            if ep%25==0:print(f'TRAIN {path.parent.name}/{path.stem} {ep}/50 val={score:.4f}',flush=True)
        net.load_state_dict(state);net.eval()
        torch.save(dict(state_dict=state,selected_epoch=selected,loss_weights=weights,training_seed=seed,latent=latent,initial_sha256=initial,selection='S1 common-reference validation objective'),path)
        pd.DataFrame(logs).to_csv(path.with_suffix('.training.csv'),index=False)
        return net

    def calibrate(self,bundle,net,latent,path):
        path=Path(path)
        if path.exists():
            with np.load(path) as z:errors={k:z[k] for k in z.files}
            return errors,json.loads(path.with_suffix('.horizons.json').read_text()) if path.with_suffix('.horizons.json').exists() else json.loads((LOCAL/'trained_horizons.json').read_text())
        seq=self.sequences(bundle,latent);errors={};horizons={}
        with torch.no_grad():
            for x in seq['train']:
                self.s.set_aspect(net,[x],self.r.DEV);b=self.r.seq_batch([x]);self.loss(net,b,REFERENCE)
                err=(b[2][0,:,:2]-net.last_position[0]).cpu().numpy()/[1,x['aspect']]
                errors.setdefault(x['layout'],[]).append(err);horizons[x['layout']]=max(horizons.get(x['layout'],1),len(x['tgt']))
        errors={k:np.vstack(v) for k,v in errors.items()};path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**errors);dump(path.with_suffix('.horizons.json'),horizons)
        return errors,horizons

    def ensure_D(self,bundle):
        path=HERE/f'models/seed_{bundle["seed"]}/transformer_no_latent.pt'
        net=self.train_net(bundle,False,path,REFERENCE,bundle['seed'])
        bundle['models']['transformer_no_latent']=net
        bundle['D_cal']=self.calibrate(bundle,net,False,path.with_suffix('.residuals.npz'))

    def ensure_bundle(self,seed):
        bundle=self.load_bundle(seed)
        if bundle is not None and 'transformer_latent' in bundle['models']:return bundle
        r=self.r;dest=HERE/f'models/seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        r.HERE=dest;r.STATS=copy.deepcopy(self.base);r.seedall(seed)
        r.legacy['DualLatentVAE']=self.env['MaskedRectangularEVAE'];r.forward_evae=self.env['forward_evae']
        exec(compile((REPAIR/'geometry_aware_training_definition.py').read_text(),str(REPAIR/'geometry_aware_training_definition.py'),'exec'),r.__dict__)
        if not (dest/'evae.pt').exists():
            ev,logs,ep=r.train_evae(self.train,self.valid,r.STATS,seed)
            torch.save(dict(state_dict=ev.state_dict(),selected_epoch=ep,training_seed=seed),dest/'evae.pt');pd.DataFrame(logs).to_csv(dest/'evae.training.csv',index=False)
        else:
            ev=self.env['MaskedRectangularEVAE'](N=128,yscale=self.base['yscale']).to(r.DEV);ev.load_state_dict(torch.load(dest/'evae.pt',map_location=r.DEV,weights_only=True)['state_dict']);ev.eval()
        if not (dest/'continuation_head.pt').exists():
            head,logs,ep=self.env['train_head'](self.train,self.valid,ev);net,mu,sd=head
            torch.save(dict(state_dict=net.state_dict(),X_mean=mu,X_std=sd,selected_epoch=ep),dest/'continuation_head.pt');pd.DataFrame(logs).to_csv(dest/'head.training.csv',index=False)
        bundle=self.load_bundle(seed)
        net=self.train_net(bundle,True,dest/'transformer_latent.pt',REFERENCE,seed)
        bundle['models']['transformer_latent']=net;self.calibrate(bundle,net,True,dest/'training_position_residuals.npz')
        r.HERE=HERE;return bundle

    def observe(self,face,layout,condition,seed):
        st=self.r.strips(layout,20);obs,parents=self.r.observe(face['arr'],st);raw_unique=len(np.unique(parents));rng=np.random.default_rng(seed+810000)
        if condition=='miss10':
            unique=np.unique(parents);kept={int(k) for k in unique if rng.random()>=.1};mask=np.array([int(k) in kept for k in parents]);obs=obs[mask];parents=parents[mask]
        elif condition=='noise5':
            noise={int(k):rng.normal(0,np.radians(5)) for k in np.unique(parents)}
            obs=noisy_aperture_clips(self.r,obs,parents,st,face['ym'],noise)
        return obs,parents,st,dict(raw_unique_observed=raw_unique,retained_unique_observed=len(np.unique(parents)),observed_trace_fraction=len(np.unique(parents))/max(len(face['arr']),1))

    def generate(self,fid,arm,seed,training_seed=42001,layout='5m',condition='clean',variant='reference',override=None):
        key=f'{fid}__{arm}__t{training_seed if arm in ["C","D","E","F"] else 0}__g{seed}__{layout}__{condition}__{variant if arm in ["E","F"] else "reference"}'
        if condition=='noise5':key+='__aperture_v2'
        dest=HERE/'cases'/key
        if (dest/'metrics.json').exists():return json.loads((dest/'metrics.json').read_text())
        dest.mkdir(parents=True,exist_ok=True);face=self.faces[fid];r=self.r
        bundle=self.load_bundle(training_seed);models=dict(bundle['models'])
        if override is not None:models['transformer_latent']=override
        obs,parents,st,obsaudit=self.observe(face,layout,condition,seed)
        public={k:v for k,v in face.items() if k!='arr'};assert 'arr' not in public
        self.s.corrected_references(r,self.base,public,self.cm)
        if arm in ['D','E','F']:
            latent=arm!='D';net=models['transformer_latent' if latent else 'transformer_no_latent'];net.aspect=torch.tensor([face['ym']],device=r.DEV)
            if not latent:errors,horizons=bundle['D_cal']
            elif override is not None:errors,horizons=self.calibrate(bundle,net,True,HERE/f'calibration/{variant}_seed{training_seed}.npz')
            else:errors,horizons=self.calibrate(bundle,net,True,bundle['dest']/'training_position_residuals.npz')
            self.current.update(errors=errors,layout=layout);self.env['HORIZON']=min(128,horizons[layout])
        # Missed detection removes the complete-negative-evidence premise. Both
        # arms relax that strip-crossing veto, retaining every other hard gate.
        def gate(row,f,lines):
            reason=self.original_gate(row,f,lines)
            return None if condition=='miss10' and reason=='observation_conflict' else reason
        r.candidate_gate=gate
        stage_start=len(self.env['STAGE_LOG'])
        final,inf,fixed,ledger,budget=self.env['reconstruct'](public,obs,st,arm,models,seed)
        p,q=r.ends(final);z=np.stack([p,q],1);bad=((z[:,:,0]<-1e-7)|(z[:,:,0]>1+1e-7)|(z[:,:,1]<-1e-7)|(z[:,:,1]>face['ym']+1e-7)).any(1)
        if bad.any():final[bad]=self.s.clip_outer(r,final[bad],public)
        anchored=np.array([any(self.env['contains'](o,a,20) for o in obs) for a in final],bool);fixed=final[anchored];inf=final[~anchored]
        p,q=r.ends(final);z=np.stack([p,q],1)
        assert np.isfinite(z).all() and (final[:,2]>0).all(),key
        assert ((z[:,:,0]>=-1e-6)&(z[:,:,0]<=1+1e-6)&(z[:,:,1]>=-1e-6)&(z[:,:,1]<=face['ym']+1e-6)).all(),key
        preserved=self.env['preservation'](obs,final,20);assert preserved==1.,key
        truth=face['arr'];hidden=truth[~np.isin(np.arange(len(truth)),np.unique(parents))]
        metrics=r.metrics(truth,final,20,face['h'],obs)
        metrics={k:v for k,v in metrics.items() if not any(k.startswith(x) for x in ['I_share','Y_share','X_share','components','largest_component','top_bottom'])}
        metrics.update(self.b.spatial(truth,final,face['ym']))
        tp,tq=r.ends(truth);truthgraph=segment_graph(np.stack([tp,tq],1)*20,face['h']);predgraph=segment_graph(z*20,face['h'])
        for k,v in truthgraph.items():metrics[k+'_truth']=v;metrics[k+'_pred']=predgraph[k];metrics[k+'_abs_error']=abs(predgraph[k]-v)
        metrics.update(P21_abs_error=abs(metrics['P21_pred']-metrics['P21_truth']),intersection_density_abs_error=abs(metrics['intersections_per_m2_pred']-metrics['intersections_per_m2_truth']),inferred_count=len(inf),hidden_truth_count=len(hidden),inferred_direction_W1=r.axial_w1(hidden[:,3],inf[:,3]),inferred_length_W1_m=r.wasserstein_distance(hidden[:,2]*20,inf[:,2]*20) if len(hidden) and len(inf) else math.nan,measured_preservation_2mm=preserved,final_trim_count=int(bad.sum()))
        boundary=(np.abs(z[:,:,0])<.0011)|(np.abs(z[:,:,0]-1)<.0011)|(np.abs(z[:,:,1])<.0011)|(np.abs(z[:,:,1]-face['ym'])<.0011)
        metrics['pred_boundary_touching_fraction']=float(boundary.any(1).mean()) if len(z) else math.nan
        metrics['random_background_fraction']=float(np.mean(r.assign(final,r.STATS['fc'])==3))
        meta=dict(case_key=key,window_id=fid,parent_face=face['parent_face'],role=face['role'],arm=arm,training_seed=training_seed if arm in ['C','D','E','F'] else 0,generation_seed=seed,layout=layout,condition=condition,variant=variant,**obsaudit)
        assert len(ledger)==budget['raw_budget']
        pred=[]
        for tier,a in [('mapped',truth),('anchored',fixed),('inferred',inf),('measured_clips',obs)]:pred.append(pd.DataFrame(a,columns=['cx','cy','length','angle']).assign(tier=tier))
        pd.concat(pred).to_csv(dest/'predictions.csv',index=False);ledger.to_csv(dest/'proposal_ledger.csv.gz',index=False)
        pd.DataFrame(obs,columns=['cx','cy','length','angle']).assign(simulator_parent_id=parents).to_csv(dest/'observations.csv',index=False)
        dump(dest/'budget.json',dict(meta,**budget));dump(dest/'continuity.json',self.env['STAGE_LOG'][stage_start:]);dump(dest/'metrics.json',dict(meta,**metrics))
        r.candidate_gate=self.original_gate
        return dict(meta,**metrics)

    def group(self,frame):
        columns=[c for c in frame.select_dtypes('number').columns if c not in ['training_seed','generation_seed']]
        bywindow=frame.groupby(['parent_face','window_id','arm','variant','layout','condition'],dropna=False)[columns].mean().reset_index()
        byface=bywindow.groupby(['parent_face','arm','variant','layout','condition'],dropna=False)[columns].mean().reset_index()
        summary=byface.groupby(['arm','variant','layout','condition'],dropna=False)[columns].mean().reset_index()
        return bywindow,byface,summary

    def save_stage(self,stage,rows,note):
        dest=HERE/stage;dest.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(dest/'metrics.csv',index=False)
        for name,frame in zip(['window_means','face_means','face_equal_summary'],self.group(df)):frame.to_csv(dest/(name+'.csv'),index=False)
        (dest/'REPORT.md').write_text(note+'\n\nAll detailed predictions and ledgers are in ../cases/<case_key>/. Face means give equal weight to windows; seeds are averaged within windows. Faces receive equal weight in the summary. No significance test treats windows/realizations as independent faces.\n',encoding='utf-8')
        dump(dest/'COMPLETE.json',dict(rows=len(df),cases=int(df.case_key.nunique()),status='complete'));return df
