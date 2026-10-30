"""One S1-only local-window retraining; two fixed S2 preview cases."""
from pathlib import Path
import ast, contextlib, copy, importlib.util, io, json, math, shutil, sys, time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
REPAIR=STUDY/'s1_s2_repair_trial_2026-10-18'
SPATIAL=STUDY/'spatial_placement_fix_trial_2026-10-19'
PREV=STUDY/'minimal_original_method_preview_2026-10-19'
DEST=Path(r'C:\Users\User\OneDrive - UNSW\[4] Files\PhD Thesis\Papers\2nd paper\final_wokr\REvision\test_files')
SEED=42001
CHECKS=[]

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def check(name,ok,detail=''):
    CHECKS.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    assert ok, name+': '+str(detail)

def windows(r):
    labels=pd.read_csv(STUDY/'data_prep/manual_trace_manifest.csv')
    geom=pd.read_csv(REPAIR/'face_split_geometry.csv')
    result={};inventory=[];records=[]
    for row in geom.sort_values('face_id').itertuples():
        data=labels[labels.face_id.eq(row.face_id)]
        w=float(row.width_m);h=float(row.height_up_face_m)
        p=data[['u1_m','v1_m']].to_numpy()+[0,h]
        q=data[['u2_m','v2_m']].to_numpy()+[0,h]
        for index in range(int(w//20)):
            left=index*20.;right=left+20;v=q-p
            den=v[:,0];parallel=np.abs(den)<1e-12;safe=np.where(parallel,1,den)
            aa=(left-p[:,0])/safe;bb=(right-p[:,0])/safe
            lo=np.maximum(0,np.minimum(aa,bb));hi=np.minimum(1,np.maximum(aa,bb))
            lo[parallel]=0;hi[parallel]=1
            good=(hi>lo+1e-9)&(~parallel|((p[:,0]>=left)&(p[:,0]<right)))
            ix=np.flatnonzero(good)
            pp=p[ix]+lo[ix,None]*v[ix]-[left,0]
            qq=p[ix]+hi[ix,None]*v[ix]-[left,0]
            arr=r.lines(pp/20,qq/20)
            fid=f'{row.face_id}__X{index:02d}'
            result[fid]=dict(id=fid,parent_face=row.face_id,role=row.role,w=20.,h=h,ym=h/20,arr=arr,x0=left)
            inventory.append(dict(window_id=fid,parent_face=row.face_id,role=row.role,section=row.section,x0_m=left,x1_m=right,width_m=20,height_m=h,traces=len(arr),boundary_clipped_traces=int(((lo[ix]>1e-8)|(hi[ix]<1-1e-8)).sum()),discarded_right_remainder_m=w-20*int(w//20)))
            for k,j in enumerate(ix):
                records.append(dict(window_id=fid,parent_face=row.face_id,role=row.role,trace_id=data.iloc[j].trace_id,boundary_clipped=bool(lo[j]>1e-8 or hi[j]<1-1e-8),cx=arr[k,0],cy=arr[k,1],length=arr[k,2],angle=arr[k,3]))
    inv=pd.DataFrame(inventory);inv.to_csv(HERE/'window_inventory.csv',index=False)
    trace=pd.DataFrame(records);trace.to_csv(HERE/'window_trace_manifest.csv',index=False)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        check('parent_face_disjoint:'+a+':'+b,not(set(inv[inv.role==a].parent_face)&set(inv[inv.role==b].parent_face)))
        check('trace_id_disjoint:'+a+':'+b,not(set(trace[trace.role==a].trace_id)&set(trace[trace.role==b].trace_id)))
    for fid,f in result.items():
        p,q=r.ends(f['arr']);z=np.vstack([p,q])
        check('crop_in_bounds:'+fid,not len(z) or (z[:,0].min()>=-1e-5 and z[:,0].max()<=1.00001 and z[:,1].min()>=-1e-5 and z[:,1].max()<=f['ym']+1e-5))
    return result

def make_examples(r,faces,role):
    out=[]
    for face in faces.values():
        if face['role']!=role or not len(face['arr']):continue
        for layout in ['centre','5m']:
            for aug in (['orig','rot180'] if role=='train' else ['orig']):
                a=r.augment(face,aug);st=r.strips(layout,face['w']);obs,parents=r.observe(a,st)
                out.append(dict(face=face,layout=layout,arr=a,obs=obs,parents=parents,strips=st))
    return out

def main():
    start=time.time();torch.set_num_threads(2)
    if (HERE/'RUN_COMPLETE.json').exists():raise RuntimeError('Preserve completed run')
    m=load('local_preview_utils',PREV/'run_preview.py')
    s=load('local_geometry_utils',SPATIAL/'run_spatial_trial.py')
    r,env,unused=m.load_state();del unused
    r.HERE=HERE
    refs=json.loads((STUDY/'FOCUSED_REVISION_PLAN_MANIFEST_2026-10-18.json').read_text())['protected_source_hashes']
    # Read the submitted manuscript itself; preserve a text reference for this task.
    from docx import Document
    doc=next(Path(p) for p in refs if p.endswith('Paudyal_et_al_DFN_Reconstruction.docx'))
    text='\n'.join(p.text for p in Document(doc).paragraphs)
    (HERE/'submitted_reference_text.txt').write_text(text,encoding='utf-8')
    inputs={str(p):m.sha(p) for p in [Path(__file__),STUDY/'data_prep/manual_trace_manifest.csv',REPAIR/'face_split_geometry.csv',REPAIR/'repair_trial.py',REPAIR/'geometry_aware_training_definition.py',PREV/'run_preview.py',SPATIAL/'run_spatial_trial.py',SPATIAL/'supported_sampled_generator.py']}
    dump('inputs_before.json',inputs)
    protocol=dict(user_request='Retrain on local S1 windows and show separate mapped versus generated S2 examples.',reviewer_links=['R1.3 spatial validation','R2.3 mapped spatial context','verified scale/support defect in whole-face rollout'],architecture='Original 12/20 dual-latent EVAE, sequential Transformer V4 generator, original seven-feature deterministic scorer; corrected adapter, not exact historical reproduction',window='20 m width, original full face height, nonoverlapping x tiles; incomplete right remainder excluded',split_before_crop=True,training_faces=r.PROTOCOL['train_faces'],validation_faces=r.PROTOCOL['validation_faces'],heldout='Physical S2; previously inspected development section, held out from this fit and selection',seed=SEED,epochs=dict(evae=150,continuation_head=50,transformer=50),layouts=['centre','5m'],strip_width_m=.1,selection='S1 validation losses only',new_architecture=False,hyperparameter_search=False,statistical_infill_fallback=False,population='S1 window density times public target window area; no target count',sampling='Original family/length/angle posterior; bounded positions with S1 window fitting residuals; uniform valid local draw; trained horizon',stopping='One configuration, one seed, two preselected S2 examples; no best-window/seed search',censoring='Window-boundary lengths are censored 2D mapped segments, not complete physical fracture lengths',limitations=['Adjacent windows from one face are not independent replicates.','Cross-section validation is same-mine, not an independent mine.','This preview does not resolve all reviewer requirements or demonstrate superiority.'])
    dump('PROTOCOL.json',protocol)
    (HERE/'TASK_LOG.md').write_text('# Local-window retraining preview\n\nUser authorized S1 local-window training and separate S2 previews, 2026-10-19. This addresses the verified mismatch between whole-face generation and local training support, plus reviewer R1.3/R2.3 spatial validation/context. Existing archived hybrid uses a different architecture; existing original-method checkpoints trained on whole faces. Therefore retrain original components on a fixed 20 m window protocol. Preserve prior failures. One seed/configuration, fixed cases, no Word, no manuscript edits. S2 was previously inspected and is development validation, not a pristine test.\n',encoding='utf-8')
    r.FACES=windows(r)
    # Lock cases using geometry/order before learning or generation.
    testids=sorted(k for k,f in r.FACES.items() if f['role']=='test')
    cases=[(testids[0],'centre'),(testids[1],'5m')]
    dump('PRESELECTED_CASES.json',cases)
    r.STATS=r.fit_stats(r.FACES)
    train=make_examples(r,r.FACES,'train');val=make_examples(r,r.FACES,'validation')
    check('training_targets_within_128',max(len(e['arr']) for e in train+val)<=128)
    check('examples_exclude_S2',all(e['face']['role']!='test' for e in train+val))
    canonical=[]
    fitting=np.vstack([f['arr'] for f in r.FACES.values() if f['role']=='train' and len(f['arr'])])
    for f in r.FACES.values():
        if f['role']=='train' and len(f['arr']):
            a=f['arr'].copy();a[:,:3]*=f['w']/36;canonical.append(a)
    env['CONT']=r.definitions(r.EXP/'hybrid_v6.py',dict(load_panels_raw=lambda:canonical,get_physics_stats=lambda:{'fam_centers_rad':r.STATS['fc']},assign_family_ids=r.assign,segments_intersect=r.intersect,FAM_ASSIGN_THRESHOLD_DEG=20.),names=['apply_continuity_rules'])
    with contextlib.redirect_stdout(io.StringIO()):gs=r.geo['_train_geology_stats'](canonical,r.STATS['fc'])
    env['SPACING_METRES']={i:float(gs['spacing_floor'][i])*36 for i in range(3)};env['SPACING_METRES'][3]=.288
    env['TRAIN_X_STD']=float(fitting[:,0].std());env['TRAIN_Y_STD']=float(fitting[:,1].std())
    r.legacy['DualLatentVAE']=env['MaskedRectangularEVAE'];r.forward_evae=env['forward_evae']
    exec(compile((REPAIR/'geometry_aware_training_definition.py').read_text(),str(REPAIR/'geometry_aware_training_definition.py'),'exec'),r.__dict__)
    r.seedall(SEED);logs=[];selected={};models={}
    print(f'Window inventory: {pd.read_csv(HERE/"window_inventory.csv").groupby("role").size().to_dict()}; {len(train)} train examples, {len(val)} validation examples; device={r.DEV}',flush=True)
    models['evae'],ll,selected['evae']=r.train_evae(train,val,r.STATS,SEED);logs+=ll
    torch.save(dict(state_dict=models['evae'].state_dict(),selected_epoch=selected['evae'],protocol=protocol),HERE/'evae.pt')
    pd.DataFrame(logs).to_csv(HERE/'training_log.csv',index=False)
    models['head'],ll,selected['head']=env['train_head'](train,val,models['evae']);logs+=ll
    head,mu,sd=models['head'];torch.save(dict(state_dict=head.state_dict(),X_mean=mu,X_std=sd,selected_epoch=selected['head']),HERE/'continuation_head.pt')
    seqs={}
    for role,exs in [('train',train),('validation',val)]:
        items=[]
        for e in exs:
            x=r.sequence_example(e,models,True)
            if not len(x['tgt']):continue
            ix=np.argsort(x['tgt'][:,0],kind='stable');x['tgt']=x['tgt'][ix];x['fam']=x['fam'][ix]
            x.update(aspect=e['face']['ym'],layout=e['layout'],face_id=e['face']['id'])
            items.append(x)
        seqs[role]=items
    Base=r.trans['SeqFracTransformerV4']
    class HeightTransformer(Base):
        def forward(self,ctx,ctx_mask,tgt,tgt_mask,sz=None,gz=None):
            p,l,a,f=super().forward(ctx,ctx_mask,tgt,tgt_mask,sz,gz)
            aspect=torch.as_tensor(self.aspect,device=p.device,dtype=p.dtype).reshape(-1,1,1)
            p=torch.cat([p[:,:,:1],p[:,:,1:]*aspect],-1);self.last_position=p
            return p,l,a,f
    net=HeightTransformer().to(r.DEV)
    tau=max(float(np.std(fitting[:,:2],axis=0).sum()),1e-3)
    opt=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-5);best=math.inf
    for ep in range(1,51):
        net.train();ls=[]
        for ix in np.array_split(np.random.permutation(len(seqs['train'])),math.ceil(len(seqs['train'])/8)):
            items=[seqs['train'][i] for i in ix];s.set_aspect(net,items,r.DEV)
            loss,parts=s.get_losses(m,net,r.seq_batch(items),r,tau)
            check('finite_transformer_loss',bool(torch.isfinite(loss)))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();ls.append(float(loss))
        net.eval();vs=[]
        with torch.no_grad():
            for x in seqs['validation']:
                s.set_aspect(net,[x],r.DEV);loss,_=s.get_losses(m,net,r.seq_batch([x]),r,tau);vs.append(float(loss))
        score=float(np.mean(vs));logs.append(dict(component='transformer',epoch=ep,train=float(np.mean(ls)),validation=score))
        if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};selected['transformer']=ep
        if ep%10==0:print(f'Local Transformer {ep}/50 validation={score:.4f}',flush=True)
    net.load_state_dict(state);net.eval();models['transformer_latent']=net
    torch.save(dict(state_dict=state,selected_epoch=selected['transformer'],public_height_scaled_output=True,protocol=protocol),HERE/'transformer_latent.pt')
    pd.DataFrame(logs).to_csv(HERE/'training_log.csv',index=False);dump('selected_epochs.json',selected)
    # Fit position uncertainty strictly on S1 fitting examples, never S2.
    errors={};horizons={}
    with torch.no_grad():
        for x in seqs['train']:
            s.set_aspect(net,[x],r.DEV);b=r.seq_batch([x]);s.get_losses(m,net,b,r,tau)
            err=(b[2][0,:,:2]-net.last_position[0]).cpu().numpy()/[1,x['aspect']]
            errors.setdefault(x['layout'],[]).append(err)
            horizons[x['layout']]=max(horizons.get(x['layout'],1),len(x['tgt']))
    errors={k:np.vstack(v) for k,v in errors.items()}
    np.savez_compressed(HERE/'training_position_residuals.npz',**errors)
    dump('trained_horizons.json',horizons)
    original=env['posterior_draws'];current={}
    def posterior(position,loglength,angle,family_logits,face,rng,n):
        a,_,fids=original(position,loglength,angle,family_logits,face,rng,n)
        pos=np.asarray(position)/[1,face['ym']];e=errors[current['layout']]
        eligible=np.flatnonzero(np.all((pos+e>=0)&(pos+e<=1),axis=1))
        if not len(eligible):a[:]=0;return a,np.zeros(n),fids
        a[:,:2]=(pos+e[rng.choice(eligible,n,replace=True)])*[1,face['ym']]
        return s.clip_outer(r,a,face),np.zeros(n),fids
    env['posterior_draws']=posterior
    exec(compile((SPATIAL/'supported_sampled_generator.py').read_text(),str(SPATIAL/'supported_sampled_generator.py'),'exec'),env)
    env['sample_neural']=env['sample_neural_sampled']
    cm,_,_=s.training_references(r);base=copy.deepcopy(r.STATS)
    metrics=[];predictions=[];ledgers=[];budgets=[]
    for num,(fid,layout) in enumerate(cases,1):
        f=r.FACES[fid];st=r.strips(layout,f['w']);obs,parents=r.observe(f['arr'],st)
        check('inference_public_geometry_only',f['role']=='test')
        # Generation receives geometry and observations only, not hidden trace arrays.
        public={k:v for k,v in f.items() if k!='arr'}
        s.corrected_references(r,base,public,cm)
        net.aspect=torch.tensor([f['ym']],device=r.DEV);current['layout']=layout;env['HORIZON']=min(128,horizons[layout])
        final,inf,fixed,ledger,budget=env['reconstruct'](public,obs,st,'E',models,SEED)
        before=final.copy();trim=s.clip_outer(r,final,public)
        check('final_trim_does_not_remove_traces',bool((trim[:,2]>0).all()))
        # Keep original if inside to avoid trimming edge observations unnecessarily.
        p,q=r.ends(final);z=np.stack([p,q],1)
        bad=((z[:,:,0]<-1e-7)|(z[:,:,0]>1+1e-7)|(z[:,:,1]<-1e-7)|(z[:,:,1]>f['ym']+1e-7)).any(1)
        final[bad]=trim[bad]
        check('measured_preservation_2mm:'+fid,env['preservation'](obs,final,20)==1.)
        p,q=r.ends(final);z=np.stack([p,q],1)
        check('final_coordinates_finite:'+fid,np.isfinite(z).all())
        check('final_coordinates_inside:'+fid,((z[:,:,0]>=-1e-6)&(z[:,:,0]<=1+1e-6)&(z[:,:,1]>=-1e-6)&(z[:,:,1]<=f['ym']+1e-6)).all())
        anchored=np.array([any(env['contains'](o,a,20) for o in obs) for a in final],bool)
        fixed=final[anchored];inf=final[~anchored]
        meta=dict(window_id=fid,parent_face=f['parent_face'],layout=layout,seed=SEED)
        met=r.metrics(f['arr'],final,20,f['h'],obs)
        # Do not promote legacy pairwise topology proxies as validated node statistics.
        met={k:v for k,v in met.items() if not any(k.startswith(x) for x in ['I_share','Y_share','X_share','components','largest_component','top_bottom'])}
        metrics.append(dict(meta,**met,inferred_count=len(inf),anchored_count=len(fixed),unique_observed_truth_traces=len(np.unique(parents)),observed_trace_fraction=len(np.unique(parents))/max(1,len(f['arr'])),post_continuity_trimmed_traces=int(bad.sum())))
        budgets.append(dict(meta,**budget));ledgers.append(ledger.assign(**meta))
        for tier,a in [('mapped',f['arr']),('anchored',fixed),('inferred',inf),('measured_clips',obs)]:predictions.append(pd.DataFrame(a,columns=['cx','cy','length','angle']).assign(tier=tier,**meta))
        fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
        def add(ax,a,color,lw=1.5):
            p,q=r.ends(a);ax.add_collection(LineCollection(np.stack([p,q],1)*20,colors=color,linewidths=lw))
        add(axes[0],f['arr'],'#333333')
        add(axes[1],fixed,'#e58a13');add(axes[1],inf,'#167ba1');add(axes[1],obs,'#cc2233',2.3)
        for ax in axes:
            for lo,hi in st:ax.axvline((lo+hi)*10,color='#aaaaaa',ls='--',lw=.8)
            ax.set(xlim=(0,20),ylim=(0,f['h']),xlabel='Local along-face (m)',ylabel='Up-face (m)');ax.set_aspect('equal')
        axes[0].set_title(f'Mapped S2 window | {len(f["arr"])} traces')
        axes[1].set_title(f'Retrained original components | {len(final)} traces\n{len(fixed)} anchored + {len(inf)} inferred')
        fig.suptitle(f'S1 local-window training → physical S2 preview {num}\n{fid} | parent x={f["x0"]:.0f}–{f["x0"]+20:.0f} m | {layout} | seed {SEED}',fontsize=11)
        fig.supxlabel(f'Orange: continued observations; blue: generated infill; red: measured clips.\nDirection W1={met["axial_w1_deg"]:.2f}°; clipped-length W1={met["length_w1_m"]:.2f} m. First fixed realization, not selected for appearance.',fontsize=10)
        image=HERE/f'LOCAL_RETRAIN_PREVIEW_{num}_{layout}.png';fig.savefig(image,dpi=180);plt.close(fig)
        print(f'Preview {fid}/{layout}: mapped={len(f["arr"])}, generated={len(final)}, inferred={len(inf)}, directionW1={met["axial_w1_deg"]:.2f}',flush=True)
    pd.DataFrame(metrics).to_csv(HERE/'preview_metrics.csv',index=False)
    pd.concat(predictions).to_csv(HERE/'preview_predictions.csv',index=False)
    pd.concat(ledgers).to_csv(HERE/'proposal_ledger.csv.gz',index=False)
    pd.DataFrame(budgets).to_csv(HERE/'count_budgets.csv',index=False)
    pd.DataFrame(env['STAGE_LOG']).to_csv(HERE/'continuity_stages.csv',index=False)
    for p,h in {**inputs,**refs}.items():check('input_preserved:'+p,m.sha(Path(p))==h)
    pd.DataFrame(CHECKS+r.CHECKS).to_csv(HERE/'execution_checks.csv',index=False)
    report='# Local-window retraining results — 2026-10-19\n\nOne fixed configuration retrained the original-parameter EVAE, continuation head and sequential Transformer on S1 20 m windows. The scorer is the original seven-feature deterministic scorer. Masking, rectangular geometry, consistent crossing statistics, x ordering and fit-residual position sampling retain the previously documented adapter corrections. This is not an exact numerical reproduction of the submitted pipeline and does not use the later GAST architecture.\n\n'
    report+=pd.DataFrame(metrics)[['window_id','layout','truth_count','predicted_count','inferred_count','axial_w1_deg','length_w1_m','observed_trace_fraction']].to_markdown(index=False)
    report+='\n\nSplit parent faces before crops; training and validation use S1 only. Physical S2 is excluded from gradient updates, priors, residual calibration, early stopping and checkpoint selection. It has been inspected in earlier experiments, so these are additional cross-section development previews rather than pristine test results. Cases were fixed before training, one seed, no best-image selection. Window fragments are dependent within faces and have censored lengths. Observed fractions are reported without claiming ≤10%. No statistical superiority or full-bench capability is established. No Word document or manuscript edit.\n\nAll predicted lines and observations are saved in preview_predictions.csv. Checkpoints, logs, input hashes, geometry/trace manifests, budget/rejection ledger and execution checks are retained here.\n'
    (HERE/'LOCAL_RETRAIN_REPORT.md').write_text(report,encoding='utf-8')
    DEST.mkdir(parents=True,exist_ok=True)
    for p in list(HERE.glob('LOCAL_RETRAIN_PREVIEW_*.png'))+[HERE/'LOCAL_RETRAIN_REPORT.md',HERE/'PROTOCOL.json',HERE/'preview_metrics.csv']:
        shutil.copy2(p,DEST/('LOCAL_WINDOW_2026-10-19_'+p.name))
    dump('RUN_COMPLETE.json',dict(status='local_window_preview_complete',epochs=selected,training_windows=sum(f['role']=='train' for f in r.FACES.values()),validation_windows=sum(f['role']=='validation' for f in r.FACES.values()),S2_preview_cases=2,new_models_trained=3,seed=SEED,checks_passed=all(c['passed'] for c in CHECKS+r.CHECKS),elapsed_seconds=time.time()-start,output_folder=str(DEST)))
    print((HERE/'RUN_COMPLETE.json').read_text(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('RUN_FAILURE.json',dict(error=repr(exc)))
        pd.DataFrame(CHECKS).to_csv(HERE/'execution_checks_on_failure.csv',index=False)
        raise
