"""Frozen two-case comparison: local KDE-like and ADFNE-like proxy controls."""
from pathlib import Path
import copy,contextlib,hashlib,importlib.util,io,json,math,shutil,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent
LOCAL=STUDY/'local_window_retrain_2026-10-19'
PREV=STUDY/'minimal_original_method_preview_2026-10-19'
SPATIAL=STUDY/'spatial_placement_fix_trial_2026-10-19'
DEST=Path((__import__('os').environ['DFN3_CORE'] + '/_external_not_used'))
CHECKS=[]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def dump(name,x):(HERE/name).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf-8')
def check(name,ok,detail=''):
    CHECKS.append(dict(check=name,passed=bool(ok),detail=str(detail)));assert ok,name

def spatial(real,pred,aspect):
    # Distributional location errors, normalized by public width/height.
    a=real[:,:2]/[1,aspect];b=pred[:,:2]/[1,aspect]
    if not len(a) or not len(b):return dict(centre_energy_distance=math.nan,grid_TV=math.nan)
    energy=max(0,2*cdist(a,b).mean()-cdist(a,a).mean()-cdist(b,b).mean())**.5
    ha=np.histogram2d(a[:,0],a[:,1],bins=[np.linspace(0,1,6),np.linspace(0,1,5)])[0]/len(a)
    hb=np.histogram2d(b[:,0],b[:,1],bins=[np.linspace(0,1,6),np.linspace(0,1,5)])[0]/len(b)
    return dict(centre_energy_distance=float(energy),grid_TV=float(abs(ha-hb).sum()/2))

def main():
    torch.set_num_threads(2)
    if (HERE/'RUN_COMPLETE.json').exists():raise RuntimeError('Preserve completed comparison')
    (HERE/'TASK_LOG.md').write_text('# Matched local-window baseline check\n\nUser asked whether the new local model improves over ADFNE/KDE. Reviewer R1.1/R1.2 require fair comparisons and attribution. Existing strict-parity proxy results belong to a different CVAE, population and observation protocol and cannot answer this question. Smallest action: freeze the two existing cases/one seed; reuse learned predictions and compare transparent KDE-like/ADFNE-like controls with shared evidence, priors/counts, draw budgets, gates, scorer and continuity. Include full statistical pipelines and EVAE-fixed infill controls. No retraining, test tuning, window selection, manuscript edit or licensed-ADFNE claim. Stop after this preliminary comparison.\n',encoding='utf-8')
    m=load('baseline_preview_utilities',PREV/'run_preview.py');s=load('baseline_geometry_utilities',SPATIAL/'run_spatial_trial.py')
    r,env,models=m.load_state();r.HERE=HERE
    inputs={str(p):m.sha(p) for p in [Path(__file__),LOCAL/'evae.pt',LOCAL/'continuation_head.pt',LOCAL/'transformer_latent.pt',LOCAL/'training_only_statistics.json',LOCAL/'window_trace_manifest.csv',LOCAL/'window_inventory.csv',LOCAL/'preview_predictions.csv',LOCAL/'preview_metrics.csv',LOCAL/'count_budgets.csv']}
    dump('input_hashes.json',inputs)
    stats=m.keyrestore(json.loads((LOCAL/'training_only_statistics.json').read_text()))
    for key in ['fc','shares']:stats[key]=np.asarray(stats[key])
    stats['pools']={k:{n:np.asarray(v) for n,v in p.items()} for k,p in stats['pools'].items()};r.STATS=stats
    inv=pd.read_csv(LOCAL/'window_inventory.csv');tr=pd.read_csv(LOCAL/'window_trace_manifest.csv')
    r.FACES={}
    for row in inv.itertuples():
        if row.role=='test':continue
        a=tr[tr.window_id.eq(row.window_id)][['cx','cy','length','angle']].to_numpy(np.float32)
        r.FACES[row.window_id]=dict(id=row.window_id,w=20.,h=row.height_m,ym=row.height_m/20,role=row.role,arr=a)
    fitting=np.vstack([f['arr'] for f in r.FACES.values() if f['role']=='train'])
    canonical=[]
    for f in r.FACES.values():
        if f['role']=='train':a=f['arr'].copy();a[:,:3]*=20/36;canonical.append(a)
    env['CONT']=r.definitions(r.EXP/'hybrid_v6.py',dict(load_panels_raw=lambda:canonical,get_physics_stats=lambda:{'fam_centers_rad':stats['fc']},assign_family_ids=r.assign,segments_intersect=r.intersect,FAM_ASSIGN_THRESHOLD_DEG=20.),names=['apply_continuity_rules'])
    with contextlib.redirect_stdout(io.StringIO()):gs=r.geo['_train_geology_stats'](canonical,stats['fc'])
    env['SPACING_METRES']={k:float(gs['spacing_floor'][k])*36 for k in range(3)};env['SPACING_METRES'][3]=.288
    env['TRAIN_X_STD']=float(fitting[:,0].std());env['TRAIN_Y_STD']=float(fitting[:,1].std())
    ev=env['MaskedRectangularEVAE'](N=128,yscale=stats['yscale']).to(r.DEV)
    ev.load_state_dict(torch.load(LOCAL/'evae.pt',map_location=r.DEV,weights_only=True)['state_dict']);ev.eval();models['evae']=ev
    ck=torch.load(LOCAL/'continuation_head.pt',map_location=r.DEV,weights_only=True)
    head=r.gen['EVAEContHead']().to(r.DEV);head.load_state_dict(ck['state_dict']);head.eval();models['head']=(head,ck['X_mean'],ck['X_std'])
    original=r.statistical_proposals
    def propose(face,total,mode,rng):
        if mode=='kde':a=original(face,total,'kde',rng)
        elif mode=='adfne_proxy':
            # Same train-derived family information; transparent histogram angle
            # mixture + lognormal length + uniform location. Not licensed ADFNE.
            families=rng.choice(4,total,p=r.STATS['shares']);rows=[]
            for family in families:
                pool=r.STATS['pools'][int(family)];hist,edges=np.histogram(pool['angles'],bins=12,range=(0,np.pi));prob=(hist+1)/(hist.sum()+12)
                k=int(rng.choice(12,p=prob));angle=rng.uniform(edges[k],edges[k+1]);logs=np.log(pool['lengths_m'])
                length=math.exp(float(rng.normal(logs.mean(),max(logs.std(),.08))))
                rows.append([rng.uniform(),rng.uniform(0,face['ym']),length/face['w'],angle])
            a=np.asarray(rows,np.float32)
        else:a=original(face,total,mode,rng)
        return s.clip_outer(r,a,face) # Same public-frame boundary clipping as neural candidates.
    r.statistical_proposals=propose
    arms={'KDE_full':['stat','kde',True], 'ADFNE_proxy_full':['stat','adfne_proxy',True],
          'KDE_EVAEfixed':['evae','kde',True], 'ADFNE_proxy_EVAEfixed':['evae','adfne_proxy',True]}
    r.PROTOCOL['arms'].update(arms)
    dump('PROTOCOL.json',dict(cases=json.loads((LOCAL/'PRESELECTED_CASES.json').read_text()),seed=42001,arms=arms,learned='Frozen saved outputs, no new neural realization',KDE='Family-conditioned product kernels on axial angle/log-length; Scott n^(-1/5) bandwidth; uniform location; matches existing revision KDE-like helper',ADFNE_proxy='Family-conditioned smoothed 12-bin axial histogram, lognormal length, uniform location; local transparent proxy, not licensed ADFNE',shared=['S1 fitted priors and family definitions','same observed 0.1m strips','same training-density total count','644 raw candidate draws per case/arm','same public boundary clipping and hard gates','same seven-feature scorer and quota/spacing checks','same continuity and final guards'],attribution='EVAEfixed controls isolate the Transformer infill pathway with identical continuation. Full controls replace both continuation and infill with statistics.',limitations=['Two adjacent windows on one S2 face, one seed; exploratory only, no independent replicates/significance.','KDE-like product-kernel control, not every possible KDE implementation.','No licensed ADFNE comparison.','Neural eligible pool retains only one valid draw per 24 local draws; statistical pool retains all valid raw draws. Same raw supply, native selection bottleneck disclosed.']))
    oldpred=pd.read_csv(LOCAL/'preview_predictions.csv');oldmet=pd.read_csv(LOCAL/'preview_metrics.csv');oldbu=pd.read_csv(LOCAL/'count_budgets.csv')
    cases=json.loads((LOCAL/'PRESELECTED_CASES.json').read_text());cm,_,_=s.training_references(r);base=copy.deepcopy(stats)
    metrics=[];outputs=[];ledgers=[];budgets=[]
    for fid,layout in cases:
        row=inv[inv.window_id.eq(fid)].iloc[0];public=dict(id=fid,w=20.,h=float(row.height_m),ym=float(row.height_m)/20)
        saved=oldpred[oldpred.window_id.eq(fid)&oldpred.layout.eq(layout)]
        truth=saved[saved.tier.eq('mapped')][['cx','cy','length','angle']].to_numpy(np.float32)
        obs=saved[saved.tier.eq('measured_clips')][['cx','cy','length','angle']].to_numpy(np.float32)
        st=r.strips(layout,20)
        # Observation simulation for scoring hidden-only reference, not generation.
        _,parents=r.observe(truth,st);hidden=truth[~np.isin(np.arange(len(truth)),np.unique(parents))]
        learned=saved[saved.tier.isin(['anchored','inferred'])][['cx','cy','length','angle']].to_numpy(np.float32)
        linf=saved[saved.tier.eq('inferred')][['cx','cy','length','angle']].to_numpy(np.float32)
        def score(arm,final,inf):
            met=r.metrics(truth,final,20,public['h'],obs)
            met={k:v for k,v in met.items() if not any(k.startswith(x) for x in ['I_share','Y_share','X_share','components','largest_component','top_bottom'])}
            met.update(spatial(truth,final,public['ym']))
            met.update(P21_abs_error=abs(met['P21_pred']-met['P21_truth']),intersection_density_abs_error=abs(met['intersections_per_m2_pred']-met['intersections_per_m2_truth']),inferred_count=len(inf),hidden_truth_count=len(hidden),inferred_direction_W1=r.axial_w1(hidden[:,3],inf[:,3]),inferred_length_W1_m=r.wasserstein_distance(hidden[:,2]*20,inf[:,2]*20) if len(hidden) and len(inf) else math.nan)
            return dict(window_id=fid,layout=layout,arm=arm,seed=42001,**met)
        metrics.append(score('Learned_frozen',learned,linf))
        check('frozen_learned_metric_reproduced:'+fid,abs(metrics[-1]['axial_w1_deg']-oldmet[oldmet.window_id.eq(fid)].iloc[0].axial_w1_deg)<1e-5)
        for arm in arms:
            s.corrected_references(r,base,public,cm)
            final,inf,fixed,ledger,budget=env['reconstruct'](public,obs,st,arm,models,42001)
            p,q=r.ends(final);z=np.stack([p,q],1);bad=((z[:,:,0]<-1e-7)|(z[:,:,0]>1+1e-7)|(z[:,:,1]<-1e-7)|(z[:,:,1]>public['ym']+1e-7)).any(1)
            if bad.any():final[bad]=s.clip_outer(r,final[bad],public)
            anchor=np.array([any(env['contains'](o,a,20) for o in obs) for a in final],bool);fixed=final[anchor];inf=final[~anchor]
            p,q=r.ends(final);z=np.stack([p,q],1)
            check('bounds:'+fid+':'+arm,np.isfinite(z).all() and ((z[:,:,0]>=-1e-6)&(z[:,:,0]<=1+1e-6)&(z[:,:,1]>=-1e-6)&(z[:,:,1]<=public['ym']+1e-6)).all())
            check('measured_clips_preserved:'+fid+':'+arm,env['preservation'](obs,final,20)==1.)
            ref=oldbu[oldbu.window_id.eq(fid)].iloc[0]
            check('same_budget_and_target:'+fid+':'+arm,budget['raw_budget']==ref.raw_budget and budget['target_count']==ref.target_count)
            if arm.endswith('EVAEfixed'):
                anchored=saved[saved.tier.eq('anchored')][['cx','cy','length','angle']].to_numpy()
                check('identical_EVAE_anchors:'+fid+':'+arm,len(fixed)==len(anchored) and np.allclose(np.sort(fixed,axis=0),np.sort(anchored,axis=0),atol=2e-6))
            metrics.append(score(arm,final,inf));budgets.append(dict(window_id=fid,layout=layout,arm=arm,**budget));ledgers.append(ledger.assign(window_id=fid,layout=layout,arm=arm))
            for tier,a in [('anchored',fixed),('inferred',inf),('measured_clips',obs)]:outputs.append(pd.DataFrame(a,columns=['cx','cy','length','angle']).assign(window_id=fid,layout=layout,arm=arm,tier=tier))
            print(arm,layout,'N',len(final),'direction',round(metrics[-1]['axial_w1_deg'],2),'length',round(metrics[-1]['length_w1_m'],2),flush=True)
    df=pd.DataFrame(metrics);df.to_csv(HERE/'paired_metrics.csv',index=False)
    pd.concat(outputs).to_csv(HERE/'baseline_predictions.csv',index=False);pd.concat(ledgers).to_csv(HERE/'baseline_proposal_ledger.csv.gz',index=False);pd.DataFrame(budgets).to_csv(HERE/'count_budget_parity.csv',index=False)
    endpoints=['axial_w1_deg','length_w1_m','matched_angle_MAE_deg','matched_length_MAE_m','count_error_pct','P21_abs_error','intersection_density_abs_error','centre_energy_distance','grid_TV','inferred_direction_W1','inferred_length_W1_m']
    delta=[]
    for fid,layout in cases:
        block=df[df.window_id.eq(fid)];learned=block[block.arm.eq('Learned_frozen')].iloc[0]
        for control in block[block.arm.ne('Learned_frozen')].itertuples():
            for endpoint in endpoints:
                lv=float(learned[endpoint]);cv=float(getattr(control,endpoint))
                delta.append(dict(window_id=fid,layout=layout,control=control.arm,endpoint=endpoint,learned=lv,control_value=cv,learned_minus_control=lv-cv,learned_better=lv<cv-1e-8,tie=abs(lv-cv)<=1e-8))
    pd.DataFrame(delta).to_csv(HERE/'paired_differences.csv',index=False)
    rows=['| Window/layout | Arm | Direction W1 (deg) ↓ | Length W1 (m) ↓ | Count error (%) ↓ | P21 abs error ↓ | Centre energy distance ↓ |','|---|---|---:|---:|---:|---:|---:|']
    for x in df.itertuples():rows.append(f'| {x.layout} | {x.arm} | {x.axial_w1_deg:.2f} | {x.length_w1_m:.2f} | {x.count_error_pct:.1f} | {x.P21_abs_error:.3f} | {x.centre_energy_distance:.3f} |')
    report='''# Matched local-window model/baseline check — 2026-10-19

This compares the frozen learned previews with local KDE-like and ADFNE-like proxy controls. It is not a comparison with licensed ADFNE. The earlier CVAE baseline table cannot be transferred to this experiment.

One fixed seed, two preselected adjacent S2 windows on one face; no retraining, parameter search or best-image selection. All methods use S1 fit-only priors, the same observations, public geometry, training-density count rule (14), raw draw budget (644), boundary clipping, hard gates, geological scorer, family/spacing checks and continuity. All target endpoints are scored only after generation. The same physical observations are preserved.

`*_full` uses statistical indicated continuation and statistical infill. `*_EVAEfixed` keeps the identical learned EVAE continuation and replaces only Transformer infill; this isolates that pathway. Learned results reuse the existing saved realization. KDE-like uses the existing family-conditioned product kernels in angle/log-length and uniform position. ADFNE-like uses family-conditioned smoothed axial histograms, lognormal length and uniform position. Both are explicit local approximations; neither label covers all external implementations. The control settings were fixed without S2 tuning.

'''+ '\n'.join(rows)+'''

All endpoints, including Hungarian-matched errors, hidden-only/inferred distributions, spatial grid discrepancy and intersection-density error, are in paired_metrics.csv. Lower errors are better. P21 and intersection endpoints are absolute errors against the mapped window, not the raw quantities; count agreement alone is not enough. Legacy node-type metrics are excluded rather than promoted as validated topology.

The neural sampler exposes at most one eligible candidate per 24 local draws, while statistical samplers expose all geometrically valid draws. Raw supply and scorer subset limits are identical; the different native proposal filtering is reported in the ledgers. This is a pipeline comparison, not an equal-eligible-pool experiment.

These small development previews cannot support significance or general superiority/equivalence claims. Adjacent windows and candidate/trace counts are not independent walls. Observed fractions remain 28.6% and 40%; no ≤10% sparsity claim. S2 was previously inspected. The report evaluates where errors are lower, without tuning to remove negative results.
'''
    (HERE/'BASELINE_COMPARISON_REPORT.md').write_text(report,encoding='utf-8')
    for p,h in inputs.items():check('frozen_input_preserved:'+p,m.sha(Path(p))==h)
    pd.DataFrame(CHECKS).to_csv(HERE/'comparison_checks.csv',index=False)
    dump('RUN_COMPLETE.json',dict(status='two_case_matched_comparison_complete',controls=4,networks_generated=8,learned_outputs_reused=2,training_runs=0,all_checks_passed=True,checks=len(CHECKS),licensed_ADFNE=False,independent_units='one face, two adjacent windows; no significance test'))
    DEST.mkdir(parents=True,exist_ok=True)
    for name in ['BASELINE_COMPARISON_REPORT.md','paired_metrics.csv','paired_differences.csv','PROTOCOL.json']:
        shutil.copy2(HERE/name,DEST/('LOCAL_BASELINE_2026-10-19_'+name))
    print(df[['layout','arm','axial_w1_deg','length_w1_m','count_error_pct','centre_energy_distance']].to_string(index=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:dump('RUN_FAILURE.json',dict(error=repr(exc)));raise
