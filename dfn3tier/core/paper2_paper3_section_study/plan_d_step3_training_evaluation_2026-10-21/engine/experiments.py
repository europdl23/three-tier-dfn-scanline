"""Saved-pool causal experiments and independent S1 evaluations."""
from runtime import *
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
COLS=['cx','cy','length','angle']
def saved_meta():
    df=pd.read_csv(OLD/'02_repeated/metrics.csv');df=df[df.arm.isin(['E','KDE','ADFNE'])]
    assert len(df)==144 and df.role.eq('validation').all() and df.window_id.str.startswith('W').all();return df
def pool(e,row):
    fid=row['window_id'];face=e.faces[fid];pub=public_face(face);r=e.r
    assert face['role']=='validation' and fid.startswith('W')
    e.s.corrected_references(r,e.base,pub,e.cm)
    obs,parents,st,_=e.observe(face,'5m','clean',int(row['generation_seed']))
    path=OLD/'cases'/row['case_key'];ledger=pd.read_csv(path/'proposal_ledger.csv.gz');raw=ledger[COLS].to_numpy(np.float32)
    pred=pd.read_csv(path/'predictions.csv');oldobs=pred[pred.tier.eq('measured_clips')][COLS].to_numpy(np.float32)
    assert np.allclose(obs,oldobs,atol=1e-7,rtol=0),'Observation replay differs'
    bundle=e.load_bundle(42001);mode,proposal,_=r.PROTOCOL['arms'][row['arm']];rng=np.random.default_rng(int(row['generation_seed']))
    fixed=e.env['indicated'](obs,pub,st,mode,bundle['models'],rng)
    if row['arm']!='E':
        generated=r.statistical_proposals(pub,len(raw),proposal,rng)
        assert np.array_equal(generated,raw),'Statistical pool replay differs'
    eligible=~ledger.disposition.eq('posterior_not_selected').to_numpy()
    # Hard-invalid eligibility is unobservable historically and immaterial; never count it as gate-valid.
    valid=np.array([r.candidate_gate(x,pub,st) is None for x in raw])
    return dict(raw=raw,eligible=eligible,valid=valid,sampled=ledger.sampled_family.to_numpy(int),step=ledger.decoder_step.to_numpy(int),source=ledger.source.tolist(),rng=copy.deepcopy(rng.bit_generator.state),fixed=fixed,obs=obs,parents=parents,st=st,pub=pub,face=face,meta=dict(row),historical=pred,path=str(path),raw_sha=hashlib.sha256(raw.tobytes()).hexdigest())
def replay(e,p,policy='legacy',eligibility='native',equal_m=None,shared=None,target=None,scoring=True,tail=False,weight=None,keyed=False):
    r=e.r;e.s.corrected_references(r,e.base,p['pub'],e.cm)
    src=ast.unparse(next(n for n in ast.parse((STUDY/'s1_s2_repair_trial_2026-10-18/repair_trial.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='reconstruct'))
    a=src.index('    before = len(STAGE_LOG)');b=src.index('    fids = r.assign',a)
    eligibility_array=p['eligible'].copy()
    if eligibility=='all':eligibility_array=np.ones(len(p['raw']),bool)
    elif eligibility=='equal':
        ids=np.flatnonzero(p['valid']);pr=streams((p['meta']['window_id'],p['meta']['generation_seed'],p['meta']['arm'],'equal'))['selection']
        kept=pr.choice(ids,int(equal_m),replace=False);eligibility_array=np.zeros(len(ids) if False else len(p['raw']),bool);eligibility_array[kept]=True
    T=max(1,round(e.base['rate']*p['pub']['w']*p['pub']['h'])) if target is None else int(target)
    replacement="    before = len(STAGE_LOG)\n    fixed = REPLAY_FIXED.copy()\n    target = REPLAY_TARGET\n    raw = REPLAY_RAW\n    eligible = REPLAY_ELIGIBLE\n    sampled = REPLAY_SAMPLED\n    step = REPLAY_STEPS\n    source = REPLAY_SOURCE\n    budget = len(raw)\n    rng.bit_generator.state = copy.deepcopy(REPLAY_RNG)\n"
    src=src[:a]+replacement+src[b:]
    src=src.replace('posterior_selected=int(eligible.sum())',"posterior_selected=int(eligible.sum()), remaining_available=len(available), unfilled_quotas=(need-pc).tolist(), feasible_remaining=sum(1 for j in available if pc[int(fids[j])] < need[int(fids[j])] and duplicate_or_spacing(raw[j], list(fixed)+selected, int(fids[j]), list(knownfam)+sfams, face) is None), stop_reason='target_accepted' if len(selected) >= missing else ('pool_exhausted' if not available else 'attempt_limit')")
    # R2b-corrected (declared): the former text substitution never matched ast.unparse's parenthesised tuple, so
    # scoring=False was ignored. Explicit AST insertion of `scoring = REPLAY_SCORING` after the tuple assignment.
    _tree=ast.parse(src);_fn=_tree.body[0]
    _ix=[i for i,n in enumerate(_fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [x.id for x in n.targets[0].elts]==['mode','proposal','scoring']]
    assert len(_ix)==1,'scorer intervention target not found exactly once'
    _fn.body.insert(_ix[0]+1,ast.parse('scoring = REPLAY_SCORING').body[0]);src=ast.unparse(_tree)
    assert src.count('scoring = REPLAY_SCORING')==1,'scorer intervention not inserted'
    if policy=='slotsafe':
        src=src.replace('    for _ in range(missing):', '    attempts = 0\n    while len(selected) < missing and attempts < budget * max(1, missing):\n        attempts += 1')
        assert 'attempts = 0' in src,'slot-safe policy not applied'  # R2b-corrected (declared)
    if weight:
        src=src.replace('    for _ in range(missing):', '    weights[REPLAY_WEIGHT_INDEX] *= REPLAY_WEIGHT_FACTOR\n    for _ in range(missing):') if policy!='slotsafe' else src.replace('    attempts = 0','    weights[REPLAY_WEIGHT_INDEX] *= REPLAY_WEIGHT_FACTOR\n    attempts = 0')
    oldfeat=r.scorer['extract_features'];r.scorer['extract_features']=corrected_feature(e,policy!='legacy',tail)
    globals_=dict(e.env,REPLAY_FIXED=p['fixed'] if shared is None else shared,REPLAY_TARGET=T,REPLAY_RAW=p['raw'],REPLAY_ELIGIBLE=eligibility_array,REPLAY_SAMPLED=p['sampled'],REPLAY_STEPS=p['step'],REPLAY_SOURCE=p['source'],REPLAY_RNG=p['rng'],REPLAY_SCORING=scoring,REPLAY_WEIGHT_INDEX=weight[0] if weight else 0,REPLAY_WEIGHT_FACTOR=weight[1] if weight else 1)
    # Identical keyed subset stream across policy contrasts; historical reference retains original RNG.
    if keyed:
        globals_['REPLAY_RNG']=streams((p['meta']['window_id'],p['meta']['generation_seed'],'selection-reference'))['selection'].bit_generator.state
    exec(src,globals_)
    t=time.monotonic()
    try:final,inf,fixed,ledger,budget=globals_['reconstruct'](p['pub'],p['obs'],p['st'],p['meta']['arm'],{},int(p['meta']['generation_seed']))
    finally:r.scorer['extract_features']=oldfeat
    if not scoring:assert int(ledger.score_evaluations.sum())==0,'scorer-off replay evaluated the scorer'  # R2b-corrected (declared)
    # Same final-frame contract as historical Engine.generate.
    p0,q0=r.ends(final);z=np.stack([p0,q0],1);bad=((z[:,:,0]<-1e-7)|(z[:,:,0]>1+1e-7)|(z[:,:,1]<-1e-7)|(z[:,:,1]>p['pub']['ym']+1e-7)).any(1)
    if bad.any():final[bad]=e.s.clip_outer(r,final[bad],p['pub'])
    anchored=np.array([any(e.env['contains'](o,a,20) for o in p['obs']) for a in final],bool)
    assert e.env['preservation'](p['obs'],final,20)==1
    assert budget['generated']==len(final)
    valid=p['valid'];budget.update(raw_valid=int(valid.sum()),eligible_valid=int((valid&eligibility_array).sum()),distinct_scored=int(ledger.score_evaluations.gt(0).sum()),scoring_calls=int(ledger.score_evaluations.sum()),runtime_s=time.monotonic()-t,model_calls=0,pool_sha=p['raw_sha'],rejection_reasons=ledger.disposition.value_counts().to_dict(),historical_invalid_candidate_eligibility='not recoverable from old ledger; eligibility of hard-invalid draws is irrelevant to scoring')
    budget['posterior_selected']=None
    assert hashlib.sha256(p['raw'].tobytes()).hexdigest()==p['raw_sha']
    return final,final[~anchored],ledger,budget
def evaluate(e,p,final,inf,label,phase,budget):
    face=p['face'];truth=face['arr'];hidden=truth[~np.isin(np.arange(len(truth)),np.unique(p['parents']))];r=e.r;meta=p['meta']
    base=dict(phase=phase,configuration=label,arm=meta['arm'],window_id=face['id'],parent_face=face['parent_face'],training_seed=int(meta.get('training_seed',0)),generation_seed=int(meta['generation_seed']))
    metrics=dict(base,N=len(truth),H=len(p['obs']),M=len(e.env['anchor_groups'](p['obs'],p['pub'],p['st'])),A=budget['indicated_parents'],T=budget['target_count'],F=len(final),count_MAE=abs(len(final)-len(truth)),count_bias=len(final)-len(truth),count_error_pct=100*abs(len(final)-len(truth))/len(truth) if len(truth) else math.nan,zero_truth=len(truth)==0,target_count_MAE=abs(budget['target_count']-len(truth)),target_count_bias=budget['target_count']-len(truth),P21_abs_error=abs(final[:,2].sum()-truth[:,2].sum())/face['h'],measured_preservation_2mm=1,**e.b.spatial(truth,final,face['ym']),**{k:budget[k] for k in ['raw_valid','eligible_valid','distinct_scored','scoring_calls','runtime_s','model_calls']})
    assert len(final)-len(truth)==(budget['target_count']-len(truth))+(budget['indicated_parents']+budget['accepted_before_continuity']-budget['target_count'])-budget['continuity_count_change']
    for scope,g,a in [('full_network',truth,final),('hidden_inferred_only',hidden,inf)]:
        ga,gb=r.ends(g);pa,pb=r.ends(a);gg=e.source_module.segment_graph(np.stack([ga,gb],1)*20,face['h']);pg=e.source_module.segment_graph(np.stack([pa,pb],1)*20,face['h'])
        for key in gg:metrics[scope+'_'+key+'_abs_error']=abs(gg[key]-pg[key])
    clusters=[]
    for scope,g,a in [('full_network',truth,final),('hidden_inferred_only',hidden,inf)]:
        gf=r.assign(g,e.base['fc']);af=r.assign(a,e.base['fc'])
        for k in range(4):
            gt=g[gf==k];pr=a[af==k];n=min(len(gt),len(pr));row=dict(base,scope=scope,cluster='C'+str(k+1) if k<3 else 'Unassigned',truth_count=len(gt),generated_count=len(pr),matched_count=n,unmatched_truth=len(gt)-n,unmatched_prediction=len(pr)-n,matched_truth_fraction=n/len(gt) if len(gt) else math.nan,missing_cluster=bool(len(gt) and not len(pr)),spurious_cluster=bool(not len(gt) and len(pr)),truth_proportion=len(gt)/max(len(g),1),pred_proportion=len(pr)/max(len(a),1))
            row.update(angle_MAE_deg=math.nan,length_MAE_m=math.nan,length_MAPE=math.nan,length_W1_m=math.nan,angle_W1_deg=math.nan)
            if n:
                ii,jj=linear_sum_assignment(cdist(gt[:,:2]*20,pr[:,:2]*20));delta=abs((gt[ii,3]-pr[jj,3]+np.pi/2)%np.pi-np.pi/2)
                row.update(angle_MAE_deg=float(np.degrees(delta).mean()),length_MAE_m=float(abs(gt[ii,2]-pr[jj,2]).mean()*20),length_MAPE=float((abs(gt[ii,2]-pr[jj,2])/gt[ii,2]).mean()*100),length_W1_m=float(wasserstein_distance(gt[:,2]*20,pr[:,2]*20)))
                centre=e.base['fc'][k] if k<3 else 0;ga=(gt[:,3]-centre+np.pi/2)%np.pi-np.pi/2;aa=(pr[:,3]-centre+np.pi/2)%np.pi-np.pi/2
                row['angle_W1_deg']=float(np.degrees(wasserstein_distance(ga,aa)))
            for name,arr in [('truth',gt),('pred',pr)]:row[name+'_axial_dispersion']=float(1-abs(np.exp(2j*arr[:,3]).mean())) if len(arr) else math.nan
            clusters.append(row)
    return metrics,clusters
def save_case(e,p,label,phase,final,inf,ledger,budget):
    dest=HERE/phase/'cases'/(label+'__'+p['meta']['case_key']);dest.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(final,columns=COLS);df['inferred']=np.array([any(np.array_equal(a,b) for b in inf) for a in final]);df.to_csv(dest/'predictions.csv',index=False)
    ledger.to_csv(dest/'proposal_ledger.csv.gz',index=False);dump(dest/'budget.json',budget)
    metrics,clusters=evaluate(e,p,final,inf,label,phase,budget);dump(dest/'metrics.json',metrics)
    dump(dest/'COMPLETE.json',dict(files={x.name:sha(x) for x in dest.iterdir() if x.name!='COMPLETE.json'},units='coordinates and length divided by20m; angles radians; metrics metres/degrees/fragments',raw_pool_sha=p['raw_sha']))
    return metrics,clusters
def aggregate(df,keys):
    numbers=[c for c in df.select_dtypes('number').columns if c not in ['training_seed','generation_seed']]
    w=df.groupby(keys+['parent_face','window_id'],dropna=False)[numbers].mean().reset_index()
    f=w.groupby(keys+['parent_face'],dropna=False)[numbers].mean().reset_index();s=f.groupby(keys,dropna=False)[numbers].mean().reset_index();return f,s
