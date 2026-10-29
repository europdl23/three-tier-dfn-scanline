from experiments import *
def phase2(run,e,pools):
    from priority_replay import finish_priority_phase
    finish_priority_phase(run,e,pools)
    run.state['phase']='P2';run.log('P2','running');t=time.monotonic();audits=[];detections={}
    fit=[f for f in e.faces.values() if f['role']=='train']
    for f in fit:
        run.guard();st=e.r.strips('5m',20);obs,parent=e.r.observe(f['arr'],st);groups=e.env['anchor_groups'](obs,public_face(f),st)
        unique=len(set(parent));wrong=sum(len(set(parent[g]))>1 for g in groups)
        train=np.vstack([x['arr'] for x in fit if x['parent_face']!=f['parent_face']]);rng=np.random.default_rng(7319)
        ix=rng.integers(len(train),size=32768);sim=train[ix].copy();sim[:,0]=rng.uniform(size=len(sim));sim[:,1]=rng.uniform(0,f['ym'],len(sim));sim=e.s.clip_outer(e.r,sim,public_face(f))
        a,b=e.r.ends(sim);lo=np.minimum(a[:,0],b[:,0]);hi=np.maximum(a[:,0],b[:,0]);hits=np.zeros(len(sim),int)
        for left,right in st:hits+=((hi>=left)&(lo<=right)&(sim[:,2]>1e-8))
        p=float((hits>0).mean());multiplicity=float(hits.mean());observed=unique/max(len(f['arr']),1);empirical=len(obs)/max(len(f['arr']),1)
        audits.append(dict(window_id=f['id'],parent_face=f['parent_face'],N=len(f['arr']),H=len(obs),M=len(groups),unique_detected_labels_diagnostic=unique,cross_parent_groups=wrong,group_count_abs_error=abs(len(groups)-unique),p_sim=p,p_fit_observed=observed,clip_multiplicity_sim=multiplicity,clip_multiplicity_observed=empirical,hit_error=abs(p-observed),multiplicity_error=abs(multiplicity-empirical),simulation_seed=7319,simulation_traces=32768,possible_double_censoring=True))
    df=pd.DataFrame(audits);df.to_csv(HERE/'count_unit_and_group_audit.csv',index=False)
    folds=[]
    for face,g in df.groupby('parent_face'):
        p=np.average(g.p_sim,weights=g.N);actual=g.unique_detected_labels_diagnostic.sum()/g.N.sum();mu=np.average(g.clip_multiplicity_sim,weights=g.N);emp=g.H.sum()/g.N.sum()
        group=g.group_count_abs_error.sum()/max(g.unique_detected_labels_diagnostic.sum(),1);merge=g.cross_parent_groups.sum()/max(g.M.sum(),1)
        good=group<=.05 and merge<=.01 and abs(p-actual)<=max(.1*actual,.02) and abs(mu-emp)<=max(.1*emp,.02)
        folds.append(dict(parent_face=face,p_sim=p,p_actual=actual,multiplicity_sim=mu,multiplicity_actual=emp,group_relative_error=group,cross_parent_merge_rate=merge,pass_calibration=bool(good)))
    pd.DataFrame(folds).to_csv(HERE/'count_calibration_folds.csv',index=False);valid=all(f['pass_calibration'] for f in folds)
    dump(HERE/'count_policy_decision.json',dict(evidence_count_valid=valid,reason='All source-face-blocked calibration screens passed' if valid else 'Observable detection simulator fails at least one predeclared face-blocked calibration screen. Evidence-conditioned estimator is not promoted.',alpha=2,oracle='Evaluation-only; no truth-based pool enlargement.'))
    # Density/oracle use identical historical density-sized pools. If valid evidence
    # requires a larger supply, defer that affected branch rather than violate supply contract.
    if valid:
        run.log('P2_evidence','deferred',reason='Validated model requires newly generated common max(N0,N_evidence) pools; density-sized saved pools cannot be silently treated as that experiment.')
    for p in pools:
        if time.monotonic()-t>=3600:run.log('P2','deferred',reason='phase time ceiling');break
        for label,target in [('density',None),('oracle_diagnostic',len(p['face']['arr']))]:
            run.case(e,p,label,'P2',policy='slotsafe',eligibility='native',tail=True,target=target,keyed=True)
    run.flush();run.log('P2','complete',evidence_promoted=False,elapsed_s=time.monotonic()-t)
def new_stat_pool(e,p,mode,bw=1,bins=12,spatial=False):
    r=e.r;e.s.corrected_references(r,e.base,p['pub'],e.cm);q=copy.copy(p);rng=streams((p['meta']['window_id'],p['meta']['generation_seed'],'strong_stat_reference'))
    n=len(p['raw']);families=rng['family'].choice(4,n,p=e.base['shares']);rows=[]
    for k in families:
        pr=e.base['pools'][int(k)];ix=int(rng['angle'].integers(len(pr['angles'])))
        if mode=='kde':
            h=len(pr['angles'])**(-1/5)*bw;A=(pr['angles'][ix]+rng['angle'].normal(0,max(np.std(pr['angles']),.01)*h))%np.pi
            L=math.exp(math.log(pr['lengths_m'][ix])+rng['length'].normal(0,max(np.std(np.log(pr['lengths_m'])),.1)*h))
        else:
            hist,edges=np.histogram(pr['angles'],bins=bins,range=(0,np.pi));idx=int(rng['angle'].choice(bins,p=(hist+1)/(hist.sum()+bins)));A=rng['angle'].uniform(edges[idx],edges[idx+1]);logs=np.log(pr['lengths_m']);L=math.exp(rng['length'].normal(logs.mean(),max(logs.std(),.08)))
        if spatial:
            candidates=[]
            for f in e.faces.values():
                if f['role']=='train':
                    sub=f['arr'][r.assign(f['arr'],e.base['fc'])==k];candidates.extend(sub[:,:2]/[1,f['ym']])
            pos=np.asarray(candidates)[int(rng['position'].integers(len(candidates)))];cx,cy=pos*[1,p['pub']['ym']]
        else:cx=rng['position'].uniform();cy=rng['position'].uniform(0,p['pub']['ym'])
        rows.append([cx,cy,L/20,A])
    raw=e.s.clip_outer(r,np.asarray(rows,np.float32),p['pub']);q.update(raw=raw,eligible=np.ones(n,bool),sampled=families,step=np.full(n,-1),source=[mode+'_paired_stream']*n,raw_sha=hashlib.sha256(raw.tobytes()).hexdigest(),valid=np.array([r.candidate_gate(x,p['pub'],p['st']) is None for x in raw]))
    return q
def angle_sampler(e,alpha,case):
    rngs=streams(case);r=e.r
    def posterior(position,loglength,angle,logits,face,rng,n):
        probs=np.exp(logits-np.max(logits));probs/=probs.sum();families=rngs['family'].choice(4,n,p=probs);rows=[]
        for k in families:
            pr=e.base['pools'][int(k)];logs=np.log(pr['lengths_m']/20);prior_sd=max(float(logs.std()),.08);model_sd=max(prior_sd*.7,.06);pmodel=1/model_sd**2;pprior=1/prior_sd**2;mu=(loglength*pmodel+logs.mean()*pprior)/(pmodel+pprior);sd=math.sqrt(1/(pmodel+pprior));L=math.exp(rngs['length'].normal(mu,sd))
            if k<3:
                kp=e.env['axial_kappa'](pr['angles']);vec=alpha*kp*np.exp(2j*angle)+kp*np.exp(2j*e.base['fc'][k]);A=(rngs['angle'].vonmises(np.angle(vec),abs(vec))/2)%np.pi
            else:A=(rngs['angle'].vonmises(2*angle,.3)/2)%np.pi
            rows.append([0,0,L,A])
        a=np.asarray(rows,np.float32);pos=np.asarray(position)/[1,face['ym']];errors=e.current['errors']['5m'];valid=np.flatnonzero(np.all((pos+errors>=0)&(pos+errors<=1),axis=1))
        if not len(valid):a[:]=0
        else:a[:,:2]=(pos+errors[rngs['position'].choice(valid,n)])*[1,face['ym']]
        e.current.setdefault('posterior_position_probes',[]).append(dict(mean_x_fraction=float(pos[0]),mean_y_fraction=float(pos[1]),sample_x_min=float(a[:,0].min()),sample_x_max=float(a[:,0].max()),sample_y_min_fraction=float(a[:,1].min()/face['ym']),sample_y_max_fraction=float(a[:,1].max()/face['ym'])))
        return e.s.clip_outer(r,a,face),np.zeros(n),families
    return posterior
def phase3(run,e,pools):
    run.state['phase']='P3';run.log('P3','running');t=time.monotonic();bundle=e.load_bundle(42001);net=bundle['models']['transformer_latent'];oldpost=e.env['posterior_draws'];calls=[]
    frozen=json.loads((HERE/'frozen_manifest.json').read_text())
    verify_identity(sha(STAGE/'training_position_residuals.npz'),frozen['input_hashes'][str(STAGE/'training_position_residuals.npz')])
    errors,horizons=e.calibrate(bundle,net,True,STAGE/'training_position_residuals.npz')
    for p in pools:
        if time.monotonic()-t>=3600:run.log('P3','deferred',reason='phase time cap');break
        arm=p['meta']['arm']
        if arm=='E':
            e.current.update(errors=errors,layout='5m');e.env['HORIZON']=horizons['5m'];net.aspect=torch.tensor([p['pub']['ym']],device=e.r.DEV)
            for alpha in [0,.5]:
                run.guard();e.current['posterior_position_probes']=[];e.env['posterior_draws']=angle_sampler(e,alpha,(p['meta']['window_id'],p['meta']['generation_seed'],'angular_contrast'))
                count=[0]
                def hook(*args):count[0]+=1
                h=net.register_forward_hook(hook);started=time.monotonic()
                try:raw,eligible,fam,step=e.env['sample_neural'](net,p['fixed'],p['pub'],len(p['raw']),True,bundle['models']['evae'],streams((p['meta']['window_id'],p['meta']['generation_seed'],'local_prefix_choice'))['selection'],p['st'])
                finally:h.remove();e.env['posterior_draws']=oldpost
                q=copy.copy(p);q.update(raw=raw,eligible=eligible,sampled=fam,step=step,source=['neural_paired_stream']*len(raw),raw_sha=hashlib.sha256(raw.tobytes()).hexdigest(),valid=np.array([e.r.candidate_gate(x,p['pub'],p['st']) is None for x in raw]))
                m,c=run.case(e,q,'angle_alpha_'+str(alpha),'P3',policy='slotsafe',eligibility='native',tail=True,keyed=True)
                probes=pd.DataFrame(e.current['posterior_position_probes']);probe_path=HERE/'P3'/'raw_mean_residual_probes';probe_path.mkdir(exist_ok=True);probes.to_csv(probe_path/(p['meta']['case_key']+'__alpha'+str(alpha)+'.csv'),index=False)
                m['generation_model_calls']=count[0];m['generation_runtime_s']=time.monotonic()-started
                calls.append(dict(case_key=p['meta']['case_key'],alpha=alpha,model_calls=count[0],generation_runtime_s=time.monotonic()-started))
        elif arm=='KDE':
            for bw in [.5,1,2]:run.case(e,new_stat_pool(e,p,'kde',bw=bw),'KDE_bw_'+str(bw),'P3',policy='slotsafe',eligibility='all',tail=True,keyed=True)
            run.case(e,new_stat_pool(e,p,'kde',spatial=True),'KDE_spatial_bw_1','P3',policy='slotsafe',eligibility='all',tail=True,keyed=True)
        else:
            for bins in [6,12,24]:run.case(e,new_stat_pool(e,p,'hist',bins=bins),'ADFNE_bins_'+str(bins),'P3',policy='slotsafe',eligibility='all',tail=True,keyed=True)
    pd.DataFrame(calls).to_csv(HERE/'P3_model_calls.csv',index=False)
    # Same-context perturbations do not change continuation or train a no-latent model.
    probes=[];seq=e.sequences(bundle,True)
    for role,items in seq.items():
        for ix,x in enumerate(items):
            run.guard();e.s.set_aspect(net,[x],e.r.DEV);c,cm,y,mask,fam,sz,gz=e.r.seq_batch([x]);di=torch.cat([torch.zeros(1,1,5,device=e.r.DEV),y[:,:-1]],1);dm=torch.zeros(di.shape[:2],dtype=torch.bool,device=e.r.DEV)
            other=items[(ix+1)%len(items)];sz2=torch.as_tensor(other['sz'],device=e.r.DEV).reshape_as(sz);gz2=torch.as_tensor(other['gz'],device=e.r.DEV).reshape_as(gz)
            for name,s,z in [('reference',sz,gz),('zero',torch.zeros_like(sz),torch.zeros_like(gz)),('permuted',sz2,gz2)]:
                with torch.no_grad():pos,L,a,logits=net(c,cm,di,dm,s,z)
                angle=torch.atan2(a[0,:,0],a[0,:,1]);target=torch.atan2(y[0,:,3],y[0,:,4]);delta=abs((angle-target+np.pi/2)%np.pi-np.pi/2)
                probes.append(dict(role=role,layout=x['layout'],window_id=x['face_id'],parent_face=e.faces[x['face_id']]['parent_face'],perturbation=name,position_RMSE_m=float(((pos-y[:,:,:2])**2).mean().sqrt()*20),y_span_fraction=float((pos[0,:,1].max()-pos[0,:,1].min())/x['aspect']),truth_y_span_fraction=float((y[0,:,1].max()-y[0,:,1].min())/x['aspect']),angle_MAE_deg=float(delta.mean()*180/np.pi),family_accuracy=float((logits[0].argmax(-1)==fam[0]).float().mean())))
    pd.DataFrame(probes).to_csv(HERE/'P3_same_context_latent_probes.csv',index=False)
    run.flush()
    from selection import nominate
    decisions=dict(KDE=nominate(run,'P3','KDE_bw_1',['KDE_bw_0.5','KDE_bw_2'],arm='KDE'),ADFNE=nominate(run,'P3','ADFNE_bins_12',['ADFNE_bins_6','ADFNE_bins_24'],arm='ADFNE'),spatial='KDE_spatial_bw_1 remains separately reported, not silently omitted when it wins.')
    dump(HERE/'baseline_selection.json',decisions)
    run.log('P3','complete',elapsed_s=time.monotonic()-t)
def neural_phases(run,e,pools):
    from training import execute_neural
    execute_neural(run,e,pools)
def finalise(run,e):
    from selection import final_audit
    # Include all pre-owner smoke replays, including the separate P0-only owner.
    # 3 manual native +144 standalone checks +144 P0-only +1 slot-safe smoke
    # +144 full-owner P0 =436. The live counter initially reserved294 for these.
    run.state['new_maps']+=143
    dump(HERE/'replay_accounting_reconciliation.json',dict(manual_native=3,standalone_P0=144,P0_only_smoke=144,slot_safe_smoke=1,full_owner_P0=144,priority_smoke=1,total_pre_phase1=437,live_counter_correction=143,budget_effect='Total remains below12000; no scientific observations are multiplied by replay counts.'))
    final_audit(run,e)
    base_report=run.report
    def complete_report(error=None):
        if run.state['status']=='applicable_programme_finished':
            confirmation=json.loads((HERE/'confirmation_decision.json').read_text()) if (HERE/'confirmation_decision.json').exists() else {}
            run.state['status']='candidate_confirmed_on_S1' if confirmation.get('candidate') else 'diagnosis_only_no_candidate'
        base_report(error)
        findings=HERE/'SCIENTIFIC_FINDINGS.md'
        if findings.exists():
            report=HERE/'OVERNIGHT_RESULTS.md';report.write_text(report.read_text(encoding='utf-8')+'\n\n'+findings.read_text(encoding='utf-8'),encoding='utf-8')
            import shutil
            dest=Path((__import__('os').environ['DFN3_CORE'] + '/_external_not_used'))
            for name in ['OVERNIGHT_RESULTS.md','SCIENTIFIC_FINDINGS.md','FINAL_COMMON_CLUSTER_COMPARISON.csv','FINAL_ALL_CASE_NETWORK_COMPARISON.csv','FINAL_ENDPOINT_WIN_LOSS.csv','FINAL_COVERAGE.csv']:
                p=HERE/name
                if p.exists():shutil.copy2(p,dest/('AUTHORISED_EXECUTION_2026-10-19_'+name))
    run.report=complete_report
