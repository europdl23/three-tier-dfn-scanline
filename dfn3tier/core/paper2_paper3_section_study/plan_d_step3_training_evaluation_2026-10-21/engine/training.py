"""Paired original-topology training with atomic epoch checkpoints and RNG state."""
from experiments import *
import random,traceback
MILESTONES={0,5,10,20,30,50}
def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'].cpu())
    if x['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in x['cuda']])
def atomic_torch(path,obj):
    q=path.with_name(path.name+'.partial');torch.save(obj,q);q.replace(path);dump(path.with_suffix('.sha256.json'),dict(sha256=sha(path)))
def safe_calibrate(e,bundle,net,path):
    path=Path(path);identity=dict(model_sha=e.m.statehash(net.state_dict()),evae_sha=e.m.statehash(bundle['models']['evae'].state_dict()),head_sha=e.m.statehash(bundle['models']['head'][0].state_dict()),split_sha=sha(STAGE/'window_trace_manifest.csv'),normalisation='width20/public_height; fitting teacher-forced residual fractions',engine_sha=sha(OLD/'engine.py'),loss_weights=e.source_module.REFERENCE)
    side=path.with_suffix('.identity.json')
    if path.exists():
        if not side.exists():raise ValueError('Unversioned residual cache cannot be reused')
        record=json.loads(side.read_text());verify_identity(record['identity'],identity);verify_identity(record['residual_sha'],sha(path))
    errors,horizons=e.calibrate(bundle,net,True,path)
    if not side.exists():dump(side,dict(identity=identity,residual_sha=sha(path),horizons=horizons))
    return errors,horizons
def epoch_job(run,e,name,net,train_epoch,validation,epochs,lr,seed,settings,diagnostic=None,wd=1e-5):
    run.guard();dest=HERE/'training'/name;dest.mkdir(parents=True,exist_ok=True)
    conf=dict(name=name,epochs=epochs,lr=lr,weight_decay=wd,seed=seed,settings=settings,initial_state=e.m.statehash(net.state_dict()),sources={p.name:sha(p) for p in HERE.glob('*.py')},inputs={p.name:sha(p) for p in STAGE.iterdir() if p.is_file()})
    fingerprint=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest();config=dest/'config.json'
    if config.exists():assert json.loads(config.read_text())==conf,'Training configuration/cache mismatch'
    else:dump(config,conf)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd);best=math.inf;beststate=cpu_state(net);logs=[];selected=0;start_epoch=0
    latest=dest/'latest.pt';bestfile=dest/'best.pt'
    if (dest/'COMPLETE.json').exists():
        record=json.loads((dest/'COMPLETE.json').read_text());assert record['fingerprint']==fingerprint and record['best_sha']==sha(bestfile)
        ck=torch.load(bestfile,map_location=e.r.DEV,weights_only=False);net.load_state_dict(ck['state_dict']);net.eval();return net,record
    if latest.exists():
        assert json.loads(latest.with_suffix('.sha256.json').read_text())['sha256']==sha(latest)
        ck=torch.load(latest,map_location=e.r.DEV,weights_only=False);assert ck['fingerprint']==fingerprint
        net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng']);start_epoch=ck['epoch'];best=ck['best_score'];selected=ck['selected_epoch'];logs=ck['logs'];beststate=ck['best_state']
    else:atomic_torch(dest/'epoch_000.pt',dict(state_dict=cpu_state(net),fingerprint=fingerprint,epoch=0))
    run.state['training_jobs']+=1
    if run.state['training_jobs']>18:raise RuntimeError('Training job ceiling')
    run.log(name,'running',fingerprint=fingerprint,initial_state=conf['initial_state']);start=time.monotonic();epoch_durations=[]
    for epoch in range(start_epoch+1,epochs+1):
        run.guard();beg=time.monotonic();run.state['progress']=name+f' epoch {epoch}/{epochs}'
        net.train();train_values=train_epoch(net,opt,epoch);net.eval()
        with torch.no_grad():v,components=validation(net,epoch)
        if not math.isfinite(v):raise FloatingPointError('Nonfinite common validation loss')
        if v<best:best=v;beststate=cpu_state(net);selected=epoch;atomic_torch(bestfile,dict(state_dict=beststate,epoch=epoch,common_validation=v,fingerprint=fingerprint))
        row=dict(epoch=epoch,common_validation=v,**train_values,**{'validation_'+k:float(x) for k,x in components.items()});logs.append(row);pd.DataFrame(logs).to_csv(dest/'curve.csv',index=False)
        ck=dict(state_dict=cpu_state(net),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,data_cursor=0,best_state=beststate,best_score=best,selected_epoch=selected,logs=logs,fingerprint=fingerprint,data_order_policy='numpy seed reset per paired job; permutation state and epoch cursor saved',initial_sha=conf['initial_state'])
        atomic_torch(latest,ck)
        if epoch in MILESTONES:atomic_torch(dest/f'epoch_{epoch:03d}.pt',dict(state_dict=cpu_state(net),fingerprint=fingerprint,epoch=epoch))
        if diagnostic and epoch in {1,5,10,20,30,50,100,150}:diagnostic(net,epoch,dest)
        epoch_durations.append(time.monotonic()-beg)
        estimate=np.mean(epoch_durations[-5:])*(epochs-epoch)
        run.guard(estimate)
        if epoch%10==0:print(name,epoch,round(v,5),flush=True)
    net.load_state_dict(beststate);net.eval();record=dict(fingerprint=fingerprint,selected_epoch=selected,validation=best,best_sha=sha(bestfile),elapsed_s=time.monotonic()-start,epochs=epochs)
    dump(dest/'COMPLETE.json',record);run.log(name,'complete',**record);return net,record
def update(net,opt,loss):
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    opt.zero_grad(set_to_none=True);loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise FloatingPointError('Nonfinite gradient')
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();return float(norm)
def tf_loss(e,net,items,weights,hidden_prior=False,reference=False):
    e.s.set_aspect(net,items,e.r.DEV);old=e.r.STATS['shares'];batch=e.r.seq_batch(items)
    if hidden_prior and not reference:
        # Mean over sequences, matching the existing aggregate softmax prior penalty.
        e.r.STATS['shares']=np.average([e.hidden_priors[x['layout']] for x in items],axis=0,weights=[len(x['tgt']) for x in items])
    try:
        ref,parts=e.s.get_losses(e.m,net,batch,e.r,e.tau);result=ref
        if not reference:
            for name,key in [('position','balanced_position'),('length','length'),('angular','angular'),('family','family'),('proportion','proportion'),('crossing','correct_crossing')]:result=result+(weights[name]-e.source_module.REFERENCE[name])*parts[key]
    finally:e.r.STATS['shares']=old
    return result,parts
def tf_diagnostic(e,net,items,epoch,dest):
    rows=[];grad=[];net.eval()
    for x in items:
        e.s.set_aspect(net,[x],e.r.DEV);c,cm,y,mask,fam,sz,gz=e.r.seq_batch([x]);di=torch.cat([torch.zeros(1,1,5,device=e.r.DEV),y[:,:-1]],1);dm=torch.zeros(di.shape[:2],device=e.r.DEV,dtype=torch.bool)
        with torch.no_grad():
            p,L,a,f=net(c,cm,di,dm,sz,gz);ta=torch.atan2(y[0,:,3],y[0,:,4]);pa=torch.atan2(a[0,:,0],a[0,:,1]);delta=abs((pa-ta+np.pi/2)%np.pi-np.pi/2)*180/np.pi
            prefix=torch.zeros(1,1,5,device=e.r.DEV);free=[]
            for j in range(len(x['tgt'])):
                pp,ll,aa,ff=net(c,cm,prefix,torch.zeros(prefix.shape[:2],dtype=torch.bool,device=e.r.DEV),sz,gz);angle=torch.atan2(aa[0,-1,0],aa[0,-1,1]);free.append(pp[0,-1]);token=torch.cat([pp[:,-1:],ll[:,-1:],angle.sin().reshape(1,1,1),angle.cos().reshape(1,1,1)],-1);prefix=torch.cat([prefix,token],1)
            free=torch.stack(free)
            for k in range(4):
                ix=fam[0]==k
                rows.append(dict(epoch=epoch,window_id=x['face_id'],layout=x['layout'],cluster=k,targets=int(ix.sum()),angle_MAE_deg=float(delta[ix].mean()) if ix.any() else math.nan,length_MAE_m=float(abs(L[0,:,0].exp()[ix]-y[0,:,2].exp()[ix]).mean()*20) if ix.any() else math.nan,family_accuracy=float((f[0].argmax(-1)[ix]==fam[0][ix]).float().mean()) if ix.any() else math.nan,position_RMSE_m=float(((p-y[:,:,:2])**2).mean().sqrt()*20),free_position_RMSE_m=float(((free-y[0,:,:2])**2).mean().sqrt()*20),y_span_fraction=float(torch.ptp(p[0,:,1])/x['aspect']) if hasattr(torch,'ptp') else float((p[0,:,1].max()-p[0,:,1].min())/x['aspect']),truth_y_span_fraction=float((y[0,:,1].max()-y[0,:,1].min())/x['aspect']),**{f'predicted_family_{j}_count':int(((f[0].argmax(-1)==j)&ix).sum()) for j in range(4)}))
    pd.DataFrame(rows).to_csv(dest/f'diagnostics_{epoch:03d}.csv',index=False)
def tiny_fit(run,e,bundle):
    run.state['phase']='P4_tiny_fit';examples=[]
    previous=HERE/'tiny_fit_decision.json'
    if previous.exists():
        record=json.loads(previous.read_text())
        for i in range(2):
            checkpoint=HERE/'training'/f'tiny_{i}'/'step_1000.pt';verify_identity(sha(checkpoint),json.loads(checkpoint.with_suffix('.sha256.json').read_text())['sha256'])
        run.log('tiny_fit','reused',reason='Existing complete diagnostics retained; no repeat-until-win or extra1000step fits.')
        return bool(record['passed'])
    # Construct original-augmentation fitting examples explicitly; no ambiguity from skipped sequence entries.
    for f in sorted((f for f in e.faces.values() if f['role']=='train'),key=lambda f:f['id']):
        st=e.r.strips('5m',20);obs,parent=e.r.observe(f['arr'],st);z=dict(face=f,arr=f['arr'],obs=obs,parents=parent,strips=st,layout='5m');x=e.r.sequence_example(z,bundle['models'],True)
        if not len(x['tgt']):continue
        ix=np.argsort(x['tgt'][:,0],kind='stable');x['tgt']=x['tgt'][ix];x['fam']=x['fam'][ix];x.update(aspect=f['ym'],layout='5m',face_id=f['id']);examples.append(x)
        if len(examples)==2:break
    input_hashes=[hashlib.sha256(x['ctx'].tobytes()+x['sz'].detach().cpu().numpy().tobytes()+x['gz'].detach().cpu().numpy().tobytes()).hexdigest() for x in examples];assert len(set(input_hashes))==2,'Identifiability failure'
    results=[]
    for i,x in enumerate(examples):
        e.r.seedall(42001);random.seed(42001);net=e.Net().to(e.r.DEV);net.latent=True
        for m in net.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.
            if isinstance(m,torch.nn.MultiheadAttention):m.dropout=0.
        opt=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=0);dest=HERE/'training'/f'tiny_{i}';dest.mkdir(parents=True,exist_ok=True);curve=[]
        weights=dict(e.source_module.REFERENCE,proportion=0.,crossing=0.)
        for step in range(1001):
            run.guard();run.state['progress']=f'tiny example{i} step{step}';net.eval();loss,parts=tf_loss(e,net,[x],weights)
            value=float(loss.detach());curve.append(dict(step=step,supervised_loss=value,**{k:float(v.detach()) for k,v in parts.items()}))
            if step==0:first=value
            if step in {0,100,300,1000}:
                atomic_torch(dest/f'step_{step:04d}.pt',dict(state_dict=cpu_state(net),optimizer=opt.state_dict(),rng=rng_state(),step=step,input_hash=input_hashes[i],settings=weights));pd.DataFrame(curve).to_csv(dest/'curve.csv',index=False);tf_diagnostic(e,net,[x],step,dest)
            if step<1000:net.train();update(net,opt,loss)
        good=value<=.1*first;results.append(dict(example=x['face_id'],input_hash=input_hashes[i],initial_loss=first,final_loss=value,reduction_fraction=1-value/first,passed=good));run.log('tiny_'+str(i),'complete',**results[-1])
    dump(HERE/'tiny_fit_decision.json',dict(examples=results,passed=all(r['passed'] for r in results),interpretation='90% positive-supervised-loss reduction is diagnostic learnability, not geological validation.'))
    return all(r['passed'] for r in results)
def train_evae(run,e,label,seed):
    e.r.seedall(seed);random.seed(seed);length=label in ['E_L','E_LB'];bg=label in ['E_B','E_LB'];net=make_evae(e,length);native,source=loss_function(e,bg);common,_=loss_function(e,True)
    def train(m,opt,ep):
        vals=[];grads=[];components=[];head_gradients=[];order=np.random.permutation(len(e.train))
        for ix in np.array_split(order,math.ceil(len(order)/8)):
            b=pack(e,[e.train[j] for j in ix]);out=e.env['forward_evae'](m,b);loss,parts=native(out,b,.001*min(ep/50,1),torch.ones(4,device=e.r.DEV));grads.append(update(m,opt,loss));vals.append(float(loss));components.append(parts)
            row={name+'_clipped_gradient_norm':math.sqrt(sum(float(p.grad.square().sum()) for p in getattr(m,name).parameters() if p.grad is not None)) for name in ['am','ak','ld','cd','smu','gmu']}
            bg_grad=m.am[-1].weight.grad.reshape(128,4,-1)[:,3];row['background_logit_clipped_gradient_norm']=float(bg_grad.norm());head_gradients.append(row)
        return dict(train=np.mean(vals),gradient_norm=np.mean(grads),order_hash=hashlib.sha256(order.tobytes()).hexdigest(),**{'train_'+k:float(v) for k,v in pd.DataFrame(components).mean().to_dict().items()},**pd.DataFrame(head_gradients).mean().to_dict())
    def valid(m,ep):
        vals=[];parts=[]
        with torch.random.fork_rng(devices=[0] if e.r.DEV.type=='cuda' else []):
            torch.manual_seed(seed+777)
            for x in e.valid:
                b=pack(e,[x]);o=e.env['forward_evae'](m,b);l,c=common(o,b,.001,torch.ones(4,device=e.r.DEV));vals.append(float(l));parts.append(c)
        return float(np.mean(vals)),pd.DataFrame(parts).mean().to_dict()
    def diagnostic(m,ep,dest):
        out=[]
        with torch.no_grad(),torch.random.fork_rng(devices=[0] if e.r.DEV.type=='cuda' else []):
            torch.manual_seed(seed+777)
            for x in e.valid:
                b=pack(e,[x]);o=e.env['forward_evae'](m,b);assign=e.r.lossenv['hungarian_match_batch'](o['pc'],o['pl'],o['ml'],b['C_tgt'],b['L_tgt'],b['F_tgt'],b['E_tgt'],b['E_tgt'])
                pi,gi=assign[0];pr=torch.as_tensor(pi,device=e.r.DEV);gt=torch.as_tensor(gi,device=e.r.DEV);true=b['F_tgt'][0,gt];pred=o['ml'][0,pr].argmax(-1)
                for k in range(4):
                    keep=true==k
                    for j in range(4):out.append(dict(epoch=ep,window_id=x['face']['id'],layout=x['layout'],truth_family=k,pred_family=j,count=int(((pred==j)&keep).sum()),length_MAE_m=float(abs(o['pl'][0,pr,0][keep]-b['L_tgt'][0,gt,0][keep]).mean()*20) if keep.any() else math.nan,unsupported_targets=int((b['L_tgt'][0,gt,0]>1).sum())))
        pd.DataFrame(out).to_csv(dest/f'supervision_{ep:03d}.csv',index=False)
    return epoch_job(run,e,f'{label}_seed{seed}',net,train,valid,150,3e-4,seed,dict(length_support=length,background=bg,loss_source_sha=hashlib.sha256(source.encode()).hexdigest(),beta_max=.001,anneal=50,freebits=.1,batch=8,clip=1),diagnostic)
def train_head(run,e,ev,seed):
    r=e.r
    def data(items):
        X=[];Y=[]
        for z in items:
            lat=torch.cat(r.tensor_encode(ev,z['obs']),-1).detach().cpu().numpy()[0]
            for members in e.env['anchor_groups'](z['obs'],public_face(z['face']),z['strips']):
                for i,side in e.env['group_outer_observations'](z['obs'],members):
                    feat,a,u,bound=e.env['continuation_features'](z['obs'],i,side,z['face'])
                    if not any(abs(a[0]-edge)<1e-6 for st in z['strips'] for edge in st):continue
                    p,q=r.ends(z['arr'][z['parents'][i]:z['parents'][i]+1]);extension=max(float((p[0]-a)@u),float((q[0]-a)@u),0)
                    if bound>1e-8:X.append(np.r_[lat,feat]);Y.append(math.log(max(extension,1e-6)))
        return torch.tensor(np.asarray(X),device=r.DEV),torch.tensor(Y,device=r.DEV)[:,None]
    X,Y=data(e.train);V,Z=data(e.valid);mu=X.mean(0);sd=X.std(0).clamp_min(1e-3);X=(X-mu)/sd;V=(V-mu)/sd;r.seedall(seed);random.seed(seed);net=r.gen['EVAEContHead']().to(r.DEV)
    def train(m,opt,ep):
        vals=[]
        for ix in torch.randperm(len(X),device=r.DEV).split(8):
            a,l=m(X[ix]);loss=(.5*((Y[ix]-a)/l.exp())**2+l).mean();update(m,opt,loss);vals.append(float(loss))
        return dict(train=np.mean(vals))
    def valid(m,ep):
        a,l=m(V);v=(.5*((Z-a)/l.exp())**2+l).mean();return float(v),dict(gaussian_NLL=float(v))
    net,meta=epoch_job(run,e,f'head_LB_seed{seed}',net,train,valid,50,1e-3,seed,dict(evae_sha=e.m.statehash(ev.state_dict()),batch=8,batch_disposition='Global overnight batch8 limit applies to all new heads; historical head used256. Combined repaired-reference contrast includes this operational change.',Xmean=mu.detach().cpu().tolist(),Xstd=sd.detach().cpu().tolist()))
    return (net,mu,sd)
def train_tf(run,e,bundle,label,seed):
    r=e.r;seq=e.sequences(bundle,True);r.seedall(seed);random.seed(seed);net=e.Net().to(r.DEV);net.latent=True;weights=dict(e.source_module.REFERENCE)
    if label=='T_P':weights['position']=4
    if label=='T_F':weights['family']=2
    if label=='T_Z':weights['proportion']=0
    hidden=label!='T0'
    def train(m,opt,ep):
        vals=[];parts=[];grads=[];order=np.random.permutation(len(seq['train']))
        for ix in np.array_split(order,math.ceil(len(order)/8)):
            items=[seq['train'][i] for i in ix];loss,p=tf_loss(e,m,items,weights,hidden);grads.append(update(m,opt,loss));vals.append(float(loss));parts.append({k:float(v) for k,v in p.items()})
        return dict(train=np.mean(vals),gradient_norm=np.mean(grads),order_hash=hashlib.sha256(order.tobytes()).hexdigest(),**{'train_'+k:float(v) for k,v in pd.DataFrame(parts).mean().to_dict().items()})
    def valid(m,ep):
        vals=[];parts=[]
        for x in seq['validation']:
            l,p=tf_loss(e,m,[x],weights,reference=True);vals.append(float(l));parts.append({k:float(v) for k,v in p.items()})
        return float(np.mean(vals)),pd.DataFrame(parts).mean().to_dict()
    def diag(m,ep,dest):
        diverse=[]
        for layout in ['centre','5m']:
            for face in sorted({x['face_id'] for x in seq['train']}):
                choices=[x for x in seq['train'] if x['face_id']==face and x['layout']==layout]
                if choices:diverse.append(choices[0])
                if len(diverse)>=8:break
        tf_diagnostic(e,m,seq['validation'],ep,dest)
        gradients=[]
        for i,x in enumerate(diverse[:8]):
            m.zero_grad(set_to_none=True);l,p=tf_loss(e,m,[x],weights,hidden)
            for key in ['balanced_position','length','angular','family','proportion','correct_crossing']:
                gs=torch.autograd.grad(p[key],list(m.parameters()),retain_graph=True,allow_unused=True);norm=math.sqrt(sum(float(g.square().sum()) for g in gs if g is not None));gradients.append(dict(epoch=ep,window_id=x['face_id'],layout=x['layout'],component=key,gradient_norm=norm))
            for head in ['pos_head','loglen_head','ang_head','fam_head']:
                gs=torch.autograd.grad(l,list(getattr(m,head).parameters()),retain_graph=True,allow_unused=True);norm=math.sqrt(sum(float(g.square().sum()) for g in gs if g is not None));gradients.append(dict(epoch=ep,window_id=x['face_id'],layout=x['layout'],component=head+'_weighted_total',gradient_norm=norm))
        pd.DataFrame(gradients).to_csv(dest/f'gradient_probe_{ep:03d}.csv',index=False)
    return epoch_job(run,e,f'{label}_seed{seed}',net,train,valid,50,3e-4,seed,dict(weights=weights,prior='fit_hidden_layout' if hidden else 'full_map',evae_sha=e.m.statehash(bundle['models']['evae'].state_dict()),head_sha=e.m.statehash(bundle['models']['head'][0].state_dict()),batch=8,clip=1,dropout=.1),diag)
def make_bundle(run,e,seed,ev=None):
    if ev is None:ev,_=train_evae(run,e,'E_LB',seed)
    head=train_head(run,e,ev,seed);key=seed+1000000
    bundle=dict(seed=key,dest=HERE/'training'/f'bundle_{seed}',models=dict(evae=ev,head=head));e.bundles[key]=bundle
    return bundle
def generate_tf(run,e,bundle,net,label,pools,seeds=None):
    errors,horizons=safe_calibrate(e,bundle,net,HERE/'training'/label/'residuals.npz');bundle['models']['transformer_latent']=net
    rows=[p for p in pools if p['meta']['arm']=='E']
    for p in rows:
        run.guard();q=copy.copy(p);q['meta']=dict(p['meta']);q['meta']['training_seed']=bundle['seed']-1000000
        if seeds is not None:
            original=int(p['meta']['generation_seed']);q['meta']['generation_seed']=seeds[[42001,42002,42003].index(original)];q['meta']['case_key']=p['meta']['case_key'].replace('__g'+str(original),'__g'+str(q['meta']['generation_seed']))
        e.s.corrected_references(e.r,e.base,q['pub'],e.cm);fixed=e.env['indicated'](q['obs'],q['pub'],q['st'],'evae',bundle['models'],np.random.default_rng(int(q['meta']['generation_seed'])))
        q['fixed']=fixed;net.aspect=torch.tensor([q['pub']['ym']],device=e.r.DEV);e.current.update(errors=errors,layout='5m');e.env['HORIZON']=horizons['5m']
        from followup import angle_sampler
        alpha=getattr(e,'locked_angle_alpha',.5);oldpost=e.env['posterior_draws'];e.current['posterior_position_probes']=[]
        e.env['posterior_draws']=angle_sampler(e,alpha,(q['meta']['window_id'],q['meta']['generation_seed'],'angular_contrast'));calls=[0]
        hook=net.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1));beg=time.monotonic()
        try:raw,eligible,fam,step=e.env['sample_neural'](net,fixed,q['pub'],len(q['raw']),True,bundle['models']['evae'],streams((q['meta']['window_id'],q['meta']['generation_seed'],'local_prefix_choice'))['selection'],q['st'])
        finally:hook.remove();e.env['posterior_draws']=oldpost
        q.update(raw=raw,eligible=eligible,sampled=fam,step=step,source=['corrected_transformer']*len(raw),raw_sha=hashlib.sha256(raw.tobytes()).hexdigest(),valid=np.array([e.r.candidate_gate(x,q['pub'],q['st']) is None for x in raw]))
        cached=HERE/'P5/cases'/(label+'__'+q['meta']['case_key'])/'COMPLETE.json'
        if cached.exists():verify_identity(json.loads(cached.read_text())['raw_pool_sha'],q['raw_sha'])
        m,c=run.case(e,q,label,'P5',policy='slotsafe',eligibility='native',tail=True,keyed=True);m['generation_model_calls']=calls[0];m['generation_runtime_s']=time.monotonic()-beg;m['locked_angle_alpha']=alpha
        import csv
        output=HERE/'P5_model_calls.csv';exists=output.exists();record=dict(configuration=label,window_id=q['meta']['window_id'],generation_seed=q['meta']['generation_seed'],model_calls=calls[0],generation_runtime_s=time.monotonic()-beg,angle_alpha=alpha)
        with output.open('a',newline='',encoding='utf-8') as fp:w=csv.DictWriter(fp,fieldnames=record);w.writeheader() if not exists else None;w.writerow(record)
def execute_neural(run,e,pools):
    from selection import nominate
    sampler=nominate(run,'P3','angle_alpha_0.5',['angle_alpha_0'],arm='E');e.locked_angle_alpha=0. if sampler.get('candidate') else .5
    dump(HERE/'SAMPLING_POLICY_LOCK.json',dict(angle_alpha=e.locked_angle_alpha,screen=sampler,streams='Independent family/length/angle/position/local-choice streams; candidate/stage keyed final selection priorities',count='S1 fit density',scorer='centroid+slot-safe+family2-only tail; legacy floors retained; uninformative intersection recalibration not promoted'))
    probes=pd.read_csv(HERE/'P3_same_context_latent_probes.csv');fit=probes[(probes.role=='train')&(probes.perturbation=='reference')]
    (HERE/'P3_RETRAINING_DECISION.md').write_text('P0–P3 are complete before neural fitting. Controlled sampling is frozen in SAMPLING_POLICY_LOCK.json. Proceed to the two individual learnability fits because the fitting reference still has mean raw angle MAE '+str(float(fit.angle_MAE_deg.mean()))+' degrees, family accuracy '+str(float(fit.family_accuracy.mean()))+', and mean vertical span '+str(float(fit.y_span_fraction.mean()))+' versus target '+str(float(fit.truth_y_span_fraction.mean()))+'. Confirmed EVAE support/background omissions independently justify four paired semantic-repair pilots. Continue to the Transformer screen only if tiny-fit learnability and the corrected E_LB reference remain usable. No job is launched merely to fill the18-job allowance; no bigger model, extra grid or new physics is added.\n',encoding='utf-8')
    run.state['phase']='P4';run.log('P4','running');historical=e.load_bundle(42001)
    if not tiny_fit(run,e,historical):
        run.log('P4','failed',reason='Single identifiable examples did not reduce positive supervised loss by90%; affected broad training stopped.');dump(HERE/'neural_dependency_stop.json',dict(P4='tiny fit gate failed',P5='deferred by dependency'));return
    repaired=None;results=[]
    for name in ['E_R','E_L','E_B','E_LB']:
        try:
            ev,meta=train_evae(run,e,name,42001);results.append(dict(variant=name,status='completed',**meta))
            if name=='E_LB':repaired=ev
        except Exception as exc:
            if exc.__class__.__name__=='ResourceStop':raise
            run.log(name,'failed',exception=repr(exc),traceback=traceback.format_exc());results.append(dict(variant=name,status='failed',exception=repr(exc)))
    dump(HERE/'EVAE_pilot_summary.json',results)
    if repaired is None:
        dump(HERE/'neural_dependency_stop.json',dict(P4='E_LB unusable',P5='deferred by dependency'));return
    bundle=make_bundle(run,e,42001,repaired)
    e.hidden_priors={}
    seq=e.sequences(bundle,True)
    for layout in ['centre','5m']:
        fam=np.concatenate([x['fam'] for x in seq['train'] if x['layout']==layout]);counts=np.bincount(fam,minlength=4);e.hidden_priors[layout]=counts/counts.sum()
    dump(HERE/'fit_hidden_priors.json',e.hidden_priors)
    run.state['phase']='P5';nets={}
    for name in ['T0','T_H','T_P','T_F','T_Z']:
        try:
            net,meta=train_tf(run,e,bundle,name,42001);nets[name]=net;generate_tf(run,e,bundle,net,name+'_seed42001',pools)
        except Exception as exc:
            if exc.__class__.__name__=='ResourceStop':raise
            run.log(name,'failed',exception=repr(exc),traceback=traceback.format_exc())
            if name=='T0':dump(HERE/'neural_dependency_stop.json',dict(P5='T0 corrected reference failed; loss search stopped'));return
    decision=nominate(run,phase='P5',reference='T0_seed42001',candidates=[x+'_seed42001' for x in ['T_H','T_P','T_F','T_Z'] if x in nets],arm='E');dump(HERE/'candidate_nomination.json',decision)
    candidate=decision.get('candidate')
    if candidate:
        label=candidate.replace('_seed42001','');parent='T0' if label=='T_H' else 'T_H'
        for name in [parent,label]:generate_tf(run,e,bundle,nets[name],name+'_seed42001_final',pools,seeds=[62001,62002,62003])
        for seed in [42002,42003]:
            run.guard(1200);b=make_bundle(run,e,seed)
            for name in [parent,label]:
                net,meta=train_tf(run,e,b,name,seed);generate_tf(run,e,b,net,name+'_seed'+str(seed)+'_final',pools,seeds=[62001,62002,62003])
        from types import SimpleNamespace
        cl=[];nm=[]
        for rows,target in [(run.clusters,cl),(run.metrics,nm)]:
            for row in rows:
                if row['phase']=='P5' and row['configuration'].endswith('_final'):
                    a=dict(row);a['phase']='confirmation';a['configuration']='candidate' if row['configuration'].startswith(label+'_seed') else 'parent';target.append(a)
        confirmation=nominate(SimpleNamespace(clusters=cl,metrics=nm),'confirmation','parent',['candidate'],arm='E');confirmation.update(training_seeds=[42001,42002,42003],generation_seeds=[62001,62002,62003],direct_parent=parent,pilot_candidate=label)
        dump(HERE/'confirmation_decision.json',confirmation)
        # Frozen controls get one set of fresh-seed outputs, not one copy per training seed.
        from followup import new_stat_pool
        control=json.loads((HERE/'baseline_selection.json').read_text());kde=control['KDE'].get('candidate') or 'KDE_bw_1';adf=control['ADFNE'].get('candidate') or 'ADFNE_bins_12'
        for p in pools:
            if p['meta']['arm'] not in ['KDE','ADFNE']:continue
            q=copy.copy(p);q['meta']=dict(p['meta']);oldseed=int(q['meta']['generation_seed']);newseed=[62001,62002,62003][[42001,42002,42003].index(oldseed)];q['meta']['generation_seed']=newseed;q['meta']['case_key']=q['meta']['case_key'].replace('__g'+str(oldseed),'__g'+str(newseed))
            e.s.corrected_references(e.r,e.base,q['pub'],e.cm);q['fixed']=e.env['indicated'](q['obs'],q['pub'],q['st'],'stat',historical['models'],np.random.default_rng(newseed))
            if q['meta']['arm']=='KDE':
                bw=float(kde.rsplit('_',1)[1]);run.case(e,new_stat_pool(e,q,'kde',bw=bw),'locked_'+kde,'final_controls',policy='slotsafe',eligibility='all',tail=True,keyed=True)
                run.case(e,new_stat_pool(e,q,'kde',spatial=True),'locked_KDE_spatial','final_controls',policy='slotsafe',eligibility='all',tail=True,keyed=True)
            else:
                bins=int(adf.rsplit('_',1)[1]);run.case(e,new_stat_pool(e,q,'hist',bins=bins),'locked_'+adf,'final_controls',policy='slotsafe',eligibility='all',tail=True,keyed=True)
        run.log('P5_confirmation','complete',candidate=label,parent=parent,decision=confirmation,interpretation='Paired S1 screen; no independent-site or universal superiority inference.')
    else:run.log('P5_confirmation','not_applicable',reason='No single candidate passed the predefined S1 screen')
