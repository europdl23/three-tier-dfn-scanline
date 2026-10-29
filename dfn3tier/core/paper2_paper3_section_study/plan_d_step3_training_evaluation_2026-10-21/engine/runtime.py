"""Isolated, S1-only adapter runtime; no experiment is launched on import."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib.util,io,json,math,os,sys,time
import numpy as np
import pandas as pd
import torch
# PLAN D (declared change 1): stage and output folders come from the job environment; engine code lives in step3/engine.
ENGINE_DIR=Path(__file__).resolve().parent; STUDY=ENGINE_DIR.parent.parent; ROOT=STUDY.parent
STAGE=Path(os.environ['PLAN_D_STAGE']).resolve(); HERE=Path(os.environ['PLAN_D_OUT']).resolve(); HERE.mkdir(parents=True,exist_ok=True)
OLD=STUDY/'stepwise_execution_2026-10-19'
STAGING_ROOT=(STUDY/'plan_d_step2_staging_2026-10-21'/'stages').resolve()
STEP3_EVAL_OUT=(ENGINE_DIR.parent/'evaluation').resolve()
PLAN_D_MODE=os.environ.get('PLAN_D_MODE','train')   # 'train' (default) or 'evaluate'
def _plan_d_guard_paths():
    norm=lambda x:Path(x).resolve().as_posix().lower()
    forbidden_prefixes=[norm(ROOT/'bench_labelling_section1'),norm(STUDY/'option3a_dataset_freeze_2026-10-20'),norm(STUDY/'data_prep'),
                        norm(STUDY/'family_freeze_F4_2026-10-20/F4_FRAGMENT_FAMILIES.csv')]
    stage_root=norm(STAGING_ROOT);own=norm(STAGE);evalout=norm(STEP3_EVAL_OUT)
    def denied(path):
        q=norm(path)
        if any(q.startswith(f) for f in forbidden_prefixes):return 'source tables / raw labels'
        if q.startswith(stage_root+'/') and not (q==own or q.startswith(own+'/')):return 'another Plan D stage'
        if PLAN_D_MODE!='evaluate' and ('/eval/' in q+'/' and q.startswith(stage_root)):return 'evaluation stage'
        if PLAN_D_MODE!='evaluate' and q.startswith(evalout):return 'evaluation outputs'
        return None
    return denied
_PLAN_D_DENIED=_plan_d_guard_paths()
def _plan_d_hook(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        why=_PLAN_D_DENIED(os.fsdecode(args[0]))
        if why:raise PermissionError('Plan D read guard ('+why+'): '+os.fsdecode(args[0]))
    if event in ('subprocess.Popen','os.system','os.posix_spawn','os.exec','os.spawn','os.startfile'):
        raise PermissionError('Plan D read guard: subprocess creation is blocked inside guarded jobs ('+event+')')
pass  # public release: the read guard of the research runs is not installed (sys.addaudithook(_plan_d_hook))
if PLAN_D_MODE=='evaluate' and STAGE.name!='eval':raise PermissionError('evaluate mode requires an eval stage')
if PLAN_D_MODE!='evaluate' and STAGE.name=='eval':raise PermissionError('training jobs may not use an eval stage')
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.partial')
    q.write_text(json.dumps(x,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8');q.replace(p)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def build_engine():
    forbidden={str((STUDY/'local_window_retrain_2026-10-19'/n).resolve()).lower() for n in ['window_inventory.csv','window_trace_manifest.csv']}
    forbidden.add(str((STUDY/'data_prep/manual_trace_manifest.csv').resolve()).lower())
    # R1 (declared data-interface change): block files that contain reserved W00/W01, excluded W02/W04 or S2 data.
    for p in ['option3a_dataset_freeze_2026-10-20/WINDOW_FRAGMENTS_3A.csv','option3a_dataset_freeze_2026-10-20/WINDOW_INVENTORY_3A.csv',
              'option3a_dataset_freeze_2026-10-20/TRACE_MANIFEST_3A.csv','family_freeze_F4_2026-10-20/F4_FRAGMENT_FAMILIES.csv',
              'authorised_execution_2026-10-19/s1_inputs/window_inventory.csv','authorised_execution_2026-10-19/s1_inputs/window_trace_manifest.csv',
              'data_prep/manual_trace_segments_world_s1_s4.csv']:
        forbidden.add(str((STUDY/p).resolve()).lower())
    def read_guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=str(Path(os.fsdecode(args[0])).resolve()).lower();normal=path.replace('\\','/')
            if path in forbidden or '/06_s2/' in normal or ('/cases/s2_' in normal):raise PermissionError('S2/mixed-data runtime access forbidden: '+path)
    pass  # public release: S2 read guard not installed (sys.addaudithook(read_guard))
    torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.environ.get('PLAN_D_GPU_FRACTION','0.15')))  # PLAN D (declared change 3): parallel jobs
    eng=module('authorised_engine',OLD/'engine.py');original=eng.load_module
    def loader(name,path):
        m=original(name,path)
        if name=='stepwise_preview':
            def load_s1_state():
                r=module('authorised_original_definitions',STUDY/'s1_s2_original_revision_2026-10-18/run_revision.py')
                r.HERE=HERE
                stats=m.keyrestore(json.loads((STAGE/'training_only_statistics.json').read_text()))
                for k in ['fc','shares']:stats[k]=np.asarray(stats[k])
                stats['pools']={k:{n:np.asarray(v) for n,v in item.items()} for k,item in stats['pools'].items()};r.STATS=stats
                canonical=[]
                inv=pd.read_csv(STAGE/'window_inventory.csv');tr=pd.read_csv(STAGE/'window_trace_manifest.csv');faces={}
                for row in inv.itertuples():
                    a=tr[tr.window_id.eq(row.window_id)][['cx','cy','length','angle']].to_numpy(np.float32)
                    faces[row.window_id]=dict(id=row.window_id,role=row.role,w=20.,h=row.height_m,ym=row.height_m/20,arr=a)
                    if row.role=='train':b=a.copy();b[:,:3]*=20/36;canonical.append(b)
                r.FACES=faces
                cont=r.definitions(r.EXP/'hybrid_v6.py',dict(load_panels_raw=lambda:canonical,get_physics_stats=lambda:{'fam_centers_rad':stats['fc']},assign_family_ids=r.assign,segments_intersect=r.intersect,FAM_ASSIGN_THRESHOLD_DEG=20.),names=['apply_continuity_rules'])
                env=r.definitions(STUDY/'s1_s2_repair_trial_2026-10-18/repair_trial.py',dict(r=r,old_features=r.continuation_features,contextlib=contextlib,io=io,copy=copy,ast=ast,CONT=cont,STAGE_LOG=[],SPACING_METRES={k:.288 for k in range(4)},TRAIN_X_STD=.2,TRAIN_Y_STD=.2))
                r.indicated=env['indicated'];r.continuation_features=env['continuation_features'];return r,env,{}
            m.load_state=load_s1_state
        return m
    eng.load_module=loader;eng.LOCAL=STAGE;eng.HERE=HERE
    e=eng.Engine()
    # PLAN D (declared change 2): the R1 27-window assertion is replaced by the stage manifest check.
    inv=pd.read_csv(STAGE/'window_inventory.csv');assert set(e.faces)==set(inv.window_id) and not any(f['role']=='test' for f in e.faces.values())
    e.r.HERE=HERE
    e.source_module=eng
    return e
def corrected_feature(e,centroid=False,tail=False):
    src=ast.unparse(next(n for n in ast.parse((e.r.EXP/'scorer.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='extract_features'))
    if centroid:src=src.replace('n_existing = max(len(pts), 1)','n_existing = len(pts)')
    if tail:src=src.replace('if L >= f_p90:', 'if fam == 2 and L >= f_p90:').replace('elif L >= f_mean:', 'elif fam == 2 and L >= f_mean:')
    env=dict(e.r.scorer);exec(src,env);return env['extract_features']
def count_posterior(M,p,N0,alpha=2):
    if not 0<=p<=1 or N0<=0 or M<0 or (p==0 and M>0):raise ValueError('Inconsistent detection/count support')
    return M+(1-p)*(M+alpha)/(p+alpha/N0)
def streams(case):
    return {name:np.random.default_rng(int.from_bytes(hashlib.sha256((str(case)+'|'+name).encode()).digest()[:8],'little')) for name in ['family','length','angle','position','selection']}
def pack(e,items):
    n=max(1,max(len(z['obs']) for z in items));B=len(items)
    arrays={k:np.zeros(sh,np.float32) for k,sh in [('C_in',(B,n,2)),('L_in',(B,n,1)),('A_in',(B,n,1)),('C_tgt',(B,128,2)),('L_tgt',(B,128,1)),('A_tgt',(B,128,1)),('E_tgt',(B,128))]}
    arrays['mask']=np.zeros((B,n),bool);arrays['F_tgt']=np.full((B,128),-1,np.int64);arrays['aspect']=np.array([z['face']['ym'] for z in items],np.float32)
    for i,z in enumerate(items):
        a=z['arr'];o=z['obs'];nt=len(a);ni=len(o);assert nt<=128
        arrays['C_in'][i,:ni]=o[:,:2];arrays['L_in'][i,:ni]=o[:,2:3];arrays['A_in'][i,:ni]=o[:,3:4];arrays['mask'][i,:ni]=True
        arrays['C_tgt'][i,:nt]=a[:,:2];arrays['L_tgt'][i,:nt]=a[:,2:3];arrays['A_tgt'][i,:nt]=a[:,3:4];arrays['E_tgt'][i,:nt]=1;arrays['F_tgt'][i,:nt]=e.r.assign(a,e.base['fc'])
    return {k:torch.as_tensor(v,device=e.r.DEV) for k,v in arrays.items()}
def loss_function(e,background=False):
    node=next(n for n in ast.parse((e.r.EXP/'model_v4.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='compute_loss')
    src=ast.unparse(node).replace('torch.tensor([0.623, 1.165, 2.221], device=pc.device)','torch.tensor(FAMILY_CENTRES, device=pc.device)').replace('-kpr[:, k] * torch.cos(2 * diff)','-kpr[:, k] * torch.cos(2 * diff) + torch.log(torch.special.i0e(kpr[:, k])) + kpr[:, k] + math.log(math.pi)')
    if background:
        start=src.index('        if (~is_random).any():');end=src.index('        gt_A =',start)
        src=src[:start]+"        loss_family += F.cross_entropy(ml[b, pi_t], gt_f, reduction='sum')\n"+src[end:]
        src=src.replace('loss_angle += (p3 * nll_stack).sum()',"loss_angle += torch.where(is_random, torch.full_like(gt_A, math.log(math.pi)), (p3 * nll_stack).sum(-1)).sum()")
    env=dict(e.r.lossenv,FAMILY_CENTRES=e.base['fc']);exec(src,env);return env['compute_loss'],src
def make_evae(e,length=False):
    Base=e.env['MaskedRectangularEVAE']
    class SupportedEVAE(Base):
        def decode(self,sz,gz,mode='B'):
            pc,pl,ml,kp,el=super().decode(sz,gz,mode)
            if length:
                aspect=torch.as_tensor(self.batch_yscale,device=pl.device,dtype=pl.dtype).reshape(-1,1,1)
                pl=pl*torch.sqrt(1+aspect.square())
            return pc,pl,ml,kp,el
    return SupportedEVAE(N=128,yscale=e.base['yscale']).to(e.r.DEV)
def public_face(face):
    return {k:face[k] for k in ['id','parent_face','role','w','h','ym','x0'] if k in face}
def intersection_calibration(e):
    """Replace only constant F7 calibration samples with fit-prefix runtime F7."""
    node=next(n for n in ast.parse((e.r.EXP/'run_ablation_v3.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='_calibrate_scorer_weights')
    src=ast.unparse(node)
    replacement="feature_samples[6].append(max(0, sum(segments_intersect(arr[idx], ex) for ex in arr[max(0, idx-20):idx])-3)/max(float(gstats['intersection_std']), .1) if idx >= 8 else 0.0)"
    assert src.count('feature_samples[6].append(0.0)')==1
    src=src.replace('feature_samples[6].append(0.0)',replacement).replace('    return result','    result[\'F7_samples\'] = feature_samples[6]\n    return result')
    env=dict(e.r.geo,segments_intersect=e.r.intersect);exec(src,env)
    canonical=[]
    for f in e.faces.values():
        if f['role']=='train':a=f['arr'].copy();a[:,:3]*=20/36;canonical.append(a[np.argsort(a[:,0],kind='stable')])
    with contextlib.redirect_stdout(io.StringIO()):result=env['_calibrate_scorer_weights'](canonical,e.base['fc'],e.base['gs'])
    values=result.pop('F7_samples');valid=len(values)>5 and np.std(values)>1e-6
    artifact=dict(status='informative' if valid else 'uninformative_do_not_promote',sample_count=len(values),std=float(np.std(values)),nonzero=int(np.count_nonzero(values)),legacy=e.base['gs']['scorer_weights'],corrected=result,definition='Actual runtime soft-cap crossings against up to20 preceding x-ordered S1-fitting traces; same >=8 prefix rule and inter_std. Other calibration definitions unchanged.',source_sha=hashlib.sha256(src.encode()).hexdigest())
    dump(HERE/'intersection_calibration.json',artifact);return artifact
def verify_identity(actual,expected):
    if actual!=expected:raise ValueError('Cache/configuration fingerprint mismatch')
    return True
