# -*- coding: utf-8 -*-
"""RA-01 learned-component ablation of LEARNED_ETP (EXPLORATORY): RUNNER.

Pre-specification: 03_REANALYSIS/results/RA01_LEARNED_ABLATION/ABLATION_PRESPEC.md (SHA-256 recorded below and checked
before every stage). The ablation arms are COMPOSITIONS of frozen functions (learned_arm, final_eval, eval_learned,
learned, gast, p2v2, p2lib, rules2); no frozen file is edited. Function compose() is the only new reconstruction code:
  tier 2       E = learned: learned.tier2_learned (evidence network); E = rule: p2v2.tier2 (as final_eval.frozen)
  intensity    p2v2.budget (R2), A_PRIOR 2 (both)
  pool         stat      p2lib.unif_proposals(60, ...)                 (STAT_F's pool)
               tp_evae   learned_arm.reconstruct lines 290-294, repeated exactly (LEARNED_ETP's pool)
               tp_stat   transformer centres with statistical (direction, whole-trace length) pairs   (added arm)
               unif_evae EVAE shapes at 60 uniform centres                                           (added arm)
  placement    eval_learned.rule_place (R1, R4, R3 via final_eval.fill; then R5 rules2.cleanup)
  termination  none / learned_arm.terminate mode 'rebuilt' (TermNet2) / mode 'rule' (C['p_pass']); observed spans of
               tier 2 protected (eval_learned.protect_spans); pieces <= 0.05 m removed after termination
With (rule, stat, none) and seed key STAT_F the composition is final_eval.frozen(..., clean=True) with STAT_F's pool;
with (learned, tp_evae, rebuilt) and seed key LEARNED_ETP it is learned_arm.reconstruct. Both are regression-tested.

Conventions: trained wall = D34; Section 2 = D27/D29 (as ra10s2_run._job_body). Scoring as the stored rows.

Modes:
  python -B ra01abl_run.py gates            digests; label copies; regression (Section 2 one job, trained-wall STAT_F one
                                            window); determinism (one trained-wall window, r 0/8/16, all arms, twice)
  python -B ra01abl_run.py single SPEC OUT  one job in this process (SPEC = JSON job); writes OUT/part.csv, part.npz, meta
  python -B ra01abl_run.py main [STAGES]    stages 1 (Section 2), 2 (trained wall 5 m), 3 (trained wall centre);
                                            10 workers, fresh process per job; resumable (finished jobs are skipped)
  python -B ra01abl_run.py assemble         rows, reconstructions, full regression against stored rows, run record
Writes per-job parts to the session scratchpad and the assembled outputs to results/RA01_LEARNED_ABLATION."""
import os, sys, json, time, math, shutil, hashlib, platform, subprocess, traceback
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import ra10s2_run as R10      # noqa: E402  frozen Section 2 runner: windows, digests, helpers (nothing is run)

VERSION = 'ra01abl_run 1.0 (2026-10-12), EXPLORATORY'
PROJECT = R10.PROJECT
OUT = PROJECT / 'work' / 'RA01_LEARNED_ABLATION'
PRESPEC = OUT / 'ABLATION_PRESPEC.md'
PRESPEC_SHA = 'f4f2f0ebba9c7bf649187c7f469046630dadcf04c72b8a1288f35fedfb10ccc7'
SCRATCH = PROJECT / 'work' / 'ra01abl'
PARTS = SCRATCH / 'parts'
LABEL_COPY = SCRATCH / 'labels'
STORED_S2SEC = R10.OUT / 'rows_RA10_S2.csv'
STORED_S2SEC_SHA = '742912572b9cb03cea03449c84c59ad75b150a79684e4f0ed9dbce127ac420f0'
STORED_MAINC = R10.STUDY / 'results' / 'final_mainC_rows.csv'
STORED_MAINC_SHA = '1f8428d24bd59bb4f25ad37cbfdbd154e7997c47923b4f85975122db4a28bc1e'
RUNNER10_SHA = '3b8a100b0af5049a2bfb55564afc3b071fce726cd6f6ded08bf960e9fd6e9a7a'
PROTOCOLP_SHA = '51c04bb193240f2231e4bff3b2714ae8a4e84460d4983309f0f9adcb98f5c919'
RA02_DIR = PROJECT / 'work' / 'RA02'
NWORKERS = 10
SEEDS = (1337, 20260903, 7)
KEY = ['scheme', 'fold', 'box', 'cfg', 'arm', 'r']

# arm = (tier 2, pool, termination); the arm name is also its seed key
ARMS = {
    'STAT_F':      ('rule', 'stat', None),
    'LEARNED_ETP': ('learned', 'tp_evae', 'rebuilt'),
    'ABL_E':       ('learned', 'stat', None),
    'ABL_P':       ('rule', 'tp_evae', None),
    'ABL_T':       ('rule', 'stat', 'rebuilt'),
    'ABL_noE':     ('rule', 'tp_evae', 'rebuilt'),
    'ABL_noP':     ('learned', 'stat', 'rebuilt'),
    'ABL_noT':     ('learned', 'tp_evae', None),
    'ABL_Trule':   ('rule', 'stat', 'rule'),
    'ABL_ruleT':   ('learned', 'tp_evae', 'rule'),
    'ABL_Pc':      ('rule', 'tp_stat', None),
    'ABL_Ps':      ('rule', 'unif_evae', None),
}
ARM_ORDER = list(ARMS)
LONG_PIECE = 10.0


# ------------------------------------------------------------------ helpers
def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def sha256(path):
    return R10.sha256(path)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=str)


def environment():
    d = R10.environment()
    d['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS'); d['MKL_NUM_THREADS'] = os.environ.get('MKL_NUM_THREADS')
    return d


# ------------------------------------------------------------------ the composition (only new reconstruction code)
def make_pool(kind, C, t1, lines, H, rng, models, evae_nets, net_offset):
    import p2lib as P
    if kind == 'stat':                                                     # final_eval.run_window l. 117
        return lambda k, rec: P.unif_proposals(60, C['tr_th'], C['tr_len3d'], H, rng)
    import gast as G
    if kind == 'tp_evae':                                                  # learned_arm.reconstruct l. 290-294
        gnet = models['gnet']

        def pf(k, rec, src=evae_nets):
            g = G.generate(gnet, C, t1, rec, lines, H, rng); s_ = src[(net_offset + k) % 24]
            if not len(g) or not len(s_): return np.zeros((0, 4))
            e = s_[rng.integers(len(s_), size=len(g))]; c = (g[:, :2] + g[:, 2:]) / 2; ce = (e[:, :2] + e[:, 2:]) / 2
            return P.clip(e + np.hstack([c - ce, c - ce]), H)
        return pf
    if kind == 'tp_stat':                                                  # added: transformer centres, statistical shapes
        gnet = models['gnet']

        def pf(k, rec):
            g = G.generate(gnet, C, t1, rec, lines, H, rng)
            if not len(g): return np.zeros((0, 4))
            c = (g[:, :2] + g[:, 2:]) / 2
            idx = rng.integers(0, len(C['tr_th']), len(g))
            th, ln = C['tr_th'][idx], C['tr_len3d'][idx]
            dx, dy = np.cos(th) * ln / 2, np.sin(th) * ln / 2
            return P.clip(np.column_stack([c[:, 0] - dx, c[:, 1] - dy, c[:, 0] + dx, c[:, 1] + dy]), H)
        return pf
    if kind == 'unif_evae':                                                # added: EVAE shapes, uniform centres
        def pf(k, rec, src=evae_nets):
            s_ = src[(net_offset + k) % 24]
            if not len(s_): return np.zeros((0, 4))
            c = np.column_stack([rng.uniform(0, P.M, 60), rng.uniform(0, H, 60)])
            e = s_[rng.integers(len(s_), size=60)]; ce = (e[:, :2] + e[:, 2:]) / 2
            return P.clip(e + np.hstack([c - ce, c - ce]), H)
        return pf
    raise ValueError(kind)


def compose(C, obs, lines, H, rng, arm, models, evae_nets, net_offset, dmc):
    """one reconstruction of an ablation arm; returns (rec, diagnostics). Random draws in the order of the frozen code."""
    import p2lib as P
    import p2v2 as V
    import final_eval as FE
    import learned as LN
    import eval_learned as EL
    import learned_arm as LA
    tier2, pool, term = ARMS[arm]
    t1 = V.tier1(obs); groups = V.merge(obs, FE.BASE['JOIN_DTH'], FE.BASE['JOIN_DPERP'])
    if tier2 == 'learned':
        t2, _ = LN.tier2_learned(models['evn'], C, groups, lines, H, rng)
    else:
        t2 = V.tier2(groups, C['th2'], C['tr_len2d'], H, rng, FE.BASE['LEN_DWIN'])
    bud, p21h = V.budget(obs, lines, H, C['tr_th'], C['tr_len3d'], C['prior'], FE.BASE['A_PRIOR'])
    pf = make_pool(pool, C, t1, lines, H, rng, models, evae_nets, net_offset)
    rec = EL.rule_place(C, t2, bud, p21h, lines, H, rng, pf, dmc, len(t2))
    diag = dict(n_t2=int(len(t2)), len_t2_m=float(P.seglen(t2).sum()), budget_m=float(bud), p21_hat=float(p21h),
                n_preT=int(len(rec)), len_preT_m=float(P.seglen(rec).sum()), n_cut=0, n_removed_short=0)
    if term is not None and len(rec) > 1:
        pr = EL.protect_spans(t2, obs) + [None] * (len(rec) - len(t2))
        before = np.array(rec, float).reshape(-1, 4).copy()
        rec = LA.terminate(rec, len(t2), rng, C['model'], H, term, net=models['term'] if term == 'rebuilt' else None,
                           p_pass=C['p_pass'], protect=pr)
        diag['n_cut'] = int((np.abs(rec - before).max(1) > 1e-12).sum()) if len(rec) == len(before) else -1
        keep = P.seglen(rec) > 0.05
        diag['n_removed_short'] = int((~keep).sum())
        rec = rec[keep]
    return rec, diag


def piece_stats(rec, P):
    S = np.asarray(rec, float).reshape(-1, 4); ln = P.seglen(S)
    if not len(ln):
        return dict(med_len_m=np.nan, mean_len_m=np.nan, share_ge10=np.nan, total_len_m=0.0, max_len_m=np.nan)
    return dict(med_len_m=float(np.median(ln)), mean_len_m=float(ln.mean()), share_ge10=float((ln >= LONG_PIECE).mean()),
                total_len_m=float(ln.sum()), max_len_m=float(ln.max()))


def extra_cols(rec, gt, H, diag, P):
    ps = piece_stats(rec, P); pg = piece_stats(gt, P)
    p21_rec = ps['total_len_m'] / (P.M * H); p21_gt = pg['total_len_m'] / (P.M * H)
    out = dict(ps)
    out['p21_signed'] = p21_rec / p21_gt - 1.0 if p21_gt > 0 else np.nan
    out.update({f'gt_{k}': v for k, v in pg.items()})
    out.update(diag)
    out['len_removed_T_m'] = diag['len_preT_m'] - ps['total_len_m']
    return out


# ------------------------------------------------------------------ jobs
def _setup_worker():
    import torch
    torch.set_num_threads(1)
    import learned_arm as LA                                               # study + release on sys.path (read only)
    return LA


def job_section2(spec):
    """Section 2: one window and one S2 fold (ra10s2_run._job_body conventions, D29)"""
    LA = _setup_worker()
    import p2lib as P
    import p2v2 as V
    import final_eval as FE
    import gast as G
    from run_v2 import sd
    tag, x0, fi = spec['tag'], float(spec['x0']), int(spec['fold'])
    arms = spec.get('arms') or ARM_ORDER
    sp = R10.PIECES[tag]
    lab = LABEL_COPY / Path(sp['labels']).name
    if sha256(lab) != sp['labels_sha256']:
        raise RuntimeError(f'label copy digest differs: {lab}')
    L, Himg = LA.piece_traces(lab)
    H = float(sp['H']) if sp['H'] is not None else Himg
    yoff = float(sp['y_offset'])
    gt = P.clip(L - np.array([x0, yoff, x0, yoff]), H)
    scheme = 'EAST' + tag; box = f'x{int(x0):02d}'
    smodel = V.setbg.load(P.REL / 'data' / 'orientation_clusters' / 'S2_f0.json')
    C = G.prep_ctx('S2', fi)
    nets = LA.evae_prior_nets(fi, H, x0, tag)
    dmc = {}; rows = []; recs = {}
    for si, seed in enumerate(P.SEEDS):
        if spec.get('seed_idx') is not None and si not in spec['seed_idx']:
            continue
        models = LA.load_models(fi, seed, 'rebuilt')
        r = fi * 3 + si
        for spacing in (5.0, None):
            cname = '5m' if spacing else 'centre'
            lines = V.lines_layout('v', spacing, FE.BASE['APERTURE'], H)
            obs = V.observe(gt, lines, H, np.random.default_rng(sd('EAST' + tag, int(x0), r, cname, 'obs')))
            for arm in arms:
                rng = np.random.default_rng(sd('EAST' + tag, int(x0), r, cname, arm)); t0 = time.time()
                rec, diag = compose(C, obs, lines, H, rng, arm, models, nets, si * 8, dmc)
                row = LA.score_row(rec, gt, H, smodel, scheme, fi, box, cname, arm, r, time.time() - t0)
                row.update(extra_cols(rec, gt, H, diag, P)); row['n_hits'] = len(obs)
                rows.append(row); recs[f'{arm}__{cname}__r{r:02d}'] = np.asarray(rec, float).reshape(-1, 4)
    return rows, recs, gt, dict(H=H, scheme=scheme, box=box, dmin_cache={str(k): v for k, v in dmc.items()})


def evae_prior_trained(fi, box, H):
    """D34: 24 EVAE prior samples per trained-wall window, seed study.gen_seed('EVAE', 'S2prior', fi, box, seed, draw)"""
    import p2lib as P
    import study as S
    import tr_gen as TG
    out = []
    for s in P.SEEDS:
        ev = TG.evae_sampler('S2', fi, s)
        for d in range(P.NDRAW):
            out.append(P.clip(ev(H, S.gen_seed('EVAE', 'S2prior', fi, box, s, d)), H))
    return out


def job_trained(spec):
    """trained wall: one S2 test window and one layout, r = 0-23 (D34 conventions)"""
    LA = _setup_worker()
    import p2lib as P
    import p2v2 as V
    import final_eval as FE
    import gast as G
    import statistical_tests as ST
    from run_v2 import sd
    fi, box, cname = int(spec['fold']), spec['box'], spec['cfg']
    arms = spec.get('arms') or ARM_ORDER
    rlist = spec.get('rlist') or list(range(24))
    C = G.prep_ctx('S2', fi); model = C['model']
    b = C['boxes'][box]; H = b['H']; gt = P.clip(b['L'], H); ym = H / 20
    nets = evae_prior_trained(fi, box, H)
    models = {s: LA.load_models(fi, s, 'rebuilt') for s in P.SEEDS}
    spacing = 5.0 if cname == '5m' else None
    lines = V.lines_layout('v', spacing, FE.BASE['APERTURE'], H)
    dmc = {}; rows = []; recs = {}
    for r in rlist:
        obs = V.observe(gt, lines, H, np.random.default_rng(sd('S2', box, r, cname, 'final_obs')))
        m = models[P.SEEDS[r // 8]]
        for arm in arms:
            rng = np.random.default_rng(sd('S2', box, r, cname, 'final', arm)); t0 = time.time()
            rec, diag = compose(C, obs, lines, H, rng, arm, m, nets, r, dmc)
            sec = time.time() - t0
            s = ST.score(np.asarray(rec).reshape(-1, 4) / 20, gt / 20, ym, model)
            s = {k: (np.nan if (v is not None and np.isinf(v)) else v) for k, v in s.items()}
            row = dict(scheme='S2', fold=fi, box=box, cfg=cname, arm=arm, r=r, n=len(rec), gt_n=len(gt),
                       map_rmse=FE.p21map_err(rec, gt, H), sec=sec, **s)
            row.update(extra_cols(rec, gt, H, diag, P)); row['n_hits'] = len(obs)
            rows.append(row); recs[f'{arm}__{cname}__r{r:02d}'] = np.asarray(rec, float).reshape(-1, 4)
    return rows, recs, gt, dict(H=H, scheme='S2', box=box, dmin_cache={str(k): v for k, v in dmc.items()})


def job_name(spec):
    if spec['area'] == 'section2':
        return f"sec_{spec['tag']}_x{int(float(spec['x0'])):03d}_f{int(spec['fold'])}"
    return f"tw_{spec['cfg']}_f{int(spec['fold'])}_{spec['box']}"


def run_job(spec):
    """one job; returns dict(rows, recs, gt, meta). Errors are caught and returned in meta."""
    started = now(); t0 = time.time()
    meta = dict(spec=spec, name=job_name(spec), started=started, pid=os.getpid())
    out = dict(rows=[], recs={}, gt=np.zeros((0, 4)), meta=meta)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn = job_section2 if spec['area'] == 'section2' else job_trained
            rows, recs, gt, m = fn(spec)
        out.update(rows=rows, recs=recs, gt=gt); meta.update(m)
    except Exception:
        meta['error'] = traceback.format_exc()
    meta.update(finished=now(), job_sec=time.time() - t0, environment=environment(), code=R10.imported_files())
    return out


def save_part(res, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    name = res['meta']['name']
    write_json(folder / f'{name}.meta.json', res['meta'])
    if 'error' in res['meta']:
        return False
    np.savez_compressed(folder / f'{name}.npz', gt=res['gt'], **res['recs'])
    pd.DataFrame(res['rows']).to_csv(folder / f'{name}.csv', index=False, float_format='%.17g')
    return True


def single(spec_json, outdir):
    spec = json.loads(spec_json)
    res = run_job(spec)
    ok = save_part(res, outdir)
    print(f"single {res['meta']['name']}: {'ok' if ok else 'ERROR'} rows {len(res['rows'])} {res['meta']['job_sec']:.0f} s", flush=True)
    if not ok:
        print(res['meta']['error'], flush=True)
        raise SystemExit(2)


# ------------------------------------------------------------------ digests (read only)
def verify_all():
    checks = []

    def add(kind, name, got, expected, ok, note=''):
        checks.append(dict(kind=kind, file=str(name), sha256=got, expected=expected, ok=bool(ok), note=note))
    g = sha256(PRESPEC); add('prespec', PRESPEC, g, PRESPEC_SHA, g == PRESPEC_SHA)
    g = sha256(Path(R10.__file__).resolve()); add('code', 'ra10s2_run.py', g, RUNNER10_SHA, g == RUNNER10_SHA)
    g = sha256(HERE / 'protocolP.py'); add('code', 'protocolP.py', g, PROTOCOLP_SHA, g == PROTOCOLP_SHA)
    for f, e in R10.EXPECTED_CODE.items():
        g = sha256(f) if f.exists() else 'MISSING'; add('code', f, g, e, g == e)
    for f, e in R10.EXPECTED_DATA.items():
        g = sha256(f) if f.exists() else 'MISSING'; add('data', f, g, e, g == e)
    for (fi, s), e in R10.TERM2.items():
        f = R10.RA32_MODELS / f'term2_S2_f{fi}_s{s}.pt'; g = sha256(f) if f.exists() else 'MISSING'; add('model', f, g, e, g == e)
    for name, (files, e) in R10.families().items():
        missing = [str(f) for f in files if not Path(f).exists()]
        if missing:
            add('family', name, 'MISSING', e, False, f'missing: {missing[:3]}'); continue
        g = R10.combined(files); add('family', name, g, e, g == e and len(files) == R10.FAMILY_COUNTS.get(name, 21), f'{len(files)} files')
    g = sha256(STORED_S2SEC); add('stored rows', STORED_S2SEC, g, STORED_S2SEC_SHA, g == STORED_S2SEC_SHA)
    g = sha256(STORED_MAINC); add('stored rows', STORED_MAINC, g, STORED_MAINC_SHA, g == STORED_MAINC_SHA)
    for tag, sp in R10.PIECES.items():
        lab = LABEL_COPY / Path(sp['labels']).name
        g = sha256(lab) if lab.exists() else 'MISSING'; add('label copy', lab, g, sp['labels_sha256'], g == sp['labels_sha256'])
    return dict(all_ok=all(c['ok'] for c in checks), n_checks=len(checks), n_failed=sum(not c['ok'] for c in checks), checks=checks)


def copy_labels():
    LABEL_COPY.mkdir(parents=True, exist_ok=True)
    for tag, sp in R10.PIECES.items():
        src = Path(sp['labels'])
        if sha256(src) != sp['labels_sha256']:
            raise SystemExit(f'STOP: label file differs from D27: {src}')
        dst = LABEL_COPY / src.name
        shutil.copy2(src, dst)
        if sha256(dst) != sp['labels_sha256']:
            raise SystemExit(f'STOP: label copy differs: {dst}')


# ------------------------------------------------------------------ comparisons
def value_columns(df):
    return [c for c in df.columns if c not in KEY + ['sec']]


def parse_image(df, cols):
    """values as they come back from the CSV round trip the stored final_mainC_rows.csv went through: written with
    pandas' default float format (shortest repr) and read with pandas' default parser (not round-trip exact)"""
    import io
    buf = io.StringIO(); df[cols].to_csv(buf, index=False); buf.seek(0)
    img = pd.read_csv(buf)
    out = df.copy()
    for c in cols:
        out[c] = img[c].to_numpy(dtype=float)
    return out


def compare_to_stored(new, old, cols=None, image=False):
    """bitwise comparison on the key; NaN equals NaN. image=True compares the default-parser image of the new values
    (see parse_image) with the stored values read exactly. Returns a dict."""
    new = new.copy(); old = old.copy()
    if image:
        cv = cols if cols is not None else [c for c in value_columns(old) if c in new.columns]
        new = parse_image(new.reset_index(drop=True), cv)
    for k in ('scheme', 'box', 'cfg', 'arm'):
        new[k] = new[k].astype(str); old[k] = old[k].astype(str)
    for k in ('fold', 'r'):
        new[k] = new[k].astype(int); old[k] = old[k].astype(int)
    if cols is None:
        cols = [c for c in value_columns(old) if c in new.columns]
    m = new.merge(old, on=KEY, how='left', suffixes=('', '__stored'), indicator=True)
    found = int((m['_merge'] == 'both').sum())
    A = m[cols].to_numpy(dtype=float); B = m[[c + '__stored' for c in cols]].to_numpy(dtype=float)
    eq = (A == B) | (np.isnan(A) & np.isnan(B))
    fin = np.isfinite(A) & np.isfinite(B)
    md = float(np.max(np.abs(A - B)[fin])) if fin.any() else 0.0
    row_eq = eq.all(1)
    return dict(rows_new=int(len(new)), rows_matched=found, n_columns=len(cols), n_values=int(eq.size), n_equal=int(eq.sum()),
                rows_all_equal=int(row_eq.sum()), max_abs_diff=md,
                differing_columns=[c for j, c in enumerate(cols) if not eq[:, j].all()],
                all_equal=bool(found == len(new) and eq.all()))


def load_stored_s2sec():
    return pd.read_csv(STORED_S2SEC, float_precision='round_trip')


def load_stored_mainc():
    d = pd.read_csv(STORED_MAINC, float_precision='round_trip')
    return d[d['scheme'] == 'S2'].reset_index(drop=True)


# ------------------------------------------------------------------ gates
def run_single_subprocess(spec, outdir):
    cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'single', json.dumps(spec), str(outdir)]
    return subprocess.Popen(cmd, env=dict(os.environ))


def fecheck(box, outdir):
    """the frozen final_eval.run_window ('mainC') run today on one S2 fold-0 window: STAT_F rows of both layouts"""
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(R10.STUDY))
    import final_eval as FE
    rows = FE.run_window(('S2', 0, box, 'mainC'))
    df = pd.DataFrame(rows)
    df = df[df.arm == 'STAT_F'].copy()
    df['cfg'] = df['cfg'].str.split('|').str[0]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(outdir) / 'fe_runwindow_STAT_F.csv', index=False, float_format='%.17g')
    print(f'fecheck {box}: {len(df)} STAT_F rows', flush=True)


def gates():
    started = now(); t0 = time.time()
    if sha256(PRESPEC) != PRESPEC_SHA:
        raise SystemExit('STOP: the pre-specification changed')
    copy_labels()
    ver = verify_all()
    print(f"digests: {ver['n_checks']} checks, {ver['n_failed']} failed", flush=True)
    if not ver['all_ok']:
        for c in ver['checks']:
            if not c['ok']:
                print('  FAILED', c, flush=True)
        write_json(OUT / 'RA01ABL_gates.json', dict(step='gates', started=started, digests=ver, PASS=False))
        raise SystemExit('STOP: a frozen digest differs; nothing was run')
    gd = SCRATCH / 'gates'
    if gd.exists():
        shutil.rmtree(gd)
    sys.path.insert(0, str(R10.STUDY))
    import p2lib as P
    box0 = P.load_split('S2', 0)['test'][0]
    specs = {
        'sec_regression': dict(area='section2', tag='S2B01P1', x0=0.0, fold=0, arms=['STAT_F', 'LEARNED_ETP']),
        'tw_regression': dict(area='trained', fold=0, box=box0, cfg='5m', arms=['STAT_F'], rlist=list(range(24))),
        'tw_determinism_1': dict(area='trained', fold=0, box=box0, cfg='5m', arms=ARM_ORDER, rlist=[0, 8, 16]),
        'tw_determinism_2': dict(area='trained', fold=0, box=box0, cfg='5m', arms=ARM_ORDER, rlist=[0, 8, 16]),
    }
    procs = {k: run_single_subprocess(s, gd / k) for k, s in specs.items()}
    procs['fe_runwindow'] = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'fecheck', box0,
                                              str(gd / 'fe_runwindow')], env=dict(os.environ))
    rc = {k: p.wait() for k, p in procs.items()}
    print('gate processes:', rc, flush=True)
    res = dict(step='gates', version=VERSION, started=started, prespec_sha256=PRESPEC_SHA, digests=ver, return_codes=rc,
               specs=specs, note=('attempt 2. Attempt 1 (RA01ABL_gates_attempt1_strict.json) failed only on the strict '
                                  'bitwise comparison with final_mainC_rows.csv (638 of 984 values, largest difference '
                                  '1.8e-15). Diagnosis: the stored file holds the pandas default-parser image of the exact '
                                  'values (written with the shortest repr, read back with the default parser); the frozen '
                                  'final_eval.run_window run today equals the composition bitwise. Attempt 2 adds that check '
                                  '(fe_runwindow) and compares the stored rows with the same CSV round trip (image).'))
    ok = all(v == 0 for v in rc.values())
    if ok:
        def rd(k):
            nm = job_name(specs[k])
            return pd.read_csv(gd / k / f'{nm}.csv', float_precision='round_trip')
        sec = rd('sec_regression'); old = load_stored_s2sec()
        res['sec_regression'] = {a: compare_to_stored(sec[sec.arm == a], old) for a in ('STAT_F', 'LEARNED_ETP')}
        tw = rd('tw_regression'); oldc = load_stored_mainc()
        res['tw_regression_STAT_F_strict'] = compare_to_stored(tw, oldc)
        res['tw_regression_STAT_F_image'] = compare_to_stored(tw, oldc, image=True)
        fe = pd.read_csv(gd / 'fe_runwindow' / 'fe_runwindow_STAT_F.csv', float_precision='round_trip')
        res['tw_regression_STAT_F_vs_frozen_run_today'] = compare_to_stored(tw, fe[fe.cfg == '5m'])
        res['fe_run_today_vs_stored_image'] = compare_to_stored(fe, oldc, image=True)
        d1, d2 = rd('tw_determinism_1'), rd('tw_determinism_2')
        res['tw_determinism'] = R10.compare_rows(d1.drop(columns=[c for c in d1.columns if c == 'sec']),
                                                 d2.drop(columns=[c for c in d2.columns if c == 'sec']))
        res['tw_determinism']['rows'] = int(len(d1))
        res['tw_determinism_STAT_F_vs_frozen_run_today'] = compare_to_stored(d1[d1.arm == 'STAT_F'], fe[fe.cfg == '5m'])
        z1 = np.load(gd / 'tw_determinism_1' / f"{job_name(specs['tw_determinism_1'])}.npz")
        z2 = np.load(gd / 'tw_determinism_2' / f"{job_name(specs['tw_determinism_2'])}.npz")
        res['tw_determinism_recon_equal'] = bool(sorted(z1.files) == sorted(z2.files) and
                                                 all(np.array_equal(z1[k], z2[k]) for k in z1.files))
        ok = bool(all(v['all_equal'] for v in res['sec_regression'].values())
                  and res['tw_regression_STAT_F_image']['all_equal']
                  and res['tw_regression_STAT_F_vs_frozen_run_today']['all_equal']
                  and res['fe_run_today_vs_stored_image']['all_equal']
                  and res['tw_determinism']['all_equal'] and res['tw_determinism_recon_equal']
                  and res['tw_determinism_STAT_F_vs_frozen_run_today']['all_equal'])
    res.update(PASS=ok, finished=now(), seconds=time.time() - t0, environment=environment())
    write_json(OUT / 'RA01ABL_gates.json', res)
    summ = {}
    for k, v in res.items():
        if isinstance(v, dict) and ('all_equal' in v or k == 'sec_regression'):
            summ[k] = ({a: (w['n_equal'], w['n_values'], w['max_abs_diff'], w['all_equal']) for a, w in v.items()}
                       if k == 'sec_regression' else (v.get('n_equal'), v.get('n_values'), v.get('max_abs_diff'), v.get('all_equal')))
    summ['tw_determinism_recon_equal'] = res.get('tw_determinism_recon_equal'); summ['PASS'] = res['PASS']
    print(json.dumps(summ, indent=1, default=str), flush=True)
    if not ok:
        raise SystemExit('STOP: a gate failed')


# ------------------------------------------------------------------ main run
def all_jobs(stages):
    sys.path.insert(0, str(R10.STUDY))
    import p2lib as P
    jobs = []
    if 1 in stages:
        jobs += [dict(area='section2', tag=tag, x0=x0, fold=fi) for tag, sp in R10.PIECES.items() for x0 in sp['x0s']
                 for fi in range(7)]
    for st, cfg in ((2, '5m'), (3, 'centre')):
        if st in stages:
            jobs += [dict(area='trained', fold=fi, box=box, cfg=cfg) for fi in range(7) for box in P.load_split('S2', fi)['test']]
    return jobs


def main(stages):
    g = OUT / 'RA01ABL_gates.json'
    if not g.exists() or not json.load(open(g))['PASS']:
        raise SystemExit('STOP: the gates have not passed')
    if sha256(PRESPEC) != PRESPEC_SHA:
        raise SystemExit('STOP: the pre-specification changed')
    ver = verify_all()
    if not ver['all_ok']:
        raise SystemExit('STOP: a frozen digest differs')
    PARTS.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs(stages)
    todo = [j for j in jobs if not (PARTS / f'{job_name(j)}.csv').exists()]
    started = now(); t0 = time.time()
    print(f'{started} main stages {stages}: {len(jobs)} jobs, {len(todo)} to run, {NWORKERS} workers', flush=True)
    from multiprocessing import Pool
    errors = []
    with Pool(NWORKERS, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(run_job, todo, chunksize=1)):
            ok = save_part(res, PARTS)
            m = res['meta']
            if not ok:
                errors.append(m['name']); print(f"ERROR {m['name']}\n{m['error']}", flush=True)
            print(f"{now()} {i + 1}/{len(todo)} {m['name']} {m['job_sec']:.0f} s (elapsed {time.time() - t0:.0f} s)", flush=True)
    rec = dict(step=f'main {stages}', version=VERSION, started=started, finished=now(), seconds=time.time() - t0,
               jobs=len(jobs), run=len(todo), errors=errors, digests_ok=ver['all_ok'], environment=environment())
    write_json(SCRATCH / f"main_record_{'_'.join(map(str, stages))}_{time.strftime('%H%M%S')}.json", rec)
    print(f"{now()} main finished: {len(todo)} jobs, {len(errors)} errors", flush=True)


# ------------------------------------------------------------------ assembly
def assemble():
    started = now()
    parts = sorted(PARTS.glob('*.csv'))
    sec_parts = [p for p in parts if p.name.startswith('sec_')]
    tw_parts = [p for p in parts if p.name.startswith('tw_')]
    out = dict(step='assemble', version=VERSION, started=started, prespec_sha256=sha256(PRESPEC))
    arm_rank = {a: i for i, a in enumerate(ARM_ORDER)}
    if sec_parts:
        df = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sec_parts], ignore_index=True)
        tag_rank = {t: i for i, t in enumerate(R10.PIECES)}
        sk = pd.DataFrame(dict(a=df.scheme.str[4:].map(tag_rank), b=df.box.str[1:].astype(int), c=df.fold, d=df.r,
                               e=df.cfg.map({'5m': 0, 'centre': 1}), f=df.arm.map(arm_rank)))
        df = df.loc[sk.sort_values(list('abcdef'), kind='mergesort').index].reset_index(drop=True)
        p = OUT / 'rows_RA01ABL_section2.csv'; df.to_csv(p, index=False, float_format='%.17g')
        back = pd.read_csv(p, float_precision='round_trip')
        old = load_stored_s2sec()
        out['section2'] = dict(parts=len(sec_parts), rows=len(df), file=p.name, sha256=sha256(p),
                               regression={a: compare_to_stored(back[back.arm == a], old) for a in ('STAT_F', 'LEARNED_ETP')})
        arrays = {}
        gts = {}
        for pp in sec_parts:
            z = np.load(pp.with_suffix('.npz'))
            meta = json.load(open(pp.parent / (pp.stem + '.meta.json')))
            win = f"{meta['spec']['tag']}_x{int(float(meta['spec']['x0'])):03d}"
            gts.setdefault(win, []).append(z['gt'])
            for k in z.files:
                if k != 'gt':
                    arrays[f'{win}__{k}'] = z[k]
        gt_same = {w: bool(all(np.array_equal(v[0], x) for x in v)) for w, v in gts.items()}
        for w, v in gts.items():
            arrays[f'{w}__gt'] = v[0]
        pz = OUT / 'recon_RA01ABL_section2.npz'
        np.savez_compressed(pz, **arrays)
        out['section2'].update(recon_file=pz.name, recon_arrays=len(arrays), recon_sha256=sha256(pz), gt_identical_over_folds=gt_same)
    if tw_parts:
        df = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in tw_parts], ignore_index=True)
        sk = pd.DataFrame(dict(a=df.cfg.map({'5m': 0, 'centre': 1}), b=df.fold, c=df.box, d=df.r, e=df.arm.map(arm_rank)))
        df = df.loc[sk.sort_values(list('abcde'), kind='mergesort').index].reset_index(drop=True)
        p = OUT / 'rows_RA01ABL_trainedwall.csv'; df.to_csv(p, index=False, float_format='%.17g')
        back = pd.read_csv(p, float_precision='round_trip')
        oldc = load_stored_mainc()
        reg = dict(STAT_F_strict=compare_to_stored(back[back.arm == 'STAT_F'], oldc),
                   STAT_F_image=compare_to_stored(back[back.arm == 'STAT_F'], oldc, image=True))
        ra02 = None
        if RA02_DIR.exists():
            cands = [f for f in os.listdir(RA02_DIR) if f.startswith('rows') and f.endswith('.csv')]
            ra02 = dict(folder_exists=True, row_files=cands)
            for f in cands:
                o2 = pd.read_csv(RA02_DIR / f, float_precision='round_trip')
                if 'arm' in o2 and (o2['arm'] == 'LEARNED_ETP').any():
                    reg[f'LEARNED_ETP_vs_{f}'] = compare_to_stored(back[back.arm == 'LEARNED_ETP'], o2[o2.arm == 'LEARNED_ETP'])
        else:
            ra02 = dict(folder_exists=False)
        out['trained_wall'] = dict(parts=len(tw_parts), rows=len(df), file=p.name, sha256=sha256(p), regression=reg, ra02=ra02,
                                   layouts=sorted(df.cfg.unique().tolist()))
        for cfg in sorted(df.cfg.unique()):
            arrays = {}
            for pp in tw_parts:
                if not pp.name.startswith(f'tw_{cfg}_'):
                    continue
                z = np.load(pp.with_suffix('.npz'))
                meta = json.load(open(pp.parent / (pp.stem + '.meta.json')))
                box = meta['spec']['box']
                arrays[f'{box}__gt'] = z['gt']
                for k in z.files:
                    if k != 'gt':
                        arrays[f'{box}__{k}'] = z[k]
            pz = OUT / f'recon_RA01ABL_trainedwall_{cfg}.npz'
            np.savez_compressed(pz, **arrays)
            out['trained_wall'][f'recon_{cfg}'] = dict(file=pz.name, arrays=len(arrays), sha256=sha256(pz))
    metas = [json.load(open(m)) for m in sorted(PARTS.glob('*.meta.json'))]
    out['jobs'] = dict(n=len(metas), errors=[m['name'] for m in metas if 'error' in m],
                       job_seconds_total=float(sum(m.get('job_sec', 0) for m in metas)),
                       code_files=sorted({(k, v['sha256']) for m in metas for k, v in m.get('code', {}).items()}))
    out['finished'] = now()
    out['environment'] = environment()
    write_json(OUT / 'RA01ABL_run_record.json', out)
    rg = []
    for area in ('section2', 'trained_wall'):
        for arm, c in out.get(area, {}).get('regression', {}).items():
            rg.append(dict(status='EXPLORATORY', area=area, comparison=arm, **{k: v for k, v in c.items() if k != 'differing_columns'},
                           differing_columns=';'.join(c['differing_columns'])))
    pd.DataFrame(rg).to_csv(OUT / 'RA01ABL_regression_full.csv', index=False)
    print(json.dumps({k: out[k] for k in out if k in ('section2', 'trained_wall', 'jobs')}, indent=1, default=str)[:6000], flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    OUT.mkdir(parents=True, exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True)
    if mode == 'gates':
        gates()
    elif mode == 'single':
        single(sys.argv[2], sys.argv[3])
    elif mode == 'fecheck':
        fecheck(sys.argv[2], sys.argv[3])
    elif mode == 'main':
        st = [int(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1, 2, 3]
        main(st)
    elif mode == 'assemble':
        assemble()
    else:
        print(__doc__)
