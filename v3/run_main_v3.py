# -*- coding: utf-8 -*-
"""v3 main runs (15 Oct 2026): every generator of the paper rerun with the v3 indicated-trace rule (v3patch.py), with
the seeds, models, pools, rules and scoring of the original runs, so that the only change is the indicated-trace rule.

Job types (one job = one test window):
  fill     EVAE_F, STAT_F, KDE_F, ADFNE_F at 5 m and centre line: final_eval.run_window(('S2', fi, box, 'mainC'))
           unchanged, reconstructions captured through final_eval.p21map_err (as ra02_run.regen_job)
  stat     STAT_FIT, TR_FIT at 5 m and centre (as hybfix_ext/statfit_run.one); STAT_FIT at 10, 7, 3 m;
           STAT_FIT_noR5 at 5 m (same seeds, map adjustment off)
  hyb      HYB_FIX2 at 5 m and centre (as hybfix_ext/hybfix2_run.one_tw)
  hyblay   HYB_FIX2 at 10, 7, 3 m (as hybfix_ext/hyb2_layouts_run.one)
  hybnoR5  HYB_FIX2_noR5 at 5 m (same seeds as HYB_FIX2, map adjustment off: final_eval.BASE['CLEAN_IT'] = 0)
Output: v3/out/<type>/<type>_f<fold>_<box>.csv and .npz (keys ARM__CFG__rNN, gt).
Usage: python -B run_main_v3.py <type|all> [workers] | test"""
import os, sys, time, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HX = HERE.parent / 'hybrid'
sys.path.insert(0, str(HX)); sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import hybfix2_run as H2                 # frozen hybfix_run (HF) -> ra01abl_run (R) with HYB_FIX2 registered
import statfit_run as SF                 # TR_FIT registered; fit / sample / segs
R = H2.R; HF = H2.HF
R._setup_worker()                        # torch threads, learned_arm (study + release on sys.path)
import v3patch                           # noqa: E402
v3patch.install()
OUT = HERE.parent / 'work' / 'v3_out'
LAYS = {'10m': 10.0, '7m': 7.0, '3m': 3.0}


def _common(fi, box):
    import p2lib as P, p2v2 as V, final_eval as FE, gast as G, statistical_tests as ST
    return P, V, FE, G, ST


def _score(ST, FE, rec, gt, H, ym, model):
    s = ST.score(np.asarray(rec, float).reshape(-1, 4) / 20, gt / 20, ym, model)
    s = {k: (np.nan if (v is not None and np.isinf(v)) else v) for k, v in s.items()}
    return dict(map_rmse=FE.p21map_err(rec, gt, H), **s)


def job_fill(fi, box):
    P, V, FE, G, ST = _common(fi, box)
    store = []
    orig = FE.p21map_err

    def cap(rec, gt, H, _o=orig):
        store.append(np.asarray(rec, float).reshape(-1, 4).copy())
        return _o(rec, gt, H)
    FE.p21map_err = cap
    try:
        rows = FE.run_window(('S2', fi, box, 'mainC'))
    finally:
        FE.p21map_err = orig
    assert len(rows) == len(store), (len(rows), len(store))
    C = FE.TC.fold_ctx('S2', fi); H = C['boxes'][box]['H']; gt = P.clip(C['boxes'][box]['L'], H)
    recs = {}
    for row, rec in zip(rows, store):
        row['cfg'] = row['cfg'].split('|')[0]
        recs[f"{row['arm']}__{row['cfg']}__r{int(row['r']):02d}"] = rec
    return rows, recs, gt


def job_stat(fi, box):
    P, V, FE, G, ST = _common(fi, box)
    from run_v2 import sd
    LA = R._setup_worker()
    CF = FE.TC.fold_ctx('S2', fi); CG = G.prep_ctx('S2', fi)
    model = CF['model']; H = CF['boxes'][box]['H']; gt = P.clip(CF['boxes'][box]['L'], H); ym = H / 20
    fams = SF.fit(CF)
    models = {s: LA.load_models(fi, s, 'rebuilt') for s in P.SEEDS}
    dmc_f, dmc_h = {}, {}; rows, recs = [], {}

    def stat_rec(obs, lines, cname, clean):
        rng = np.random.default_rng(sd('S2', box, r, cname, 'final', 'STAT_FIT'))

        def pool(k, rng=rng):
            c = np.column_stack([rng.uniform(0, P.M, 60), rng.uniform(0, H, 60)]); th, L = SF.sample(fams, 60, rng)
            return SF.segs(c, th, L)
        return FE.frozen(CF, obs, lines, H, rng, pool, dict(FE.BASE), dmc_f, clean)

    for cname, sp in (('5m', 5.0), ('centre', None), ('10m', 10.0), ('7m', 7.0), ('3m', 3.0)):
        lines = V.lines_layout('v', sp, FE.BASE['APERTURE'], H)
        for r in range(24):
            obs = V.observe(gt, lines, H, np.random.default_rng(sd('S2', box, r, cname, 'final_obs')))
            arms = ['STAT_FIT'] + (['TR_FIT'] if cname in ('5m', 'centre') else []) + (['STAT_FIT_noR5'] if cname == '5m' else [])
            for arm in arms:
                if arm == 'STAT_FIT':
                    rec = stat_rec(obs, lines, cname, True)
                elif arm == 'STAT_FIT_noR5':
                    rec = stat_rec(obs, lines, cname, False)
                else:
                    rng = np.random.default_rng(sd('S2', box, r, cname, 'final', arm))
                    rec, _ = R.compose(CG, obs, lines, H, rng, arm, models[P.SEEDS[r // 8]], [np.zeros((0, 4))] * 24, r, dmc_h)
                rec = np.asarray(rec, float).reshape(-1, 4)
                rows.append(dict(scheme='S2', fold=fi, box=box, cfg=cname, arm=arm, r=r, n=len(rec), gt_n=len(gt),
                                 n_hits=len(obs), **_score(ST, FE, rec, gt, H, ym, model)))
                recs[f'{arm}__{cname}__r{r:02d}'] = rec
    return rows, recs, gt


def job_hyb(fi, box, cfgs=('5m', 'centre'), noR5=False):
    import final_eval as FE
    rows, recs, gt = [], {}, None
    old = FE.BASE['CLEAN_IT']
    try:
        if noR5:
            FE.BASE['CLEAN_IT'] = 0
        for cfg in cfgs:
            rw, rc, gt, meta = R.job_trained(dict(fold=fi, box=box, cfg=cfg, arms=[H2.ARM]))
            for x in rw:
                if noR5: x['arm'] = 'HYB_FIX2_noR5'
                rows.append(x)
            for k, v in rc.items():
                recs[k.replace('HYB_FIX2__', 'HYB_FIX2_noR5__') if noR5 else k] = v
    finally:
        FE.BASE['CLEAN_IT'] = old
    return rows, recs, gt


def job_hyblay(fi, box):
    P, V, FE, G, ST = _common(fi, box)
    from run_v2 import sd
    LA = R._setup_worker()
    CG = G.prep_ctx('S2', fi); model = CG['model']; b = CG['boxes'][box]; H = b['H']; gt = P.clip(b['L'], H); ym = H / 20
    nets = HF.evae_whole_trained(fi, box, H)
    models = {s: LA.load_models(fi, s, 'rebuilt') for s in P.SEEDS}
    dmc = {}; rows, recs = [], {}
    for lay, sp in LAYS.items():
        lines = V.lines_layout('v', sp, FE.BASE['APERTURE'], H)
        for r in range(24):
            obs = V.observe(gt, lines, H, np.random.default_rng(sd('S2', box, r, lay, 'final_obs')))
            rng = np.random.default_rng(sd('S2', box, r, lay, 'final', H2.ARM))
            rec, _ = R.compose(CG, obs, lines, H, rng, H2.ARM, models[P.SEEDS[r // 8]], nets, r, dmc)
            rec = np.asarray(rec, float).reshape(-1, 4)
            rows.append(dict(scheme='S2', fold=fi, box=box, cfg=lay, arm=H2.ARM, r=r, n=len(rec), gt_n=len(gt),
                             n_hits=len(obs), **_score(ST, FE, rec, gt, H, ym, model)))
            recs[f'{H2.ARM}__{lay}__r{r:02d}'] = rec
    return rows, recs, gt


JOBS = {'fill': job_fill, 'stat': job_stat, 'hyb': job_hyb, 'hyblay': job_hyblay,
        'hybnoR5': lambda fi, box: job_hyb(fi, box, ('5m',), noR5=True)}


def one(spec):
    import warnings
    warnings.simplefilter('ignore')
    kind, fi, box = spec
    t0 = time.time()
    try:
        rows, recs, gt = JOBS[kind](fi, box)
        d = OUT / kind; d.mkdir(parents=True, exist_ok=True)
        name = f'{kind}_f{fi}_{box}'
        pd.DataFrame(rows).to_csv(d / f'{name}.tmp', index=False, float_format='%.17g')
        np.savez_compressed(d / f'{name}.npz', gt=gt, **recs)
        os.replace(d / f'{name}.tmp', d / f'{name}.csv')
        return 'ok', name, time.time() - t0, ''
    except Exception:
        return 'error', f'{kind}_f{fi}_{box}', time.time() - t0, traceback.format_exc()


def windows():
    """the 50 test windows: (fold, window) of the seven leave-one-bench-out folds"""
    import p2lib as P
    return [(fi, box) for fi in range(7) for box in P.load_split('S2', fi)['test']]


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor, as_completed
    kinds = list(JOBS) if sys.argv[1] in ('all', 'test') else sys.argv[1].split(',')
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 22
    ws = windows()
    if sys.argv[1] == 'test':
        ws = [w for w in ws if w[1] == 'W02_C3']
    order = ['hyb', 'hyblay', 'hybnoR5', 'stat', 'fill']                    # longest first
    specs = [(k, f, b) for k in order if k in kinds for f, b in ws if not (OUT / k / f'{k}_f{f}_{b}.csv').exists()]
    print(len(specs), 'jobs', flush=True); t0 = time.time()
    with ProcessPoolExecutor(nw) as ex:
        futs = [ex.submit(one, s) for s in specs]
        for i, f in enumerate(as_completed(futs)):
            st, name, sec, err = f.result()
            print(f'{i + 1}/{len(specs)} {st} {name} {sec:.0f} s (elapsed {time.time() - t0:.0f} s)', flush=True)
            if st != 'ok':
                print(err, flush=True)
