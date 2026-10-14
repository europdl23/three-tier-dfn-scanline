# -*- coding: utf-8 -*-
"""PLAN step 5c-L: HYB_FIX2 = HYB_FIX + data-driven length limits on the inferred candidates (EXPLORATORY).
Per direction cluster k (fold cluster model, training whole-trace lengths tr_len3d): Lmin_k = shortest, Lmax_k = longest.
For each transformer trace, an EVAE whole trace of the same cluster is placed at its centre (as HYB_FIX) and cut once at
the window edge. It is redrawn (same cluster, up to 10 tries, then left out) when
  - its whole length exceeds Lmax_k, or
  - the placed piece has both ends inside the window (not cut by the edge) and is shorter than Lmin_k.
Pieces cut by the window edge are exempt. Everything else identical to HYB_FIX; seed key HYB_FIX2.
Usage: python hybfix2_run.py tw CFG [workers] | s2 [workers]"""
import os, sys, math, time
from pathlib import Path
FROZEN = Path(__file__).resolve().parent
sys.path.insert(0, str(FROZEN))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import hybfix_run as HF
import hybfix_s2_run as HS2
R = HF.R
ARM = 'HYB_FIX2'
TRIES = 10
EDGE_TOL = 0.05
_prev_make_pool = R.make_pool                                     # HF's make_pool (handles tp_evae_whole and the frozen kinds)


def make_pool(kind, C, t1, lines, H, rng, models, evae_nets, net_offset):
    if kind != 'tp_evae_whole_len':
        return _prev_make_pool(kind, C, t1, lines, H, rng, models, evae_nets, net_offset)
    import p2lib as P
    import p2v2 as V
    import gast as G
    gnet = models['gnet']
    cl_tr = np.asarray(V.setbg.classify(C['tr_th'], C['model']))
    LMIN = {int(k): float(C['tr_len3d'][cl_tr == k].min()) for k in np.unique(cl_tr)}
    LMAX = {int(k): float(C['tr_len3d'][cl_tr == k].max()) for k in np.unique(cl_tr)}

    def pf(k, rec, src=evae_nets):
        g = G.generate(gnet, C, t1, rec, lines, H, rng); s_ = src[(net_offset + k) % 24]
        if not len(g) or not len(s_): return np.zeros((0, 4))
        tg, te = HF.seg_theta(g), HF.seg_theta(s_)
        cg = np.asarray(V.setbg.classify(tg, C['model'])); ce_ = np.asarray(V.setbg.classify(te, C['model']))
        Le = P.seglen(s_)
        out = []
        for i in range(len(g)):
            idx = np.flatnonzero(ce_ == cg[i])
            if not len(idx):
                d = np.abs(((te - tg[i]) + math.pi / 2) % math.pi - math.pi / 2); idx = np.argsort(d)[:3]
            c = (g[i, :2] + g[i, 2:]) / 2
            kk = int(cg[i]); lmin = LMIN.get(kk, min(LMIN.values())); lmax = LMAX.get(kk, max(LMAX.values()))
            for _ in range(TRIES):
                j = idx[rng.integers(len(idx))]; e = s_[j]
                if Le[j] > lmax:
                    continue
                pc = P.clip((e + np.r_[c - (e[:2] + e[2:]) / 2, c - (e[:2] + e[2:]) / 2])[None], H)
                if not len(pc):
                    continue
                q = pc[0]
                inside = (min(q[0], q[2]) > EDGE_TOL and max(q[0], q[2]) < P.M - EDGE_TOL and
                          min(q[1], q[3]) > EDGE_TOL and max(q[1], q[3]) < H - EDGE_TOL)
                if inside and P.seglen(pc)[0] < lmin:
                    continue
                out.append(q); break
        return np.asarray(out, float).reshape(-1, 4)
    return pf


R.make_pool = make_pool
R.ARMS[ARM] = ('rule', 'tp_evae_whole_len', None)


def one_tw(spec):
    import warnings
    warnings.simplefilter('ignore')
    t0 = time.time()
    rows, recs, gt, meta = R.job_trained(dict(spec, arms=[ARM]))
    name = f"tw_{spec['cfg']}_f{spec['fold']}_{spec['box']}"
    pd.DataFrame(rows).to_csv(HERE / 'out_hyb2' / f'{name}.csv', index=False, float_format='%.17g')
    np.savez_compressed(HERE / 'out_hyb2' / f'{name}.npz', gt=gt, **recs)
    return name, time.time() - t0


def one_s2(spec):
    import warnings
    warnings.simplefilter('ignore')
    R._setup_worker()
    import learned_arm as LA
    LA.evae_prior_nets = HS2.evae_whole_s2
    t0 = time.time()
    rows, recs, gt, meta = R.job_section2(dict(spec, arms=[ARM]))
    name = R.job_name(dict(spec, area='section2'))
    pd.DataFrame(rows).to_csv(HERE / 'out_hyb2_s2' / f'{name}.csv', index=False, float_format='%.17g')
    np.savez_compressed(HERE / 'out_hyb2_s2' / f'{name}.npz', gt=gt, **recs)
    return name, time.time() - t0


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor, as_completed
    mode = sys.argv[1]
    if mode == 'tw':
        (HERE / 'out_hyb2').mkdir(exist_ok=True); cfg = sys.argv[2]; nw = int(sys.argv[3]) if len(sys.argv) > 3 else 22
        jobs = pd.read_csv(R.OUT / 'rows_RA01ABL_trainedwall.csv', usecols=['fold', 'box'])[['fold', 'box']].drop_duplicates()
        specs = [dict(fold=int(f), box=b, cfg=cfg) for f, b in jobs.itertuples(index=False)
                 if not (HERE / 'out_hyb2' / f'tw_{cfg}_f{int(f)}_{b}.csv').exists()]
        fn = one_tw
    else:
        (HERE / 'out_hyb2_s2').mkdir(exist_ok=True); nw = int(sys.argv[2]) if len(sys.argv) > 2 else 22
        old = pd.read_csv(R.OUT / 'rows_RA01ABL_section2.csv', usecols=['scheme', 'box', 'fold'])[['scheme', 'box', 'fold']].drop_duplicates()
        specs = [dict(tag=s.replace('EAST', ''), x0=float(b[1:]), fold=int(f)) for s, b, f in old.itertuples(index=False)]
        specs = [s for s in specs if not (HERE / 'out_hyb2_s2' / f"{R.job_name(dict(s, area='section2'))}.csv").exists()]
        fn = one_s2
    print(len(specs), 'jobs', flush=True); t0 = time.time()
    with ProcessPoolExecutor(nw) as ex:
        for i, f in enumerate(as_completed([ex.submit(fn, s) for s in specs])):
            n, s = f.result(); print(f'{i+1}/{len(specs)} {n} {s:.0f} s (elapsed {time.time()-t0:.0f} s)', flush=True)
