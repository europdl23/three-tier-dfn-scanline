# -*- coding: utf-8 -*-
"""PLAN step 1a: HYB_FIX on the 13 Section 2 windows (RA01 Section 2 conventions, ra01abl_run.job_section2).
The fixed pool and whole-trace EVAE sampler are imported unchanged from the frozen hybfix_run.py; the only addition is
the Section 2 EVAE prior (learned_arm.evae_prior_nets) with the same seeds, not clipped."""
import os, sys, time
from pathlib import Path
FROZEN = Path(__file__).resolve().parent
sys.path.insert(0, str(FROZEN))
import numpy as np
import pandas as pd
import hybfix_run as HF                         # patches ra01abl_run: make_pool, ARMS['HYB_FIX'], evae_prior_trained
R = HF.R
OUT = Path(__file__).resolve().parents[1] / 'work' / 'hybfix_s2'


def evae_whole_s2(fi, H, x0, piece=''):
    """learned_arm.evae_prior_nets with identical seeds, without the final clip"""
    import p2lib as P
    import study as S
    out = []
    for s in P.SEEDS:
        ev = HF.evae_whole_sampler(fi, s)
        for d in range(P.NDRAW):
            out.append(ev(H, S.gen_seed('EVAE', 'east' + piece, fi, int(x0), s, d)))
    return out


def one(spec):
    import warnings
    warnings.simplefilter('ignore')
    R._setup_worker()
    import learned_arm as LA
    LA.evae_prior_nets = evae_whole_s2
    t0 = time.time()
    rows, recs, gt, meta = R.job_section2(dict(spec, arms=[HF.ARM]))
    name = R.job_name(dict(spec, area='section2'))
    pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False, float_format='%.17g')
    np.savez_compressed(OUT / f'{name}.npz', gt=gt, **recs)
    return name, time.time() - t0


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    old = pd.read_csv(R.OUT / 'rows_RA01ABL_section2.csv', usecols=['scheme', 'box', 'fold'])
    specs = []
    for sch, box, fi in old[['scheme', 'box', 'fold']].drop_duplicates().itertuples(index=False):
        sp = dict(tag=sch.replace('EAST', ''), x0=float(box[1:]), fold=int(fi))
        if not (OUT / f"{R.job_name(dict(sp, area='section2'))}.csv").exists():
            specs.append(sp)
    print(len(specs), 'jobs', flush=True); t0 = time.time()
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 22) as ex:
        fs = [ex.submit(one, s) for s in specs]
        for i, f in enumerate(as_completed(fs)):
            n, s = f.result(); print(f'{i+1}/{len(specs)} {n} {s:.0f} s (elapsed {time.time()-t0:.0f} s)', flush=True)
