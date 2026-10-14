# -*- coding: utf-8 -*-
"""Current-method evidence test (15 Oct 2026): fitted statistical sampling (STAT_FIT pool, as run_main_v3 job_stat) on the
same 50 test windows WITHOUT scanline evidence, to replace the earlier pre-specified test H1 with the current pipeline.
  STAT_FIT_NOSCAN  no scanlines (no measured or indicated traces); all placement rules, with the intensity budget falling
                   back to the training intensity (Equation 2 with no hits and no scanlines gives P21tr)
  STAT_FIT_FREE    no scanlines and no placement rules: traces drawn from the same pool until the total length reaches
                   the training intensity times the window area
Compared in analysis with STAT_FIT at 5 m and with the centre line (main v3 runs). Output: v3/out/noscan/.
Usage: python -B run_noscan_v3.py [workers]"""
import os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import run_main_v3 as M


def job_noscan(fi, box):
    P, V, FE, G, ST = M._common(fi, box)
    from run_v2 import sd
    CF = FE.TC.fold_ctx('S2', fi)
    model = CF['model']; H = CF['boxes'][box]['H']; gt = P.clip(CF['boxes'][box]['L'], H); ym = H / 20
    fams = M.SF.fit(CF)
    dmc = {}; rows, recs = [], {}
    for r in range(24):
        for arm in ('STAT_FIT_NOSCAN', 'STAT_FIT_FREE'):
            rng = np.random.default_rng(sd('S2', box, r, 'noscan', 'final', arm))

            def pool(k, rng=rng):
                c = np.column_stack([rng.uniform(0, P.M, 60), rng.uniform(0, H, 60)]); th, L = M.SF.sample(fams, 60, rng)
                return M.SF.segs(c, th, L)
            if arm == 'STAT_FIT_NOSCAN':
                rec = FE.frozen(CF, [], [], H, rng, pool, dict(FE.BASE), dmc, True)
            else:
                bud = CF['prior'] * P.M * H; rec = np.zeros((0, 4)); have = 0.0
                for k in range(24):
                    if have >= bud: break
                    cand = P.clip(pool(k), H)
                    for i in rng.permutation(len(cand)):
                        if have >= bud: break
                        rec = np.vstack([rec, cand[i][None]]); have += P.seglen(cand[i][None])[0]
            rec = np.asarray(rec, float).reshape(-1, 4)
            rows.append(dict(scheme='S2', fold=fi, box=box, cfg='none', arm=arm, r=r, n=len(rec), gt_n=len(gt), n_hits=0,
                             **M._score(ST, FE, rec, gt, H, ym, model)))
            recs[f'{arm}__none__r{r:02d}'] = rec
    return rows, recs, gt


M.JOBS['noscan'] = job_noscan

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor, as_completed
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    specs = [('noscan', f, b) for f, b in M.windows() if not (M.OUT / 'noscan' / f'noscan_f{f}_{b}.csv').exists()]
    print(len(specs), 'jobs', flush=True); t0 = time.time()
    with ProcessPoolExecutor(nw) as ex:
        for i, f in enumerate(as_completed([ex.submit(M.one, s) for s in specs])):
            st, name, sec, err = f.result()
            print(f'{i + 1}/{len(specs)} {st} {name} {sec:.0f} s (elapsed {time.time() - t0:.0f} s)', flush=True)
            if st != 'ok': print(err, flush=True)
