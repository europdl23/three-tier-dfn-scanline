# -*- coding: utf-8 -*-
"""PLAN step 5j: statistical reference with standard fitted distributions (no resampling of training traces, no learning).
Per fold, fitted on that fold's training traces (C['tr_th'] radians, C['tr_len3d'] whole-trace lengths, metres), families from
the fold's cluster model: family shares; per family von Mises on 2*theta (axial orientation; kappa by the Best-Fisher
approximation) and lognormal length (mean and sd of log length); background (no family): uniform orientation, lognormal length.
Arms (study wall, 5 m and centre line, 24 realisations, same observations as every other arm):
  STAT_FIT  final_eval.frozen(..., clean=True) with uniform centres + fitted traces (as STAT_F, pool of 60)
  TR_FIT    ra01abl_run composition (rule tier 2, no termination) with sequential-transformer centres + fitted traces (as ABL_Pc)"""
import sys, time, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
FROZEN = Path(__file__).resolve().parent
sys.path.insert(0, str(FROZEN)); sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import hybfix_run as HF
R = HF.R
OUT = HERE.parent / 'work' / 'statfit'


def a1inv(Rb):
    if Rb < 0.53: return 2 * Rb + Rb ** 3 + 5 * Rb ** 5 / 6
    if Rb < 0.85: return -0.4 + 1.39 * Rb + 0.43 / (1 - Rb)
    return 1 / (Rb ** 3 - 4 * Rb ** 2 + 3 * Rb)


def fit(C):
    import p2v2 as V
    th, L = np.asarray(C['tr_th'], float), np.asarray(C['tr_len3d'], float)
    cl = np.asarray(V.setbg.classify(th, C['model']))
    fams = []
    for k in np.unique(cl):
        m = cl == k; z = np.exp(2j * th[m]); Rb = float(np.abs(z.mean()))
        fams.append(dict(k=int(k), share=float(m.mean()), bg=bool(k >= 4), mu2=float(np.angle(z.mean())), kappa=float(a1inv(min(Rb, 0.999))),
                         lmu=float(np.log(L[m]).mean()), lsd=float(np.log(L[m]).std(ddof=1)), n=int(m.sum())))
    return fams


def sample(fams, n, rng):
    p = np.array([f['share'] for f in fams]); idx = rng.choice(len(fams), n, p=p / p.sum())
    th = np.empty(n); L = np.empty(n)
    for i, j in enumerate(idx):
        f = fams[j]
        th[i] = rng.uniform(0, math.pi) if f['bg'] else (rng.vonmises(f['mu2'], f['kappa']) / 2) % math.pi
        L[i] = math.exp(rng.normal(f['lmu'], f['lsd']))
    return th, L


def segs(c, th, L):
    dx, dy = np.cos(th) * L / 2, np.sin(th) * L / 2
    return np.column_stack([c[:, 0] - dx, c[:, 1] - dy, c[:, 0] + dx, c[:, 1] + dy])


_prev = R.make_pool


def make_pool(kind, C, t1, lines, H, rng, models, evae_nets, net_offset):
    if kind != 'tp_fit':
        return _prev(kind, C, t1, lines, H, rng, models, evae_nets, net_offset)
    import p2lib as P
    import gast as G
    gnet = models['gnet']; fams = fit(C)

    def pf(k, rec):                                     # as tp_stat (ra01abl_run.make_pool), traces from the fitted distributions
        g = G.generate(gnet, C, t1, rec, lines, H, rng)
        if not len(g): return np.zeros((0, 4))
        c = (g[:, :2] + g[:, 2:]) / 2; th, L = sample(fams, len(g), rng)
        return P.clip(segs(c, th, L), H)
    return pf


R.make_pool = make_pool
R.ARMS['TR_FIT'] = ('rule', 'tp_fit', None)


def one(spec):
    import warnings
    warnings.simplefilter('ignore')
    LA = R._setup_worker()
    import p2lib as P, p2v2 as V, final_eval as FE, gast as G, statistical_tests as ST
    from run_v2 import sd
    t0 = time.time(); fi, box = int(spec['fold']), spec['box']
    CF = FE.TC.fold_ctx('S2', fi); CG = G.prep_ctx('S2', fi)
    model = CF['model']; H = CF['boxes'][box]['H']; gt = P.clip(CF['boxes'][box]['L'], H); ym = H / 20
    fams = fit(CF)
    models = {s: LA.load_models(fi, s, 'rebuilt') for s in P.SEEDS}
    dmc_f, dmc_h = {}, {}; rows, recs = [], {}
    for cname, sp in (('5m', 5.0), ('centre', None)):
        lines = V.lines_layout('v', sp, FE.BASE['APERTURE'], H)
        for r in range(24):
            obs = V.observe(gt, lines, H, np.random.default_rng(sd('S2', box, r, cname, 'final_obs')))
            for arm in ('STAT_FIT', 'TR_FIT'):
                rng = np.random.default_rng(sd('S2', box, r, cname, 'final', arm))
                if arm == 'STAT_FIT':
                    def pool(k, rng=rng):
                        c = np.column_stack([rng.uniform(0, P.M, 60), rng.uniform(0, H, 60)]); th, L = sample(fams, 60, rng)
                        return segs(c, th, L)
                    rec = FE.frozen(CF, obs, lines, H, rng, pool, dict(FE.BASE), dmc_f, True)
                else:
                    rec, _ = R.compose(CG, obs, lines, H, rng, arm, models[P.SEEDS[r // 8]], [np.zeros((0, 4))] * 24, r, dmc_h)
                rec = np.asarray(rec, float).reshape(-1, 4)
                s = ST.score(rec / 20, gt / 20, ym, model); s = {k: (np.nan if (v is not None and np.isinf(v)) else v) for k, v in s.items()}
                rows.append(dict(scheme='S2', fold=fi, box=box, cfg=cname, arm=arm, r=r, n=len(rec), gt_n=len(gt),
                                 map_rmse=FE.p21map_err(rec, gt, H), n_hits=len(obs), **s))
                recs[f'{arm}__{cname}__r{r:02d}'] = rec
    name = f'tw_statfit_f{fi}_{box}'
    pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False, float_format='%.17g')
    np.savez_compressed(OUT / f'{name}.npz', gt=gt, **recs)
    return name, time.time() - t0


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    jobs = pd.read_csv(R.OUT / 'rows_RA01ABL_trainedwall.csv', usecols=['fold', 'box'])[['fold', 'box']].drop_duplicates()
    if sys.argv[1] == 'fits':
        R._setup_worker()
        import final_eval as FE
        allf = {int(f): fit(FE.TC.fold_ctx('S2', int(f))) for f in sorted(jobs.fold.unique())}
        json.dump(allf, open(OUT / 'fitted_parameters_per_fold.json', 'w'), indent=1)
        for f, fams in allf.items():
            print('fold', f, ' | '.join(f"k{x['k']} share {x['share']:.2f} mean {math.degrees(x['mu2'] / 2) % 180:.0f}deg kappa {x['kappa']:.1f} "
                                        f"len median {math.exp(x['lmu']):.1f} m sd(log) {x['lsd']:.2f}" for x in fams))
        sys.exit()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    specs = [dict(fold=int(f), box=b) for f, b in jobs.itertuples(index=False) if not (OUT / f'tw_statfit_f{int(f)}_{b}.csv').exists()]
    if sys.argv[1] == 'test': specs = specs[:1]
    print(len(specs), 'jobs', flush=True); t0 = time.time()
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 22) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, s) for s in specs])):
            n, s = f.result(); print(f'{i+1}/{len(specs)} {n} {s:.0f} s (elapsed {time.time()-t0:.0f} s)', flush=True)
