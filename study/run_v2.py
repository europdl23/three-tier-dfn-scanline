# -*- coding: utf-8 -*-
"""Full study, window scale. One process per held-out window. Fixed grid (PAPER2_TEST_PLAN_2026-10-06.md):
  main     (S1,S2,S3)  final method + fair conditioned baselines + no-evidence references + one-part-removed ablations
  spacing  (S2)        centre / 10 / 7 / 5 / 3 m, layout shifted at random per realisation
  survey   (S2)        horizontal lines, vertical + horizontal
  realism  (S2)        missed hits (with / without known-rate correction), direction noise, set mislabel, position
                       error, aperture width, irregular layout
  domain   (S2)        held-out window rotated 20 / 40 deg (simulated change of geology), 5 m and centre scanline
  params   (S2)        method settings over pre-set ranges
Writes results/<scheme>_v2_rows.csv and results/calib/<scheme>_<box>.npz (presence probabilities)."""
import sys, math, time, zlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
from run_extra import train_cross_rate

OUT = HERE / 'results'; (OUT / 'calib').mkdir(parents=True, exist_ok=True)
BASE = dict(kind='v', spacing=5.0, width=0.10, shift=False, nh=2, miss=0.0, noise=0.0, mislabel=0.0, pos=0.0, known=0.0,
            rot=0, a_prior=2.0, xfac=1.0, dth=6.0, dperp=0.35, dwin=15.0)
ABL = ['F_UNIF_nogate', 'F_UNIF_nobudget', 'F_UNIF_noxlim', 'F_UNIF_notier2', 'F_UNIF_len3d', 'F_UNIF_nobias']
MAIN_ARMS = ['F_UNIF', 'F_LOCAL', 'F_EVAE', 'F_KDE', 'F_ADFNE', 'EVID', 'STAT', 'RAW_EVAE', 'RAW_KDE', 'RAW_ADFNE'] + ABL


def configs(scheme):
    C = [('main', dict(BASE), MAIN_ARMS)]
    if scheme != 'S2':
        return C
    for s in (None, 10.0, 7.0, 5.0, 3.0):
        C.append((f'sp_{s or "centre"}', dict(BASE, spacing=s, shift=True), ['F_UNIF', 'F_LOCAL', 'EVID', 'STAT']))
    C += [('survey_h2', dict(BASE, kind='h', nh=2), ['F_UNIF', 'EVID']),
          ('survey_h4', dict(BASE, kind='h', nh=4), ['F_UNIF', 'EVID']),
          ('survey_v5h2', dict(BASE, kind='vh', nh=2), ['F_UNIF', 'EVID']),
          ('survey_centre_h2', dict(BASE, kind='vh', spacing=None, nh=2), ['F_UNIF', 'EVID'])]
    R = [('miss_0.1', dict(miss=0.1)), ('miss_0.2', dict(miss=0.2)), ('miss_0.4', dict(miss=0.4)),
         ('miss_0.2_known', dict(miss=0.2, known=0.2)), ('miss_0.4_known', dict(miss=0.4, known=0.4)),
         ('noise_5', dict(noise=5.0)), ('noise_10', dict(noise=10.0)), ('mislabel_0.1', dict(mislabel=0.1)),
         ('mislabel_0.2', dict(mislabel=0.2)), ('pos_0.2', dict(pos=0.2)), ('pos_0.5', dict(pos=0.5)),
         ('width_0.05', dict(width=0.05)), ('width_0.3', dict(width=0.3)), ('irregular', dict(shift=True))]
    C += [(f'real_{n}', dict(BASE, **d), ['F_UNIF', 'F_LOCAL']) for n, d in R]
    for rot in (20, 40):
        for s in (5.0, None):
            C.append((f'rot{rot}_{s or "centre"}', dict(BASE, rot=rot, spacing=s), ['F_UNIF', 'F_LOCAL', 'EVID', 'STAT']))
    Pm = [('a_prior', 0.0), ('a_prior', 5.0), ('xfac', 0.75), ('xfac', 1.5), ('dth', 4.0), ('dth', 10.0),
          ('dperp', 0.2), ('dperp', 0.6), ('dwin', 10.0), ('dwin', 25.0)]
    C += [(f'par_{k}_{v}', dict(BASE, **{k: v}), ['F_UNIF']) for k, v in Pm]
    return C


def sd(*a):
    return zlib.crc32('|'.join(map(str, a)).encode()) & 0x7fffffff


def run_window(job):
    scheme, fi, box = job
    boxes = P.load_library(); sp = P.load_split(scheme, fi)
    model = V.setbg.load(P.REL / 'data' / 'orientation_clusters' / f'{scheme}_f{fi}.json')
    tr_th, tr_len3d = P.train_traces(boxes, sp); prior = P.train_p21(boxes, sp)
    xrate, tr_len2d = train_cross_rate(boxes, sp)
    held = set().union(*[set(boxes[t]['ids']) for t in sp['test']]); th2 = []
    for k in sp['fitting'] + sp['validation']:
        b = boxes[k]; keep = np.array([i not in held for i in b['ids']], bool); th2 += list(P.segdir(P.clip(b['L'][keep], b['H'])))
    th2 = np.array(th2)
    b = boxes[box]; H = b['H']; gt0 = P.clip(b['L'], H)
    draws = [(s, d) for s in P.SEEDS for d in range(P.NDRAW)]
    stored = {m: [P.clip(P.stored_net(scheme, m, fi, box, s, d), H) for s, d in draws] for m in ('EVAE', 'KDE', 'ADFNE')}
    rows, calib = [], {}
    for cname, cf, arms in configs(scheme):
        gt = V.rotate(gt0, H, cf['rot']); g = P.metrics(gt, H)
        pres = {a: [] for a in arms}
        for r in range(len(draws)):
            orng = np.random.default_rng(sd(scheme, box, r, cname, 'obs'))
            lines = V.lines_layout(cf['kind'], cf['spacing'], cf['width'], H, rng=orng, shift=cf['shift'], nh=cf['nh'])
            obs = V.observe(gt, lines, H, orng, miss=cf['miss'], noise=cf['noise'], mislabel=cf['mislabel'], pos_err=cf['pos'],
                            tr_th=tr_th, model=model)
            groups = V.merge(obs, cf['dth'], cf['dperp'])
            bud, p21h = V.budget(obs, lines, H, tr_th, tr_len3d, prior, cf['a_prior'], miss_known=cf['known'])
            xmax0 = xrate * W * H * (p21h / prior) ** 2 * cf['xfac']
            gpre = None
            seen = sorted({o['tid'] for o in obs})
            for arm in arms:
                rng = np.random.default_rng(sd(scheme, box, r, cname, arm))
                tiers = []
                if arm.startswith('RAW_'):
                    rec = stored[arm[4:]][r]; tiers = [3] * len(rec)
                elif arm == 'STAT':          # no evidence: training statistics, training P21 budget, crossing limit
                    rec = np.zeros((0, 4))
                    for k in range(24):
                        if P.seglen(rec).sum() >= prior * W * H:
                            break
                        rec, tiers = V.fill(P.unif_proposals(60, tr_th, tr_len3d, H, rng), [], rec, tiers, prior * W * H, H, rng,
                                            xmax=xrate * W * H * cf['xfac'], gate=False)
                else:
                    parts = arm.split('_'); filler = parts[1] if arm.startswith('F_') else None; ab = parts[2] if len(parts) > 2 else ''
                    if ab == 'notier2':
                        rec = V.tier1(obs)
                    else:
                        pool_len = tr_len3d if ab == 'len3d' else tr_len2d
                        pool_th = tr_th if ab == 'len3d' else th2
                        rec = V.tier2(groups, pool_th, pool_len, H, rng, cf['dwin']) if ab != 'nobias' else \
                            _tier2_nobias(groups, pool_th, pool_len, H, rng, cf['dwin'])
                    tiers = [2] * len(rec)
                    if gpre is None and ab == '':
                        gpre = V.gt_pre(gt, H, lines, rec, model)
                    if filler:
                        B = prior * W * H if ab == 'nobudget' else bud
                        xm = None if ab == 'noxlim' else xmax0
                        for k in range(len(draws)):
                            if P.seglen(rec).sum() >= B:
                                break
                            if filler == 'UNIF':
                                pool = P.unif_proposals(60, tr_th, tr_len3d, H, rng)
                            elif filler == 'LOCAL':
                                pool = V.local_proposals(60, obs, th2, tr_len2d, H, rng, dwin=cf['dwin'])
                            else:
                                pool = stored[filler][(r + k) % len(draws)]
                            rec, tiers = V.fill(pool, lines, rec, tiers, B, H, rng, xmax=xm, gate=(ab != 'nogate'))
                if gpre is None:
                    gpre = V.gt_pre(gt, H, lines, np.zeros((0, 4)), model)
                m = P.metrics(rec, H); sc = P.score(m, g)
                ex, pr = V.extra_scores(rec, tiers, gt, H, lines, model, gpre)
                pres[arm].append(pr)
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cname, arm=arm, r=r, n=m['n'], P21=m['P21'], I100=m['I100'],
                                 gt_n=g['n'], gt_P21=g['P21'], gt_I100=g['I100'], n_hits=len(obs), n_seen=len(seen),
                                 frac_seen=len(seen) / max(len(gt), 1), **sc, **ex))
        if cname == 'main':
            calib = {a: np.mean(pres[a], 0) for a in arms}
            np.savez_compressed(OUT / 'calib' / f'{scheme}_{box}.npz', gt=V.presence(gt0, H), **calib)
    return rows


def _tier2_nobias(groups, tr_th, tr_len, H, rng, dwin):
    """tier 2 with uniform (not length-weighted) length choice"""
    out = []
    for gr in groups:
        th = math.atan2(sum(math.sin(2 * o['th']) for o in gr), sum(math.cos(2 * o['th']) for o in gr)) / 2 % math.pi
        ux, uy = math.cos(th), math.sin(th)
        s = np.array([o['x'] * ux + o['y'] * uy for o in gr]); span = s.max() - s.min()
        d = np.abs(((tr_th - th) + math.pi / 2) % math.pi - math.pi / 2)
        pool = tr_len[d <= math.radians(dwin)]
        pool = pool if len(pool) >= 5 else tr_len
        pool = pool[pool >= span] if (pool >= span).any() else np.array([span * 1.1 + 0.2])
        Lf = float(rng.choice(pool)); a = rng.uniform(0, Lf - span); o0 = gr[int(np.argmin(s))]
        p1 = (o0['x'] - a * ux, o0['y'] - a * uy); out.append([p1[0], p1[1], p1[0] + Lf * ux, p1[1] + Lf * uy])
    return P.clip(np.array(out, float).reshape(-1, 4), H)


W = P.M


def main(scheme):
    nfold = {'S1': 4, 'S2': 7, 'S3': 4}[scheme]
    jobs = [(scheme, fi, box) for fi in range(nfold) for box in P.load_split(scheme, fi)['test']]
    R = []; t0 = time.time()
    with Pool(16) as pool:
        for i, r in enumerate(pool.imap_unordered(run_window, jobs)):
            R += r; print(f'{scheme} v2 {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(R).to_csv(OUT / f'{scheme}_v2_rows.csv', index=False)


if __name__ == '__main__':
    if sys.argv[1:2] == ['smoke']:
        t = time.time(); r = run_window(('S2', 0, P.load_split('S2', 0)['test'][2])); df = pd.DataFrame(r)
        print(f'{time.time() - t:.0f}s rows {len(df)}')
        print(df[df.cfg == 'main'].groupby('arm')[['n', 'P21_err_pct', 'dir_W1', 'I100_ratio', 'set_share_L1', 'perc_mismatch',
                                                    'zone_meas_rmse', 'zone_ind_rmse', 'zone_inf_rmse']].median().round(3).to_string())
        print(df.groupby('cfg')[['P21_err_pct', 'dir_W1', 'frac_seen']].median().round(3).to_string())
    else:
        for s in sys.argv[1:]:
            main(s)
