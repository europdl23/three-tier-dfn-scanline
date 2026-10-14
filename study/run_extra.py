# -*- coding: utf-8 -*-
"""Extra arms aimed at the over-connectivity of the three-tier reconstruction (5 m spacing, same evidence):
  T3_<P>_xgate : tier-3 proposals also rejected when they would push the number of trace crossings above the
                 crossing count expected from the training windows (scaled by the estimated P21 squared);
  T3_EVAE_whole: keep the EVAE's spatial organisation: choose, among the 24 stored EVAE realisations, the one whose
                 own scanline hits best match the observed hits (count and trace directions), drop only its traces
                 that cross an aperture, add tier 1+2, then fill to the budget (other draws, same rules);
  T3_EVAE_whole_xgate: both.
  T3_<P>_len2d : tier-2 lengths drawn from the 2D in-window training pieces instead of full_length_m."""
import sys, math, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import p2lib as P
from run_tests import seed_of, BASE

OUT = Path(__file__).parent / 'results'
ARMS = ['T3_EVAE', 'T3_UNIF', 'T3_EVAE_xgate', 'T3_UNIF_xgate', 'T3_KDE_xgate', 'T3_EVAE_whole', 'T3_EVAE_whole_xgate',
        'T3_EVAE_len2d', 'T3_UNIF_len2d', 'T3_UNIF_len2d_xgate', 'T3_EVAE_whole_len2d_xgate', 'RAW_EVAE']


def train_cross_rate(boxes, sp):
    held = set().union(*[set(boxes[t]['ids']) for t in sp['test']])
    X, A, L2 = 0, 0.0, []
    for k in sp['fitting'] + sp['validation']:
        b = boxes[k]; keep = np.array([i not in held for i in b['ids']], bool); L = P.clip(b['L'][keep], b['H'])
        X += len(P.crossing_pairs(L)); A += P.M * b['H']; L2 += list(P.seglen(L))
    return X / A, np.array(L2)


def n_cross_new(t, rec):
    return int(P.common.seg_cross_one(t, rec).sum()) if len(rec) else 0


def fill(pool, ap, rec, budget, H, rng, xmax=None, gate=True):
    have = P.seglen(rec).sum(); X = len(P.crossing_pairs(rec)) if len(rec) > 1 else 0
    Pp = P.clip(pool, H)
    for i in rng.permutation(len(Pp)):
        if have >= budget:
            break
        t = Pp[i]
        if gate and any(P.crosses_band(t[None], lo, hi)[0] for lo, hi in ap):
            continue
        if xmax is not None:
            nx = n_cross_new(t, rec)
            if nx > 0 and X + nx > xmax:
                continue
            X += nx
        rec = np.vstack([rec, t[None]]) if len(rec) else t[None]; have += P.seglen(t[None])[0]
    return rec


def run_window(job):
    scheme, fi, box = job
    boxes = P.load_library(); sp = P.load_split(scheme, fi)
    tr_th, tr_len = P.train_traces(boxes, sp); prior = P.train_p21(boxes, sp)
    xrate, tr_len2d = train_cross_rate(boxes, sp)
    tr_th2d = []
    held = set().union(*[set(boxes[t]['ids']) for t in sp['test']])
    for k in sp['fitting'] + sp['validation']:
        b = boxes[k]; keep = np.array([i not in held for i in b['ids']], bool); tr_th2d += list(P.segdir(P.clip(b['L'][keep], b['H'])))
    tr_th2d = np.array(tr_th2d)
    b = boxes[box]; H = b['H']; gt = P.clip(b['L'], H); g = P.metrics(gt, H)
    draws = [(s, d) for s in P.SEEDS for d in range(P.NDRAW)]
    stored = {m: [P.clip(P.stored_net(scheme, m, fi, box, s, d), H) for s, d in draws] for m in ('EVAE', 'KDE')}
    cf = BASE; rows = []
    for r in range(len(draws)):
        orng = np.random.default_rng(seed_of(scheme, box, r, cf['spacing'], cf['width'], cf['miss'], cf['noise'], cf['mode']))
        ap = P.apertures(cf['spacing'], cf['width'], mode=cf['mode'], rng=orng, jitter=1.5)
        obs = P.observe(gt, ap, H, orng); groups = P.merge_obs(obs)
        obs_th = np.array([o['th'] for o in obs])
        for arm in ARMS:
            rrng = np.random.default_rng(seed_of(scheme, box, r, 'extra', arm))
            if arm == 'RAW_EVAE':
                rec = stored['EVAE'][r]
            else:
                prop = arm.split('_')[1]; flags = set(arm.split('_')[2:])
                if 'len2d' in flags:
                    t2 = P.tier2_extend(groups, tr_th2d, tr_len2d, H, rrng)
                else:
                    t2 = P.tier2_extend(groups, tr_th, tr_len, H, rrng)
                budget, p21h = P.p21_budget(obs, ap, H, tr_th, tr_len, prior)
                xmax = xrate * P.M * H * (p21h / prior) ** 2 if 'xgate' in flags else None
                rec = t2
                order = [(r + k) % len(draws) for k in range(len(draws))]
                if 'whole' in flags:
                    cost = []
                    for k in range(len(draws)):
                        R_ = stored['EVAE'][k]
                        hits = [i for i in range(len(R_)) if any(P.crosses_band(R_[i:i + 1], lo, hi)[0] for lo, hi in ap)]
                        nh = sum(sum(P.crosses_band(R_[i:i + 1], lo, hi)[0] for lo, hi in ap) for i in hits)
                        dW = P.common.w1_circular_doubled(P.segdir(R_[hits]), obs_th) if len(hits) > 1 and len(obs_th) > 1 else 90.0
                        cost.append(abs(nh - len(obs)) / max(len(obs), 1) + dW / 30.0)
                    top = list(np.argsort(cost)[:3]); k0 = int(top[r % 3])
                    order = [k0] + [k for k in order if k != k0]
                for k in order:
                    if P.seglen(rec).sum() >= budget:
                        break
                    pool = P.unif_proposals(60, tr_th, tr_len, H, rrng) if prop == 'UNIF' else stored[prop][k]
                    rec = fill(pool, ap, rec, budget, H, rrng, xmax=xmax)
            m = P.metrics(rec, H); sc = P.score(m, g)
            rows.append(dict(scheme=scheme, fold=fi, box=box, cfg='extra', arm=arm, r=r, n=m['n'], P21=m['P21'], I100=m['I100'],
                             gt_n=g['n'], gt_P21=g['P21'], gt_I100=g['I100'], train_I100=100 * xrate, **sc))
    return rows


def main(scheme):
    nfold = {'S1': 4, 'S2': 7, 'S3': 4}[scheme]
    jobs = [(scheme, fi, box) for fi in range(nfold) for box in P.load_split(scheme, fi)['test']]
    R = []; t0 = time.time()
    with Pool(16) as pool:
        for i, r in enumerate(pool.imap_unordered(run_window, jobs)):
            R += r; print(f'{scheme} extra {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(R).to_csv(OUT / f'{scheme}_extra_rows.csv', index=False)


if __name__ == '__main__':
    for s in sys.argv[1:] or ['S1']:
        main(s)
