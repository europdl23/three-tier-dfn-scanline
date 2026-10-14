# -*- coding: utf-8 -*-
"""A vs B: final method (three tiers + termination + crossing limit) with statistical filler (A) or Paper 1 EVAE filler (B).
Based on round 3: Real traces often end on another trace (Y node, 45% of encounters in the
training windows). Tier 2 extensions and tier 3 traces are cut at a crossing with probability 1 - p_pass (p_pass
from the fold's training windows); the observed span of an indicated trace is never cut.
Arms: F_UNIF (round-2 final), F_TERM (termination + crossing limit), F_TERMNX (termination instead of the crossing
limit), EVID, EVID_TERM. Configs: main 5 m (S1,S2,S3); spacing sweep with shifted layouts (S2)."""
import sys, math, time, zlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
from run_extra import train_cross_rate
from run_v2 import BASE, sd

OUT = HERE / 'results'
ARMS = ['A_STAT', 'B_EVAE']


def cfgs(scheme):
    C = [('main', dict(BASE))]
    if scheme == 'S2':
        C += [(f'sp_{s or "centre"}', dict(BASE, spacing=s, shift=True)) for s in (None, 10.0, 7.0, 5.0, 3.0)]
    C += [('sp_centre_fixed', dict(BASE, spacing=None))]
    return C


def protect_spans(t2, obs):
    out = []
    for x1, y1, x2, y2 in t2:
        L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / max(L, 1e-9), (y2 - y1) / max(L, 1e-9)
        a = [((o['x'] - x1) * ux + (o['y'] - y1) * uy) for o in obs
             if abs(-uy * (o['x'] - x1) + ux * (o['y'] - y1)) < 0.15 and -0.1 <= (o['x'] - x1) * ux + (o['y'] - y1) * uy <= L + 0.1]
        out.append((min(a) - 0.06, max(a) + 0.06) if a else None)
    return out


def run_window(job):
    scheme, fi, box = job
    boxes = P.load_library(); sp = P.load_split(scheme, fi)
    model = V.setbg.load(P.REL / 'data' / 'orientation_clusters' / f'{scheme}_f{fi}.json')
    tr_th, tr_len3d = P.train_traces(boxes, sp); prior = P.train_p21(boxes, sp)
    xrate, tr_len2d = train_cross_rate(boxes, sp)
    held = set().union(*[set(boxes[t]['ids']) for t in sp['test']]); th2, TW, TH = [], [], []
    for k in sp['fitting'] + sp['validation']:
        b = boxes[k]; keep = np.array([i not in held for i in b['ids']], bool); Lk = P.clip(b['L'][keep], b['H'])
        th2 += list(P.segdir(Lk)); TW.append(Lk); TH.append(b['H'])
    th2 = np.array(th2); p_pass = V.pass_prob(TW, TH)
    b = boxes[box]; H = b['H']; gt = P.clip(b['L'], H); g = P.metrics(gt, H)
    draws = [(s_, d_) for s_ in P.SEEDS for d_ in range(P.NDRAW)]
    evae = [P.clip(P.stored_net(scheme, 'EVAE', fi, box, s_, d_), H) for s_, d_ in draws]
    rows = []
    for cname, cf in cfgs(scheme):
        for r in range(24):
            orng = np.random.default_rng(sd(scheme, box, r, cname, 'obs'))
            lines = V.lines_layout(cf['kind'], cf['spacing'], cf['width'], H, rng=orng, shift=cf['shift'], nh=cf['nh'])
            obs = V.observe(gt, lines, H, orng); groups = V.merge(obs)
            bud, p21h = V.budget(obs, lines, H, tr_th, tr_len3d, prior)
            xmax = xrate * P.M * H * (p21h / prior) ** 2
            gpre = None
            for arm in ARMS:
                rng = np.random.default_rng(sd(scheme, box, r, cname, arm))
                t2 = V.tier2(groups, th2, tr_len2d, H, rng)
                term = True
                if term and len(t2) > 1:
                    t2 = V.terminate(t2, 0, rng, p_pass, protect=protect_spans(t2, obs))
                rec, tiers = t2, [2] * len(t2)
                if gpre is None:
                    gpre = V.gt_pre(gt, H, lines, t2, model)
                if True:
                    xm = xmax
                    for rnd in range(3 if term else 1):
                        for k in range(24):
                            if P.seglen(rec).sum() >= bud:
                                break
                            pool = P.unif_proposals(60, tr_th, tr_len3d, H, rng) if arm == 'A_STAT' else evae[(r + k + 7 * rnd) % 24]
                            rec, tiers = V.fill(pool, lines, rec, tiers, bud, H, rng, xmax=xm)
                        if term:
                            n2 = len(t2)
                            rec = V.terminate(rec, n2, rng, p_pass)
                            rec = rec[P.seglen(rec) > 0.3]; tiers = tiers[:len(rec)]
                m = P.metrics(rec, H); sc = P.score(m, g)
                ex, _ = V.extra_scores(rec, tiers, gt, H, lines, model, gpre)
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cname, arm=arm, r=r, n=m['n'], P21=m['P21'], I100=m['I100'],
                                 gt_n=g['n'], gt_P21=g['P21'], gt_I100=g['I100'], p_pass=p_pass, **sc, **ex))
    return rows


def main(scheme):
    nfold = {'S1': 4, 'S2': 7, 'S3': 4}[scheme]
    jobs = [(scheme, fi, box) for fi in range(nfold) for box in P.load_split(scheme, fi)['test']]
    R = []; t0 = time.time()
    with Pool(12) as pool:
        for i, r in enumerate(pool.imap_unordered(run_window, jobs)):
            R += r; print(f'{scheme} ab {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(R).to_csv(OUT / f'{scheme}_ab_rows.csv', index=False)


if __name__ == '__main__':
    if sys.argv[1:2] == ['smoke']:
        df = pd.DataFrame(run_window(('S1', 0, P.load_split('S1', 0)['test'][0])))
        print(df.groupby(['cfg', 'arm'])[['n', 'gt_n', 'P21_err_pct', 'dir_W1', 'len_W1', 'I100_ratio', 'CL_err', 'map_rmse']].median().round(3).to_string())
    else:
        for s in sys.argv[1:]:
            main(s)
