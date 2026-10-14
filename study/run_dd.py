# -*- coding: utf-8 -*-
"""Round 4: duplicate/overlap filter and full filler comparison.
Final method for every filler: tier 1 + tier 2 (2D lengths) + geological termination + tier 3 with no trace through a
scanline, scanline P21 budget, crossing limit. NEW: duplicate gate - a filled trace is rejected if it is nearly parallel
(< 5 deg), overlapping (> 30% of the shorter) and closer than d_min to an existing trace; d_min = the 5th percentile
of the gaps between overlapping near-parallel pairs in the fold's training windows (about 1.5 m). The filler's own
realisation is also cleaned of its duplicates first.
Arms: A_STAT, A_STAT_DD, B_EVAE, B_EVAE_DD, K_KDE_DD, D_ADFNE_DD; no scanlines: RAW_EVAE, RAW_KDE, RAW_ADFNE, RAW_EVAE_DD;
REFERENCE (8 nearest training windows, S1 only).
Configs: main 5 m and fixed centre line (S1,S2,S3); 10 / 7 / 3 m shifted (S2).
Writes results/<scheme>_dd_rows.csv and results/dist/<scheme>_<box>_<cfg>.npz (pooled lengths, directions, set counts)."""
import sys, math, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
from run_extra import train_cross_rate
from run_v2 import BASE, sd
from run_v3_term import protect_spans

OUT = HERE / 'results'; (OUT / 'dist').mkdir(parents=True, exist_ok=True)
EV = ['A_STAT', 'A_STAT_DD', 'B_EVAE', 'B_EVAE_DD', 'K_KDE_DD', 'D_ADFNE_DD']
NOEV = ['RAW_EVAE', 'RAW_KDE', 'RAW_ADFNE', 'RAW_EVAE_DD']
DTH = math.radians(5.0)


def cfgs(scheme):
    C = [('main', dict(BASE)), ('centre', dict(BASE, spacing=None))]
    if scheme == 'S2':
        C += [(f'sp_{s}', dict(BASE, spacing=s, shift=True)) for s in (10.0, 7.0, 3.0)]
    return C


def fill_dd(pool, lines, rec, bud, H, rng, xmax, dmin, dd):
    have = P.seglen(rec).sum(); X = len(P.crossing_pairs(rec)) if len(rec) > 1 else 0
    Pp = P.clip(pool, H)
    if dd:
        Pp = V.dedup(Pp, DTH, dmin)
    for i in rng.permutation(len(Pp)):
        if have >= bud:
            break
        t = Pp[i]
        if any(V.crosses(t[None], ln)[0] for ln in lines):
            continue
        if dd and len(rec) and V.is_dup(t, rec, DTH, dmin):
            continue
        nx = int(P.common.seg_cross_one(t, rec).sum()) if len(rec) else 0
        if nx > 0 and X + nx > xmax:
            continue
        X += nx; rec = np.vstack([rec, t[None]]) if len(rec) else t[None]; have += P.seglen(t[None])[0]
    return rec


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
    dmin = float(np.percentile(V.near_parallel_gaps(TW), 5))
    b = boxes[box]; H = b['H']; gt = P.clip(b['L'], H); g = P.metrics(gt, H)
    draws = [(s_, d_) for s_ in P.SEEDS for d_ in range(P.NDRAW)]
    stored = {m: [P.clip(P.stored_net(scheme, m, fi, box, s_, d_), H) for s_, d_ in draws] for m in ('EVAE', 'KDE', 'ADFNE')}
    rows = []
    for cname, cf in cfgs(scheme):
        dist = {a: dict(len=[], th=[], cls=np.zeros(model['K'] + 1)) for a in EV + NOEV}
        arms = EV + (NOEV if cname == 'main' else [])
        for r in range(24):
            orng = np.random.default_rng(sd(scheme, box, r, cname, 'obs'))
            lines = V.lines_layout(cf['kind'], cf['spacing'], cf['width'], H, rng=orng, shift=cf['shift'], nh=cf['nh'])
            obs = V.observe(gt, lines, H, orng); groups = V.merge(obs)
            bud, p21h = V.budget(obs, lines, H, tr_th, tr_len3d, prior); xmax = xrate * P.M * H * (p21h / prior) ** 2
            gpre = None
            for arm in arms:
                rng = np.random.default_rng(sd(scheme, box, r, cname, arm))
                dd = arm.endswith('_DD')
                if arm.startswith('RAW_'):
                    rec = stored[arm.split('_')[1]][r]
                    if dd:
                        rec = V.dedup(rec, DTH, dmin)
                else:
                    t2 = V.tier2(groups, th2, tr_len2d, H, rng)
                    if len(t2) > 1:
                        t2 = V.terminate(t2, 0, rng, p_pass, protect=protect_spans(t2, obs))
                    if gpre is None:
                        gpre = V.gt_pre(gt, H, lines, t2, model)
                    rec = t2; src = arm.split('_')[1]
                    for rnd in range(3):
                        for k in range(24):
                            if P.seglen(rec).sum() >= bud:
                                break
                            pool = P.unif_proposals(60, tr_th, tr_len3d, H, rng) if src == 'STAT' else stored[src][(r + k + 7 * rnd) % 24]
                            rec = fill_dd(pool, lines, rec, bud, H, rng, xmax, dmin, dd)
                        rec = V.terminate(rec, len(t2), rng, p_pass); rec = rec[P.seglen(rec) > 0.3]
                if gpre is None:
                    gpre = V.gt_pre(gt, H, lines, np.zeros((0, 4)), model)
                m = P.metrics(rec, H); sc = P.score(m, g)
                ex, _ = V.extra_scores(rec, [], gt, H, lines, model, gpre)
                gaps = V.near_parallel_gaps([rec])
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cname, arm=arm, r=r, n=m['n'], P21=m['P21'], I100=m['I100'],
                                 gt_n=g['n'], gt_P21=g['P21'], gt_I100=g['I100'], dmin=dmin, close_pairs=int((gaps < dmin).sum()), **sc, **ex))
                dist[arm]['len'] += list(P.seglen(rec)); dist[arm]['th'] += list(P.segdir(rec))
                if len(rec):
                    dist[arm]['cls'] += np.bincount(V.setbg.classify(P.segdir(rec), model), minlength=model['K'] + 1)
        if cname == 'main' and scheme == 'S1':     # natural variability reference
            cand = [k for k in boxes if k not in sp['test']]
            near = sorted(cand, key=lambda k: (math.hypot(boxes[k]['cu'] - b['cu'], boxes[k]['cs'] - b['cs']), k))[:8]
            dist['REFERENCE'] = dict(len=[], th=[], cls=np.zeros(model['K'] + 1))
            for j, k in enumerate(near):
                nb = boxes[k]; keep = np.array([i not in held for i in nb['ids']], bool); Ln = P.clip(nb['L'][keep], nb['H'])
                m = P.metrics(Ln, nb['H']); sc = P.score(m, g)
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg='main', arm='REFERENCE', r=j, n=m['n'], P21=m['P21'], I100=m['I100'],
                                 gt_n=g['n'], gt_P21=g['P21'], gt_I100=g['I100'], dmin=dmin, close_pairs=int((V.near_parallel_gaps([Ln]) < dmin).sum()), **sc))
                dist['REFERENCE']['len'] += list(P.seglen(Ln)); dist['REFERENCE']['th'] += list(P.segdir(Ln))
                dist['REFERENCE']['cls'] += np.bincount(V.setbg.classify(P.segdir(Ln), model), minlength=model['K'] + 1)
        sv = {'gt_len': P.seglen(gt), 'gt_th': P.segdir(gt), 'gt_cls': np.bincount(V.setbg.classify(P.segdir(gt), model), minlength=model['K'] + 1),
              'centres_deg': np.array(model['centres_deg'])}
        for a, v in dist.items():
            sv[f'{a}__len'] = np.array(v['len']); sv[f'{a}__th'] = np.array(v['th']); sv[f'{a}__cls'] = v['cls']
        np.savez_compressed(OUT / 'dist' / f'{scheme}_{box}_{cname}.npz', **sv)
    return rows


def main(scheme):
    nfold = {'S1': 4, 'S2': 7, 'S3': 4}[scheme]
    jobs = [(scheme, fi, box) for fi in range(nfold) for box in P.load_split(scheme, fi)['test']]
    R = []; t0 = time.time()
    with Pool(16) as pool:
        for i, r in enumerate(pool.imap_unordered(run_window, jobs)):
            R += r; print(f'{scheme} dd {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(R).to_csv(OUT / f'{scheme}_dd_rows.csv', index=False)


if __name__ == '__main__':
    if sys.argv[1:2] == ['smoke']:
        t = time.time(); df = pd.DataFrame(run_window(('S1', 0, P.load_split('S1', 0)['test'][1]))); print(f'{time.time()-t:.0f}s')
        print(df.groupby(['cfg', 'arm'])[['n', 'gt_n', 'close_pairs', 'P21_err_pct', 'dir_W1', 'len_W1', 'count_err_pct', 'med_len_err', 'map_rmse']].median().round(2).to_string())
    else:
        for s in sys.argv[1:]:
            main(s)
