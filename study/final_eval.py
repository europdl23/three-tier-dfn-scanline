# -*- coding: utf-8 -*-
"""FINAL evaluation of the frozen method (Paper 2 revision), scored with Paper 1's 38 error scores (scripts/
statistical_tests.score, unchanged) plus the P21 map error.

Frozen method:
  tier 1 measured   observed scanline pieces
  tier 2 indicated  hits joined along one straight trace (direction tolerance JOIN_DTH, distance JOIN_DPERP), extended
                    with a 2D training length of a similar direction (window LEN_DWIN), length-weighted
  tier 3 inferred   EVAE proposals (Paper 1 EVAE of the fold, stored realisations) placed one by one under four rules:
                    R1 scanline consistency  - no inferred trace through a scanline (aperture APERTURE)
                    R2 hit-based intensity   - total length from the hits per metre of scanline (prior weight A_PRIOR)
                    R3 crossing limit        - crossings <= training crossing rate x XFAC (scaled by intensity)
                    R4 duplicate removal     - no near-parallel (< DUP_DTH), overlapping (> DUP_OV) trace closer than
                                               the DUP_PCT-th percentile of real parallel gaps
Arms (same tiers and rules): EVAE_F (frozen), STAT_F, KDE_F, ADFNE_F; transformer option TRF_F (the portable interaction
transformer accepts and orders the EVAE proposals before the rules); no-scanline generators RAW_EVAE / RAW_KDE / RAW_ADFNE;
natural variation REFERENCE (8 nearest training windows, hold-out window test only).
Sensitivity: one hand-set number changed at a time (EVAE_F), all three tests, 5 m and centre line."""
import sys, math, time, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
import tr_common as TC
from run_v2 import sd, _tier2_nobias

sys.path.insert(0, str(P.REL / 'scripts'))
OUT = HERE / 'results'

BASE = dict(APERTURE=0.10, JOIN_DTH=6.0, JOIN_DPERP=0.35, LEN_DWIN=15.0, LEN_BIAS=1, A_PRIOR=2.0, XFAC=1.0, DUP_DTH=5.0,
            DUP_OV=0.3, DUP_PCT=5.0, CLEAN_IT=300)
SENS = [('APERTURE', 0.05), ('APERTURE', 0.30), ('JOIN_DTH', 4.0), ('JOIN_DTH', 10.0), ('JOIN_DPERP', 0.20), ('JOIN_DPERP', 0.60),
        ('LEN_DWIN', 10.0), ('LEN_DWIN', 25.0), ('LEN_BIAS', 0), ('A_PRIOR', 0.0), ('A_PRIOR', 5.0), ('XFAC', 0.75), ('XFAC', 1.5),
        ('DUP_DTH', 3.0), ('DUP_DTH', 8.0), ('DUP_OV', 0.2), ('DUP_OV', 0.5), ('DUP_PCT', 1.0), ('DUP_PCT', 10.0), ('CLEAN_IT', 150), ('CLEAN_IT', 600)]
MAIN_ARMS = ['EVAE_F', 'STAT_F', 'KDE_F', 'ADFNE_F', 'TRF_F', 'RAW_EVAE', 'RAW_KDE', 'RAW_ADFNE']


def fill(pool, lines, rec, bud, H, rng, xmax, dmin, prm):
    have = P.seglen(rec).sum(); X = len(P.crossing_pairs(rec)) if len(rec) > 1 else 0
    Pp = P.clip(pool, H)
    dth = math.radians(prm['DUP_DTH'])
    if len(Pp):
        Pp = V.dedup(Pp, dth, dmin, prm['DUP_OV'])
    for i in rng.permutation(len(Pp)):
        if have >= bud: break
        t = Pp[i]
        if any(V.crosses(t[None], ln)[0] for ln in lines): continue                       # R1
        if len(rec) and V.is_dup(t, rec, dth, dmin, prm['DUP_OV']): continue             # R4
        nx = int(P.common.seg_cross_one(t, rec).sum()) if len(rec) else 0
        if nx > 0 and X + nx > xmax: continue                                             # R3
        X += nx; rec = np.vstack([rec, t[None]]) if len(rec) else t[None]; have += P.seglen(t[None])[0]
    return rec


def frozen(C, obs, lines, H, rng, pool_fn, prm, dmin_cache, clean=False):
    groups = V.merge(obs, prm['JOIN_DTH'], prm['JOIN_DPERP'])
    t2 = (V.tier2(groups, C['th2'], C['tr_len2d'], H, rng, prm['LEN_DWIN']) if prm['LEN_BIAS']
          else _tier2_nobias(groups, C['th2'], C['tr_len2d'], H, rng, prm['LEN_DWIN']))
    bud, p21h = V.budget(obs, lines, H, C['tr_th'], C['tr_len3d'], C['prior'], prm['A_PRIOR'])     # R2
    xmax = C['xrate'] * 20 * H * (p21h / C['prior']) ** 2 * prm['XFAC']
    dmin = dmin_cache.setdefault(prm['DUP_PCT'], float(np.percentile(V.near_parallel_gaps(C['TW']), prm['DUP_PCT'])))
    rec = t2
    for k in range(24):
        if P.seglen(rec).sum() >= bud: break
        rec = fill(pool_fn(k), lines, rec, bud, H, rng, xmax, dmin, prm)
    if clean:                                                         # R5 short iterative clean-up (round 6)
        import rules2 as RU
        if 'n_rate' not in C:
            A = sum(20 * h for h in C['TH']); C['n_rate'] = sum(len(w) for w in C['TW']) / A
            C['pY'] = float(np.nanmean([P.topo(w, h)['pY'] for w, h in zip(C['TW'], C['TH']) if len(w) > 1]))
        T = dict(p21=p21h, n=C['n_rate'] * 20 * H * (p21h / C['prior']), X=xmax, pY=C['pY'], dmin=dmin, mix=C['cls_share'])
        def _prop():
            arr = np.asarray(pool_fn(int(rng.integers(24)))).reshape(-1, 4)
            return arr[int(rng.integers(len(arr)))][None] if len(arr) else np.zeros((0, 4))
        rec = RU.cleanup(rec, len(t2), T, lines, C['model'], H, rng, _prop, iters=int(prm.get('CLEAN_IT', 300)))
    return rec


def p21map_err(rec, gt, H):
    return float(np.sqrt(np.mean((P.p21_map(rec, H) - P.p21_map(gt, H)) ** 2)))


def run_window(job):
    import torch; torch.set_num_threads(1)
    import statistical_tests as ST
    import tr_tip as TT
    scheme, fi, box, mode = job
    C = TC.fold_ctx(scheme, fi); model = C['model']
    b = C['boxes'][box]; H = b['H']; gt = P.clip(b['L'], H); ym = H / 20
    draws = [(s_, d_) for s_ in P.SEEDS for d_ in range(P.NDRAW)]
    stored = {m: [P.clip(P.stored_net(scheme, m, fi, box, s_, d_), H) for s_, d_ in draws] for m in ('EVAE', 'KDE', 'ADFNE')}
    tips = {s: TT.load(fi, s, scheme) for s in P.SEEDS} if mode == 'main' else None
    fall = float(np.angle(np.mean(np.exp(2j * C['th2']))) / 2) % math.pi
    dmin_cache = {}
    rows = []
    def rec_row(cfg, arm, r, rec):
        s = ST.score(np.asarray(rec).reshape(-1, 4) / 20, gt / 20, ym, model)
        s = {k: (np.nan if (v is not None and np.isinf(v)) else v) for k, v in s.items()}
        rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cfg, arm=arm, r=r, n=len(rec), gt_n=len(gt), map_rmse=p21map_err(rec, gt, H), **s))
    variants = [('base', dict(BASE))] if mode in ('main', 'mainC') else [(f'{k}={v}', dict(BASE, **{k: v})) for k, v in SENS]
    CLEAN = mode in ('mainC', 'sensC')
    for vname, prm in variants:
        for spacing in (5.0, None):
            cname = ('5m' if spacing else 'centre') + ('' if mode == 'main' else '|' + vname)
            lines = V.lines_layout('v', spacing, prm['APERTURE'], H)
            for r in range(24):
                obs = V.observe(gt, lines, H, np.random.default_rng(sd(scheme, box, r, '5m' if spacing else 'centre', 'final_obs')))
                arms = MAIN_ARMS if mode == 'main' else (['EVAE_F', 'STAT_F', 'KDE_F', 'ADFNE_F'] if mode == 'mainC' else ['EVAE_F'])
                for arm in arms:
                    rng = np.random.default_rng(sd(scheme, box, r, '5m' if spacing else 'centre', 'final', arm))
                    if arm.startswith('RAW_'):
                        rec = stored[arm[4:]][r]
                    elif arm == 'STAT_F':
                        rec = frozen(C, obs, lines, H, rng, lambda k: P.unif_proposals(60, C['tr_th'], C['tr_len3d'], H, rng), prm, dmin_cache, CLEAN)
                    elif arm in ('EVAE_F', 'KDE_F', 'ADFNE_F'):
                        src = stored[arm.split('_')[0]]
                        rec = frozen(C, obs, lines, H, rng, lambda k, src=src: src[(r + k) % 24], prm, dmin_cache, CLEAN)
                    elif arm == 'TRF_F':
                        net = tips[P.SEEDS[r // 8]]; tref = TT.theta_ref(obs, fall)
                        def tpool(k, net=net, tref=tref):
                            cand = stored['EVAE'][(r + k) % 24]
                            cand = cand[[not any(V.crosses(t[None], l)[0] for l in lines) for t in cand]] if len(cand) else cand
                            if not len(cand): return cand
                            ctx = TT.tok(V.tier1(obs), 0, lines, tref)
                            seqs = [np.vstack([ctx, TT.tok(c[None], 3, lines, tref)])[-120:] for c in cand]
                            T_ = max(len(s) for s in seqs); tk = np.zeros((len(seqs), T_, TT.FD), np.float32); pad = np.ones((len(seqs), T_), bool)
                            for i, s in enumerate(seqs): tk[i, :len(s)] = s; pad[i, :len(s)] = False
                            with torch.no_grad():
                                h, acc, pri = net(torch.as_tensor(tk), torch.as_tensor(pad))
                            ci = torch.as_tensor([len(s) - 1 for s in seqs])
                            pa = torch.sigmoid(acc[torch.arange(len(seqs)), ci]).numpy()
                            return cand[rng.random(len(cand)) < pa]
                        rec = frozen(C, obs, lines, H, rng, tpool, prm, dmin_cache)
                    rec_row(cname, arm, r, rec)
    if mode == 'main' and scheme == 'S1':
        sp = C['sp']; held = set().union(*[set(C['boxes'][t]['ids']) for t in sp['test']])
        cand = [k for k in C['boxes'] if k not in sp['test']]
        near = sorted(cand, key=lambda k: (math.hypot(C['boxes'][k]['cu'] - b['cu'], C['boxes'][k]['cs'] - b['cs']), k))[:8]
        for j, k in enumerate(near):
            nb = C['boxes'][k]; keep = np.array([i not in held for i in nb['ids']], bool); Ln = P.clip(nb['L'][keep], nb['H'])
            s = ST.score(Ln / 20, gt / 20, ym, model); s = {kk: (np.nan if (v is not None and np.isinf(v)) else v) for kk, v in s.items()}
            mr = p21map_err(Ln, gt, H) if abs(nb['H'] - H) < 1e-6 else np.nan
            for cfg in ('5m', 'centre'):
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cfg, arm='REFERENCE', r=j, n=len(Ln), gt_n=len(gt), map_rmse=mr, **s))
    return rows


def main(mode):
    jobs = [(s, fi, box, mode) for s, n in (('S1', 4), ('S2', 7), ('S3', 4)) for fi in range(n) for box in P.load_split(s, fi)['test']]
    R = []; t0 = time.time()
    with Pool(16) as pool:
        for i, r in enumerate(pool.imap_unordered(run_window, jobs)):
            R += r; print(f'{mode} {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(R).to_csv(OUT / f'final_{mode}_rows.csv', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        t = time.time(); df = pd.DataFrame(run_window(('S1', 0, P.load_split('S1', 0)['test'][1], 'main'))); print(f'{time.time()-t:.0f}s', df.shape)
        print(df.groupby(['cfg', 'arm'])[['n', 'gt_n', 'orient_w1_deg', 'len_w1_m', 'P21_relerr', 'P20_relerr', 'P22_relerr', 'nn_w1_m', 'node_tv', 'map_rmse']].median().round(3).to_string())
    else:
        main(sys.argv[1])
