# -*- coding: utf-8 -*-
"""Round 9: component-by-component LEARNED vs RULE at a NEW BENCH (S2, real geology only), paired with final_newbench.py
(same windows, scanlines and observation seeds; cfg rot0_5m / rot0_centre).
Each arm swaps ONE component of the frozen method (BASE = EVAE_F: rule tiers + EVAE + R1-R5) for its learned version:
  L1_EXT   indicated tier: evidence network (length + hit position of every joined hit group)   vs rule tier 2
  L2_CNT   how much fracturing: evidence network hidden-length head                             vs R2 hit budget
  L3_GEN   inferred fractures: sequential transformer (position, direction, set) + EVAE lengths vs EVAE realisations
  L4_ACC   accept / reject: critic ranks candidates, R1 kept as a mask                         vs R3 + R4 + R5
  L5_TERM  termination at crossings: learned termination network                                vs none (frozen method)
  ALL_L    all five learned (R1 kept as a mask);  ALL_L_NT  all learned except termination"""
import sys, math, time, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
import final_eval as FE
from run_v2 import sd
from run_ab import protect_spans
sys.path.insert(0, str(P.REL / 'scripts'))
OUT = HERE / 'results'
ARMS = os.environ.get('LARMS', 'BASE,L1_EXT,L2_CNT,L3_GEN,L4_ACC,L5_TERM,ALL_L,ALL_L_NT').split(',')
FLAGS = dict(BASE='', L1_EXT='E', L2_CNT='N', L3_GEN='G', L4_ACC='A', L5_TERM='T', ALL_L='ENGAT', ALL_L_NT='ENGA', MIX_ENT='ENT', MIX_ET='ET', MIX_ENTP='ENTP', L6_POS='P')


def rule_place(C, rec, bud, p21h, lines, H, rng, pool_fn, dmc, n_fixed):
    """R2 budget given; R1, R3, R4 in fill; R5 clean-up (as final_eval.frozen)"""
    prm = FE.BASE
    xmax = C['xrate'] * 20 * H * (p21h / C['prior']) ** 2 * prm['XFAC']
    dmin = dmc.setdefault(5.0, float(np.percentile(V.near_parallel_gaps(C['TW']), prm['DUP_PCT'])))
    for k in range(24):
        if P.seglen(rec).sum() >= bud: break
        rec = FE.fill(pool_fn(k, rec), lines, rec, bud, H, rng, xmax, dmin, prm)
    import rules2 as RU
    if 'n_rate' not in C:
        A = sum(20 * h for h in C['TH']); C['n_rate'] = sum(len(w) for w in C['TW']) / A
        C['pY'] = float(np.nanmean([P.topo(w, h)['pY'] for w, h in zip(C['TW'], C['TH']) if len(w) > 1]))
    T = dict(p21=p21h, n=C['n_rate'] * 20 * H * (p21h / C['prior']), X=xmax, pY=C['pY'], dmin=dmin, mix=C['cls_share'])
    def _prop():
        arr = np.asarray(pool_fn(int(rng.integers(24)), rec)).reshape(-1, 4)
        return arr[int(rng.integers(len(arr)))][None] if len(arr) else np.zeros((0, 4))
    return RU.cleanup(rec, n_fixed, T, lines, C['model'], H, rng, _prop, iters=prm['CLEAN_IT'])


def learned_place(LN, crt, C, t1, rec, bud, lines, H, rng, pool_fn):
    """critic as a learned ranker: candidates scored against the current record, one accepted per round (sampled in
    proportion to the accept probability); a round with every probability < 0.02 = all rejected -> fresh candidates"""
    def r1(c):
        c = P.clip(np.asarray(c).reshape(-1, 4), H)
        return c[[not any(V.crosses(t[None], l)[0] for l in lines) for t in c]] if len(c) else c
    k, cand = 0, np.zeros((0, 4))
    for _ in range(300):
        if P.seglen(rec).sum() >= bud: break
        if len(cand) < 10 and k < 24:
            cand = np.vstack([cand, r1(pool_fn(k, rec))]); k += 1
            if len(cand) < 10 and k < 24: continue
        if not len(cand): break
        ctx = np.vstack([t1, rec]) if len(rec) else np.asarray(t1).reshape(-1, 4)
        tiers = np.r_[np.zeros(len(t1)), np.ones(len(rec))]
        p = LN.critic_prob(crt, C, ctx, tiers, cand, lines)
        if p.max() < 0.02:
            if k >= 24: break
            cand = np.zeros((0, 4)); continue
        i = rng.choice(len(cand), p=p / p.sum())
        rec = np.vstack([rec, cand[i][None]]) if len(rec) else cand[i][None]; cand = np.delete(cand, i, 0)
    return rec


def run_window(job):
    import torch; torch.set_num_threads(1)
    import statistical_tests as ST
    import gast as G
    import learned as LN
    import tr_gen as TG
    scheme, fi, box = job
    C = G.prep_ctx(scheme, fi); model = C['model']
    b = C['boxes'][box]; H = b['H']; gt = P.clip(b['L'], H); ym = H / 20
    ev_r = [P.clip(P.stored_net(scheme, 'EVAE', fi, box, s_, d_), H) for s_ in P.SEEDS for d_ in range(P.NDRAW)]
    ev = np.vstack(ev_r); evl, evc = P.seglen(ev), V.setbg.classify(P.segdir(ev), model)
    pools = {k: (evl[evc == k] if (evc == k).sum() >= 5 else evl) for k in range(model['K'] + 1)}
    len_fn = lambda k, rng: float(rng.choice(pools.get(k, evl)))
    nets = {s: (G.load(fi, s, scheme, 'gast2'),) + LN.load(fi, s, scheme) + (TG.load_models(scheme, fi, s)[0],) for s in P.SEEDS}
    dmc = {}; rows = []
    for spacing in (5.0, None):
        cname = f'rot0_{"5m" if spacing else "centre"}'
        lines = V.lines_layout('v', spacing, FE.BASE['APERTURE'], H)
        for r in range(24):
            obs = V.observe(gt, lines, H, np.random.default_rng(sd(scheme, box, r, cname, 'nb_obs')))
            gnet, evn, crt, term = nets[P.SEEDS[r // 8]]; t1 = V.tier1(obs)
            groups = V.merge(obs, FE.BASE['JOIN_DTH'], FE.BASE['JOIN_DPERP'])
            for arm in ARMS:
                f = FLAGS[arm]; t0 = time.time()
                rng = np.random.default_rng(sd(scheme, box, r, cname, 'nb', 'EVAE_F' if arm == 'BASE' else arm))
                if 'E' in f: t2, _ = LN.tier2_learned(evn, C, groups, lines, H, rng)
                else: t2 = V.tier2(groups, C['th2'], C['tr_len2d'], H, rng, FE.BASE['LEN_DWIN'])
                bud, p21h = V.budget(obs, lines, H, C['tr_th'], C['tr_len3d'], C['prior'], FE.BASE['A_PRIOR'])
                if 'N' in f:
                    bud = P.seglen(t2).sum() + LN.hidden_length(evn, C, groups, lines, H, rng); p21h = bud / (20 * H)
                if 'P' in f:     # sequential transformer = WHERE (centre), EVAE = WHAT (direction-length pair of one EVAE trace)
                    def pool_fn(k, rec, gnet=gnet):
                        g = G.generate(gnet, C, t1, rec, lines, H, rng)
                        src = ev_r[(r + k) % 24]
                        if not len(g) or not len(src): return np.zeros((0, 4))
                        e = src[rng.integers(len(src), size=len(g))]; c = (g[:, :2] + g[:, 2:]) / 2; ce = (e[:, :2] + e[:, 2:]) / 2
                        return P.clip(e + np.hstack([c - ce, c - ce]), H)
                elif 'G' in f:
                    pool_fn = lambda k, rec, gnet=gnet: G.generate(gnet, C, t1, rec, lines, H, rng, len_fn=len_fn)
                else:
                    pool_fn = lambda k, rec: ev_r[(r + k) % 24]
                if 'A' in f:
                    rec = learned_place(LN, crt, C, t1, t2, bud, lines, H, rng, pool_fn)
                else:
                    rec = rule_place(C, t2, bud, p21h, lines, H, rng, pool_fn, dmc, len(t2))
                if 'T' in f and len(rec) > 1:
                    pr = protect_spans(t2, obs) + [None] * (len(rec) - len(t2))
                    rec = TG.terminate_learned(rec, len(t2), rng, term, model, protect=pr)
                    rec = rec[P.seglen(rec) > 0.05]
                s = ST.score(np.asarray(rec).reshape(-1, 4) / 20, gt / 20, ym, model)
                s = {kk: (np.nan if (v is not None and np.isinf(v)) else v) for kk, v in s.items()}
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cname, arm=arm, r=r, n=len(rec), gt_n=len(gt),
                                 map_rmse=FE.p21map_err(rec, gt, H), sec=time.time() - t0, **s))
    return rows


def main():
    jobs = [('S2', fi, box) for fi in range(7) for box in P.load_split('S2', fi)['test']]
    R = []; t0 = time.time(); tag = os.environ.get('LTAG', 'learned')
    with Pool(int(os.environ.get('LPOOL', 22))) as pool:
        for i, r in enumerate(pool.imap_unordered(run_window, jobs)):
            R += r; print(f'{tag} {i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
            if (i + 1) % 10 == 0: pd.DataFrame(R).to_csv(OUT / f'final_{tag}_rows.partial.csv', index=False)
    pd.DataFrame(R).to_csv(OUT / f'final_{tag}_rows.csv', index=False)


if __name__ == '__main__':
    if sys.argv[1:] == ['smoke']:
        P_ = P.load_split('S2', 3)['test'][0]; t = time.time()
        import eval_learned as EL
        rows = EL.run_window(('S2', 3, P_)); df = pd.DataFrame(rows)
        print(f'{time.time()-t:.0f}s'); print(df.groupby(['cfg', 'arm'])[['n', 'gt_n', 'sec', 'orient_w1_deg', 'len_w1_m', 'P21_relerr', 'node_tv', 'map_rmse']].median().round(3).to_string())
    else:
        main()
