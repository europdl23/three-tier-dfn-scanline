# -*- coding: utf-8 -*-
"""HYB_FIX (EXPLORATORY, 14 Oct 2026): combined transformer + EVAE pool with the two faults fixed.
Same composition as ABL_P (rule tier 2, no termination); only the pool differs:
  1. EVAE traces are taken WHOLE: identical draw to generators.generate_evae (same model, same seed) but the final
     window clip is skipped, so each trace is cut once only, at the real window edge, after placement.
  2. At each transformer trace, the EVAE trace is drawn from the SAME direction cluster as the transformer's
     predicted direction (setbg.classify with the fold's cluster model); fallback = nearest axial direction.
Frozen code is imported, never edited. Usage: python hybfix_run.py check | run CFG"""
import os, sys, json, math, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1] / 'reanalysis'
sys.path.insert(0, str(SCRIPTS))
import numpy as np
import pandas as pd
import ra01abl_run as R

OUT = Path(__file__).resolve().parents[1] / 'work' / 'hybfix'
ARM = 'HYB_FIX'
_EVW = {}


def evae_whole_sampler(fi, seed):
    import p2lib as P
    key = (fi, seed)
    if key not in _EVW:
        sys.path.insert(0, str(P.REL / 'src' / 'evaluation'))
        import generators as G
        G.BRANCH = P.REL / 'src' / 'model'
        m, dev = G.load_evae(P.REL / 'models' / 'evae' / f'S2_f{fi}_s{seed}' / 'checkpoint.pt')
        _EVW[key] = (m, dev)
    m, dev = _EVW[key]

    def gen(H, seed_gen):                                   # generators.generate_evae, lines 89-125, without the clip
        import torch
        import torch.nn.functional as F
        torch.manual_seed(seed_gen); rng = np.random.default_rng(seed_gen)
        with torch.no_grad():
            sz = torch.randn(1, 24, device=dev); gz = torch.randn(1, 32, device=dev)
            centers = torch.sigmoid(m.spatial_decoder(sz)).view(-1, m.max_lines, 2)[0]
            lengths = torch.sigmoid(m.length_decoder(gz)).view(-1, m.max_lines)[0]
            logits = m.angle_logits_head(gz).view(-1, m.max_lines, m.K_angle)[0]
            mu_raw = m.angle_mu_head(gz).view(-1, m.max_lines, m.K_angle)[0]
            kappa_raw = m.angle_kappa_head(gz).view(-1, m.max_lines, m.K_angle)[0]
            alpha = F.softmax(logits, dim=-1).cpu().numpy(); mu = (math.pi * torch.tanh(mu_raw)).cpu().numpy()
            kappa = (F.softplus(kappa_raw) + 1e-4).cpu().numpy()
            exist = m.existence_head(torch.cat([sz, gz], 1)).view(-1, m.max_lines)[0]
            keep = (torch.sigmoid(exist) >= 0.5).cpu().numpy()
            if keep.sum() < 2:
                top = torch.topk(exist, 10).indices.cpu().numpy(); keep = np.zeros(m.max_lines, bool); keep[top] = True
            c = centers.cpu().numpy()[keep]; L = lengths.cpu().numpy()[keep]
            aa, am, ak = alpha[keep], mu[keep], kappa[keep]
        th = []
        for j in range(len(c)):
            comp = rng.choice(m.K_angle, p=aa[j] / aa[j].sum())
            th.append((rng.vonmises(2.0 * am[j, comp], max(ak[j, comp], 1e-4)) / 2.0) % math.pi)
        th = np.asarray(th); dx = np.cos(th) * L / 2; dy = np.sin(th) * L / 2
        return np.column_stack([c[:, 0] - dx, c[:, 1] - dy, c[:, 0] + dx, c[:, 1] + dy]) * 20.0
    return gen


def evae_whole_trained(fi, box, H):
    """same 24 draws and seeds as ra01abl_run.evae_prior_trained (D34), not clipped"""
    import p2lib as P
    import study as S
    out = []
    for s in P.SEEDS:
        ev = evae_whole_sampler(fi, s)
        for d in range(P.NDRAW):
            out.append(ev(H, S.gen_seed('EVAE', 'S2prior', fi, box, s, d)))
    return out


def seg_theta(a):
    return np.arctan2(a[:, 3] - a[:, 1], a[:, 2] - a[:, 0]) % math.pi


_orig_make_pool = R.make_pool


def make_pool(kind, C, t1, lines, H, rng, models, evae_nets, net_offset):
    if kind != 'tp_evae_whole':
        return _orig_make_pool(kind, C, t1, lines, H, rng, models, evae_nets, net_offset)
    import p2lib as P
    import p2v2 as V
    import gast as G
    gnet = models['gnet']

    def pf(k, rec, src=evae_nets):
        g = G.generate(gnet, C, t1, rec, lines, H, rng); s_ = src[(net_offset + k) % 24]
        if not len(g) or not len(s_): return np.zeros((0, 4))
        tg, te = seg_theta(g), seg_theta(s_)
        cg = V.setbg.classify(tg, C['model']); ce_ = V.setbg.classify(te, C['model'])
        pick = []
        for i in range(len(g)):
            idx = np.flatnonzero(ce_ == cg[i])
            if not len(idx):
                d = np.abs(((te - tg[i]) + math.pi / 2) % math.pi - math.pi / 2); idx = np.argsort(d)[:3]
            pick.append(idx[rng.integers(len(idx))])
        e = s_[np.asarray(pick)]; c = (g[:, :2] + g[:, 2:]) / 2; ce = (e[:, :2] + e[:, 2:]) / 2
        return P.clip(e + np.hstack([c - ce, c - ce]), H)
    return pf


R.make_pool = make_pool
R.ARMS[ARM] = ('rule', 'tp_evae_whole', None)
R.evae_prior_trained = evae_whole_trained


def one(spec):
    import warnings
    warnings.simplefilter('ignore')
    t0 = time.time()
    rows, recs, gt, meta = R.job_trained(dict(spec, arms=[ARM]))
    name = f"tw_{spec['cfg']}_f{spec['fold']}_{spec['box']}"
    pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False, float_format='%.17g')
    np.savez_compressed(OUT / f'{name}.npz', gt=gt, **recs)
    return name, time.time() - t0


def check():
    """pool length check on one window per fold: whole EVAE traces after one cut at a random centre"""
    R._setup_worker()
    import p2lib as P
    import gast as G
    jobs = pd.read_csv(R.OUT / 'rows_RA01ABL_trainedwall.csv', usecols=['fold', 'box']).drop_duplicates()
    L_old, L_new = [], []
    rng = np.random.default_rng(0)
    for fi, b in jobs.groupby('fold').head(1).itertuples(index=False):
        C = G.prep_ctx('S2', int(fi)); H = C['boxes'][b]['H']
        for whole in (False, True):
            nets = evae_whole_trained(int(fi), b, H)
            for s_ in nets:
                if whole is False: s_ = P.clip(s_, H)
                c = np.column_stack([rng.uniform(0, P.M, len(s_)), rng.uniform(0, H, len(s_))]); ce = (s_[:, :2] + s_[:, 2:]) / 2
                x = P.clip(s_ + np.hstack([c - ce, c - ce]), H)
                (L_new if whole else L_old).extend(P.seglen(x))
    for nm, L in (('old: cut, moved, cut again', L_old), ('fixed: whole, moved, cut once', L_new)):
        L = np.asarray(L); print(f'{nm}: n {len(L)} median {np.median(L):.2f} m, under 3 m {np.mean(L < 3):.1%}, 10 m+ {np.mean(L >= 10):.1%}')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == 'check':
        check()
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        cfg = sys.argv[2]
        jobs = pd.read_csv(R.OUT / 'rows_RA01ABL_trainedwall.csv', usecols=['fold', 'box', 'cfg'])
        jobs = jobs[jobs.cfg == cfg].drop_duplicates()
        specs = [dict(fold=int(f), box=b, cfg=cfg) for f, b, _ in jobs.itertuples(index=False)
                 if not (OUT / f'tw_{cfg}_f{int(f)}_{b}.csv').exists()]
        print(len(specs), 'jobs', flush=True); t0 = time.time()
        with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 20) as ex:
            fs = [ex.submit(one, s) for s in specs]
            for i, f in enumerate(as_completed(fs)):
                n, s = f.result(); print(f'{i+1}/{len(specs)} {n} {s:.0f} s (elapsed {time.time()-t0:.0f} s)', flush=True)
