# -*- coding: utf-8 -*-
"""Shared fold context and the final rule-based rebuild (tiers 1-3, termination, crossing limit, duplicate filter) used
by all transformer-direction tests, so every direction is compared under identical rules, folds, seeds and evidence."""
import math
from pathlib import Path
import numpy as np
import p2lib as P
import p2v2 as V
from run_extra import train_cross_rate
from run_v3_term import protect_spans
from run_dd import fill_dd

DTH = math.radians(5.0)


def fold_ctx(scheme, fi):
    boxes = P.load_library(); sp = P.load_split(scheme, fi)
    model = V.setbg.load(P.REL / 'data' / 'orientation_clusters' / f'{scheme}_f{fi}.json')
    tr_th, tr_len3d = P.train_traces(boxes, sp); prior = P.train_p21(boxes, sp)
    xrate, tr_len2d = train_cross_rate(boxes, sp)
    held = set().union(*[set(boxes[t]['ids']) for t in sp['test']]); th2, TW, TH = [], [], []
    for k in sp['fitting'] + sp['validation']:
        b = boxes[k]; keep = np.array([i not in held for i in b['ids']], bool); Lk = P.clip(b['L'][keep], b['H'])
        th2 += list(P.segdir(Lk)); TW.append(Lk); TH.append(b['H'])
    th2 = np.array(th2)
    cls = V.setbg.classify(th2, model); share = np.bincount(cls, minlength=model['K'] + 1) / len(cls)
    return dict(boxes=boxes, sp=sp, model=model, tr_th=tr_th, tr_len3d=tr_len3d, prior=prior, xrate=xrate, tr_len2d=tr_len2d,
                th2=th2, TW=TW, TH=TH, p_pass=V.pass_prob(TW, TH), dmin=float(np.percentile(V.near_parallel_gaps(TW), 5)),
                cls_share=share)


def rebuild(C, obs, lines, H, rng, pool_fn, miss_known=0.0, n_hits_override=None, t2=None, editor=None, xlim=True, term=True, dd=True,
            fill_rules=True, term_fn=None):
    """final method. pool_fn(k, rnd) -> candidate traces (metres). editor(rec, cand, n2) -> edited candidate list or None."""
    groups = V.merge(obs)
    if t2 is None:
        t2 = V.tier2(groups, C['th2'], C['tr_len2d'], H, rng)
        if term and len(t2) > 1:
            t2 = (term_fn(t2, 0, rng, protect_spans(t2, obs)) if term_fn else V.terminate(t2, 0, rng, C['p_pass'], protect=protect_spans(t2, obs)))
    obs_n = obs if n_hits_override is None else [None] * n_hits_override
    bud, p21h = V.budget(obs_n, lines, H, C['tr_th'], C['tr_len3d'], C['prior'], miss_known=miss_known)
    xmax = C['xrate'] * P.M * H * (p21h / C['prior']) ** 2 if xlim else 1e9
    rec = t2
    for rnd in range(3 if term else 1):
        for k in range(24):
            if P.seglen(rec).sum() >= bud:
                break
            pool = pool_fn(k, rnd)
            if editor is not None:
                pool = editor(rec, pool, len(t2))
                if pool is None or not len(pool):
                    continue
            if fill_rules:
                rec = fill_dd(pool, lines, rec, bud, H, rng, xmax, C['dmin'], dd)
            else:
                add = P.clip(pool, H)
                add = add[[not any(V.crosses(t[None], l)[0] for l in lines) for t in add]] if len(add) else add
                rec = np.vstack([rec, add]) if len(add) else rec
                break
        if term:
            rec = term_fn(rec, len(t2), rng, None) if term_fn else V.terminate(rec, len(t2), rng, C['p_pass'])
            rec = rec[P.seglen(rec) > 0.3]
    return rec, len(t2)


def scores(rec, gt, H, g=None):
    g = g or P.metrics(gt, H); m = P.metrics(rec, H); s = P.score(m, g)
    tr, tg = P.topo(rec, H) if len(rec) else dict(pY=np.nan), P.topo(gt, H)
    s['pY_err'] = abs(tr['pY'] - tg['pY']) if np.isfinite(tr.get('pY', np.nan)) and np.isfinite(tg['pY']) else np.nan
    s['n'] = m['n']; s['gt_n'] = g['n']
    return s


def stat_pool(C, H, rng):
    return lambda k, rnd: P.unif_proposals(60, C['tr_th'], C['tr_len3d'], H, rng)


def evae_pool(scheme, fi, box, H, r):
    draws = [(s_, d_) for s_ in P.SEEDS for d_ in range(P.NDRAW)]
    nets = [P.clip(P.stored_net(scheme, 'EVAE', fi, box, s_, d_), H) for s_, d_ in draws]
    return lambda k, rnd: nets[(r + k + 7 * rnd) % 24], nets
