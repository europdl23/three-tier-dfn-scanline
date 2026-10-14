# -*- coding: utf-8 -*-
"""Geology-first rule extensions (round 6). All statistics from the fold's training windows only.
  EDGE   censoring-corrected lengths: 2D trace pieces touching a window edge are right-censored; the length distribution
         is the Kaplan-Meier product-limit estimate (censored pieces count as 'at least this long'); used for tier 2
         extensions and tier 3 proposals instead of raw pieces / 3D lengths.
  TERZ   Terzaghi set-bias correction: the window's set mix is estimated from its scanline hits weighted by
         1 / (L_set * |cos theta|) (the chance that a vertical line cuts a trace of that direction), with |cos| capped at
         0.25 and shrunk toward the training set mix with a prior of 3 traces; tier 3 then fills each set according to
         its deficit against this target (tier 2 + tier 3 together reach the corrected mix).
  ASYM   set-specific asymmetric termination: from real Y-nodes (trace of set a ends on a trace of set b) and real
         crossings (a, b), P(a stops on b) = Y[a,b] / (Y[a,b] + X[a,b]) with +1 smoothing; sets are ranked by how
         rarely they end on others (through-going = older), placed first, and younger sets are cut against them.
  CLEAN  short iterative clean-up (greedy, 300 moves): add / delete / replace tier 3 traces to lower a joint misfit of
         P21, count, crossings, Y-node share, near-duplicate pairs and set mix against their targets."""
import math
import numpy as np
import p2lib as P
import p2v2 as V
from run_v3_term import protect_spans

CAP = 0.25


# ------------------------------------------------------------------ EDGE
def km_lengths(windows, Hs, e=0.2):
    L, c = [], []
    for W_, H in zip(windows, Hs):
        for x1, y1, x2, y2 in W_:
            L.append(math.hypot(x2 - x1, y2 - y1))
            c.append(min(x1, x2) < e or max(x1, x2) > 20 - e or min(y1, y2) < e or max(y1, y2) > H - e)
    L = np.array(L); c = np.array(c)
    o = np.argsort(L); L, c = L[o], c[o]
    n = len(L); S = 1.0; surv = []
    for i in range(n):
        if not c[i]:
            S *= (1 - 1 / (n - i))
        surv.append(S)
    surv = np.array(surv)
    # probability mass at each event; the remaining tail (censored beyond the largest event) is put at 1.3x max length
    p = np.r_[1.0, surv[:-1]] - surv
    tail = surv[-1]
    vals = np.r_[L, L.max() * 1.3]; pr = np.r_[p, tail]; pr = np.clip(pr, 0, None)
    return vals, pr / pr.sum()


def km_sample(vals, pr, rng, n, minlen=0.0, weight_by_len=False):
    w = pr * (vals >= minlen)
    if w.sum() <= 0:
        return np.full(n, minlen * 1.1 + 0.2)
    if weight_by_len:
        w = w * vals
    return rng.choice(vals, size=n, p=w / w.sum())


def km_by_set(windows, Hs, model):
    """KM length distribution per set (falls back to all traces when a set has < 15 pieces)"""
    allv = km_lengths(windows, Hs)
    out = {}
    for k in range(model['K'] + 1):
        sub = [np.array([t for t in W_ if V.setbg.classify(np.array([P.segdir(t[None])[0]]), model)[0] == k]).reshape(-1, 4) for W_ in windows]
        n = sum(len(s) for s in sub)
        out[k] = km_lengths(sub, Hs) if n >= 15 else allv
    out['all'] = allv
    return out


def tier2_km(groups, kms, model, H, rng):
    out = []
    for g in groups:
        th = math.atan2(sum(math.sin(2 * o['th']) for o in g), sum(math.cos(2 * o['th']) for o in g)) / 2 % math.pi
        ux, uy = math.cos(th), math.sin(th)
        s = np.array([o['x'] * ux + o['y'] * uy for o in g]); span = s.max() - s.min()
        k = V.setbg.classify(np.array([th]), model)[0]; vals, pr = kms[k]
        Lf = float(km_sample(vals, pr, rng, 1, minlen=span, weight_by_len=True)[0])
        a = rng.uniform(0, max(Lf - span, 0)); o0 = g[int(np.argmin(s))]
        p1 = (o0['x'] - a * ux, o0['y'] - a * uy); out.append([p1[0], p1[1], p1[0] + Lf * ux, p1[1] + Lf * uy])
    return P.clip(np.array(out, float).reshape(-1, 4), H)


def proposals(n, C, H, rng, kms=None, set_probs=None):
    """statistical proposals; with kms: censoring-corrected lengths per set; with set_probs: directions drawn from the
    training traces of a chosen set (set chosen with set_probs)."""
    model = C['model']; th_all = C['th2']; cls = C['cls2']
    out = []
    for _ in range(n):
        if set_probs is not None:
            k = rng.choice(len(set_probs), p=set_probs); pool = th_all[cls == k]
            th = float(rng.choice(pool)) if len(pool) else float(rng.choice(th_all))
        else:
            th = float(rng.choice(th_all)); k = V.setbg.classify(np.array([th]), model)[0]
        if kms is not None:
            L = float(km_sample(*kms[k], rng, 1)[0])
        else:
            L = float(rng.choice(C['tr_len3d']))
        cx, cy = rng.uniform(0, 20), rng.uniform(0, H)
        out.append([cx - math.cos(th) * L / 2, cy - math.sin(th) * L / 2, cx + math.cos(th) * L / 2, cy + math.sin(th) * L / 2])
    return np.array(out)


# ------------------------------------------------------------------ TERZ
def terzaghi_target(obs, lines, C, kms, prior_n=3.0):
    model = C['model']; K1 = model['K'] + 1
    base = C['cls_share']
    if not obs:
        return base
    th = np.array([o['th'] for o in obs]); cls = V.setbg.classify(th, model)
    w = np.zeros(K1)
    for t, k in zip(th, cls):
        Lk = float(np.sum(kms[k][0] * kms[k][1]))         # mean length of the set
        w[k] += 1.0 / (Lk * max(abs(math.cos(t)), CAP))
    w = w / w.sum() * len(obs)                              # weighted counts, same total as hits
    mix = (w + prior_n * base) / (len(obs) + prior_n)
    return mix / mix.sum()


def set_deficit_probs(rec, target, model):
    K1 = model['K'] + 1
    cur = np.bincount(V.setbg.classify(P.segdir(rec), model), minlength=K1) if len(rec) else np.zeros(K1)
    n = max(len(rec), 1)
    d = np.clip(target * (n + 5) - cur, 0.05, None)
    return d / d.sum()


# ------------------------------------------------------------------ ASYM
def asym_tables(windows, Hs, model, tol=0.3):
    K1 = model['K'] + 1
    Y = np.ones((K1, K1)); X = np.ones((K1, K1))           # +1 smoothing
    for L, H in zip(windows, Hs):
        n = len(L)
        if n < 2: continue
        c = V.setbg.classify(P.segdir(L), model)
        for i, j in P.crossing_pairs(L):
            X[c[i], c[j]] += 1; X[c[j], c[i]] += 1
        E = np.vstack([L[:, :2], L[:, 2:]]); own = np.r_[np.arange(n), np.arange(n)]
        inside = (E[:, 0] > 0.1) & (E[:, 0] < 19.9) & (E[:, 1] > 0.1) & (E[:, 1] < H - 0.1)
        D = P._pt_seg(E, L[:, :2], L[:, 2:]); D[np.arange(2 * n), own] = np.inf
        for e in np.where((D.min(1) <= tol) & inside)[0]:
            Y[c[own[e]], c[int(D[e].argmin())]] += 1
    pstop = Y / (Y + X / 2)      # X counted from both sides above
    ends_on_others = Y.sum(1) / (Y.sum(1) + Y.sum(0))
    age_rank = np.argsort(ends_on_others)                   # first = rarely ends on others = older / through-going
    return pstop, age_rank


def terminate_asym(rec, keep_n, rng, model, pstop, age_rank, protect=None):
    rec = np.array(rec, float).reshape(-1, 4); keep_n = min(keep_n, len(rec))
    cls = V.setbg.classify(P.segdir(rec), model) if len(rec) else np.zeros(0, int)
    rank = {k: i for i, k in enumerate(age_rank)}
    tail = list(range(keep_n, len(rec))); rng.shuffle(tail)
    tail.sort(key=lambda i: rank[cls[i]])                   # older sets placed first
    order = list(range(keep_n)) + tail; done = []
    for i in order:
        x1, y1, x2, y2 = rec[i]; L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-6 or not done:
            done.append(i); continue
        ps = protect[i] if protect is not None and i < len(protect) and protect[i] is not None else (L / 2, L / 2)
        lo, hi = 0.0, L
        for j in done:
            q = rec[j]; p, r = np.array([x1, y1]), np.array([x2 - x1, y2 - y1]); qq, s = q[:2], q[2:] - q[:2]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12: continue
            d = qq - p; t = (d[0] * s[1] - d[1] * s[0]) / den; u = (d[0] * r[1] - d[1] * r[0]) / den
            if not (0 < t < 1 and 0 < u < 1): continue
            a = t * L
            if ps[0] <= a <= ps[1]: continue
            if rng.random() >= pstop[cls[i], cls[j]]: continue
            if a < ps[0]: lo = max(lo, a)
            else: hi = min(hi, a)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        rec[i] = [x1 + lo * ux, y1 + lo * uy, x1 + hi * ux, y1 + hi * uy]; done.append(i)
    return rec


# ------------------------------------------------------------------ CLEAN
def misfit(rec, T, model, H):
    if not len(rec):
        return 1e9
    A = 20 * H; L = P.seglen(rec)
    X = len(P.crossing_pairs(rec)) if len(rec) > 1 else 0
    t = P.topo(rec, H); pY = t['pY'] if np.isfinite(t['pY']) else 0
    dup = int((V.near_parallel_gaps([rec]) < T['dmin']).sum())
    K1 = model['K'] + 1; c = np.bincount(V.setbg.classify(P.segdir(rec), model), minlength=K1) / len(rec)
    return (abs(L.sum() / A - T['p21']) / T['p21'] + 0.5 * abs(len(rec) - T['n']) / max(T['n'], 1)
            + 0.5 * abs(X - T['X']) / max(T['X'], 1) + abs(pY - T['pY']) / 0.2 + 0.5 * dup + 0.5 * np.abs(c - T['mix']).sum())


def cleanup(rec, n_fixed, T, lines, model, H, rng, propose, iters=300):
    rec = np.array(rec, float).reshape(-1, 4); best = misfit(rec, T, model, H)
    for it in range(iters):
        mv = rng.choice(3); new = rec
        if mv == 0 or len(rec) <= n_fixed:                   # add
            c = P.clip(propose(), H)
            c = c[[not any(V.crosses(t[None], l)[0] for l in lines) for t in c]] if len(c) else c
            if not len(c): continue
            new = np.vstack([rec, c[:1]])
        elif mv == 1:                                       # delete a tier 3 trace
            j = rng.integers(n_fixed, len(rec)); new = np.delete(rec, j, 0)
        else:                                               # replace
            c = P.clip(propose(), H)
            c = c[[not any(V.crosses(t[None], l)[0] for l in lines) for t in c]] if len(c) else c
            if not len(c): continue
            j = rng.integers(n_fixed, len(rec)); new = rec.copy(); new[j] = c[0]
        m = misfit(new, T, model, H)
        if m < best:
            rec, best = new, m
    return rec
