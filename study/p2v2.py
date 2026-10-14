# -*- coding: utf-8 -*-
"""Paper 2 full study engine (window scale). Extends p2lib (first round, unchanged) with:
  * survey types: vertical scanlines, horizontal lines, both; random layout shifts
  * observation realism: missed hits, direction noise, set mislabel, position error; known-miss-rate correction
  * simulated domain shift: held-out window rotated by a fixed angle (training unrotated)
  * fillers: UNIF (training statistics), LOCAL (directions from the scanline hits when >= 3 hits), EVAE (Paper 1 stored),
    STAT (no evidence: training statistics, training P21, crossing limit), RAW_EVAE (Paper 1, no evidence)
  * measures added: set shares and unclustered traces (fold's Paper 1 cluster model), uncensored lengths, top-to-bottom
    connection, errors by confidence zone (measured / indicated / inferred) and by distance from the nearest scanline,
    1 m presence-probability grids for calibration
All lengths in metres; window 20 m wide x H."""
import math, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path[:0] = [str(HERE)]
import p2lib as P
sys.path.insert(0, str(P.REL / 'src' / 'study'))
import setbg  # Paper 1 learned set model (sets + background)

W = P.M


# ------------------------------------------------------------------ survey lines
def lines_layout(kind, spacing, width, H, rng=None, shift=False, nh=2):
    """list of dicts {o:'v'|'h', lo, hi, c}. kind: v (vertical scanlines), h (nh horizontal lines), vh (both).
    spacing None -> one vertical centre line."""
    out = []
    if 'v' in kind:
        ap = P.apertures(spacing, width)
        if shift and rng is not None:
            s = spacing or W
            d = rng.uniform(-s / 2, s / 2)
            ap = [(min(max(lo + d, 0.02), W - width - 0.02), min(max(lo + d, 0.02), W - width - 0.02) + width) for lo, hi in ap]
        out += [dict(o='v', lo=lo, hi=hi, c=(lo + hi) / 2) for lo, hi in ap]
    if 'h' in kind:
        for k in range(nh):
            c = H * (k + 1) / (nh + 1)
            out.append(dict(o='h', lo=c - width / 2, hi=c + width / 2, c=c))
    return out


def crosses(L, ln):
    L = np.asarray(L, float).reshape(-1, 4)
    a, b = (0, 2) if ln['o'] == 'v' else (1, 3)
    return (np.maximum(L[:, a], L[:, b]) >= ln['lo']) & (np.minimum(L[:, a], L[:, b]) <= ln['hi'])


def observe(gt, lines, H, rng, miss=0.0, noise=0.0, mislabel=0.0, pos_err=0.0, tr_th=None, model=None):
    obs = []
    for j, ln in enumerate(lines):
        for i in np.where(crosses(gt, ln))[0]:
            if rng.random() < miss:
                continue
            x1, y1, x2, y2 = gt[i]
            th = math.atan2(y2 - y1, x2 - x1) % math.pi
            if ln['o'] == 'v':
                t = np.clip((ln['c'] - x1) / (x2 - x1), 0, 1) if abs(x2 - x1) > 1e-9 else 0.5
                x, y = ln['c'], y1 + t * (y2 - y1)
                y = float(np.clip(y + rng.normal(0, pos_err), 0, H)) if pos_err else y
            else:
                t = np.clip((ln['c'] - y1) / (y2 - y1), 0, 1) if abs(y2 - y1) > 1e-9 else 0.5
                x, y = x1 + t * (x2 - x1), ln['c']
                x = float(np.clip(x + rng.normal(0, pos_err), 0, W)) if pos_err else x
            if noise:
                th = (th + math.radians(rng.normal(0, noise))) % math.pi
            if mislabel and rng.random() < mislabel and tr_th is not None and model is not None:
                own = setbg.classify(np.array([th]), model)[0]
                cls = setbg.classify(tr_th, model)
                pool = tr_th[cls != own]
                if len(pool):
                    th = float(rng.choice(pool))
            obs.append(dict(ap=j, o=ln['o'], x=x, y=y, th=th, tid=int(i), lo=ln['lo'], hi=ln['hi']))
    return obs


def tier1(obs):
    out = []
    for o in obs:
        w = (o['hi'] - o['lo']) / 2
        c, s = math.cos(o['th']), math.sin(o['th'])
        d = abs(c) if o['o'] == 'v' else abs(s)
        if d < 1e-6:
            continue
        t = min(w / d, 1.0)
        out.append([o['x'] - t * c, o['y'] - t * s, o['x'] + t * c, o['y'] + t * s])
    return np.array(out, float).reshape(-1, 4)


def merge(obs, dth=6.0, dperp=0.35):
    groups = []
    for o in sorted(obs, key=lambda q: (q['x'], q['y'])):
        placed = False
        for g in groups:
            r = g[-1]
            if any(q['ap'] == o['ap'] for q in g):
                continue
            if abs(((o['th'] - r['th']) + math.pi / 2) % math.pi - math.pi / 2) > math.radians(dth):
                continue
            ux, uy = math.cos(r['th']), math.sin(r['th'])
            if abs(-uy * (o['x'] - r['x']) + ux * (o['y'] - r['y'])) <= dperp:
                g.append(o); placed = True; break
        if not placed:
            groups.append([o])
    return groups


def tier2(groups, tr_th, tr_len, H, rng, dwin=15.0):
    out = []
    for g in groups:
        th = math.atan2(sum(math.sin(2 * o['th']) for o in g), sum(math.cos(2 * o['th']) for o in g)) / 2 % math.pi
        ux, uy = math.cos(th), math.sin(th)
        s = np.array([o['x'] * ux + o['y'] * uy for o in g]); span = s.max() - s.min()
        d = np.abs(((tr_th - th) + math.pi / 2) % math.pi - math.pi / 2)
        pool = tr_len[d <= math.radians(dwin)]
        if len(pool) < 5:
            pool = tr_len
        pool = pool[pool >= span] if (pool >= span).any() else np.array([span * 1.1 + 0.2])
        Lf = float(rng.choice(pool, p=pool / pool.sum()))
        a = rng.uniform(0, Lf - span)
        o0 = g[int(np.argmin(s))]
        p1 = (o0['x'] - a * ux, o0['y'] - a * uy)
        out.append([p1[0], p1[1], p1[0] + Lf * ux, p1[1] + Lf * uy])
    return P.clip(np.array(out, float).reshape(-1, 4), H)


def budget(obs, lines, H, tr_th, tr_len, prior, a_prior=2.0, use=True, miss_known=0.0):
    if not use:
        return prior * W * H, prior
    w = tr_len / tr_len.sum()
    ev = float(np.sum(w * np.abs(np.cos(tr_th))))
    eh = float(np.sum(w * np.abs(np.sin(tr_th))))
    expo = sum((H if ln['o'] == 'v' else W) * (ev if ln['o'] == 'v' else eh) for ln in lines)   # P21 -> expected hits
    n = len(obs) / max(1.0 - miss_known, 1e-3)
    p21 = (n + a_prior) / (expo + (a_prior / prior if a_prior > 0 else 0.0))
    return p21 * W * H, p21


def ncross_new(t, rec):
    return int(P.common.seg_cross_one(t, rec).sum()) if len(rec) else 0


def fill(pool, lines, rec, tiers, bud, H, rng, xmax=None, gate=True):
    have = P.seglen(rec).sum(); X = len(P.crossing_pairs(rec)) if len(rec) > 1 else 0
    Pp = P.clip(pool, H)
    for i in rng.permutation(len(Pp)):
        if have >= bud:
            break
        t = Pp[i]
        if gate and any(crosses(t[None], ln)[0] for ln in lines):
            continue
        if xmax is not None:
            nx = ncross_new(t, rec)
            if nx > 0 and X + nx > xmax:
                continue
            X += nx
        rec = np.vstack([rec, t[None]]) if len(rec) else t[None]; tiers.append(3); have += P.seglen(t[None])[0]
    return rec, tiers


def local_proposals(n, obs, tr_th, tr_len, H, rng, p_local=0.7, dwin=15.0):
    """like UNIF but the direction is drawn from the scanline hits (with a small spread) with probability p_local
    when there are >= 3 hits; length from training traces of similar direction."""
    hth = np.array([o['th'] for o in obs])
    c = np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H, n)])
    out = []
    for k in range(n):
        if len(hth) >= 3 and rng.random() < p_local:
            th = (rng.choice(hth) + math.radians(rng.normal(0, 5))) % math.pi
        else:
            th = tr_th[rng.integers(len(tr_th))]
        d = np.abs(((tr_th - th) + math.pi / 2) % math.pi - math.pi / 2)
        pool = tr_len[d <= math.radians(dwin)]
        ln = float(rng.choice(pool if len(pool) >= 5 else tr_len))
        out.append([c[k, 0] - math.cos(th) * ln / 2, c[k, 1] - math.sin(th) * ln / 2, c[k, 0] + math.cos(th) * ln / 2, c[k, 1] + math.sin(th) * ln / 2])
    return np.array(out)


def rotate(L, H, deg):
    if not deg:
        return L
    a = math.radians(deg); cx, cy = W / 2, H / 2
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p1 = (L[:, :2] - [cx, cy]) @ R.T + [cx, cy]; p2 = (L[:, 2:] - [cx, cy]) @ R.T + [cx, cy]
    return P.clip(np.hstack([p1, p2]), H)


# ------------------------------------------------------------------ extra measures
def presence(L, H, cell=1.0):
    nx, ny = int(round(W / cell)), max(1, int(round(H / cell)))
    g = np.zeros((ny, nx), bool)
    for x1, y1, x2, y2 in L:
        k = max(2, int(math.hypot(x2 - x1, y2 - y1) / 0.1))
        t = np.linspace(0, 1, k)
        ix = np.clip(((x1 + t * (x2 - x1)) / (W / nx)).astype(int), 0, nx - 1)
        iy = np.clip(((y1 + t * (y2 - y1)) / (H / ny)).astype(int), 0, ny - 1)
        g[iy, ix] = True
    return g


def percolates(L, H, tol=0.3):
    """True if one connected group of traces touches both the top and the bottom of the window."""
    n = len(L)
    if n == 0:
        return False
    par = list(range(n))
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j in P.crossing_pairs(L):
        par[f(i)] = f(j)
    if n > 1:
        E = np.vstack([L[:, :2], L[:, 2:]]); own = np.r_[np.arange(n), np.arange(n)]
        D = P._pt_seg(E, L[:, :2], L[:, 2:]); D[np.arange(2 * n), own] = np.inf
        for e in np.where(D.min(1) <= tol)[0]:
            par[f(own[e])] = f(int(D[e].argmin()))
    top = {f(i) for i in range(n) if min(L[i, 1], L[i, 3]) <= tol}
    bot = {f(i) for i in range(n) if max(L[i, 1], L[i, 3]) >= H - tol}
    return bool(top & bot)


def set_measures(L, model):
    th = P.segdir(L)
    if len(th) == 0:
        return np.full(model['K'] + 1, np.nan), np.zeros(0)
    c = setbg.classify(th, model)
    return np.bincount(c, minlength=model['K'] + 1) / len(c), c


def uncensored(L, H, e=0.2):
    if not len(L):
        return np.zeros(0)
    ok = (np.minimum(L[:, 0], L[:, 2]) > e) & (np.maximum(L[:, 0], L[:, 2]) < W - e) & \
         (np.minimum(L[:, 1], L[:, 3]) > e) & (np.maximum(L[:, 1], L[:, 3]) < H - e)
    return P.seglen(L[ok])


def zone_grid(lines, t2, H, cell=2.0):
    """0 measured (cell overlaps a scanline), 1 indicated (an indicated trace passes), 2 inferred."""
    nx, ny = int(round(W / cell)), max(1, int(round(H / cell)))
    z = np.full((ny, nx), 2)
    ind = presence(t2, H, cell) if len(t2) else np.zeros((ny, nx), bool)
    z[ind] = 1
    xs = (np.arange(nx) + 0.5) * W / nx; ys = (np.arange(ny) + 0.5) * H / ny
    for ln in lines:
        if ln['o'] == 'v':
            z[:, np.abs(xs - ln['c']) <= W / nx / 2 + 1e-9] = 0
        else:
            z[np.abs(ys - ln['c']) <= H / ny / 2 + 1e-9, :] = 0
    return z


def dist_grid(lines, H, cell=2.0):
    nx, ny = int(round(W / cell)), max(1, int(round(H / cell)))
    xs = (np.arange(nx) + 0.5) * W / nx; ys = (np.arange(ny) + 0.5) * H / ny
    X, Y = np.meshgrid(xs, ys); D = np.full(X.shape, np.inf)
    for ln in lines:
        D = np.minimum(D, np.abs(X - ln['c']) if ln['o'] == 'v' else np.abs(Y - ln['c']))
    return D


def p21_grid(L, H, cell=2.0):
    return P.p21_map(L, H, cell=cell)


BANDS = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 99)]


def extra_scores(rec, tiers, gt, H, lines, model, gpre):
    """confidence-zone, distance-band, set and other added measures for one realisation."""
    out = {}
    s_rec, c_rec = set_measures(rec, model)
    out['set_share_L1'] = float(np.nansum(np.abs(s_rec - gpre['shares']))) if len(rec) else np.nan
    out['uncl_share_err'] = abs(s_rec[-1] - gpre['shares'][-1]) if len(rec) else np.nan
    lu, gu = P.seglen(rec[c_rec == model['K']]) if len(rec) else np.zeros(0), gpre['uncl_len']
    out['uncl_len_W1'] = P.common.w1(lu, gu) if len(lu) and len(gu) else np.nan
    uc = uncensored(rec, H)
    out['len_W1_uncens'] = P.common.w1(uc, gpre['uncens']) if len(uc) and len(gpre['uncens']) else np.nan
    out['perc_mismatch'] = float(percolates(rec, H) != gpre['perc'])
    g2 = p21_grid(rec, H)
    z = gpre['zone']; D = gpre['dist']
    for k, nm in enumerate(('meas', 'ind', 'inf')):
        m = z == k
        out[f'zone_{nm}_rmse'] = float(np.sqrt(np.mean((g2[m] - gpre['p21g'][m]) ** 2))) if m.any() else np.nan
        out[f'zone_{nm}_frac'] = float(m.mean())
    for lo, hi in BANDS:
        m = (D >= lo) & (D < hi)
        out[f'band_{lo}_{hi}_rmse'] = float(np.sqrt(np.mean((g2[m] - gpre['p21g'][m]) ** 2))) if m.any() else np.nan
    # trace-level: indicated traces vs the true trace they extend (tier 2 only)
    return out, presence(rec, H)


def gt_pre(gt, H, lines, t2, model):
    shares, c = set_measures(gt, model)
    return dict(shares=shares, uncl_len=P.seglen(gt[c == model['K']]) if len(gt) else np.zeros(0),
                uncens=uncensored(gt, H), perc=percolates(gt, H), zone=zone_grid(lines, t2, H), dist=dist_grid(lines, H),
                p21g=p21_grid(gt, H))


# ------------------------------------------------------------------ geological termination (added after round 1)
def pass_prob(train_windows, H_list):
    """share of encounters between two traces that are crossings (X) rather than terminations (Y), from the training
    windows (Paper 1 node definitions, TOL 0.3 m, ends near the window edge not counted)."""
    nX = nY = 0
    for L, H in zip(train_windows, H_list):
        if len(L) < 2:
            continue
        t = P.topo(L, H)
        tot_ends = 2 * len(L)
        nX += t['nX']
        nY += int(round(t['pY'] * (t['nX'] / max(t['pX'], 1e-9)))) if t['pX'] and t['pX'] > 0 else 0
    return nX / max(nX + nY, 1)


def terminate(rec, keep_n, rng, p_pass, protect=None):
    """walk through traces keep_n.. in random order; where a trace crosses an earlier trace outside its protected
    span, it passes with probability p_pass, otherwise it is cut there (it ends on the other trace, a Y node).
    protect: list of (s_min, s_max) arc-length intervals (from the first endpoint) that must stay (observed span)."""
    rec = np.array(rec, float).reshape(-1, 4)
    keep_n = min(keep_n, len(rec))
    order = list(range(keep_n)) + list(keep_n + rng.permutation(len(rec) - keep_n))
    done = []
    for i in order:
        x1, y1, x2, y2 = rec[i]; L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-6 or not done:
            done.append(i); continue
        ps = protect[i] if protect is not None and i < len(protect) and protect[i] is not None else (L / 2, L / 2)
        lo, hi = 0.0, L
        for j in done:
            q = rec[j]
            p, r = np.array([x1, y1]), np.array([x2 - x1, y2 - y1]); qq, s = q[:2], q[2:] - q[:2]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12:
                continue
            d = qq - p; t = (d[0] * s[1] - d[1] * s[0]) / den; u = (d[0] * r[1] - d[1] * r[0]) / den
            if not (0 < t < 1 and 0 < u < 1):
                continue
            a = t * L
            if ps[0] <= a <= ps[1] or rng.random() < p_pass:
                continue
            if a < ps[0]:
                lo = max(lo, a)
            else:
                hi = min(hi, a)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        rec[i] = [x1 + lo * ux, y1 + lo * uy, x1 + hi * ux, y1 + hi * uy]
        done.append(i)
    return rec


# ------------------------------------------------------------------ duplicate / overlap filter (round 4)
def _pair_geom(a, b):
    """direction difference (rad), perpendicular distance of b's midpoint to a's line (m), overlap along a as a
    share of the shorter trace."""
    ax, ay, bx, by = a; L = math.hypot(bx - ax, by - ay)
    if L < 1e-9:
        return math.pi, 1e9, 0.0
    ux, uy = (bx - ax) / L, (by - ay) / L
    t1 = math.atan2(by - ay, bx - ax) % math.pi; t2 = math.atan2(b[3] - b[1], b[2] - b[0]) % math.pi
    dth = abs(((t1 - t2) + math.pi / 2) % math.pi - math.pi / 2)
    mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dperp = abs(-uy * (mx - ax) + ux * (my - ay))
    s1 = (b[0] - ax) * ux + (b[1] - ay) * uy; s2 = (b[2] - ax) * ux + (b[3] - ay) * uy
    ov = max(0.0, min(L, max(s1, s2)) - max(0.0, min(s1, s2)))
    Lb = math.hypot(b[2] - b[0], b[3] - b[1])
    return dth, dperp, ov / max(min(L, Lb), 1e-9)


def near_parallel_gaps(windows, dth_deg=5.0, min_ov=0.3):
    """perpendicular gaps between overlapping near-parallel trace pairs in real windows"""
    out = []
    for L in windows:
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                dth, dp, ov = _pair_geom(L[i], L[j])
                if dth <= math.radians(dth_deg) and ov >= min_ov:
                    out.append(dp)
    return np.array(out)


def is_dup(t, rec, dth, dmin, min_ov=0.3):
    for q in rec:
        a, b = _pair_geom(q, t), _pair_geom(t, q)
        if a[0] <= dth and min(a[1], b[1]) < dmin and max(a[2], b[2]) >= min_ov:
            return True
    return False


def dedup(L, dth, dmin, min_ov=0.3):
    """keep traces longest first; drop a trace that duplicates a kept one (near-parallel, closer than dmin, overlapping)"""
    L = np.asarray(L, float).reshape(-1, 4)
    keep = []
    for i in np.argsort(-P.seglen(L)):
        if not is_dup(L[i], [L[k] for k in keep], dth, dmin, min_ov):
            keep.append(i)
    return L[sorted(keep)]
