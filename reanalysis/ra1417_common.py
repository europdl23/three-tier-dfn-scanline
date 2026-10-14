# -*- coding: utf-8 -*-
r"""ra1417_common.py: shared code of RA-14, RA-15, RA-16 and RA-17 (observation density and observation realism).

Read-only use of the revision study (<local path>
release. Nothing is written into either folder: run Python with -B (PYTHONDONTWRITEBYTECODE=1); every output goes to
03_REANALYSIS\results\RA14_15_16_17\.

Contents
  * frozen_g    copy of final_eval.frozen (lines 58-79) that also returns the number of indicated (tier-2) traces and
                adds three options: known-rate correction of R2 (budget miss_known), detection-aware R1 (an inferred
                trace crossing k strips is kept with probability m_hat^k) and inclined lines with known geometry
                (R1 and R2 on the true line axes). With the default options it must reproduce final_eval.frozen exactly
                (checked by ra1417_validate / the batch-1 comparison with final_mainC_rows.csv and final_main_rows.csv).
  * observe_g   observation with misses, structured misses, contiguous gaps, position noise, direction noise,
                cluster mislabelling, inclined lines (known or unknown deviation) and false hits.
  * layouts     vertical (p2lib.apertures), mixed vertical + horizontal (p2v2.lines_layout), inclined, irregular.
  * score_row   final_eval.rec_row equivalent (Paper 1 statistical_tests.score + p21map_err, inf -> NaN).
  * local       RA-17 local measures (point samples, distance bands, tier zones, weighted circular W1).
"""
import sys, math, zlib
from pathlib import Path

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parents[1] / 'study'
REL = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'work' / 'RA14_15_16_17'
SCRIPTS = PROJECT / 'reanalysis'
for p in (str(STUDY), str(REL / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                     # noqa: E402
import p2lib as P                      # noqa: E402
import p2v2 as V                       # noqa: E402
import final_eval as FE                # noqa: E402  (module import only; main() is never called)
import tr_common as TC                 # noqa: E402
import rules2 as RU                    # noqa: E402
from run_v2 import sd                  # noqa: E402

W = P.M                                # 20 m window width
BASE = dict(FE.BASE)

STUDY_MODULES = ['p2lib.py', 'p2v2.py', 'final_eval.py', 'tr_common.py', 'rules2.py', 'run_v2.py', 'run_extra.py',
                 'run_v3_term.py', 'run_dd.py', 'run_tests.py', 'face.py']
REL_MODULES = ['scripts/statistical_tests.py', 'src/study/study.py', 'src/study/setbg.py', 'src/study/topology.py',
               'src/evaluation/common.py', 'src/evaluation/generators.py', 'src/repo_paths.py']


def module_paths():
    return [STUDY / m for m in STUDY_MODULES] + [REL / m for m in REL_MODULES]


def check_W():
    """face.py sets p2v2.W to the face width; window-scale work needs 20 m."""
    if V.W != W:
        V.W = W


# ============================================================================================ geometry of line strips
def incl_line(c, ang_deg, H, width):
    """inclined line collared at (c, 0) on the top edge, angle ang_deg from the vertical (positive: x grows with depth),
    axis clipped where it leaves the window [0, 20] x [0, H]."""
    a = math.radians(ang_deg)
    ux, uy = math.sin(a), math.cos(a)                  # unit vector down the line
    smax = H / uy
    if ux > 1e-12:
        smax = min(smax, (W - c) / ux)
    elif ux < -1e-12:
        smax = min(smax, (0.0 - c) / ux)
    smax = max(smax, 0.0)
    return dict(o='i', c=c, x0=c, y0=0.0, x1=c + smax * ux, y1=smax * uy, ux=ux, uy=uy, L=smax, ang=ang_deg,
                beta=math.atan2(uy, ux), w=width, lo=c - width / 2, hi=c + width / 2)


def _seg_seg_dist(L, A, B):
    """minimum distance between each segment of L (n x 4) and the segment A-B; also returns, for each segment, the
    closest point on A-B (as the arc length along A-B) and whether they intersect."""
    L = np.asarray(L, float).reshape(-1, 4)
    p, q = L[:, :2], L[:, 2:]
    A = np.asarray(A, float); B = np.asarray(B, float)
    d = B - A; Ld = math.hypot(d[0], d[1])
    n = len(L)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, bool)
    r = q - p
    den = r[:, 0] * d[1] - r[:, 1] * d[0]
    w0 = A[None, :] - p
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w0[:, 0] * d[1] - w0[:, 1] * d[0]) / den          # along L
        u = (w0[:, 0] * r[:, 1] - w0[:, 1] * r[:, 0]) / den    # along A-B
    inter = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    # candidate distances: endpoints of L to A-B, A and B to L
    def pt_seg(Pt, S0, S1):
        dd = S1 - S0; l2 = np.maximum((dd ** 2).sum(-1), 1e-18)
        tt = np.clip(((Pt - S0) * dd).sum(-1) / l2, 0, 1)
        proj = S0 + tt[..., None] * dd
        return np.linalg.norm(Pt - proj, axis=-1), tt
    dp, up = pt_seg(p, A[None], B[None])
    dq, uq = pt_seg(q, A[None], B[None])
    dA, tA = pt_seg(A[None].repeat(n, 0), p, q)
    dB, tB = pt_seg(B[None].repeat(n, 0), p, q)
    D = np.column_stack([dp, dq, dA, dB])
    k = D.argmin(1)
    dist = D[np.arange(n), k]
    s_on_axis = np.where(k == 0, up, np.where(k == 1, uq, np.where(k == 2, 0.0, 1.0))) * Ld
    s_on_axis = np.where(inter, np.clip(u, 0, 1) * Ld, s_on_axis)
    dist = np.where(inter, 0.0, dist)
    return dist, s_on_axis, inter


def crosses_g(L, ln):
    """True where a segment overlaps the strip of the line (vertical, horizontal or inclined)."""
    if ln['o'] in ('v', 'h'):
        return V.crosses(L, ln)
    L = np.asarray(L, float).reshape(-1, 4)
    if ln['L'] <= 1e-9:
        return np.zeros(len(L), bool)
    dist, _, _ = _seg_seg_dist(L, (ln['x0'], ln['y0']), (ln['x1'], ln['y1']))
    return dist <= ln['w'] / 2 + 1e-12


def n_crossed(t, lines):
    return int(sum(bool(crosses_g(t[None], ln)[0]) for ln in lines))


# ============================================================================================ layouts
def layout(name, H, width=0.10, rng=None, incl_deg=0.0, jitter=0.0):
    """name: 'centre', '10m', '7m', '5m', '3m', '5m+h2', 'centre+h2'. Optional inclination (uniform within
    +-incl_deg per line, drawn from rng) or per-line jitter (uniform within +-jitter m, drawn from rng)."""
    sp = {'centre': None, '10m': 10.0, '7m': 7.0, '5m': 5.0, '3m': 3.0, '5m+h2': 5.0, 'centre+h2': None}[name]
    kind = 'vh' if name.endswith('+h2') else 'v'
    lines = V.lines_layout(kind, sp, width, H, nh=2)
    if jitter:
        out = []
        for ln in lines:
            if ln['o'] == 'v':
                c = ln['c'] + rng.uniform(-jitter, jitter)
                c = min(max(c, width / 2 + 0.02), W - width / 2 - 0.02)
                out.append(dict(o='v', lo=c - width / 2, hi=c + width / 2, c=c))
            else:
                out.append(ln)
        lines = out
    if incl_deg:
        out = []
        for ln in lines:
            if ln['o'] == 'v':
                out.append(incl_line(ln['c'], rng.uniform(-incl_deg, incl_deg), H, width))
            else:
                out.append(ln)
        lines = out
    return lines


def effort(lines, H):
    """total line length in the window (m)."""
    tot = 0.0
    for ln in lines:
        tot += H if ln['o'] == 'v' else (W if ln['o'] == 'h' else ln['L'])
    return tot


# ============================================================================================ observation with errors
def draw_gaps(Lline, frac, seg_lo, seg_hi, rng):
    """contiguous gap intervals along a line of length Lline: segments U(seg_lo, seg_hi) at uniform positions until
    their union reaches frac * Lline; the last segment is shortened so that the union equals the target exactly."""
    target = frac * Lline
    if target <= 0 or Lline <= 0:
        return []
    iv = []
    def union_len(v):
        if not v:
            return 0.0
        v = sorted(v); tot = 0.0; a, b = v[0]
        for x, y in v[1:]:
            if x > b:
                tot += b - a; a, b = x, y
            else:
                b = max(b, y)
        return tot + (b - a)
    for _ in range(1000):
        ln = min(rng.uniform(seg_lo, seg_hi), Lline)
        s0 = rng.uniform(0, Lline - ln)
        cur = union_len(iv)
        new = union_len(iv + [(s0, s0 + ln)])
        if new >= target - 1e-12:
            need = target - cur
            # shorten the new segment from its end until the union equals the target (bisection on the length)
            lo_, hi_ = 0.0, ln
            for _ in range(60):
                mid = (lo_ + hi_) / 2
                if union_len(iv + [(s0, s0 + mid)]) < cur + need:
                    lo_ = mid
                else:
                    hi_ = mid
            iv.append((s0, s0 + hi_))
            break
        iv.append((s0, s0 + ln))
    return iv


def in_gaps(s, iv):
    return any(a <= s <= b for a, b in iv)


def observe_g(gt, lines, H, rng, *, miss=0.0, struct=None, gaps=None, pos_err=0.0, noise=0.0, mislabel=0.0,
              tr_th=None, model=None, full=None, false_frac=0.0, false_dirs=None, unknown_dev=False):
    """Scanline observation of the mapped window with errors.

    lines: dicts with o = 'v', 'h' (p2v2 conventions) or 'i' (incl_line). For 'v' lines without errors the hits equal
    p2v2.observe exactly (x = line centre, y on the trace or at its nearest end).
    Order per line and true crossing: gap check, random miss, structured miss, position noise along the line,
    direction noise, mislabel. Then false hits. Each hit: dict(ap, o, x, y, th, tid, lo, hi, s).
    struct: None or ('angle', a) [p_miss = min(1, a cos^2 phi)] or ('length', b) [p_miss = min(1, b exp(-l / 5))],
        phi = acute angle between trace and line, l = whole-trace length (full[i]).
    gaps: None or (frac, seg_lo, seg_hi).
    unknown_dev: for inclined lines, report each hit on the nominal vertical line (x = collar, y = along-line distance)
        with its direction rotated into the believed vertical frame (theta + deviation angle)."""
    obs = []
    cls_tr = None
    if mislabel and tr_th is not None and model is not None:
        cls_tr = V.setbg.classify(tr_th, model)
    gaps_by_line = []
    for j, ln in enumerate(lines):
        Lline = H if ln['o'] == 'v' else (W if ln['o'] == 'h' else ln['L'])
        iv = draw_gaps(Lline, gaps[0], gaps[1], gaps[2], rng) if gaps else []
        gaps_by_line.append(iv)
        idx = np.where(crosses_g(gt, ln))[0]
        if ln['o'] == 'i' and len(idx):
            _, s_all, _ = _seg_seg_dist(gt[idx], (ln['x0'], ln['y0']), (ln['x1'], ln['y1']))
        for kk, i in enumerate(idx):
            x1, y1, x2, y2 = gt[i]
            th = math.atan2(y2 - y1, x2 - x1) % math.pi
            if ln['o'] == 'v':
                t = np.clip((ln['c'] - x1) / (x2 - x1), 0, 1) if abs(x2 - x1) > 1e-9 else 0.5
                x, y = ln['c'], y1 + t * (y2 - y1); s = y
                beta = math.pi / 2
            elif ln['o'] == 'h':
                t = np.clip((ln['c'] - y1) / (y2 - y1), 0, 1) if abs(y2 - y1) > 1e-9 else 0.5
                x, y = x1 + t * (x2 - x1), ln['c']; s = x
                beta = 0.0
            else:
                s = float(s_all[kk]); x, y = ln['x0'] + s * ln['ux'], ln['y0'] + s * ln['uy']
                beta = ln['beta']
            if iv and in_gaps(s, iv):
                continue
            if miss and rng.random() < miss:
                continue
            if struct is not None:
                if struct[0] == 'angle':
                    dphi = abs(((th - beta) + math.pi / 2) % math.pi - math.pi / 2)   # 0..pi/2, 0 = parallel
                    pm = min(1.0, struct[1] * math.cos(dphi) ** 2)
                else:
                    l = float(full[i]) if full is not None and np.isfinite(full[i]) else math.hypot(x2 - x1, y2 - y1)
                    pm = min(1.0, struct[1] * math.exp(-l / 5.0))
                if rng.random() < pm:
                    continue
            if pos_err:
                s = float(np.clip(s + rng.normal(0, pos_err), 0, Lline))
                if ln['o'] == 'v':
                    y = s
                elif ln['o'] == 'h':
                    x = s
                else:
                    x, y = ln['x0'] + s * ln['ux'], ln['y0'] + s * ln['uy']
            if noise:
                th = (th + math.radians(rng.normal(0, noise))) % math.pi
            if mislabel and rng.random() < mislabel and cls_tr is not None:
                own = V.setbg.classify(np.array([th]), model)[0]
                pool = tr_th[cls_tr != own]
                if len(pool):
                    th = float(rng.choice(pool))
            o = ln['o']
            if unknown_dev and ln['o'] == 'i':
                x, y = ln['c'], float(np.clip(s, 0, H))
                th = (th + math.radians(ln['ang'])) % math.pi
                o = 'v'
            obs.append(dict(ap=j, o=o, x=x, y=y, th=th, tid=int(i), lo=ln['lo'], hi=ln['hi'], s=s))
    if false_frac and len(lines):
        nf = int(round(false_frac * len(obs)))
        for k in range(nf):
            j = int(rng.integers(len(lines))); ln = lines[j]
            Lline = H if ln['o'] == 'v' else (W if ln['o'] == 'h' else ln['L'])
            s = rng.uniform(0, Lline)
            if ln['o'] == 'v':
                x, y = ln['c'], s
            elif ln['o'] == 'h':
                x, y = s, ln['c']
            else:
                x, y = ln['x0'] + s * ln['ux'], ln['y0'] + s * ln['uy']
            th = float(false_dirs[rng.integers(len(false_dirs))])
            obs.append(dict(ap=j, o=ln['o'], x=x, y=y, th=th, tid=-(k + 1), lo=ln['lo'], hi=ln['hi'], s=s))
    return obs


def believed_lines(lines, H, width=0.10):
    """the vertical lines the method believes in when the deviation is unknown (collar position, vertical)."""
    out = []
    for ln in lines:
        if ln['o'] == 'i':
            out.append(dict(o='v', lo=ln['c'] - width / 2, hi=ln['c'] + width / 2, c=ln['c']))
        else:
            out.append(ln)
    return out


# ============================================================================================ frozen method (copy)
def budget_g(obs, lines, H, tr_th, tr_len, prior, a_prior=2.0, miss_known=0.0):
    """R2. Equals p2v2.budget for vertical and horizontal lines; inclined lines use their axis length and the
    length-weighted E|sin(theta - line direction)|."""
    if all(ln['o'] in ('v', 'h') for ln in lines):
        return V.budget(obs, lines, H, tr_th, tr_len, prior, a_prior, miss_known=miss_known)
    w = tr_len / tr_len.sum()
    ev = float(np.sum(w * np.abs(np.cos(tr_th)))); eh = float(np.sum(w * np.abs(np.sin(tr_th))))
    expo = 0.0
    for ln in lines:
        if ln['o'] == 'v':
            expo += H * ev
        elif ln['o'] == 'h':
            expo += W * eh
        else:
            expo += ln['L'] * float(np.sum(w * np.abs(np.sin(tr_th - ln['beta']))))
    n = len(obs) / max(1.0 - miss_known, 1e-3)
    p21 = (n + a_prior) / (expo + (a_prior / prior if a_prior > 0 else 0.0))
    return p21 * W * H, p21


def make_gate(lines, r1='std', m_hat=0.0, rng=None):
    """R1 gate: returns f(t) -> True if the inferred trace t may be kept.
    std: no trace through any strip (identical to final_eval.fill for vertical/horizontal lines; no random call).
    da : detection-aware, kept with probability m_hat^k for k crossed strips (one random call when k > 0)."""
    std_vh = all(ln['o'] in ('v', 'h') for ln in lines)
    if r1 == 'std':
        if std_vh:
            return lambda t: not any(V.crosses(t[None], ln)[0] for ln in lines)
        return lambda t: not any(crosses_g(t[None], ln)[0] for ln in lines)
    def gate(t):
        k = n_crossed(t, lines)
        if k == 0:
            return True
        if m_hat <= 0:
            return False
        return rng.random() < m_hat ** k
    return gate


def fill_g(pool, rec, bud, H, rng, xmax, dmin, prm, gate):
    """copy of final_eval.fill (lines 41-55) with the R1 test replaced by gate(t)."""
    have = P.seglen(rec).sum(); X = len(P.crossing_pairs(rec)) if len(rec) > 1 else 0
    Pp = P.clip(pool, H)
    dth = math.radians(prm['DUP_DTH'])
    if len(Pp):
        Pp = V.dedup(Pp, dth, dmin, prm['DUP_OV'])
    for i in rng.permutation(len(Pp)):
        if have >= bud: break
        t = Pp[i]
        if not gate(t): continue                                                           # R1
        if len(rec) and V.is_dup(t, rec, dth, dmin, prm['DUP_OV']): continue             # R4
        nx = int(P.common.seg_cross_one(t, rec).sum()) if len(rec) else 0
        if nx > 0 and X + nx > xmax: continue                                             # R3
        X += nx; rec = np.vstack([rec, t[None]]) if len(rec) else t[None]; have += P.seglen(t[None])[0]
    return rec


def frozen_g(C, obs, lines, H, rng, pool_fn, prm, dmin_cache, clean=False, miss_known=0.0, r1='std', m_hat=0.0):
    """copy of final_eval.frozen (lines 58-79) returning (rec, number of indicated traces, budget, P21 estimate).
    Options: miss_known (R2 known-rate correction), r1 'std' or 'da' with m_hat, inclined lines in `lines`."""
    groups = V.merge(obs, prm['JOIN_DTH'], prm['JOIN_DPERP'])
    t2 = (V.tier2(groups, C['th2'], C['tr_len2d'], H, rng, prm['LEN_DWIN']) if prm['LEN_BIAS']
          else FE._tier2_nobias(groups, C['th2'], C['tr_len2d'], H, rng, prm['LEN_DWIN']))
    bud, p21h = budget_g(obs, lines, H, C['tr_th'], C['tr_len3d'], C['prior'], prm['A_PRIOR'], miss_known)   # R2
    xmax = C['xrate'] * 20 * H * (p21h / C['prior']) ** 2 * prm['XFAC']
    dmin = dmin_cache.setdefault(prm['DUP_PCT'], float(np.percentile(V.near_parallel_gaps(C['TW']), prm['DUP_PCT'])))
    gate = make_gate(lines, r1, m_hat, rng)
    rec = t2
    for k in range(24):
        if P.seglen(rec).sum() >= bud: break
        rec = fill_g(pool_fn(k), rec, bud, H, rng, xmax, dmin, prm, gate)
    if clean:                                                         # R5 short iterative clean-up (round 6)
        if 'n_rate' not in C:
            A = sum(20 * h for h in C['TH']); C['n_rate'] = sum(len(w) for w in C['TW']) / A
            C['pY'] = float(np.nanmean([P.topo(w, h)['pY'] for w, h in zip(C['TW'], C['TH']) if len(w) > 1]))
        T = dict(p21=p21h, n=C['n_rate'] * 20 * H * (p21h / C['prior']), X=xmax, pY=C['pY'], dmin=dmin, mix=C['cls_share'])
        std_vh = (r1 == 'std') and all(ln['o'] in ('v', 'h') for ln in lines)
        if std_vh:
            def _prop():
                arr = np.asarray(pool_fn(int(rng.integers(24)))).reshape(-1, 4)
                return arr[int(rng.integers(len(arr)))][None] if len(arr) else np.zeros((0, 4))
            rec = RU.cleanup(rec, len(t2), T, lines, C['model'], H, rng, _prop, iters=int(prm.get('CLEAN_IT', 300)))
        else:
            def _prop_g():
                arr = np.asarray(pool_fn(int(rng.integers(24)))).reshape(-1, 4)
                if not len(arr):
                    return np.zeros((0, 4))
                c = P.clip(arr[int(rng.integers(len(arr)))][None], H)
                if not len(c) or not gate(c[0]):
                    return np.zeros((0, 4))
                return c
            rec = RU.cleanup(rec, len(t2), T, [], C['model'], H, rng, _prop_g, iters=int(prm.get('CLEAN_IT', 300)))
    return rec, len(t2), bud, p21h


def stat_pool(C, H, rng):
    return lambda k: P.unif_proposals(60, C['tr_th'], C['tr_len3d'], H, rng)


# ============================================================================================ scoring
_ST = None


def ST():
    global _ST
    if _ST is None:
        import statistical_tests as st
        _ST = st
    return _ST


def score_row(rec, gt, H, model):
    """final_eval.rec_row equivalent: Paper 1 scores (inf -> NaN) + map_rmse + n + gt_n."""
    s = ST().score(np.asarray(rec).reshape(-1, 4) / 20, gt / 20, H / 20, model)
    s = {k: (np.nan if (v is not None and np.isinf(v)) else v) for k, v in s.items()}
    return dict(n=len(rec), gt_n=len(gt), map_rmse=FE.p21map_err(rec, gt, H), **s)


# ============================================================================================ RA-17 local measures
STEP = 0.05
BANDS_C = [(0, 1), (1, 3), (3, 5), (5, 10), (10, np.inf)]
BINS_5 = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001)]


def points(L):
    """point samples as p2lib.p21_map: k = max(2, int(l / 0.05)), fractions (j + 0.5) / k, weight l / k."""
    L = np.asarray(L, float).reshape(-1, 4)
    xs, ys, th, wt = [], [], [], []
    for x1, y1, x2, y2 in L:
        l = math.hypot(x2 - x1, y2 - y1); k = max(2, int(l / STEP))
        t = (np.arange(k) + 0.5) / k
        xs.append(x1 + t * (x2 - x1)); ys.append(y1 + t * (y2 - y1))
        th.append(np.full(k, math.atan2(y2 - y1, x2 - x1) % math.pi)); wt.append(np.full(k, l / k))
    if not xs:
        z = np.zeros(0)
        return z, z, z, z
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(th), np.concatenate(wt)


def w1_circ_doubled_w(th_a, w_a, th_b, w_b):
    """weighted version of Paper 1 common.w1_circular_doubled (degrees of theta); equal weights reproduce it."""
    a = np.mod(2 * np.asarray(th_a, float), 2 * np.pi); b = np.mod(2 * np.asarray(th_b, float), 2 * np.pi)
    wa = np.asarray(w_a, float); wb = np.asarray(w_b, float)
    xs = np.concatenate([a, b]); order = np.argsort(xs, kind='mergesort'); xs_sorted = xs[order]
    steps = np.concatenate([wa / wa.sum(), -wb / wb.sum()])[order]
    D = np.cumsum(steps)
    seg = np.diff(np.concatenate([xs_sorted, [xs_sorted[0] + 2 * np.pi]]))
    o = np.argsort(D); cw = np.cumsum(seg[o])
    t = D[o][np.searchsorted(cw, cw[-1] / 2.0)]
    return math.degrees(float(np.sum(np.abs(D - t) * seg)) / 2.0)


def nearest_dist(x, lines):
    c = np.array([ln['c'] for ln in lines if ln['o'] == 'v'], float)
    return np.min(np.abs(np.asarray(x, float)[:, None] - c[None, :]), axis=1) if len(c) else np.full(len(x), np.inf)


def band_width(lines, lo, hi, nx=20000):
    """width (m) of the set {x in [0, 20]: lo <= distance to the nearest vertical line < hi} (fine grid)."""
    xs = (np.arange(nx) + 0.5) * W / nx
    d = nearest_dist(xs, lines)
    return float(((d >= lo) & (d < hi)).mean() * W)


def region_measures(pr, pg, mr, mg, area, min_len=0.5):
    """local P21 error (abs, rel) and local direction W1 for one region; pr/pg = (x, y, th, w) point arrays,
    mr/mg = boolean masks of the points inside the region."""
    lr, lg = float(pr[3][mr].sum()), float(pg[3][mg].sum())
    out = dict(p21_rec=lr / area if area > 0 else np.nan, p21_map=lg / area if area > 0 else np.nan)
    out['p21_abs_err'] = abs(out['p21_rec'] - out['p21_map']) if area > 0 else np.nan
    out['p21_rel_err'] = (out['p21_abs_err'] / out['p21_map']) if (area > 0 and lg > 0) else np.nan
    if lr >= min_len and lg >= min_len:
        out['dir_w1_deg'] = w1_circ_doubled_w(pr[2][mr], pr[3][mr], pg[2][mg], pg[3][mg])
    else:
        out['dir_w1_deg'] = np.nan
    out['len_rec'] = lr; out['len_map'] = lg
    return out


def local_bands(rec, gt, H, lines, layout_name, pg=None):
    """RA-17 bands (centre line) or normalised bins (5 m) for one reconstruction against the mapped window."""
    pr = points(rec); pg = pg if pg is not None else points(gt)
    dr, dg = nearest_dist(pr[0], lines), nearest_dist(pg[0], lines)
    rows = []
    if layout_name == 'centre':
        for lo, hi in BANDS_C:
            wdt = band_width(lines, lo, hi)
            m = region_measures(pr, pg, (dr >= lo) & (dr < hi), (dg >= lo) & (dg < hi), wdt * H)
            rows.append(dict(kind='band', region=f'{lo}-{hi if np.isfinite(hi) else "inf"} m', order=len(rows) + 1,
                             width_m=wdt, **m))
    else:
        half = 2.5
        for lo, hi in BINS_5:
            wdt = band_width(lines, lo * half, hi * half)
            m = region_measures(pr, pg, (dr / half >= lo) & (dr / half < hi), (dg / half >= lo) & (dg / half < hi), wdt * H)
            rows.append(dict(kind='bin', region=f'{lo:.1f}-{min(hi, 1.0):.1f}', order=len(rows) + 1, width_m=wdt, **m))
    return rows


def local_zones(rec, gt, H, lines, t2, pg=None, cell=2.0):
    """RA-17 tier zones (p2v2.zone_grid with this realisation's indicated traces t2): cell RMSE of 2 m-cell P21 (as
    p2v2.extra_scores), zone-level local P21 error and local direction W1."""
    check_W()
    z = V.zone_grid(lines, t2, H, cell)
    g_r, g_g = V.p21_grid(rec, H), V.p21_grid(gt, H)
    nx, ny = z.shape[1], z.shape[0]; cx, cy = W / nx, H / ny
    pr = points(rec); pg = pg if pg is not None else points(gt)
    def zone_of(p):
        ix = np.clip((p[0] / cx).astype(int), 0, nx - 1); iy = np.clip((p[1] / cy).astype(int), 0, ny - 1)
        return z[iy, ix]
    zr, zg = zone_of(pr), zone_of(pg)
    rows = []
    for k, nm in enumerate(('measured', 'indicated', 'inferred')):
        m = z == k
        area = float(m.sum()) * cx * cy
        rm = dict(kind='zone', region=nm, order=k + 1, width_m=np.nan, zone_frac=float(m.mean()),
                  cell_rmse=float(np.sqrt(np.mean((g_r[m] - g_g[m]) ** 2))) if m.any() else np.nan)
        rm.update(region_measures(pr, pg, zr == k, zg == k, area) if area > 0 else
                  dict(p21_rec=np.nan, p21_map=np.nan, p21_abs_err=np.nan, p21_rel_err=np.nan, dir_w1_deg=np.nan,
                       len_rec=np.nan, len_map=np.nan))
        rows.append(rm)
    # existing 2 m-cell band RMSE (p2v2.extra_scores BANDS), continuity only
    D = V.dist_grid(lines, H, cell)
    for lo, hi in V.BANDS:
        m = (D >= lo) & (D < hi)
        rows.append(dict(kind='cellband', region=f'{lo}-{hi} m', order=len(rows) - 2, width_m=np.nan,
                         zone_frac=float(m.mean()),
                         cell_rmse=float(np.sqrt(np.mean((g_r[m] - g_g[m]) ** 2))) if m.any() else np.nan))
    return rows


def seed_text(*a):
    return f"sd({', '.join(map(repr, a))}) = {sd(*a)}"
