# -*- coding: utf-8 -*-
"""v3 indicated-trace rule (15 Oct 2026, author decision: indicated traces may not cross a scanline that recorded nothing).

The frozen rule (p2v2.tier2) extends each group of hits into one indicated trace along the mean measured orientation,
through the first hit, with a length drawn (length-biased) from the training pieces within +-15 deg and a uniform offset.
It never checks the other scanlines, so at 5 m spacing 14.5% of the indicated traces crossed a scanline that recorded
none of their hits, which error-free detection forbids.

v3 rule (this module): for each group, the interval along the trace line that is free of scanlines without a hit of the
group is computed (scanlines carrying one of the group's hits are allowed; a strip lying between two hits of the group
cannot be avoided and is ignored). The length is drawn, as before, from the training pieces of similar orientation
that are at least as long as the hit span, but now also no longer than the free interval; the offset is drawn
uniformly among the positions that keep the trace inside the free interval while passing through all hits. With a
detection-aware rule (observation-error runs, r1 = 'da'), each bounding scanline is ignored with probability m_hat, as
for inferred traces. Everything else (grouping, orientation, window clipping) is unchanged.

The patch is installed by importing this module (install() is idempotent). It wraps the three placement entry points so
that the scanlines believed by the reconstruction are known to tier2: final_eval.frozen, ra01abl_run.compose and
ra1417_common.frozen_g. Frozen files are not edited."""
import math
import numpy as np

CTX = {'lines': None, 'm_hat': 0.0, 'W': 20.0}
EPS = 0.01


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _geom(ln, H, W):
    """(Bx, By, vx, vy, length, half width) of a scanline axis"""
    hw = 0.5 * (ln['hi'] - ln['lo']) if 'hi' in ln else 0.5 * ln.get('w', 0.10)
    if ln['o'] == 'v':
        return ln['c'], 0.0, 0.0, 1.0, H, hw
    if ln['o'] == 'h':
        return 0.0, ln['c'], 1.0, 0.0, W, hw
    return ln['x0'], ln['y0'], ln['ux'], ln['uy'], ln['L'], 0.5 * ln['w']


def bounds(x0, y0, ux, uy, span, own, lines, H, W, rng=None, m_hat=0.0):
    """free interval [s_lo, s_hi] (s along u from the anchor hit) without strips of scanlines lacking a hit of the group"""
    s_lo, s_hi = -math.inf, math.inf
    for j, ln in enumerate(lines):
        if j in own:
            continue
        Bx, By, vx, vy, L, hw = _geom(ln, H, W)
        den = _cross(ux, uy, vx, vy)
        if abs(den) < 1e-9:
            continue
        dx, dy = Bx - x0, By - y0
        s = _cross(dx, dy, vx, vy) / den
        t = _cross(dx, dy, ux, uy) / den
        if t < -hw or t > L + hw:
            continue                                    # the trace line meets the scanline outside the window
        hs = hw / abs(den)
        a, b = s - hs, s + hs
        if m_hat > 0 and rng is not None and rng.random() < m_hat:
            continue                                    # detection-aware: this line may have missed the trace
        if b < 0:
            s_lo = max(s_lo, b)
        elif a > span:
            s_hi = min(s_hi, a)
    return s_lo, s_hi


def tier2_bounded(groups, tr_th, tr_len, H, rng, dwin=15.0):
    import p2lib as P
    lines = CTX['lines']
    if lines is None:
        raise RuntimeError('v3 tier2 called without scanline context')
    W = CTX.get('W', P.M)
    m_hat = float(CTX.get('m_hat', 0.0) or 0.0)
    out = []
    for g in groups:
        th = math.atan2(sum(math.sin(2 * o['th']) for o in g), sum(math.cos(2 * o['th']) for o in g)) / 2 % math.pi
        ux, uy = math.cos(th), math.sin(th)
        s = np.array([o['x'] * ux + o['y'] * uy for o in g]); span = float(s.max() - s.min())
        d = np.abs(((tr_th - th) + math.pi / 2) % math.pi - math.pi / 2)
        pool = tr_len[d <= math.radians(dwin)]
        if len(pool) < 5:
            pool = tr_len
        o0 = g[int(np.argmin(s))]
        own = {o['ap'] for o in g}
        s_lo, s_hi = bounds(o0['x'], o0['y'], ux, uy, span, own, lines, H, W, rng, m_hat)
        lo = s_lo + EPS if math.isfinite(s_lo) else -math.inf
        hi = s_hi - EPS if math.isfinite(s_hi) else math.inf
        free = hi - lo
        cand = pool[(pool >= span) & (pool <= free)]
        if len(cand):
            Lf = float(rng.choice(cand, p=cand / cand.sum()))
        else:
            Lf = float(max(span, min(span * 1.1 + 0.2, free if math.isfinite(free) else span * 1.1 + 0.2)))
        a_min = max(0.0, Lf - hi) if math.isfinite(hi) else 0.0
        a_max = min(Lf - span, -lo) if math.isfinite(lo) else Lf - span
        a = float(rng.uniform(a_min, a_max)) if a_max > a_min else float(max(0.0, min(a_min, Lf - span)))
        p1 = (o0['x'] - a * ux, o0['y'] - a * uy)
        out.append([p1[0], p1[1], p1[0] + Lf * ux, p1[1] + Lf * uy])
    return P.clip(np.array(out, float).reshape(-1, 4), H)


def install():
    import p2v2 as V
    if getattr(V, '_v3_installed', False):
        return
    V._tier2_frozen = V.tier2
    V.tier2 = tier2_bounded
    V._v3_installed = True
    import final_eval as FE
    _frozen = FE.frozen

    def frozen(C, obs, lines, H, rng, pool_fn, prm, dmin_cache, clean=False):
        CTX['lines'] = lines; CTX['m_hat'] = 0.0
        return _frozen(C, obs, lines, H, rng, pool_fn, prm, dmin_cache, clean)
    FE.frozen = frozen
    try:
        import ra01abl_run as R
        _compose = R.compose

        def compose(C, obs, lines, H, rng, arm, models, evae_nets, net_offset, dmc):
            CTX['lines'] = lines; CTX['m_hat'] = 0.0
            return _compose(C, obs, lines, H, rng, arm, models, evae_nets, net_offset, dmc)
        R.compose = compose
    except ImportError:
        pass
    try:
        import ra1417_common as RC
        _fg = RC.frozen_g

        def frozen_g(C, obs, lines, H, rng, pool_fn, prm, dmin_cache, clean=False, miss_known=0.0, r1='std', m_hat=0.0):
            CTX['lines'] = lines; CTX['m_hat'] = float(m_hat) if r1 == 'da' else 0.0
            return _fg(C, obs, lines, H, rng, pool_fn, prm, dmin_cache, clean, miss_known=miss_known, r1=r1, m_hat=m_hat)
        RC.frozen_g = frozen_g
    except ImportError:
        pass
