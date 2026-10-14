"""Shared machinery for the frozen per-set fair comparison (2026-09-07).

Implements exactly the definitions in 02_frozen_protocol\FAIR_2D_COMPARISON_PROTOCOL_FROZEN.md.
Copied/adapted pieces are marked with their source. No comparison number is computed at import.
"""
from __future__ import annotations

import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import iv
from scipy.stats import wasserstein_distance

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from repo_paths import MAPPED_TRACES_DIR as _MAPPED
DATA = _MAPPED        # released: repo-relative, override with PAPER1_MAPPED_TRACES_DIR
EM_SEED = 20260907          # JOINT_FAMILY_RULE_AMENDMENT_A1_2026-09-07
SEEDS = (1337, 20260903, 7) # protocol Section 2
DRAWS = 3
GRID_CELL = 0.1             # FIXED_IMPLEMENTATION (frozen metrics, corrected campaign)
VALIDATION = {"A": "DJI_20240404103116_0222_Vbottom", "B": "DJI_20240404103113_0221_Vtop"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gen_seed(*parts) -> int:
    """Protocol Section 2: crc32 of 'part|part|...' (unsigned)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# ----------------------------------------------------------------------- data

def load_panels():
    """Route B isotropic loading: x/s, y/s with s = max(W, H) = 5280 (Gate 1 T18)."""
    out = []
    for p in sorted(DATA.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        W, H = int(d["imageWidth"]), int(d["imageHeight"])
        s = float(max(W, H))
        rows = [[q[0][0] / s, q[0][1] / s, q[1][0] / s, q[1][1] / s]
                for q in (sh["points"] for sh in d.get("shapes", [])
                          if sh.get("shape_type") == "line" and len(sh.get("points", [])) == 2)]
        out.append({"name": p.stem, "W": W, "H": H, "scale": s, "y_max": H / s,
                    "lines": np.asarray(rows, dtype=np.float64),
                    "band": "upper" if p.stem.endswith("_Vtop") else "lower",
                    "sha256": sha256(p)})
    assert len(out) == 9 and sum(len(p["lines"]) for p in out) == 512
    return out


def split_sets(panels, direction):
    """Protocol Section 2. Returns (fitting_windows, validation_window, test_windows)."""
    band = "lower" if direction == "A" else "upper"
    train_band = [p for p in panels if p["band"] == band]
    val = [p for p in train_band if p["name"] == VALIDATION[direction]]
    fit = [p for p in train_band if p["name"] != VALIDATION[direction]]
    test = [p for p in panels if p["band"] != band]
    assert len(val) == 1
    return fit, val[0], test


def geom(L):
    L = np.asarray(L, dtype=np.float64)
    dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    c = np.column_stack(((L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2))
    return c, np.hypot(dx, dy), np.mod(np.arctan2(dy, dx), np.pi)


def clip_segment(line, y_max):
    """Liang-Barsky clip to [0,1] x [0,y_max]. Source: run_benchmark.py:104 (audited T20)."""
    x1, y1, x2, y2 = map(float, line[:4])
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1, 1 - x1, y1, y_max - y1)
    low, high = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        v = qi / pi
        if pi < 0:
            low = max(low, v)
        else:
            high = min(high, v)
        if low > high:
            return None
    r = np.asarray([x1 + low * dx, y1 + low * dy, x1 + high * dx, y1 + high * dy])
    if np.hypot(r[2] - r[0], r[3] - r[1]) < 1e-4:
        return None
    return r


def clip_lines(lines, y_max):
    out = [c for ln in lines if (c := clip_segment(np.asarray(ln, float), y_max)) is not None]
    return np.asarray(out, float) if out else np.zeros((0, 4))


# ------------------------------------------------------- set rule (Section 3)

def _vm_pdf(z, mu, kappa):
    return np.exp(kappa * np.cos(z - mu)) / (2 * np.pi * iv(0, kappa))


def _kappa_from_R(R):
    R = min(max(float(R), 1e-6), 0.999)
    try:
        return brentq(lambda k: iv(1, k) / iv(0, k) - R, 1e-6, 500.0)
    except ValueError:
        return 0.1


def fit_set_rule(angles, k=2, seed=EM_SEED, restarts=5, iters=300):
    """Axial von Mises mixture on z = 2*theta, EM, per JOINT_FAMILY_RULE_PROTOCOL + A1.
    Returns dict with axial centres (radians of theta, sorted), kappas (on 2theta), weights."""
    z = np.mod(2.0 * np.asarray(angles, float), 2 * np.pi)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        mu = rng.uniform(0, 2 * np.pi, k)
        kap = np.full(k, 2.0)
        w = np.full(k, 1.0 / k)
        ll_old = -np.inf
        for _ in range(iters):
            dens = np.stack([w[j] * _vm_pdf(z, mu[j], kap[j]) for j in range(k)], 1)
            tot = dens.sum(1, keepdims=True) + 1e-300
            resp = dens / tot
            ll = float(np.log(tot).sum())
            w = resp.mean(0)
            for j in range(k):
                rj = resp[:, j]
                C = float((rj * np.cos(z)).sum())
                S = float((rj * np.sin(z)).sum())
                mu[j] = math.atan2(S, C) % (2 * np.pi)
                R = math.hypot(C, S) / max(rj.sum(), 1e-12)
                kap[j] = _kappa_from_R(R)
            if abs(ll - ll_old) < 1e-9:
                break
            ll_old = ll
        if best is None or ll > best["ll"]:
            best = {"ll": ll, "mu": mu.copy(), "kappa": kap.copy(), "w": w.copy()}
    order = np.argsort(np.mod(best["mu"] / 2.0, np.pi))
    return {"centres_theta": np.mod(best["mu"][order] / 2.0, np.pi),
            "mu2": np.mod(best["mu"][order], 2 * np.pi),
            "kappa2": best["kappa"][order], "weights": best["w"][order],
            "loglik": best["ll"], "em_seed": seed}


def assign_sets(angles, rule):
    """Hard assignment to the highest-responsibility component (frozen family rule;
    protocol Section 3 as amended after review finding 1)."""
    z = np.mod(2.0 * np.asarray(angles, float), 2 * np.pi)
    dens = np.stack([rule["weights"][j] * _vm_pdf(z, rule["mu2"][j], rule["kappa2"][j])
                     for j in range(len(rule["mu2"]))], 1)
    return np.argmax(dens, 1)


# ------------------------------------------------- outcome metrics (Section 4)

def w1_circular_doubled(theta_a, theta_b):
    """O1: cut-point-minimised circular W1 between empirical distributions of 2*theta
    on the circle of circumference 2*pi, reported in DEGREES of theta.
    W1_circ = min_t INT |F_a - F_b - t| dx; t* is the (length-)weighted median of the
    piecewise-constant difference D(x) = F_a(x) - F_b(x)."""
    a = np.sort(np.mod(2 * np.asarray(theta_a, float), 2 * np.pi))
    b = np.sort(np.mod(2 * np.asarray(theta_b, float), 2 * np.pi))
    xs = np.concatenate([a, b])
    order = np.argsort(xs, kind="mergesort")
    xs_sorted = xs[order]
    # step of +1/n for a-points, -1/m for b-points
    steps = np.concatenate([np.full(len(a), 1.0 / len(a)), np.full(len(b), -1.0 / len(b))])[order]
    D = np.cumsum(steps)                       # D(x) on [xs_i, xs_{i+1})
    seg = np.diff(np.concatenate([xs_sorted, [xs_sorted[0] + 2 * np.pi]]))  # wrap segment
    # weighted median of D with weights seg
    o = np.argsort(D)
    cw = np.cumsum(seg[o])
    t = D[o][np.searchsorted(cw, cw[-1] / 2.0)]
    w1_rad2 = float(np.sum(np.abs(D - t) * seg))     # in radians of 2*theta
    return math.degrees(w1_rad2 / 2.0)               # degrees of theta


def axial_csd_deg(theta):
    """O2: axial circular standard deviation, degrees: 0.5*sqrt(-2 ln R2)."""
    z = np.exp(2j * np.asarray(theta, float))
    R2 = min(max(abs(z.mean()), 1e-12), 1 - 1e-12)
    return math.degrees(0.5 * math.sqrt(-2.0 * math.log(R2)))


def length_stats(lengths):
    q = np.percentile(lengths, [5, 25, 50, 75, 95])
    return {"p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
            "max": float(np.max(lengths))}


def nn_distances(centres):
    c = np.asarray(centres, float)
    if len(c) < 2:
        return np.zeros(0)
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min(1)


def projected_gaps(centres, set_theta):
    """O5 secondary (M4): centre positions projected on the set's axial normal,
    consecutive sorted gaps, DROP the single largest and single smallest gap
    (the protocol's stated edge rule). Requires n >= 4 -> at least 1 gap left."""
    c = np.asarray(centres, float)
    if len(c) < 4:
        return None
    normal = np.array([-math.sin(set_theta), math.cos(set_theta)])
    proj = np.sort(c @ normal)
    gaps = np.diff(proj)
    if len(gaps) <= 2:
        return None
    keep = np.delete(gaps, [int(np.argmin(gaps)), int(np.argmax(gaps))])
    return keep if len(keep) else None


def w1(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(wasserstein_distance(a, b))


# ------------------------------------------------- crossing machinery (Section 7)
# Source: audited hybrid_b0_plus_rule.py / frozen_metrics_2026-09-07.py, mechanics unchanged.

def seg_cross_one(L1, L2):
    if len(L2) == 0:
        return np.zeros(0, bool)
    p1, r1 = L1[:2], L1[2:] - L1[:2]
    p2, r2 = L2[:, :2], L2[:, 2:] - L2[:, :2]
    den = r1[0] * r2[:, 1] - r1[1] * r2[:, 0]
    safe = np.where(np.abs(den) < 1e-12, np.nan, den)
    d = p2 - p1
    t = (d[:, 0] * r2[:, 1] - d[:, 1] * r2[:, 0]) / safe
    u = (d[:, 0] * r1[1] - d[:, 1] * r1[0]) / safe
    return (t > 0) & (t < 1) & (u > 0) & (u < 1)


def crossings_xy(L):
    """All interior crossing points (i, j, x, y)."""
    out = []
    n = len(L)
    for i in range(n):
        p1, r1 = L[i, :2], L[i, 2:] - L[i, :2]
        for j in range(i + 1, n):
            p2, r2 = L[j, :2], L[j, 2:] - L[j, :2]
            den = r1[0] * r2[1] - r1[1] * r2[0]
            if abs(den) < 1e-12:
                continue
            d = p2 - p1
            t = (d[0] * r2[1] - d[1] * r2[0]) / den
            u = (d[0] * r1[1] - d[1] * r1[0]) / den
            if 0 < t < 1 and 0 < u < 1:
                x, y = p1 + t * r1
                out.append((i, j, float(x), float(y)))
    return out


def crossing_decompose(L, y_max, lab):
    """Observed/analytic-expected crossing ratios, total and same-set (audited convention:
    E = sum_pairs l_i l_j |sin(a_i - a_j)| / y_max)."""
    n = len(L)
    _, lens, ang = geom(L)
    obs_s = obs_x = 0
    for (i, j, _, _) in crossings_xy(L):
        if lab[i] == lab[j]:
            obs_s += 1
        else:
            obs_x += 1
    s = np.abs(np.sin(ang[:, None] - ang[None, :]))
    same = lab[:, None] == lab[None, :]
    iu = np.triu(np.ones((n, n), bool), 1)
    ll = np.outer(lens, lens) * s / y_max
    exp_s, exp_x = ll[iu & same].sum(), ll[iu & ~same].sum()
    ratio = lambda o, e: o / max(e, 1e-9)
    return {"ratio_total": ratio(obs_s + obs_x, exp_s + exp_x),
            "ratio_same": ratio(obs_s, exp_s),
            "ratio_cross": ratio(obs_x, exp_x),
            "obs_total": obs_s + obs_x}


def grid_cell_max(L, y_max, cell=GRID_CELL):
    xs = crossings_xy(L)
    nx, ny = max(1, int(np.ceil(1.0 / cell))), max(1, int(np.ceil(y_max / cell)))
    h = np.zeros((nx, ny), int)
    for _, _, x, y in xs:
        h[min(int(x / cell), nx - 1), min(int(y / cell), ny - 1)] += 1
    return int(h.max())


# ----------------------------------------------------- parametric fits (Section 6/8)

def kappa_directional_from_axial(angles):
    """Audited axial-to-directional mapping (run_benchmark.py:256-263): solve
    I2(k)/I0(k) = R2, R2 = |mean(exp(2i*theta))|."""
    R2 = min(abs(np.mean(np.exp(2j * np.asarray(angles, float)))), 0.999)
    try:
        return brentq(lambda v: iv(2, v) / iv(0, v) - R2, 1e-6, 200.0)
    except ValueError:
        return 0.1


def trunc_exp_fit(values):
    """Audited truncated-exponential fit (run_benchmark.py:268-281): bounds = 1%/99%
    quantiles, mu mean-matched by bisection. Returns (low, mu, high)."""
    v = np.asarray(values, float)
    low, high = np.quantile(v, [0.01, 0.99])
    low, high = max(float(low), 1e-4), max(float(high), float(low) + 1e-4)
    target = float(np.mean(v))

    def mean(mu):
        ea, eb = math.exp(-low / mu), math.exp(-high / mu)
        return mu + (low * ea - high * eb) / max(ea - eb, 1e-12)

    try:
        mu = brentq(lambda m: mean(m) - target, 1e-5, 100.0)
    except ValueError:
        mu = max(target, 1e-4)
    return low, mu, high


def trunc_exp_cdf(x, low, mu, high):
    x = np.clip(np.asarray(x, float), low, high)
    ea, eb = math.exp(-low / mu), math.exp(-high / mu)
    return (ea - np.exp(-x / mu)) / max(ea - eb, 1e-12)


def trunc_exp_sample(rng, n, low, mu, high):
    u = rng.uniform(0, 1, n)
    ea, eb = math.exp(-low / mu), math.exp(-high / mu)
    return -mu * np.log(ea - u * (ea - eb))


def vm_axial_mle(theta):
    """Single axial von Mises MLE on 2*theta: mu2 = circular mean, kappa via I1/I0=R."""
    z = np.mod(2 * np.asarray(theta, float), 2 * np.pi)
    C, S = np.cos(z).mean(), np.sin(z).mean()
    mu2 = math.atan2(S, C) % (2 * np.pi)
    kap = _kappa_from_R(math.hypot(C, S))
    return mu2, kap


def vm_cdf_vals(z, mu, kappa, ngrid=4096):
    """CDF of vM(mu, kappa) on [0, 2pi) evaluated at points z, by trapezoid grid."""
    grid = np.linspace(0, 2 * np.pi, ngrid + 1)
    pdf = _vm_pdf(grid, mu, kappa)
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(np.mod(z, 2 * np.pi), grid, cdf)


def watson_u2(u):
    """Watson U^2 from PIT values u in [0,1]."""
    u = np.sort(np.asarray(u, float))
    n = len(u)
    i = np.arange(1, n + 1)
    return float(np.sum((u - (2 * i - 1) / (2 * n)) ** 2) + 1 / (12 * n)
                 - n * (u.mean() - 0.5) ** 2)


def ks_stat(x, cdf_vals):
    x = np.asarray(x, float)
    o = np.argsort(x)
    c = np.asarray(cdf_vals, float)[o]
    n = len(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - c), np.max(c - (i - 1) / n)))
