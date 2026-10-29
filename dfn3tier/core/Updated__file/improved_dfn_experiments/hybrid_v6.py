"""
hybrid_v6.py — 3-tier hybrid DFN reconstruction.

Ported from the proven pipeline in:
  Hybrid_evae_sequential/run_final_onepanel_check.py
  Hybrid_evae_sequential/run_final_onepanel_variants.py

With ML model integration (EVAE + transformer).
"""
from __future__ import annotations
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(Path((__import__('os').environ['DFN3_CORE'] + r"\Evae_sequential_hybrid_revised".replace(chr(92), '/')))))

from exp_common import (load_panels_raw, get_physics_stats, split_panel,
                         strips_for_interval, assign_family_ids, N_MAX_EVAE,
                         segments_intersect, intersection_density)
from model import DualLatentVAE
from model_v3 import StatisticalSampler
from transformer_v4 import SeqFracTransformerV4, generate_v4
from run_ablation_v3 import (_greedy_select_soft, _train_geology_stats,
                              _load_set_predictor)
from _task2_validate import decode_v4
from random_family_extension import append_random_family
from scorer import HCScorer, extract_features, apply_hard_gates

EPS = 1e-6
ANTI_LEAK_TOL = 1e-6
CANDIDATE_MULT = 80

# Canonical family assignment threshold: 20° is the standard geological
# threshold used throughout the project. Lines >20° from all centres = random.
# This single value is used for ALL family assignment calls in this file.
FAM_ASSIGN_THRESHOLD_DEG = 20.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: _touch_points — ported from run_final_onepanel_check.py:158-180
# ══════════════════════════════════════════════════════════════════════════════
def _touch_points(x1, y1, x2, y2, strips):
    """Find endpoints that lie on or near a strip boundary.

    TOUCH_TOL is derived from the strip aperture (10% of half-width),
    not hardcoded, so this adapts to any borehole configuration.
    """
    # Derive tolerance from strip aperture (data-driven)
    if strips:
        strip_half_width = abs(float(strips[0][1]) - float(strips[0][0])) * 0.5
        touch_tol = max(strip_half_width * 0.1, 1e-5)  # 10% of half-width, floor 1e-5
    else:
        touch_tol = EPS

    out = []
    for lo, hi in strips:
        for side, xb in (("left", float(lo)), ("right", float(hi))):
            d1 = abs(x1 - xb)
            d2 = abs(x2 - xb)
            if d1 <= touch_tol and d2 <= touch_tol:
                continue  # both endpoints on boundary = segment is inside strip
            if d1 <= touch_tol:
                out.append((side, (x1, y1)))
            elif d2 <= touch_tol:
                out.append((side, (x2, y2)))
    uniq = []
    seen = set()
    for s, p in out:
        key = (s, round(p[0], 5), round(p[1], 5))
        if key not in seen:
            seen.add(key)
            uniq.append((s, p))
    return uniq


# ══════════════════════════════════════════════════════════════════════════════
# Helper: _max_t_to_bounds
# ══════════════════════════════════════════════════════════════════════════════
def _max_t_to_bounds(x0, y0, ux, uy, y_max):
    t_hi = np.inf
    if abs(ux) > 1e-12:
        t_hi = min(t_hi, (1.0 - x0) / ux if ux > 0 else (0.0 - x0) / ux)
    if abs(uy) > 1e-12:
        t_hi = min(t_hi, (y_max - y0) / uy if uy > 0 else (0.0 - y0) / uy)
    if not np.isfinite(t_hi):
        return 0.0
    return max(0.0, float(t_hi))


# ══════════════════════════════════════════════════════════════════════════════
# Helper: _inside_strip
# ══════════════════════════════════════════════════════════════════════════════
def _inside_strip(x, strips):
    for lo, hi in strips:
        if float(lo) - EPS <= float(x) <= float(hi) + EPS:
            return True
    return False


def segment_crosses_any_borehole(x1, y1, x2, y2, strips, source_strip=None):
    """Check if a line segment physically crosses any borehole strip.

    Uses parametric line-vs-vertical-boundary intersection.  For each strip
    boundary (left and right edges), compute the parameter t at which the
    segment crosses that vertical line.  If both boundaries have valid t
    values (0 < t < 1) with the segment entering and exiting the strip,
    the segment crosses that borehole.

    Parameters
    ----------
    source_strip : tuple or None
        (lo, hi) of the borehole the segment originates from.  This strip
        is excluded so that an extension starting exactly on its boundary
        is not flagged as crossing its own source.

    Returns
    -------
    crossed_strip : tuple or None
        The first (lo, hi) strip that the segment crosses, or None.
    """
    dx = float(x2) - float(x1)
    for lo, hi in strips:
        lo_f = float(lo)
        hi_f = float(hi)
        # Skip the borehole the segment originates from
        if source_strip is not None:
            if abs(lo_f - float(source_strip[0])) < EPS and \
               abs(hi_f - float(source_strip[1])) < EPS:
                continue
        if abs(dx) < 1e-12:
            # Vertical segment: crosses if x is inside the strip
            xc = float(x1)
            if lo_f - EPS <= xc <= hi_f + EPS:
                return (lo_f, hi_f)
            continue
        # Parametric t where segment crosses the left and right edges
        t_lo = (lo_f - float(x1)) / dx
        t_hi = (hi_f - float(x1)) / dx
        t_enter = min(t_lo, t_hi)
        t_exit  = max(t_lo, t_hi)
        # Segment crosses this strip if there is overlap with [0, 1]
        t_enter_c = max(t_enter, 0.0)
        t_exit_c  = min(t_exit,  1.0)
        if t_enter_c < t_exit_c - EPS:
            return (lo_f, hi_f)
    return None


def _clamp_extension_before_strip(bx, by, x2_ext, y2_ext, crossed_strip, side):
    """Shorten an indicated extension so it stops just before the crossed strip.

    Returns (new_x2, new_y2) clamped to the near edge of the crossed strip
    with a small margin so the endpoint sits outside the strip.
    """
    lo_f, hi_f = crossed_strip
    dx = float(x2_ext) - float(bx)
    if abs(dx) < 1e-12:
        return float(x2_ext), float(y2_ext)
    dy = float(y2_ext) - float(by)
    # Near edge is the boundary closest to the origin point
    if float(bx) < lo_f:
        # Approaching from the left, stop just before lo
        near_edge = lo_f
    else:
        # Approaching from the right, stop just before hi
        near_edge = hi_f
    margin = 0.001  # ~0.036 m in physical coords — just outside the strip
    if float(bx) < near_edge:
        target_x = near_edge - margin
    else:
        target_x = near_edge + margin
    t = (target_x - float(bx)) / dx
    t = max(0.0, min(1.0, t))
    new_x2 = float(bx) + t * dx
    new_y2 = float(by) + t * dy
    return new_x2, new_y2


# ══════════════════════════════════════════════════════════════════════════════
# Helper: _line_from_xy
# FIX 2: family stored as int, not str
# ══════════════════════════════════════════════════════════════════════════════
def _line_from_xy(x1, y1, x2, y2, family, role, parent):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    dx = x2 - x1
    dy = y2 - y1
    ln = float(math.hypot(dx, dy))
    ang = float(math.atan2(dy, dx) % math.pi)
    return {
        "x1": float(x1), "y1": float(y1),
        "x2": float(x2), "y2": float(y2),
        "cx": float(cx), "cy": float(cy),
        "length": float(ln), "angle": float(ang),
        "family": int(family),          # FIX 2: always int
        "semantic_role": str(role),
        "parent_trace_id": str(parent),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: _build_indicated_independent
# ══════════════════════════════════════════════════════════════════════════════
def _build_indicated_independent(
    measured_dicts, fam_lengths, fam_p95_indicated, y_max, strips, rng, ind_budget,
    cont_model=None, cont_X_mean=None, cont_X_std=None,
):
    indicated = []
    from_touch = 0
    ind_count = {f: 0 for f in range(3)}
    n_clips_total = 0
    n_clips_no_touch = 0

    # Derive minimum extension length from borehole aperture (data-driven).
    # Stubs of 0.03-0.05 are real extensions for clips near borehole edges.
    if strips:
        borehole_half_width = abs(float(strips[0][1]) - float(strips[0][0])) * 0.5
    else:
        borehole_half_width = 0.5 * BOREHOLE_WIDTH_M / 36.0  # fallback
    min_ext_len = borehole_half_width * 0.5  # half the borehole half-width

    for m in measured_dicts:
        fam = int(m["family"])
        if fam > 2:
            continue
        # No budget cap for indicated tier: every eligible measured clip
        # that touches a strip boundary gets an extension. Indicated lines
        # are hidden continuations of OBSERVED fractures, not new fractures.
        # Count quota only applies to inferred tier (Tier 3).
        dx = float(m["x2"]) - float(m["x1"])
        dy = float(m["y2"]) - float(m["y1"])
        ln = float(math.hypot(dx, dy))
        if ln <= 1e-9:
            continue
        ux = dx / ln
        uy = dy / ln
        parent = str(m["parent_trace_id"])

        n_clips_total += 1
        touches = _touch_points(float(m["x1"]), float(m["y1"]),
                                float(m["x2"]), float(m["y2"]), strips)
        if len(touches) == 0:
            n_clips_no_touch += 1
            continue

        for side, (bx, by) in touches:
            sign = -1.0 if side == "left" else 1.0
            if abs(ux) < 1e-9:
                continue
            if ux * sign > 0:
                dux, duy = ux, uy
            else:
                dux, duy = -ux, -uy
            if dux * sign <= 0:
                continue

            t_bound = _max_t_to_bounds(float(bx), float(by),
                                       float(dux), float(duy), y_max)
            # FIX 1: use canonical p95 (renamed to avoid shadowing)
            max_len = min(float(fam_p95_indicated[fam]), float(t_bound))
            if max_len <= 1e-9:
                continue

            sample_len = None
            # Learned continuation-length prediction if model available
            if cont_model is not None and cont_X_mean is not None:
                import torch as _torch
                clip_A = float(m["angle"])
                n_nearby = sum(1 for m2 in measured_dicts if m2 is not m
                               and math.hypot(float(m2["cx"]) - float(m["cx"]),
                                              float(m2["cy"]) - float(m["cy"])) < 0.1)
                feat = np.array([[
                    float(np.log(max(float(m["length"]), 1e-8))),
                    float(np.sin(clip_A)),
                    float(np.cos(clip_A)),
                    1.0 if fam == 0 else 0.0,
                    1.0 if fam == 1 else 0.0,
                    1.0 if fam == 2 else 0.0,
                    float(sign),
                    float(bx),
                    float(by),
                    float(t_bound),
                    float(n_nearby),
                ]], dtype=np.float32)
                feat_t = (_torch.from_numpy(feat) - cont_X_mean) / cont_X_std
                with _torch.no_grad():
                    mu_pred, sig_pred = cont_model.predict(feat_t)
                mu_v = float(mu_pred[0, 0])
                sig_v = max(float(sig_pred[0, 0]), 0.05)
                # Use 75th percentile of predicted distribution:
                # for indicated continuation, the outward-facing remainder
                # is typically longer than the average remainder, because
                # the average includes both the short and long sides.
                logL_sample = mu_v + 0.67 * sig_v  # p75 of N(mu, sigma)
                sample_len = float(np.exp(logL_sample))
                sample_len = max(sample_len, 0.03)  # minimum ~1m continuation
                sample_len = min(sample_len, max_len)
                if sample_len <= 1e-9:
                    sample_len = None  # fall through to empirical
            # Empirical fallback
            if sample_len is None:
                arr = fam_lengths[fam]
                for _ in range(64):
                    cand = float(arr[rng.integers(0, len(arr))])
                    cand = min(cand, max_len)
                    if cand <= 1e-9:
                        continue
                    sample_len = cand
                    break
            if sample_len is None:
                sample_len = max(1e-5, 0.73 * max_len)

            x2_ext = float(bx + sample_len * dux)
            y2_ext = float(by + sample_len * duy)

            # FIX R5: Identify the source borehole so we skip it
            source_bh = None
            for s_lo, s_hi in strips:
                if abs(float(bx) - float(s_lo)) < EPS or abs(float(bx) - float(s_hi)) < EPS:
                    source_bh = (float(s_lo), float(s_hi))
                    break

            # FIX R5: If extension crosses any OTHER borehole, clamp it
            crossed = segment_crosses_any_borehole(
                float(bx), float(by), x2_ext, y2_ext, strips,
                source_strip=source_bh)
            if crossed is not None:
                x2_ext, y2_ext = _clamp_extension_before_strip(
                    float(bx), float(by), x2_ext, y2_ext, crossed, side)

            line = _line_from_xy(float(bx), float(by), x2_ext, y2_ext,
                                 fam, "indicated", parent)

            # Skip extensions shorter than data-derived floor (p5 of training lengths)
            if float(line["length"]) < min_ext_len:
                continue

            # Safety: also reject if midpoint still lands inside a strip
            # (can happen with near-vertical lines or numerical edge cases)
            if _inside_strip(0.5 * (float(line["x1"]) + float(line["x2"])), strips):
                continue

            indicated.append(line)
            ind_count[fam] += 1
            from_touch += 1

    print(f"    [tier2] clips={n_clips_total} | no_touch={n_clips_no_touch} | "
          f"extensions={from_touch} | min_ext_len={min_ext_len:.4f}")
    return indicated, from_touch


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: _farthest_endpoint_pair + _merge_fixed_lines
# FIX 2: Counter uses int family consistently
# ══════════════════════════════════════════════════════════════════════════════
def _farthest_endpoint_pair(lines):
    pts = []
    for l in lines:
        pts.append((float(l["x1"]), float(l["y1"])))
        pts.append((float(l["x2"]), float(l["y2"])))
    if len(pts) == 0:
        return (0.0, 0.0), (0.0, 0.0)
    if len(pts) == 1:
        return pts[0], pts[0]
    best_i, best_j = 0, 1
    best_d2 = -1.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        for j in range(i + 1, len(pts)):
            x2, y2 = pts[j]
            d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if d2 > best_d2:
                best_d2 = d2
                best_i, best_j = i, j
    return pts[best_i], pts[best_j]


def _merge_fixed_lines(measured_dicts, indicated):
    grouped = {}
    roles = {}
    for l in measured_dicts:
        p = str(l.get("parent_trace_id", ""))
        grouped.setdefault(p, []).append(dict(l))
        roles.setdefault(p, set()).add("measured")
    for l in indicated:
        p = str(l.get("parent_trace_id", ""))
        grouped.setdefault(p, []).append(dict(l))
        roles.setdefault(p, set()).add("indicated")

    out = []
    for parent, segs in grouped.items():
        has_meas = "measured" in roles.get(parent, set())
        has_ind = "indicated" in roles.get(parent, set())
        if has_meas and has_ind and len(segs) >= 2:
            p1, p2 = _farthest_endpoint_pair(segs)
            # FIX 2: use int family consistently
            fam = Counter([int(s.get("family", 0)) for s in segs]).most_common(1)[0][0]
            merged = _line_from_xy(p1[0], p1[1], p2[0], p2[1],
                                    fam, "fixed_merged", parent)
            if float(merged["length"]) > 1e-12:
                out.append(merged)
            else:
                out.extend(segs)
        else:
            out.extend(segs)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: _line_intrudes_strip
# ══════════════════════════════════════════════════════════════════════════════
def _line_intrudes_strip(cx, cy, L, A, strips):
    """Reject if midpoint OR either endpoint falls inside any strip."""
    if _inside_strip(cx, strips):
        return True
    x1 = cx - 0.5 * L * math.cos(A)
    x2 = cx + 0.5 * L * math.cos(A)
    xlo = min(x1, x2)
    xhi = max(x1, x2)
    for lo, hi in strips:
        if xhi >= float(lo) and xlo <= float(hi):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Calibration helpers
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_f2_scale(evae_model, panels_train, ps, device,
                        fragment_len_thresh: float = 0.05) -> float:
    fam_centers = np.asarray(ps["fam_centers_rad"], dtype=float)
    evae_f2_all = []
    evae_f2_filtered = []
    for panel in panels_train:
        strips = strips_for_interval(5.0)
        m, _ = split_panel(panel, strips)
        ev = decode_v4(evae_model, m, n_out=120, strips=None)
        fam = assign_family_ids(ev, fam_centers, threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
        f2 = ev[fam == 2, 2]
        evae_f2_all.extend(list(f2))
        evae_f2_filtered.extend(list(f2[f2 > fragment_len_thresh]))
    if len(evae_f2_filtered) >= 3:
        evae_mean = float(np.mean(evae_f2_filtered))
    else:
        evae_mean = float(np.mean(evae_f2_all)) if evae_f2_all else 0.141
    gt_f2_lens = []
    for panel in panels_train:
        fam = assign_family_ids(panel, fam_centers, threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
        gt_f2_lens.extend(list(panel[fam == 2, 2]))
    gt_mean = float(np.mean(gt_f2_lens)) if gt_f2_lens else 0.185
    print(f"[calibrate_f2_scale]"
          f"  all_F2_pred={len(evae_f2_all)}"
          f"  used_F2 (>{fragment_len_thresh:.3f})={len(evae_f2_filtered)}"
          f"  evae_mean_used={evae_mean:.4f}"
          f"  gt_mean={gt_mean:.4f}"
          f"  scale={gt_mean/max(evae_mean,1e-6):.3f}")
    return gt_mean / max(evae_mean, 1e-6)


def apply_f2_scale(cand_arr, scale, fam_centers):
    if len(cand_arr) == 0:
        return cand_arr
    arr = cand_arr.copy()
    fams = assign_family_ids(arr, fam_centers, threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    for i in np.where(fams == 2)[0]:
        arr[i, 2] *= scale
    return arr


def pick_soft_cap_mult(n_fragments, thresh=15, tight=1.1, loose=1.2):
    return tight if n_fragments > thresh else loose


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE: build_hybrid_v6
# ══════════════════════════════════════════════════════════════════════════════
def build_hybrid_v6(real, measured, strips, n_target, ps, gstats, fam_centers,
                     evae_v4, trans_v4, sp_model, sp_ps, f2_scale: float,
                     device, seed: int = 42001,
                     scorer_fn=None) -> Tuple[np.ndarray, Dict]:
    """
    3-tier hybrid generation.
      Tier 1 — MEASURED:  strip-clipped fragments, frozen
      Tier 2 — INDICATED: one-direction touchpoint extensions
      Tier 3 — INFERRED:  transformer + statistical 80x pool, spacing-filtered
      Merge:  measured+indicated → one parent line per trace
      Random: appended last, subject to strip exclusion
    """
    y_max = float(ps.get("yscale", 0.315))
    rng = np.random.default_rng(seed)

    # Data-derived parameters from physics_stats
    fam_logmu  = np.asarray(ps.get("fam_logmu",  [-2.33, -2.30, -1.99]))
    fam_logsig = np.asarray(ps.get("fam_logsig", [0.43, 0.35, 0.58]))

    # FIX 5: Fit von Mises kappa from calibration data per family (data-driven).
    # Previously hardcoded [8.69, 9.51, 10.61]. Now fitted at runtime so the
    # pipeline adapts to any site with different angular dispersion.
    from scipy.stats import vonmises as _vonmises_dist
    _all_cal_lines = np.vstack(load_panels_raw()[:7])
    _all_cal_fams = assign_family_ids(_all_cal_lines, fam_centers,
                                       threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    _hardcoded_kappa = [8.69, 9.51, 10.61]
    fam_kappa_sampling = np.zeros(3)
    for _f in range(3):
        _angles = _all_cal_lines[_all_cal_fams == _f, 3]
        if len(_angles) >= 5:
            # Fit kappa using circular variance: kappa ~ 1/circular_var
            _C = np.cos(_angles).mean()
            _S = np.sin(_angles).mean()
            _R = math.sqrt(_C**2 + _S**2)
            # Approximation: kappa ~ R*(2 - R^2) / (1 - R^2) for R > 0
            if _R > 0.01 and _R < 0.999:
                fam_kappa_sampling[_f] = _R * (2.0 - _R**2) / (1.0 - _R**2)
            else:
                fam_kappa_sampling[_f] = _hardcoded_kappa[_f]
        else:
            fam_kappa_sampling[_f] = _hardcoded_kappa[_f]
    print(f"    [kappa] Fitted: F0={fam_kappa_sampling[0]:.2f} F1={fam_kappa_sampling[1]:.2f} "
          f"F2={fam_kappa_sampling[2]:.2f} (was hardcoded: {_hardcoded_kappa})")

    # FIX 1 + FIX 4: Load panels ONCE, compute canonical p95 from ALL 9 panels.
    # This single set of bounds is used for: indicated clamp, post-merge clamp,
    # and inferred candidate clamp. No shadowing.
    all_panels_cache = load_panels_raw()
    all_lines_all = np.vstack(all_panels_cache)
    all_fams_all = assign_family_ids(all_lines_all, fam_centers,
                                      threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    canonical_p05 = np.zeros(3)
    canonical_p95 = np.zeros(3)
    for f in range(3):
        sub = all_lines_all[all_fams_all == f, 2]
        sub = sub[sub > 1e-6]
        if len(sub) >= 10:
            canonical_p05[f] = float(np.percentile(sub, 5))
            canonical_p95[f] = float(np.percentile(sub, 95))
        else:
            all_sys = all_lines_all[all_fams_all < 3, 2]
            canonical_p05[f] = float(np.percentile(all_sys, 5))
            canonical_p95[f] = float(np.percentile(all_sys, 95))

    target_counts = dict(gstats["target_counts"])
    spacing_floor = dict(gstats["spacing_floor"])

    srcs = {"indicated": 0, "trans": 0, "stat": 0}

    # Load continuation-length predictor if available
    _cont_model = None
    _cont_X_mean = None
    _cont_X_std = None
    _cont_ckpt = EXP / "outputs" / "models" / "continuation_predictor.pt"
    if _cont_ckpt.exists():
        try:
            from continuation_model import ContinuationLengthPredictor
            import torch as _t
            _ck = _t.load(_cont_ckpt, map_location="cpu", weights_only=False)
            _cont_model = ContinuationLengthPredictor(input_dim=_ck.get("input_dim", 11))
            _cont_model.load_state_dict(_ck["model_state_dict"])
            _cont_model.eval()
            _cont_X_mean = _ck["X_mean"]
            _cont_X_std = _ck["X_std"]
        except Exception:
            _cont_model = None

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1 — MEASURED
    # ══════════════════════════════════════════════════════════════════════════
    measured_arr = np.asarray(measured, dtype=np.float32).reshape(-1, 4)
    # FIX 6: single canonical threshold for all family assignment
    meas_fams = assign_family_ids(measured_arr, fam_centers,
                                   threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)

    measured_dicts = []
    for mi, (row, fam) in enumerate(zip(measured_arr, meas_fams)):
        cx, cy, L, A = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        cosA, sinA = math.cos(A), math.sin(A)
        x1 = cx - 0.5 * L * cosA
        y1 = cy - 0.5 * L * sinA
        x2 = cx + 0.5 * L * cosA
        y2 = cy + 0.5 * L * sinA
        measured_dicts.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy, "length": L, "angle": A,
            "family": int(fam),   # FIX 2: int throughout
            "semantic_role": "measured",
            "parent_trace_id": f"meas_{mi:04d}",
        })

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 — INDICATED
    # FIX 4: reuse all_panels_cache instead of calling load_panels_raw() again
    # ══════════════════════════════════════════════════════════════════════════
    all_lines_train = np.vstack(all_panels_cache[:7])
    all_fams_train = assign_family_ids(all_lines_train, fam_centers,
                                        threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    fam_lengths = {}
    for f in range(3):
        sub = all_lines_train[all_fams_train == f, 2]
        sub = sub[np.isfinite(sub)]
        fam_lengths[f] = sub if len(sub) > 0 else np.array([0.08])

    # Tier 2 extensions are near-certain continuations of observed fractures,
    # not speculative infill. No budget limit — every eligible clip gets extended.
    ind_budget = {f: 9999 for f in range(3)}

    # FIX 1: pass canonical_p95 (not a shadowed variable)
    indicated, n_touch = _build_indicated_independent(
        measured_dicts=measured_dicts,
        fam_lengths=fam_lengths,
        fam_p95_indicated={f: float(canonical_p95[f]) for f in range(3)},
        y_max=y_max,
        strips=strips,
        rng=np.random.default_rng(seed + 199000),
        ind_budget=ind_budget,
        cont_model=_cont_model,
        cont_X_mean=_cont_X_mean,
        cont_X_std=_cont_X_std,
    )
    srcs["indicated"] = n_touch

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: MERGE measured + indicated → parent lines
    # ══════════════════════════════════════════════════════════════════════════
    merged = _merge_fixed_lines(measured_dicts, indicated)

    # FIX 1: Post-merge geology clamp uses canonical_p95 (same source as inferred)
    for l in merged:
        fam_id = int(l["family"])  # FIX 2: always int now, no isinstance check
        if fam_id <= 2 and float(l["length"]) > canonical_p95[fam_id]:
            l["length"] = float(canonical_p95[fam_id])
            A = float(l["angle"])
            cx, cy = float(l["cx"]), float(l["cy"])
            half = 0.5 * float(l["length"])
            l["x1"] = cx - half * math.cos(A)
            l["y1"] = cy - half * math.sin(A)
            l["x2"] = cx + half * math.cos(A)
            l["y2"] = cy + half * math.sin(A)

    # Convert merged dicts to arrays
    merged_lines = []
    merged_fams = []
    merged_tiers = []
    for l in merged:
        role = str(l.get("semantic_role", "measured"))
        merged_lines.append(np.array([l["cx"], l["cy"], l["length"], l["angle"]],
                                      dtype=np.float32))
        merged_fams.append(int(l["family"]))  # FIX 2: direct int, no fallback
        if role == "fixed_merged":
            merged_tiers.append("indicated")
        elif role == "measured":
            merged_tiers.append("measured")
        else:
            merged_tiers.append("indicated")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3.5: Cross-borehole indicated merge
    # Collinear indicated extensions from different boreholes are joined
    # BEFORE needed count is computed. Eliminates double-counting.
    # ══════════════════════════════════════════════════════════════════════════
    if len(merged_lines) >= 2:
        _ml_arr = np.array(merged_lines, dtype=np.float32)
        _mf_arr = np.array(merged_fams, dtype=int)
        _mt_arr = np.array([1 if t == "measured" else (2 if t == "indicated" else 3)
                            for t in merged_tiers], dtype=int)
        _p95_pre = {}
        for _f in range(3):
            _sub = all_lines_all[all_fams_all == _f, 2]
            _sub = _sub[_sub > 1e-6]
            _p95_pre[_f] = float(np.percentile(_sub, 95)) if len(_sub) >= 10 else 1.0
        _ml_arr, _mf_arr, _mt_arr, _s35 = apply_continuity_rules(
            _ml_arr, _mf_arr,
            angle_tol_deg=8.0, perp_tol=0.01, gap_frac=0.6, gap_floor=0.04,
            check_t_junctions=True, max_len_p95=_p95_pre, tier_ids=_mt_arr)
        # Rebuild lists from merged arrays
        merged_lines = [_ml_arr[i] for i in range(len(_ml_arr))]
        merged_fams = [int(_mf_arr[i]) for i in range(len(_mf_arr))]
        merged_tiers = []
        for _t in _mt_arr:
            if int(_t) == 1: merged_tiers.append("measured")
            elif int(_t) == 2: merged_tiers.append("indicated")
            else: merged_tiers.append("inferred")
        _n35 = _s35.get("n_chains", 0)
        print(f"    [stage3.5] cross-borehole merges: {_n35}")
    else:
        _n35 = 0

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 3 — INFERRED (upgraded multi-objective scorer)
    # ══════════════════════════════════════════════════════════════════════════
    current = {f: 0 for f in range(3)}
    for f in merged_fams:
        if f < 3:
            current[f] += 1
    needed = {f: max(0, target_counts[f] - current[f]) for f in range(3)}

    # Transformer proposals
    tv4 = generate_v4(trans_v4, evae_v4, measured_arr, int(n_target * 1.5),
                       ps, device, strips=strips, temperature=1.0, seed=seed)
    # FIX 6: use canonical threshold (not 90°) so random lines stay random
    tv4_fams = assign_family_ids(tv4, fam_centers,
                                  threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)

    known = merged_lines[:]
    known_fams = merged_fams[:]

    inferred_lines = []
    inferred_fams = []

    # ── Training-derived reference statistics for scoring ────────────────────
    cm_ref = gstats.get("centre_mass_ref", {f: {"cx_mean": 0.5, "cx_std": 0.1,
                                                  "cy_mean": 0.15, "cy_std": 0.05}
                                              for f in range(3)})
    xbin_ref = np.asarray(gstats.get("xbin_ref", np.ones(5) / 5.0))
    xbin_std_ref = np.asarray(gstats.get("xbin_std", np.ones(5) * 0.05))
    count_std = gstats.get("count_std", {f: 3.0 for f in range(3)})
    inter_mean = float(gstats.get("intersection_mean", 0.5))
    inter_std = float(gstats.get("intersection_std", 0.1))
    inter_limit = float(gstats.get("intersection_limit", inter_mean + 2 * inter_std))
    fam_len_ref = gstats.get("fam_length_ref", {f: {"p90": 0.2, "p95": 0.3,
                                                       "mean": 0.1, "std": 0.05}
                                                   for f in range(3)})

    # ── Production scorer (HC with auto-calibrated inverse-variance weights) ──
    hc_scorer = HCScorer.from_gstats(gstats)
    W_SPACING  = hc_scorer.W_SPACING
    W_XBIN     = hc_scorer.W_XBIN
    W_FAM_DEF  = hc_scorer.W_FAM_DEF
    W_CM_DEV   = hc_scorer.W_CM_DEV
    W_F2_TAIL  = hc_scorer.W_F2_TAIL
    W_LEN_QUAL = hc_scorer.W_LEN_QUAL
    W_INTER    = hc_scorer.W_INTER

    # ── Multi-objective inferred placement ───────────────────────────────────
    XBINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def _xbin_props(lines_list):
        if len(lines_list) == 0:
            return np.zeros(5)
        cx_vals = np.array([float(l[0]) for l in lines_list])
        h = np.histogram(cx_vals, bins=XBINS)[0].astype(float)
        s = h.sum()
        return h / s if s > 0 else np.zeros(5)

    def _centre_mass(lines_list, fam_filter=None, fam_list=None):
        """Compute cx_mean, cy_mean for lines (optionally filtered by family)."""
        if fam_filter is not None and fam_list is not None:
            pts = [(float(l[0]), float(l[1])) for l, f in zip(lines_list, fam_list)
                   if int(f) == fam_filter]
        else:
            pts = [(float(l[0]), float(l[1])) for l in lines_list]
        if not pts:
            return 0.5, 0.15
        cx_arr = np.array([p[0] for p in pts])
        cy_arr = np.array([p[1] for p in pts])
        return float(cx_arr.mean()), float(cy_arr.mean())

    # F2-specific length distribution from calibration data (data-driven)
    # F2 is structurally different (50% longer, different stress regime)
    all_lines_for_f2 = np.vstack(load_panels_raw()[:7])
    all_fams_for_f2 = assign_family_ids(all_lines_for_f2, fam_centers,
                                         threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    f2_cal_lens = all_lines_for_f2[all_fams_for_f2 == 2, 2]
    f2_cal_lens = f2_cal_lens[f2_cal_lens > 1e-6]
    if len(f2_cal_lens) >= 10:
        f2_logmu_cal = float(np.log(f2_cal_lens).mean())
        f2_logsig_cal = float(np.log(f2_cal_lens).std())
    else:
        f2_logmu_cal = float(fam_logmu[2])
        f2_logsig_cal = float(fam_logsig[2])

    # Generate candidate pool for all families
    all_cands = []
    cand_counts = {f: 0 for f in range(3)}
    for fam in range(3):
        n_need = needed[fam]
        if n_need <= 0:
            continue
        # F2 gets 2x oversampling (systematically undercounted)
        mult = CANDIDATE_MULT * 2 if fam == 2 else CANDIDATE_MULT
        attempts = 0
        max_stat = n_need * mult
        stat_count = 0
        while stat_count < max_stat and attempts < max_stat * 2:
            attempts += 1
            cx = float(rng.uniform(0.0, 1.0))
            cy = float(rng.uniform(0.0, y_max))
            # F2: use calibration-derived length distribution (not pooled)
            if fam == 2:
                L = float(np.exp(rng.normal(f2_logmu_cal, f2_logsig_cal)))
            else:
                L = float(np.exp(rng.normal(float(fam_logmu[fam]), float(fam_logsig[fam]))))
            L = float(np.clip(L, canonical_p05[fam], canonical_p95[fam]))
            A = float(rng.vonmises(float(fam_centers[fam]),
                                   float(fam_kappa_sampling[fam])) % math.pi)
            if _line_intrudes_strip(cx, cy, L, A, strips):
                continue
            x1 = cx - 0.5 * L * math.cos(A)
            x2 = cx + 0.5 * L * math.cos(A)
            y1 = cy - 0.5 * L * math.sin(A)
            y2 = cy + 0.5 * L * math.sin(A)
            if min(x1, x2) < -0.01 or max(x1, x2) > 1.01:
                continue
            if min(y1, y2) < -0.01 or max(y1, y2) > y_max + 0.01:
                continue
            all_cands.append((cx, cy, L, A, fam, "stat"))
            stat_count += 1
            cand_counts[fam] += 1
    print(f"    [tier3] candidate pool: F0={cand_counts[0]} F1={cand_counts[1]} F2={cand_counts[2]} "
          f"(F2 logmu={f2_logmu_cal:.3f} vs pooled={float(fam_logmu[2]):.3f})")

    # FIX 8: Add transformer proposals to the candidate pool.
    # Previously, tv4 was generated (line 625) but never added to all_cands,
    # meaning the transformer contributed zero fractures to the final output.
    for row, fam in zip(tv4, tv4_fams):
        fam_i = int(fam)
        if fam_i > 2:
            continue  # skip random-classified proposals
        if needed.get(fam_i, 0) <= 0:
            continue
        cx, cy, L, A = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        # Enforce same strip-exclusion and bounds as statistical candidates
        if _line_intrudes_strip(cx, cy, L, A, strips):
            continue
        L = float(np.clip(L, canonical_p05[fam_i], canonical_p95[fam_i]))
        x1 = cx - 0.5 * L * math.cos(A)
        x2 = cx + 0.5 * L * math.cos(A)
        y1 = cy - 0.5 * L * math.sin(A)
        y2 = cy + 0.5 * L * math.sin(A)
        if min(x1, x2) < -0.01 or max(x1, x2) > 1.01:
            continue
        if min(y1, y2) < -0.01 or max(y1, y2) > y_max + 0.01:
            continue
        all_cands.append((cx, cy, L, A, fam_i, "trans"))
        srcs["trans"] += 1

    # Data-derived length bounds for hard geological gates (p2.5 and p97.5)
    all_lines_for_bounds = np.vstack(load_panels_raw())
    all_fams_for_bounds = assign_family_ids(all_lines_for_bounds, fam_centers,
                                             threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    fam_len_p025 = {}
    fam_len_p975 = {}
    for f in range(3):
        sub = all_lines_for_bounds[all_fams_for_bounds == f, 2]
        sub = sub[sub > 1e-6]
        if len(sub) >= 10:
            fam_len_p025[f] = float(np.percentile(sub, 2.5))
            fam_len_p975[f] = float(np.percentile(sub, 97.5))
        else:
            fam_len_p025[f] = 0.01
            fam_len_p975[f] = 0.5

    # Greedy placement with multi-objective scoring
    total_needed = sum(needed.values())
    placed_count = {f: 0 for f in range(3)}
    all_placed = list(known)  # existing context for occupancy scoring
    all_placed_fams = list(known_fams)

    for _ in range(total_needed):
        if not all_cands:
            break
        best_idx = -1
        best_score = -np.inf
        # Score a random subsample for speed (score top 200 candidates)
        indices = list(range(len(all_cands)))
        if len(indices) > 200:
            indices = list(rng.choice(len(all_cands), 200, replace=False))

        cur_props = _xbin_props(all_placed)

        for ci in indices:
            cx, cy, L, A, fam, src = all_cands[ci]
            if placed_count[fam] >= needed[fam]:
                continue

            # ── Hard geological gates (physics constraints, applied before scoring) ──

            # Gate 1: Spacing floor (stress shadow — always applies)
            same_fam = [ex for ex, ef in zip(known + inferred_lines,
                                              known_fams + inferred_fams) if int(ef) == fam]
            sf_val = float(spacing_floor[fam])
            if same_fam:
                dmin = min(math.sqrt((cx - float(ex[0]))**2 + (cy - float(ex[1]))**2)
                           for ex in same_fam)
                if dmin < sf_val * 0.25:
                    continue

            # Gate 2: X-bin saturation (reject if bin already >1.5x reference)
            b = int(np.clip(np.searchsorted(XBINS, cx, side="right") - 1, 0, 4))
            if xbin_ref[b] > 0.05 and cur_props[b] > 1.5 * xbin_ref[b]:
                continue

            # Gate 3: Length sanity (reject if outside data-derived p2.5-p97.5)
            if L < fam_len_p025[fam] or L > fam_len_p975[fam]:
                continue

            # ── Score candidate ──
            if scorer_fn is not None:
                # ABLATION ONLY: external scorer_fn (e.g. MLPScorer from ablation_scorer.py)
                # Production path uses the HC scorer below (scorer_fn=None).
                feats = np.zeros(7, dtype=np.float32)
                # F1: spacing
                if same_fam:
                    feats[0] = min(dmin / max(sf_val, 1e-6), 3.0)
                else:
                    feats[0] = 1.0
                # F2: x-bin
                b = int(np.clip(np.searchsorted(XBINS, cx, side="right") - 1, 0, 4))
                feats[1] = float(np.clip((xbin_ref[b] - cur_props[b]) / max(float(xbin_std_ref[b]), 0.02), -2.0, 3.0))
                # F3: family deficit
                feats[2] = min(float(needed[fam] - placed_count[fam]) / max(float(count_std[fam]), 1.0), 3.0)
                # F4: centre-mass
                cur_cx, cur_cy = _centre_mass(known + inferred_lines, fam, known_fams + inferred_fams)
                n_existing = max(sum(1 for f in (known_fams + inferred_fams) if int(f) == fam), 1)
                proj_cx = (cur_cx * n_existing + cx) / (n_existing + 1)
                proj_cy = (cur_cy * n_existing + cy) / (n_existing + 1)
                feats[3] = min(math.sqrt(((proj_cx - float(cm_ref[fam]["cx_mean"])) / float(cm_ref[fam]["cx_std"]))**2 +
                                         ((proj_cy - float(cm_ref[fam]["cy_mean"])) / float(cm_ref[fam]["cy_std"]))**2), 4.0)
                # F5: F2 tail
                if fam == 2:
                    f2_p90 = float(fam_len_ref[2]["p90"])
                    if L >= f2_p90:
                        feats[4] = min((L - f2_p90) / max(float(fam_len_ref[2]["std"]), 0.01), 2.0)
                    elif L >= float(fam_len_ref[2]["mean"]):
                        feats[4] = 0.2
                # F6: length quality
                feats[5] = min(abs(math.log(max(L, 1e-8)) - float(fam_logmu[fam])) / max(float(fam_logsig[fam]), 0.1), 3.0)
                # F7: intersection proxy
                if len(all_placed) >= 8:
                    cand_row = np.array([cx, cy, L, A], dtype=np.float32)
                    n_cross = sum(1 for ex in all_placed[-20:]
                                  if segments_intersect(cand_row, ex))
                    if n_cross > 3:
                        feats[6] = (n_cross - 3) / max(inter_std, 0.1)
                # Match HC sign convention: F4(cm_dev), F6(len_dev), F7(inter)
                # are penalties in HC (negative weight). Negate so MLP sees
                # the same direction: higher feature = worse candidate.
                feats[3] = -feats[3]  # cm_dev: lower is better
                feats[5] = -feats[5]  # len_dev: lower is better
                feats[6] = -feats[6]  # intersection: lower is better
                score = scorer_fn(feats)
            else:
                # PRODUCTION: HC scorer (inverse-variance calibrated, deterministic)
                score = 0.0

                # 1. Spacing reward
                if same_fam:
                    score += W_SPACING * min(dmin / max(sf_val, 1e-6), 3.0)
                else:
                    score += abs(W_SPACING)

                # 2. X-bin occupancy
                b = int(np.clip(np.searchsorted(XBINS, cx, side="right") - 1, 0, 4))
                xbin_deficit = (xbin_ref[b] - cur_props[b]) / max(float(xbin_std_ref[b]), 0.02)
                score += W_XBIN * np.clip(xbin_deficit, -2.0, 3.0)

                # 3. Family deficit
                fam_deficit = float(needed[fam] - placed_count[fam])
                fam_deficit_norm = fam_deficit / max(float(count_std[fam]), 1.0)
                score += W_FAM_DEF * min(fam_deficit_norm, 3.0)

                # 4. Centre-mass balance
                ref_cx = float(cm_ref[fam]["cx_mean"])
                ref_cy = float(cm_ref[fam]["cy_mean"])
                ref_cx_std = float(cm_ref[fam]["cx_std"])
                ref_cy_std = float(cm_ref[fam]["cy_std"])
                cur_cx, cur_cy = _centre_mass(known + inferred_lines, fam, known_fams + inferred_fams)
                n_existing = max(sum(1 for f in (known_fams + inferred_fams) if int(f) == fam), 1)
                proj_cx = (cur_cx * n_existing + cx) / (n_existing + 1)
                proj_cy = (cur_cy * n_existing + cy) / (n_existing + 1)
                cm_dev = math.sqrt(((proj_cx - ref_cx) / ref_cx_std) ** 2 +
                                   ((proj_cy - ref_cy) / ref_cy_std) ** 2)
                score += W_CM_DEV * min(cm_dev, 4.0)  # W_CM_DEV is negative

                # 5. Tail persistence (any family with long-tail distribution)
                f_p90 = float(fam_len_ref.get(fam, {}).get("p90", 0.2))
                f_std_val = float(fam_len_ref.get(fam, {}).get("std", 0.05))
                f_mean = float(fam_len_ref.get(fam, {}).get("mean", 0.1))
                if L >= f_p90:
                    score += abs(W_F2_TAIL) * min((L - f_p90) / max(f_std_val, 0.01), 2.0)
                elif L >= f_mean:
                    score += abs(W_F2_TAIL) * 0.2

                # 6. Length quality
                len_dev = abs(math.log(max(L, 1e-8)) - float(fam_logmu[fam])) / max(float(fam_logsig[fam]), 0.1)
                score += W_LEN_QUAL * min(len_dev, 3.0)  # W_LEN_QUAL is negative

                # 7. Intersection density soft cap
                if len(all_placed) >= 8:
                    cand_row = np.array([cx, cy, L, A], dtype=np.float32)
                    n_cross = sum(1 for ex in all_placed[-20:]
                                  if segments_intersect(cand_row, ex))
                    if n_cross > 3:
                        score += W_INTER * (n_cross - 3) / max(inter_std, 0.1)  # W_INTER is negative

            if score > best_score:
                best_score = score
                best_idx = ci

        if best_idx < 0:
            break
        cx, cy, L, A, fam, src = all_cands.pop(best_idx)
        inferred_lines.append(np.array([cx, cy, L, A], dtype=np.float32))
        inferred_fams.append(fam)
        all_placed.append(np.array([cx, cy, L, A], dtype=np.float32))
        all_placed_fams.append(fam)
        placed_count[fam] += 1
        # Fix source attribution: track transformer vs statistical correctly
        if src == "trans":
            srcs["trans"] += 1
        else:
            srcs["stat"] += 1

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6.5: Post-greedy merge — inferred lines that continue indicated
    # ══════════════════════════════════════════════════════════════════════════
    if inferred_lines:
        _all_lines_65 = np.array(merged_lines + inferred_lines, dtype=np.float32)
        _all_fams_65 = np.array(merged_fams + inferred_fams, dtype=int)
        _all_tiers_65 = np.array(
            [1 if t == "measured" else (2 if t == "indicated" else 3) for t in merged_tiers] +
            [3] * len(inferred_lines), dtype=int)
        _p95_65 = {}
        for _f in range(3):
            _sub = all_lines_all[all_fams_all == _f, 2]
            _sub = _sub[_sub > 1e-6]
            _p95_65[_f] = float(np.percentile(_sub, 95)) if len(_sub) >= 10 else 1.0
        _merged_65, _mf_65, _mt_65, _s65 = apply_continuity_rules(
            _all_lines_65, _all_fams_65,
            angle_tol_deg=8.0, perp_tol=0.01, gap_frac=0.6, gap_floor=0.04,
            check_t_junctions=True, max_len_p95=_p95_65, tier_ids=_all_tiers_65)
        _n65 = _s65.get("n_chains", 0)
        # Rebuild merged_lines and inferred_lines from merged result
        n_merged_before = len(merged_lines)
        merged_lines = []
        merged_fams = []
        merged_tiers = []
        inferred_lines = []
        inferred_fams = []
        for _i in range(len(_merged_65)):
            _t = int(_mt_65[_i])
            if _t <= 2:
                merged_lines.append(_merged_65[_i])
                merged_fams.append(int(_mf_65[_i]))
                merged_tiers.append("measured" if _t == 1 else "indicated")
            else:
                inferred_lines.append(_merged_65[_i])
                inferred_fams.append(int(_mf_65[_i]))
        print(f"    [stage6.5] post-greedy merges: {_n65}")
    else:
        _n65 = 0

    # Topup — if any family still below 70% of target (relaxed spacing)
    for fam in range(3):
        cur_count = sum(1 for f in (merged_fams + inferred_fams) if f == fam)
        floor = int(round(target_counts[fam] * 0.7))
        if cur_count >= floor:
            continue
        topup_need = floor - cur_count
        topup_added = 0
        for _ in range(topup_need * CANDIDATE_MULT):
            if topup_added >= topup_need:
                break
            cx = float(rng.uniform(0.0, 1.0))
            cy = float(rng.uniform(0.0, y_max))
            L = float(np.exp(rng.normal(float(fam_logmu[fam]), float(fam_logsig[fam]))))
            L = float(np.clip(L, canonical_p05[fam], canonical_p95[fam]))
            A = float(rng.vonmises(float(fam_centers[fam]),
                                   float(fam_kappa_sampling[fam])) % math.pi)
            if _line_intrudes_strip(cx, cy, L, A, strips):
                continue
            x1 = cx - 0.5 * L * math.cos(A)
            x2 = cx + 0.5 * L * math.cos(A)
            y1 = cy - 0.5 * L * math.sin(A)
            y2 = cy + 0.5 * L * math.sin(A)
            if min(x1, x2) < -0.01 or max(x1, x2) > 1.01:
                continue
            if min(y1, y2) < -0.01 or max(y1, y2) > y_max + 0.01:
                continue
            valid = True
            for ex, exf in zip(known + inferred_lines, known_fams + inferred_fams):
                if int(exf) != fam:
                    continue
                d = math.sqrt((cx - float(ex[0]))**2 + (cy - float(ex[1]))**2)
                if d < float(spacing_floor[fam]) * 0.15:
                    valid = False
                    break
            if valid:
                inferred_lines.append(np.array([cx, cy, L, A], dtype=np.float32))
                inferred_fams.append(fam)
                srcs["stat"] += 1
                topup_added += 1

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINE: merged parents + inferred
    # ══════════════════════════════════════════════════════════════════════════
    parts = []
    if merged_lines:
        parts.append(np.vstack(merged_lines))
    if inferred_lines:
        parts.append(np.vstack(inferred_lines))
    final_sys = np.vstack(parts).astype(np.float32) if parts else measured_arr.copy()

    # FIX 3: pass strips to random family generation (was strips=None)
    final, _n_rand = append_random_family(final_sys, measured_arr,
                                           target_counts, ps, strips,
                                           rand_fraction=0.06, seed=seed + 50)

    # Tier labels
    tiers = list(merged_tiers)
    tiers += ["inferred"] * len(inferred_lines)
    n_before_random = len(merged_lines) + len(inferred_lines)
    n_random = len(final) - n_before_random
    tiers += ["inferred"] * n_random

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 9 (moved here): Final continuity merge inside build_hybrid_v6
    # so that score_panel_quality sees merged panels for best-of-K.
    # ══════════════════════════════════════════════════════════════════════════
    _final_fams = assign_family_ids(final, fam_centers,
                                     threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    _tier_ids_9 = np.array([1 if t == "measured" else (2 if t == "indicated" else 3)
                            for t in tiers], dtype=int)
    if len(_tier_ids_9) < len(final):
        _tier_ids_9 = np.concatenate([_tier_ids_9,
                                       np.full(len(final) - len(_tier_ids_9), 3, dtype=int)])
    _p95_9 = {}
    for _f in range(3):
        _sub = all_lines_all[all_fams_all == _f, 2]
        _sub = _sub[_sub > 1e-6]
        _p95_9[_f] = float(np.percentile(_sub, 95)) if len(_sub) >= 10 else 1.0
    final, _final_fams, _tier_ids_9, _s9 = apply_continuity_rules(
        final, _final_fams,
        angle_tol_deg=8.0, perp_tol=0.01, gap_frac=0.6, gap_floor=0.04,
        check_t_junctions=True, max_len_p95=_p95_9, tier_ids=_tier_ids_9)
    _n9 = _s9.get("n_chains", 0)

    # Rebuild tiers from tier_ids
    tiers = []
    for _t in _tier_ids_9:
        if int(_t) == 1: tiers.append("measured")
        elif int(_t) == 2: tiers.append("indicated")
        else: tiers.append("inferred")

    print(f"    [merge summary] stage3.5={_n35} | stage6.5={_n65} | stage9={_n9}")

    # FIX 7: honest source attribution
    srcs["n_frag"] = len(measured_arr)
    srcs["n_merged"] = sum(1 for t in tiers if t == "indicated")
    srcs["n_standalone"] = sum(1 for t in tiers if t == "measured")
    srcs["n_inferred"] = sum(1 for t in tiers if t == "inferred")
    srcs["tiers"] = tiers
    srcs["merge_stages"] = {"stage3.5": _n35, "stage6.5": _n65, "stage9": _n9}
    return final, srcs


# ══════════════════════════════════════════════════════════════════════════════
# Panel-quality scoring (training-derived, NOT ground-truth)
# ══════════════════════════════════════════════════════════════════════════════
def score_panel_quality(generated: np.ndarray, gstats: dict,
                         fam_centers: np.ndarray) -> Tuple[float, Dict]:
    """
    Score a generated panel against training-derived reference statistics.
    Lower score = better match. All penalties are SD-normalised.
    """
    gen = np.asarray(generated, dtype=float).reshape(-1, 4)
    fams = assign_family_ids(gen, fam_centers, threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    penalties = {}

    target_counts = gstats["target_counts"]
    count_std = gstats.get("count_std", {f: 3.0 for f in range(3)})
    cm_ref = gstats.get("centre_mass_ref", {})
    xbin_ref = np.asarray(gstats.get("xbin_ref", np.ones(5) / 5.0))
    xbin_std_ref = np.asarray(gstats.get("xbin_std", np.ones(5) * 0.05))
    inter_mean = float(gstats.get("intersection_mean", 0.5))
    inter_std = float(gstats.get("intersection_std", 0.1))
    spacing_floor = gstats.get("spacing_floor", {f: 0.01 for f in range(3)})
    fam_len_ref = gstats.get("fam_length_ref", {})

    # 1. Family count penalty (SD-normalised)
    count_pen = 0.0
    for f in range(3):
        n_gen = int((fams == f).sum())
        n_tgt = int(target_counts[f])
        cs = float(count_std.get(f, 3.0))
        count_pen += abs(n_gen - n_tgt) / max(cs, 1.0)
    count_pen /= 3.0
    penalties["count"] = count_pen

    # 2. Family proportion L1
    gen_prop = np.array([(fams == k).sum() for k in range(4)], float)
    gen_prop /= max(gen_prop.sum(), 1.0)
    # Training proportion from target_counts
    train_total = sum(target_counts[f] for f in range(3))
    train_prop = np.array([target_counts.get(f, 0) / max(train_total, 1) for f in range(3)]
                           + [0.06], float)  # ~6% random
    train_prop /= train_prop.sum()
    penalties["fam_prop_l1"] = float(np.abs(gen_prop - train_prop).sum())

    # 3. Spatial occupancy (x-bin SD-normalised L1)
    XBINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if len(gen) > 0:
        h = np.histogram(gen[:, 0], bins=XBINS)[0].astype(float)
        h /= max(h.sum(), 1.0)
    else:
        h = np.zeros(5)
    xbin_dev = np.abs(h - xbin_ref) / np.maximum(xbin_std_ref, 0.02)
    penalties["xbin"] = float(xbin_dev.mean())

    # 4. Centre-mass deviation (SD-normalised)
    cm_pen = 0.0
    for f in range(3):
        sub = gen[fams == f]
        if len(sub) < 2 or f not in cm_ref:
            continue
        ref = cm_ref[f]
        cx_dev = abs(float(sub[:, 0].mean()) - ref["cx_mean"]) / ref["cx_std"]
        cy_dev = abs(float(sub[:, 1].mean()) - ref["cy_mean"]) / ref["cy_std"]
        cm_pen += (cx_dev + cy_dev) / 2.0
    cm_pen /= 3.0
    penalties["centre_mass"] = cm_pen

    # 5. Spacing proxy (mean same-family pairwise distance)
    spacing_pen = 0.0
    for f in range(3):
        sub = gen[fams == f, 0:2]
        sf = float(spacing_floor.get(f, 0.01))
        violations = 0
        total_pairs = 0
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                d = float(np.linalg.norm(sub[i] - sub[j]))
                total_pairs += 1
                if d < sf * 0.5:
                    violations += 1
        if total_pairs > 0:
            spacing_pen += violations / total_pairs
    spacing_pen /= 3.0
    penalties["spacing_violations"] = spacing_pen

    # 6. Intersection density (SD-normalised)
    if len(gen) >= 2:
        int_dens = intersection_density(gen)
    else:
        int_dens = 0.0
    int_pen = abs(int_dens - inter_mean) / max(inter_std, 1e-6)
    penalties["intersection"] = float(int_pen)

    # 7. Length distribution match (mean log-length deviation per family)
    len_pen = 0.0
    for f in range(3):
        sub = gen[fams == f, 2]
        if len(sub) < 2 or f not in fam_len_ref:
            continue
        ref = fam_len_ref[f]
        gen_mean = float(np.log(np.maximum(sub, 1e-8)).mean())
        ref_mean = float(np.log(max(ref["mean"], 1e-8)))
        ref_std = float(max(ref["std"], 0.01))
        len_pen += abs(gen_mean - ref_mean) / ref_std
    len_pen /= 3.0
    penalties["length"] = len_pen

    # Weighted total (lower = better)
    # Use same physics-constrained weights as greedy scorer if available
    sw = gstats.get("scorer_weights", None)
    if sw is not None:
        _w_count = abs(float(sw["w_fam_deficit"]))
        _w_prop = abs(float(sw["w_fam_deficit"])) * 1.5  # proportion ~ deficit * 1.5
        _w_xbin = abs(float(sw["w_xbin"]))
        _w_cm = abs(float(sw["w_cm_dev"]))
        _w_sp = abs(float(sw["w_spacing"]))
        _w_int = abs(float(sw["w_intersection"]))
        _w_len = abs(float(sw["w_len_quality"]))
    else:
        _w_count = 2.0
        _w_prop = 1.5
        _w_xbin = 1.5
        _w_cm = 1.0
        _w_sp = 1.0
        _w_int = 1.0
        _w_len = 0.8
    total = (_w_count * count_pen +
             _w_prop * penalties["fam_prop_l1"] +
             _w_xbin * penalties["xbin"] +
             _w_cm * cm_pen +
             _w_sp * spacing_pen +
             _w_int * float(int_pen) +
             _w_len * len_pen)
    penalties["total"] = total
    return total, penalties


# ══════════════════════════════════════════════════════════════════════════════
# Multi-candidate wrapper: generate K candidates, pick best
# ══════════════════════════════════════════════════════════════════════════════
def build_hybrid_v6_best_of_k(
    real, measured, strips, n_target, ps, gstats, fam_centers,
    evae_v4, trans_v4, sp_model, sp_ps, f2_scale: float,
    device, seed: int = 42001, K: int = 5,
    scorer_fn=None,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate K candidate panels with different seeds,
    score each against training-derived statistics,
    return the best one.

    scorer_fn: optional callable(features_7d) -> float score.
               If provided, replaces the HC scorer in the greedy loop.
               All other pipeline stages (Tier 1, 2, topup, continuity) are identical.
    """
    best_final = None
    best_srcs = None
    best_score = np.inf
    best_penalties = None

    for k in range(K):
        kseed = seed + k * 1000
        final_k, srcs_k = build_hybrid_v6(
            real, measured, strips, n_target, ps, gstats, fam_centers,
            evae_v4, trans_v4, sp_model, sp_ps, f2_scale,
            device, seed=kseed, scorer_fn=scorer_fn,
        )
        score_k, pen_k = score_panel_quality(final_k, gstats, fam_centers)
        if score_k < best_score:
            best_score = score_k
            best_final = final_k
            best_srcs = srcs_k
            best_penalties = pen_k

    best_srcs["selection_K"] = K
    best_srcs["selection_score"] = float(best_score)
    best_srcs["selection_penalties"] = best_penalties

    best_fam_ids = assign_family_ids(best_final, fam_centers,
                                      threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    tiers = best_srcs.get("tiers", [])
    tier_ids = np.array([1 if t == "measured" else (2 if t == "indicated" else 3)
                         for t in tiers], dtype=int)
    if len(tier_ids) < len(best_final):
        tier_ids = np.concatenate([tier_ids,
                                   np.full(len(best_final) - len(tier_ids), 3, dtype=int)])

    all_lines_all = np.vstack(load_panels_raw())
    all_fams_all = assign_family_ids(all_lines_all, fam_centers,
                                      threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    p95_map = {}
    for f in range(3):
        sub = all_lines_all[all_fams_all == f, 2]
        sub = sub[sub > 1e-6]
        p95_map[f] = float(np.percentile(sub, 95)) if len(sub) >= 10 else 1.0

    best_final, best_fam_ids, tier_ids, merge_stats = apply_continuity_rules(
        best_final, best_fam_ids,
        angle_tol_deg=8.0, perp_tol=0.01, gap_frac=0.6, gap_floor=0.04,
        check_t_junctions=True, max_len_p95=p95_map, tier_ids=tier_ids)
    best_srcs["continuity_merge_stats"] = merge_stats

    new_tiers = []
    for t in tier_ids:
        if t == 1:
            new_tiers.append("measured")
        elif t == 2:
            new_tiers.append("indicated")
        else:
            new_tiers.append("inferred")
    best_srcs["tiers"] = new_tiers

    return best_final, best_srcs


def apply_continuity_rules(
    panel, fam_ids,
    angle_tol_deg=5.0, perp_tol=0.005,
    gap_frac=0.5, gap_floor=0.02,
    check_t_junctions=True, max_len_p95=None,
    tier_ids=None,
):
    panel = np.asarray(panel, dtype=float).reshape(-1, 4)
    fam_ids = np.asarray(fam_ids, dtype=int).reshape(-1)
    N = len(panel)
    if tier_ids is None:
        tier_ids = np.full(N, 3, dtype=int)
    else:
        tier_ids = np.asarray(tier_ids, dtype=int).reshape(-1)

    # Per-family angle tolerance from calibration data (2 * circular_std)
    # Clamped to [2, 6] degrees — never merge across sub-families
    _cal_lines = np.vstack(load_panels_raw()[:7])
    _cal_fams = assign_family_ids(_cal_lines, np.asarray(
        get_physics_stats()["fam_centers_rad"], dtype=float),
        threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    fam_tol_rad = {}
    for _f in range(3):
        _angs = _cal_lines[_cal_fams == _f, 3]
        if len(_angs) >= 5:
            _C = np.cos(2 * _angs).mean()
            _S = np.sin(2 * _angs).mean()
            _R = math.sqrt(_C**2 + _S**2)
            _circ_std_deg = math.degrees(math.sqrt(-2.0 * math.log(max(_R, 1e-6))) / 2.0)
            _tol_deg = max(2.0, min(6.0, 2.0 * _circ_std_deg))
        else:
            _tol_deg = float(angle_tol_deg)
        fam_tol_rad[_f] = math.radians(_tol_deg)
    print(f"    [angle_tol] F0={math.degrees(fam_tol_rad[0]):.1f}deg "
          f"F1={math.degrees(fam_tol_rad[1]):.1f}deg "
          f"F2={math.degrees(fam_tol_rad[2]):.1f}deg")
    # Fallback for ep_dists sampling (use max tolerance)
    tol_rad = max(fam_tol_rad.values())

    def endpoints(row):
        cx, cy, L, A = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        dx = 0.5 * L * math.cos(A)
        dy = 0.5 * L * math.sin(A)
        return (cx - dx, cy - dy), (cx + dx, cy + dy)

    def angle_diff(a1, a2):
        d = abs(float(a1) - float(a2))
        return min(d, math.pi - d)

    def perp_distance(p, line_pt, line_dir):
        vx = float(p[0]) - float(line_pt[0])
        vy = float(p[1]) - float(line_pt[1])
        proj = vx * math.cos(line_dir) + vy * math.sin(line_dir)
        px = vx - proj * math.cos(line_dir)
        py = vy - proj * math.sin(line_dir)
        return math.sqrt(px * px + py * py)

    def along_gap(ep1a, ep1b, ep2a, ep2b, direction):
        def proj(pt):
            return float(pt[0]) * math.cos(direction) + float(pt[1]) * math.sin(direction)
        vals1 = sorted([proj(ep1a), proj(ep1b)])
        vals2 = sorted([proj(ep2a), proj(ep2b)])
        gap = max(0.0, max(vals2[0] - vals1[1], vals1[0] - vals2[1]))
        return gap

    # FIX 1: Derive endpoint distance threshold from data
    # Use median of shortest 10% inter-segment distances as baseline
    all_eps = [endpoints(panel[i]) for i in range(N)]
    ep_dists_sample = []
    for i in range(min(N, 200)):
        if int(fam_ids[i]) > 2:
            continue
        for j in range(i + 1, min(N, 200)):
            if int(fam_ids[j]) > 2 or int(fam_ids[i]) != int(fam_ids[j]):
                continue
            if angle_diff(panel[i, 3], panel[j, 3]) > tol_rad:
                continue
            ep_i, ep_j = all_eps[i], all_eps[j]
            d_min = min(
                math.sqrt((ep_i[0][0]-ep_j[0][0])**2 + (ep_i[0][1]-ep_j[0][1])**2),
                math.sqrt((ep_i[0][0]-ep_j[1][0])**2 + (ep_i[0][1]-ep_j[1][1])**2),
                math.sqrt((ep_i[1][0]-ep_j[0][0])**2 + (ep_i[1][1]-ep_j[0][1])**2),
                math.sqrt((ep_i[1][0]-ep_j[1][0])**2 + (ep_i[1][1]-ep_j[1][1])**2),
            )
            ep_dists_sample.append(d_min)
    if len(ep_dists_sample) > 10:
        ep_dists_arr = np.array(ep_dists_sample)
        ep_threshold = float(np.percentile(ep_dists_arr[ep_dists_arr > 0.001], 15))
        ep_threshold = max(ep_threshold, 0.02)  # floor
        ep_threshold = min(ep_threshold, 0.06)  # cap
    else:
        ep_threshold = 0.04  # fallback

    n_candidates_found = 0
    n_blocked_tjunction = 0

    adj = [[] for _ in range(N)]
    for i in range(N):
        if int(fam_ids[i]) > 2:
            continue
        ep_i = all_eps[i]
        for j in range(i + 1, N):
            if int(fam_ids[j]) > 2:
                continue
            if int(fam_ids[i]) != int(fam_ids[j]):
                continue
            _fam_ij = int(fam_ids[i])
            if angle_diff(panel[i, 3], panel[j, 3]) > fam_tol_rad.get(_fam_ij, tol_rad):
                continue

            # FIX 1: Use endpoint-to-endpoint distance instead of midpoint perpendicular
            ep_j = all_eps[j]
            d00 = math.sqrt((ep_i[0][0]-ep_j[0][0])**2 + (ep_i[0][1]-ep_j[0][1])**2)
            d01 = math.sqrt((ep_i[0][0]-ep_j[1][0])**2 + (ep_i[0][1]-ep_j[1][1])**2)
            d10 = math.sqrt((ep_i[1][0]-ep_j[0][0])**2 + (ep_i[1][1]-ep_j[0][1])**2)
            d11 = math.sqrt((ep_i[1][0]-ep_j[1][0])**2 + (ep_i[1][1]-ep_j[1][1])**2)
            d_endpoint = min(d00, d01, d10, d11)

            if d_endpoint > ep_threshold:
                continue

            # Also check that merged length wouldn't be absurd
            # (skip if segments overlap significantly)
            avg_dir = float(panel[i, 3])
            gap = along_gap(ep_i[0], ep_i[1], ep_j[0], ep_j[1], avg_dir)
            # Adaptive gap floor: if endpoints are close, allow gap up to 1.2x ep_dist
            # This catches collinear pairs where gap ~ endpoint distance
            adaptive_floor = max(gap_floor, d_endpoint * 1.2)
            max_gap = max(gap_frac * min(float(panel[i, 2]), float(panel[j, 2])), adaptive_floor)
            if gap > max_gap:
                continue

            n_candidates_found += 1

            # FIX 2: T-junction check — only block if cross-family fracture
            # actually intersects the GAP SEGMENT (line between near endpoints)
            if check_t_junctions:
                # Find the two near-endpoints that form the gap
                pairs = [(d00, ep_i[0], ep_j[0]), (d01, ep_i[0], ep_j[1]),
                         (d10, ep_i[1], ep_j[0]), (d11, ep_i[1], ep_j[1])]
                pairs.sort(key=lambda x: x[0])
                _, near_a, near_b = pairs[0]
                # Build gap segment
                gap_cx = 0.5 * (near_a[0] + near_b[0])
                gap_cy = 0.5 * (near_a[1] + near_b[1])
                gap_L = math.sqrt((near_b[0]-near_a[0])**2 + (near_b[1]-near_a[1])**2)
                if gap_L < 1e-8:
                    gap_A = avg_dir
                else:
                    gap_A = math.atan2(near_b[1]-near_a[1], near_b[0]-near_a[0]) % math.pi
                gap_row = np.array([gap_cx, gap_cy, gap_L + 1e-6, gap_A], dtype=np.float32)

                blocked = False
                for k in range(N):
                    if k == i or k == j:
                        continue
                    if int(fam_ids[k]) == int(fam_ids[i]):
                        continue  # same family can't block
                    # Only block if cross-family segment actually intersects the gap
                    if segments_intersect(gap_row, panel[k]):
                        blocked = True
                        break
                if blocked:
                    n_blocked_tjunction += 1
                    continue

            adj[i].append(j)
            adj[j].append(i)

    visited = [False] * N
    chains = []
    for start in range(N):
        if visited[start] or int(fam_ids[start]) > 2:
            continue
        if not adj[start]:
            continue
        chain = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            chain.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    stack.append(nb)
        if len(chain) >= 2:
            chains.append(chain)

    merged_indices = set()
    new_rows = []
    new_fams = []
    new_tiers = []
    n_merged_chains = 0
    n_segments_merged = 0

    angle_deviations = []

    for chain in chains:
        fam = int(np.bincount([int(fam_ids[idx]) for idx in chain]).argmax())

        # Projection-based merge (from _merge_collinear_group pattern):
        # 1. Compute dominant direction as length-weighted circular mean
        sum_sin = 0.0
        sum_cos = 0.0
        total_weight = 0.0
        for idx in chain:
            seg_L = float(panel[idx, 2])
            seg_A = float(panel[idx, 3])
            # Use doubled-angle for circular mean (angles are mod pi)
            sum_sin += seg_L * math.sin(2.0 * seg_A)
            sum_cos += seg_L * math.cos(2.0 * seg_A)
            total_weight += seg_L
        dom_dir = 0.5 * math.atan2(sum_sin, sum_cos) % math.pi

        # 2. Project all endpoints onto dominant direction axis
        all_pts = []
        for idx in chain:
            ep = endpoints(panel[idx])
            all_pts.append(ep[0])
            all_pts.append(ep[1])

        cos_d = math.cos(dom_dir)
        sin_d = math.sin(dom_dir)
        projections = []
        for pt in all_pts:
            proj_val = pt[0] * cos_d + pt[1] * sin_d
            projections.append((proj_val, pt))

        # 3. Merged line = min_proj to max_proj along dominant direction
        projections.sort(key=lambda x: x[0])
        proj_min_val, _ = projections[0]
        proj_max_val, _ = projections[-1]

        # Reconstruct endpoints from projections onto the dominant direction
        # Use the centre line perpendicular position (mean of all midpoints)
        perp_vals = [pt[0] * (-sin_d) + pt[1] * cos_d for pt in all_pts]
        perp_mean = sum(perp_vals) / len(perp_vals)

        p1_x = proj_min_val * cos_d + perp_mean * (-sin_d)
        p1_y = proj_min_val * sin_d + perp_mean * cos_d
        p2_x = proj_max_val * cos_d + perp_mean * (-sin_d)
        p2_y = proj_max_val * sin_d + perp_mean * cos_d

        merged_L = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
        if max_len_p95 is not None and fam in max_len_p95:
            if merged_L > max_len_p95[fam]:
                continue

        m_cx = 0.5 * (p1_x + p2_x)
        m_cy = 0.5 * (p1_y + p2_y)

        # Track angle deviation for diagnostics
        for idx in chain:
            dev = abs(float(panel[idx, 3]) - dom_dir)
            dev = min(dev, math.pi - dev)
            angle_deviations.append(math.degrees(dev))

        for idx in chain:
            merged_indices.add(idx)
        new_rows.append([m_cx, m_cy, merged_L, dom_dir])
        new_fams.append(fam)
        new_tiers.append(max(int(tier_ids[idx]) for idx in chain))
        n_merged_chains += 1
        n_segments_merged += len(chain)

    keep = [i for i in range(N) if i not in merged_indices]
    kept_panel = panel[keep] if keep else np.zeros((0, 4), dtype=float)
    kept_fams = fam_ids[keep] if keep else np.array([], dtype=int)
    kept_tiers = tier_ids[keep] if keep else np.array([], dtype=int)

    if new_rows:
        new_panel = np.array(new_rows, dtype=float)
        out_panel = np.vstack([kept_panel, new_panel]).astype(np.float32)
        out_fams = np.concatenate([kept_fams, np.array(new_fams, dtype=int)])
        out_tiers = np.concatenate([kept_tiers, np.array(new_tiers, dtype=int)])
    else:
        out_panel = kept_panel.astype(np.float32)
        out_fams = kept_fams
        out_tiers = kept_tiers

    # ── PASS 2: Narrow-intersection merge ──────────────────────────────────
    # Two same-family segments that physically intersect at a very narrow angle
    # are geologically one trace — merge them into one line.
    # Threshold: half of the narrowest real intersection angle in calibration data.
    _cal = np.vstack(load_panels_raw()[:7])
    _cal_f = assign_family_ids(_cal, np.asarray(
        get_physics_stats()["fam_centers_rad"], dtype=float),
        threshold_deg=FAM_ASSIGN_THRESHOLD_DEG)
    # Find narrowest same-family intersection angle in calibration
    narrow_angles = []
    for _f in range(3):
        _sub = _cal[_cal_f == _f]
        for _i in range(len(_sub)):
            for _j in range(_i + 1, min(len(_sub), _i + 30)):
                if segments_intersect(_sub[_i], _sub[_j]):
                    _ad = abs(float(_sub[_i, 3]) - float(_sub[_j, 3]))
                    _ad = min(_ad, math.pi - _ad)
                    if _ad > 0.001:  # skip exact duplicates
                        narrow_angles.append(math.degrees(_ad))
    if len(narrow_angles) > 5:
        min_real_angle = float(np.percentile(narrow_angles, 5))
        narrow_thresh_deg = max(2.0, min_real_angle * 0.5)
    else:
        narrow_thresh_deg = 3.0
    narrow_thresh_rad = math.radians(narrow_thresh_deg)

    # Scan output panel for narrow-intersection same-family pairs
    n_narrow_merged = 0
    M = len(out_panel)
    narrow_remove = set()
    narrow_new_rows = []
    narrow_new_fams = []
    narrow_new_tiers = []

    for i in range(M):
        if i in narrow_remove or int(out_fams[i]) > 2:
            continue
        for j in range(i + 1, M):
            if j in narrow_remove or int(out_fams[j]) > 2:
                continue
            if int(out_fams[i]) != int(out_fams[j]):
                continue
            ad = angle_diff(out_panel[i, 3], out_panel[j, 3])
            if ad >= narrow_thresh_rad:
                continue
            # Check if they actually intersect
            if not segments_intersect(out_panel[i], out_panel[j]):
                continue
            # Merge: projection-based along length-weighted mean direction
            fam_m = int(out_fams[i])
            L_i, A_i = float(out_panel[i, 2]), float(out_panel[i, 3])
            L_j, A_j = float(out_panel[j, 2]), float(out_panel[j, 3])
            w_sin = L_i * math.sin(2*A_i) + L_j * math.sin(2*A_j)
            w_cos = L_i * math.cos(2*A_i) + L_j * math.cos(2*A_j)
            dom = 0.5 * math.atan2(w_sin, w_cos) % math.pi
            ep_i = endpoints(out_panel[i])
            ep_j = endpoints(out_panel[j])
            all_p = [ep_i[0], ep_i[1], ep_j[0], ep_j[1]]
            cd, sd = math.cos(dom), math.sin(dom)
            projs = [(p[0]*cd + p[1]*sd, p) for p in all_p]
            projs.sort(key=lambda x: x[0])
            perps = [p[0]*(-sd) + p[1]*cd for p in all_p]
            pm = sum(perps) / len(perps)
            p1x = projs[0][0]*cd + pm*(-sd)
            p1y = projs[0][0]*sd + pm*cd
            p2x = projs[-1][0]*cd + pm*(-sd)
            p2y = projs[-1][0]*sd + pm*cd
            mL = math.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)
            # Length cap
            if max_len_p95 is not None and fam_m in max_len_p95:
                if mL > max_len_p95[fam_m]:
                    continue
            narrow_remove.add(i)
            narrow_remove.add(j)
            narrow_new_rows.append([0.5*(p1x+p2x), 0.5*(p1y+p2y), mL, dom])
            narrow_new_fams.append(fam_m)
            narrow_new_tiers.append(max(int(out_tiers[i]), int(out_tiers[j])))
            n_narrow_merged += 1
            break  # segment i is consumed, move to next i

    if narrow_remove:
        keep2 = [i for i in range(M) if i not in narrow_remove]
        out_panel_2 = out_panel[keep2] if keep2 else np.zeros((0, 4), dtype=np.float32)
        out_fams_2 = out_fams[keep2] if keep2 else np.array([], dtype=int)
        out_tiers_2 = out_tiers[keep2] if keep2 else np.array([], dtype=int)
        if narrow_new_rows:
            out_panel = np.vstack([out_panel_2, np.array(narrow_new_rows, dtype=np.float32)]).astype(np.float32)
            out_fams = np.concatenate([out_fams_2, np.array(narrow_new_fams, dtype=int)])
            out_tiers = np.concatenate([out_tiers_2, np.array(narrow_new_tiers, dtype=int)])
        else:
            out_panel = out_panel_2
            out_fams = out_fams_2
            out_tiers = out_tiers_2

    stats = {
        "n_chains": n_merged_chains,
        "n_segments_merged": n_segments_merged,
        "n_new_lines": len(new_rows),
        "n_before": N,
        "n_after": len(out_panel),
        "count_change_pct": (len(out_panel) - N) / max(N, 1) * 100,
        "n_candidates_found": n_candidates_found,
        "n_blocked_tjunction": n_blocked_tjunction,
        "ep_threshold": ep_threshold,
        "n_narrow_merged": n_narrow_merged,
        "narrow_thresh_deg": narrow_thresh_deg,
    }
    avg_dev = float(np.mean(angle_deviations)) if angle_deviations else 0.0
    print(f"    [continuity] ep_thresh={ep_threshold:.4f} | "
          f"candidates={n_candidates_found} | blocked_T={n_blocked_tjunction} | "
          f"merged={n_segments_merged} segs -> {n_merged_chains} chains | "
          f"narrow_intersect={n_narrow_merged} (thresh={narrow_thresh_deg:.1f}deg) | "
          f"before={N} after={len(out_panel)} | "
          f"avg_angle_dev={avg_dev:.1f}deg")
    return out_panel, out_fams, out_tiers, stats
