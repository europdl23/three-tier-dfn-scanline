"""
scorer.py — Production HC (Hill-Climbing) geological scorer.

Physics-constrained, deterministic candidate selection for Tier 3 inferred
fracture placement. No neural network, no gradient-based optimisation.

Weights are auto-calibrated from calibration bench face statistics via
inverse-variance weighting (w_k = 1 / std(f_k)), with physics floor
constraints encoding rock mechanics laws.

Reference: Section 2.3.3 of the methodology paper.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Feature names (canonical order, f_1 to f_7)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "spacing",          # F1: stress-shadow spacing reward
    "xbin",             # F2: x-bin spatial occupancy deficit
    "fam_deficit",      # F3: family count deficit
    "cm_dev",           # F4: centre-of-mass alignment penalty
    "tail_persistence", # F5: long-tail length persistence reward
    "len_quality",      # F6: log-length deviation penalty
    "intersection",     # F7: intersection density penalty
]

# Sign convention: +1 = reward (higher = better), -1 = penalty (higher = worse)
FEATURE_SIGNS = np.array([+1, +1, +1, -1, +1, -1, -1], dtype=float)

# Physics floor constraints — encode rock mechanics, NOT data preferences.
# These are hard lower bounds on weight magnitude, independent of variance.
PHYSICS_FLOORS = {
    0: +1.5,   # spacing: stress shadow always applies (tectonic)
    1: +2.0,   # xbin: spatial homogeneity is tectonic
    2: +1.0,   # fam_deficit: family identity is tectonic
    3: -0.5,   # cm_dev: penalty floor
    4: +0.5,   # tail: persistence is geological
    5: -0.3,   # len_quality: penalty floor
    6: -0.3,   # intersection: penalty floor
}


# ═══════════════════════════════════════════════════════════════════════════════
# Inverse-variance weight calibration
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_weights(train_panels: List[np.ndarray],
                      fam_centers: np.ndarray,
                      gstats: dict,
                      assign_fn=None,
                      target_sum: float = 11.5) -> dict:
    """
    Auto-calibrate scorer weights from calibration data.

    Strategy: w_k = 1 / std(f_k) across calibration samples.
    Features with low variance (stable geological signal) get high weight.
    Features with high variance (noisy) get low weight.
    Sign encodes reward (+) vs penalty (-).
    Physics floors enforce minimum importance from rock mechanics.

    Parameters
    ----------
    train_panels : list of np.ndarray
        Calibration bench face arrays, each (N, 4) = [cx, cy, L, theta].
    fam_centers : np.ndarray
        Family centre angles in radians.
    gstats : dict
        Geology statistics (spacing_floor, centre_mass_ref, xbin_ref, etc.).
    assign_fn : callable, optional
        Family assignment function. If None, uses default threshold.
    target_sum : float
        Normalisation target for total weight magnitude.

    Returns
    -------
    dict with keys w_spacing, w_xbin, w_fam_deficit, w_cm_dev,
    w_f2_tail, w_len_quality, w_intersection.
    """
    if assign_fn is None:
        from exp_common import assign_family_ids
        assign_fn = lambda arr: assign_family_ids(arr, fam_centers)

    spacing_floor = gstats["spacing_floor"]
    cm_ref = gstats["centre_mass_ref"]
    xbin_ref_arr = np.asarray(gstats["xbin_ref"])
    xbin_std_arr = np.asarray(gstats["xbin_std"])
    count_std_d = gstats["count_std"]
    target_counts = gstats["target_counts"]
    fam_len_ref = gstats["fam_length_ref"]

    XBINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    feature_samples = {k: [] for k in range(7)}

    for panel in train_panels:
        arr = np.asarray(panel, dtype=float)
        fams = assign_fn(arr)
        if len(arr) < 5:
            continue

        fam_logmu, fam_logsig = {}, {}
        for f in range(3):
            sub = arr[fams == f, 2]
            sub = sub[sub > 1e-6]
            if len(sub) > 2:
                fam_logmu[f] = float(np.log(sub).mean())
                fam_logsig[f] = float(np.log(sub).std())
            else:
                fam_logmu[f] = -2.2
                fam_logsig[f] = 0.4

        h = np.histogram(arr[:, 0], bins=XBINS)[0].astype(float)
        s = h.sum()
        cur_props = h / s if s > 0 else np.zeros(5)

        for idx in range(len(arr)):
            cx, cy, L, A = arr[idx]
            fam = int(fams[idx])
            if fam > 2:
                continue

            # F0: spacing
            same = arr[fams == fam]
            if len(same) > 1:
                dists = np.sqrt((same[:, 0] - cx)**2 + (same[:, 1] - cy)**2)
                dists = dists[dists > 1e-6]
                dmin = float(dists.min()) if len(dists) > 0 else 0.5
                sf = float(spacing_floor[fam])
                feature_samples[0].append(min(dmin / max(sf, 1e-6), 3.0))
            else:
                feature_samples[0].append(1.0)

            # F1: x-bin
            b = int(np.clip(np.searchsorted(XBINS, cx, side="right") - 1, 0, 4))
            xbd = (xbin_ref_arr[b] - cur_props[b]) / max(float(xbin_std_arr[b]), 0.02)
            feature_samples[1].append(float(np.clip(xbd, -2.0, 3.0)))

            # F2: family deficit
            placed = int((fams == fam).sum())
            need = max(0, target_counts[fam] - placed)
            feature_samples[2].append(min(need / max(float(count_std_d[fam]), 1.0), 3.0))

            # F3: centre-mass deviation
            ref_cx = float(cm_ref[fam]["cx_mean"])
            ref_cy = float(cm_ref[fam]["cy_mean"])
            ref_cx_std = float(cm_ref[fam]["cx_std"])
            ref_cy_std = float(cm_ref[fam]["cy_std"])
            cm_dev = math.sqrt(((cx - ref_cx) / ref_cx_std)**2 +
                               ((cy - ref_cy) / ref_cy_std)**2)
            feature_samples[3].append(min(cm_dev, 4.0))

            # F4: tail persistence
            if fam == 2:
                f2_p90 = float(fam_len_ref[2]["p90"])
                f2_std = float(fam_len_ref[2]["std"])
                if L >= f2_p90:
                    feature_samples[4].append(min((L - f2_p90) / max(f2_std, 0.01), 2.0))
                elif L >= float(fam_len_ref[2]["mean"]):
                    feature_samples[4].append(0.2)
                else:
                    feature_samples[4].append(0.0)
            else:
                feature_samples[4].append(0.0)

            # F5: length quality
            len_dev = abs(math.log(max(L, 1e-8)) - fam_logmu[fam]) / max(fam_logsig[fam], 0.1)
            feature_samples[5].append(min(len_dev, 3.0))

            # F6: intersection (approximate — no placement context at calibration time)
            feature_samples[6].append(0.0)

    # Compute inverse-variance weights
    weights = np.zeros(7)
    for k in range(7):
        vals = np.array(feature_samples[k])
        if len(vals) > 5 and vals.std() > 1e-6:
            weights[k] = 1.0 / vals.std()
        else:
            weights[k] = 1.0

    w_sum = weights.sum()
    if w_sum > 1e-6:
        weights = weights * (target_sum / w_sum)

    signed_weights = weights * FEATURE_SIGNS

    # Apply physics floors
    for k in range(7):
        floor = PHYSICS_FLOORS[k]
        if floor > 0:
            signed_weights[k] = max(signed_weights[k], floor)
        else:
            signed_weights[k] = min(signed_weights[k], floor)

    result = {
        "w_spacing":      float(signed_weights[0]),
        "w_xbin":         float(signed_weights[1]),
        "w_fam_deficit":  float(signed_weights[2]),
        "w_cm_dev":       float(signed_weights[3]),
        "w_f2_tail":      float(signed_weights[4]),
        "w_len_quality":  float(signed_weights[5]),
        "w_intersection": float(signed_weights[6]),
    }

    print("  [HC scorer: inverse-variance calibrated weights]")
    for k, name in enumerate(FEATURE_NAMES):
        floor = PHYSICS_FLOORS[k]
        clamped = " (CLAMPED)" if (floor > 0 and weights[k]*FEATURE_SIGNS[k] < floor) or \
                                   (floor < 0 and weights[k]*FEATURE_SIGNS[k] > floor) else ""
        print(f"    {name:<18}: {signed_weights[k]:+.3f} "
              f"(raw={weights[k]*FEATURE_SIGNS[k]:+.3f}, floor={floor:+.1f}){clamped}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction (7-dimensional SD-normalised penalty vector)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(cx: float, cy: float, L: float, A: float,
                     fam: int,
                     same_fam_lines: list,
                     placed_lines: list,
                     spacing_floor: dict,
                     xbin_ref: np.ndarray,
                     xbin_std_ref: np.ndarray,
                     cur_props: np.ndarray,
                     needed: dict,
                     placed_count: dict,
                     count_std: dict,
                     cm_ref: dict,
                     known_lines: list,
                     inferred_lines: list,
                     known_fams: list,
                     inferred_fams: list,
                     fam_logmu: dict,
                     fam_logsig: dict,
                     fam_len_ref: dict,
                     inter_std: float,
                     segments_intersect_fn=None,
                     XBINS=None) -> np.ndarray:
    """
    Compute the 7-dimensional SD-normalised feature vector for one candidate.

    Returns
    -------
    np.ndarray of shape (7,) — the features f_1 to f_7.
    """
    if XBINS is None:
        XBINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    feats = np.zeros(7, dtype=np.float32)
    sf_val = float(spacing_floor[fam])

    # F1: spacing reward
    if same_fam_lines:
        dmin = min(math.sqrt((cx - float(ex[0]))**2 + (cy - float(ex[1]))**2)
                   for ex in same_fam_lines)
        feats[0] = min(dmin / max(sf_val, 1e-6), 3.0)
    else:
        feats[0] = 1.0

    # F2: x-bin spatial occupancy deficit
    b = int(np.clip(np.searchsorted(XBINS, cx, side="right") - 1, 0, 4))
    feats[1] = float(np.clip(
        (xbin_ref[b] - cur_props[b]) / max(float(xbin_std_ref[b]), 0.02),
        -2.0, 3.0))

    # F3: family count deficit
    feats[2] = min(float(needed[fam] - placed_count[fam]) / max(float(count_std[fam]), 1.0), 3.0)

    # F4: centre-of-mass alignment
    all_lines = known_lines + inferred_lines
    all_fams = known_fams + inferred_fams
    pts = [(float(l[0]), float(l[1])) for l, f in zip(all_lines, all_fams) if int(f) == fam]
    n_existing = max(len(pts), 1)
    if pts:
        cur_cx = np.mean([p[0] for p in pts])
        cur_cy = np.mean([p[1] for p in pts])
    else:
        cur_cx, cur_cy = 0.5, 0.15
    proj_cx = (cur_cx * n_existing + cx) / (n_existing + 1)
    proj_cy = (cur_cy * n_existing + cy) / (n_existing + 1)
    feats[3] = min(math.sqrt(
        ((proj_cx - float(cm_ref[fam]["cx_mean"])) / float(cm_ref[fam]["cx_std"]))**2 +
        ((proj_cy - float(cm_ref[fam]["cy_mean"])) / float(cm_ref[fam]["cy_std"]))**2), 4.0)

    # F5: tail persistence
    f_p90 = float(fam_len_ref.get(fam, {}).get("p90", 0.2))
    f_std_val = float(fam_len_ref.get(fam, {}).get("std", 0.05))
    f_mean = float(fam_len_ref.get(fam, {}).get("mean", 0.1))
    if L >= f_p90:
        feats[4] = min((L - f_p90) / max(f_std_val, 0.01), 2.0)
    elif L >= f_mean:
        feats[4] = 0.2

    # F6: log-length deviation
    feats[5] = min(abs(math.log(max(L, 1e-8)) - float(fam_logmu[fam])) /
                   max(float(fam_logsig[fam]), 0.1), 3.0)

    # F7: intersection density (soft cap)
    if segments_intersect_fn is not None and len(placed_lines) >= 8:
        cand_row = np.array([cx, cy, L, A], dtype=np.float32)
        n_cross = sum(1 for ex in placed_lines[-20:]
                      if segments_intersect_fn(cand_row, ex))
        if n_cross > 3:
            feats[6] = (n_cross - 3) / max(inter_std, 0.1)

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# Hard geological gates (physics pre-filters)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_hard_gates(cx: float, cy: float, L: float,
                     fam: int,
                     same_fam_lines: list,
                     spacing_floor: dict,
                     cur_props: np.ndarray,
                     xbin_ref: np.ndarray,
                     fam_len_p025: dict,
                     fam_len_p975: dict,
                     XBINS=None) -> bool:
    """
    Apply three hard geological gates. Returns True if candidate passes.

    Gate 1 — Spacing floor (stress shadow): reject if too close to same-family.
    Gate 2 — X-bin saturation: reject if spatial bin already >1.5x reference.
    Gate 3 — Length sanity: reject if outside data-derived p2.5–p97.5 range.
    """
    if XBINS is None:
        XBINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    sf_val = float(spacing_floor[fam])

    # Gate 1: Spacing floor
    if same_fam_lines:
        dmin = min(math.sqrt((cx - float(ex[0]))**2 + (cy - float(ex[1]))**2)
                   for ex in same_fam_lines)
        if dmin < sf_val * 0.25:
            return False

    # Gate 2: X-bin saturation
    b = int(np.clip(np.searchsorted(XBINS, cx, side="right") - 1, 0, 4))
    if xbin_ref[b] > 0.05 and cur_props[b] > 1.5 * xbin_ref[b]:
        return False

    # Gate 3: Length sanity
    if L < fam_len_p025[fam] or L > fam_len_p975[fam]:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# HC Scorer — deterministic weighted scoring
# ═══════════════════════════════════════════════════════════════════════════════

class HCScorer:
    """
    Hill-Climbing geological scorer with auto-calibrated inverse-variance weights.

    Deterministic: given the same features and weights, always returns the same
    score. No neural network, no learned parameters, no training loop.

    The scorer computes J(c) = sum(w_k * f_k) for each candidate, where:
    - f_k are the 7 SD-normalised geological features
    - w_k are the auto-calibrated weights (inverse-variance + physics floors)
    - Higher score = better candidate (rewards are positive, penalties negative)
    """

    def __init__(self, weights: dict):
        """
        Parameters
        ----------
        weights : dict
            Keys: w_spacing, w_xbin, w_fam_deficit, w_cm_dev, w_f2_tail,
            w_len_quality, w_intersection.
        """
        self.w = np.array([
            float(weights["w_spacing"]),
            float(weights["w_xbin"]),
            float(weights["w_fam_deficit"]),
            float(weights["w_cm_dev"]),
            float(weights["w_len_quality"]),
            float(weights["w_f2_tail"]),
            float(weights["w_intersection"]),
        ], dtype=np.float64)

        # Store individual weights for the HC inline scoring path
        self.W_SPACING  = float(weights["w_spacing"])
        self.W_XBIN     = float(weights["w_xbin"])
        self.W_FAM_DEF  = float(weights["w_fam_deficit"])
        self.W_CM_DEV   = float(weights["w_cm_dev"])
        self.W_F2_TAIL  = float(weights["w_f2_tail"])
        self.W_LEN_QUAL = float(weights["w_len_quality"])
        self.W_INTER    = float(weights["w_intersection"])

    def score(self, features: np.ndarray) -> float:
        """
        Score a single candidate from its 7-feature vector.
        Higher = better candidate.
        """
        return float(np.dot(self.w, features))

    @classmethod
    def from_gstats(cls, gstats: dict) -> "HCScorer":
        """Create from gstats dict (production path)."""
        sw = gstats.get("scorer_weights", None)
        if sw is not None:
            return cls(sw)
        # Fallback to hardcoded defaults
        return cls({
            "w_spacing":      2.0,
            "w_xbin":         3.0,
            "w_fam_deficit":  2.0,
            "w_cm_dev":      -1.5,
            "w_f2_tail":      1.5,
            "w_len_quality": -0.5,
            "w_intersection":-1.0,
        })

    def __repr__(self):
        return (f"HCScorer(spacing={self.W_SPACING:+.2f}, xbin={self.W_XBIN:+.2f}, "
                f"fam_def={self.W_FAM_DEF:+.2f}, cm_dev={self.W_CM_DEV:+.2f}, "
                f"tail={self.W_F2_TAIL:+.2f}, len={self.W_LEN_QUAL:+.2f}, "
                f"inter={self.W_INTER:+.2f})")
