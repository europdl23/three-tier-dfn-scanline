"""
run_ablation_v3.py  –  Comprehensive ablation experiment runner.

Tests 12 generation methods across all borehole intervals and seeds.
All results saved to outputs/tables/ as CSV files.
Does NOT overwrite any file in legacy folders.

Methods tested
--------------
 1. input_only            – Only measured borehole fractures (lower bound)
 2. statistical_sampler   – Random sampling from training distribution (null hypothesis)
 3. evae_oracle           – EVAE given FULL real panel (upper bound, not achievable)
 4. evae_sparse_legacy    – Legacy EVAE decode from measured, angle snapped to family centres
 5. evae_sparse_fixed     – Fixed EVAE decode: continuous angle (no snapping)
 6. seq_v2_global         – v2 sequential: global EVAE latent, geology weights on (baseline)
 7. seq_no_geo_panel      – New seq: panel-specific latent, NO geology losses
 8. seq_geo_panel         – New seq: panel-specific latent, geology weights x1
 9. seq_high_geo_panel    – New seq: panel-specific latent, geology weights x5
10. set_predictor         – New SetPredictorDFN (DETR-style one-shot)
11. hybrid_no_ml          – Semantic rule-based only (no ML candidates)
12. hybrid_ml_improved    – Geology-constrained hybrid with all improved ML candidates
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from exp_common import (
    ROOT, REVISED_DIR, HYBRID_DIR, DATA_DIR, MODEL_DIR, TABLE_DIR, REPORT_DIR,
    SPARSE_INTERVALS_M, VALIDATION_PANELS, CONTROL_SEEDS, N_MAX_EVAE,
    MAX_CTX, MAX_GEN,
    load_revised_modules, load_panels_raw, get_physics_stats, ensure_dirs,
    strips_for_interval, split_panel, central_strip, condition_specs,
    tok_array, detok_array, assign_family_ids, save_json, device as get_device,
    EVAE_CKPT_LEGACY, SEQ_CKPT_V2,
    NEW_SET_PRED_CKPT, NEW_SEQ_PANEL_CKPT, NEW_SEQ_HI_GEO_CKPT,
)
from exp_metrics import compute_all_metrics
from model_v3 import (
    SetPredictorDFN, N_QUERIES,
    StatisticalSampler,
    decode_evae_continuous,
    encode_panel_specific_latent,
)

# ── Additional checkpoint paths ──────────────────────────────────────────────
SEQ_NO_GEO_CKPT  = MODEL_DIR / "seq_no_geology_v3.pth"
SEQ_GEO_CKPT     = MODEL_DIR / "seq_panel_specific_v3.pth"
SEQ_HIGHGEO_CKPT = MODEL_DIR / "seq_high_geology_v3.pth"


# ════════════════════════════════════════════════════════════════════════════════
# Model loaders (cached so each checkpoint is loaded once)
# ════════════════════════════════════════════════════════════════════════════════

_model_cache: Dict[str, object] = {}


def _load_evae(dev):
    key = "evae"
    if key not in _model_cache:
        utils_mod, model_mod, _te, _seq = load_revised_modules()
        ps = get_physics_stats()
        evae = model_mod.DualLatentVAE(sd=12, gd=20, N=N_MAX_EVAE,
                                       yscale=ps["yscale"]).to(dev)
        evae.load_state_dict(torch.load(EVAE_CKPT_LEGACY, map_location=dev))
        evae.eval()
        _model_cache[key] = (evae, ps)
    return _model_cache[key]


def _load_seq(ckpt_path: Path, dev):
    key = str(ckpt_path)
    if key not in _model_cache:
        utils_mod, model_mod, _te, seq_mod = load_revised_modules()
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        ps = ck.get("physics_stats", get_physics_stats())
        # Restore geology env vars from checkpoint
        for k, v in ck.get("geo_env", {}).items():
            os.environ[k] = str(v)
        import importlib as _il
        sys.modules.pop("sequential_transformer", None)
        seq_mod2 = _il.import_module("sequential_transformer")
        model = seq_mod2.SeqFracTransformer().to(dev)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.eval()
        _model_cache[key] = (model, seq_mod2, ps, ck)
    return _model_cache[key]


def _load_set_predictor(dev):
    key = "set_predictor"
    if key not in _model_cache:
        ck = torch.load(NEW_SET_PRED_CKPT, map_location=dev, weights_only=False)
        ps = ck.get("physics_stats", get_physics_stats())
        model = SetPredictorDFN().to(dev)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        _model_cache[key] = (model, ps)
    return _model_cache[key]


# ════════════════════════════════════════════════════════════════════════════════
# Generation helpers
# ════════════════════════════════════════════════════════════════════════════════

def _seq_generate(seq_model, seq_mod, measured: np.ndarray,
                  n_gen: int, ps: dict, strips, dev,
                  sz_np=None, gz_np=None) -> np.ndarray:
    """Wrapper: call seq_mod.generate() and return (M,4) array."""
    if len(measured) < 1 or n_gen < 1:
        return np.zeros((0, 4), dtype=np.float32)
    def _flat(arr):
        """Return 1-D numpy array regardless of input shape (1,D) or (D,)."""
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    gen = seq_mod.generate(
        model=seq_model,
        ctx_lines=list(measured[:MAX_CTX]),
        n_gen=min(n_gen, MAX_GEN),
        ps=ps,
        device=dev,
        sz=_flat(sz_np) if sz_np is not None else None,
        gz=_flat(gz_np) if gz_np is not None else None,
        temp=1.0,
        strips=strips,
        enforce_outside_strips=True,
        y_max=float(ps.get("yscale", 0.32)),
        n_candidates=24,
    )
    if not gen:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array([[g["cx"], g["cy"], g["length"], g["angle_rad"]]
                     for g in gen], dtype=np.float32)


def _outside_strip_filter(lines: np.ndarray, strips) -> np.ndarray:
    if len(lines) == 0:
        return lines
    keep = []
    for row in lines:
        cx = float(row[0])
        if not any(float(lo) <= cx <= float(hi) for lo, hi in strips):
            keep.append(row)
    return np.array(keep, dtype=np.float32) if keep else np.zeros((0,4), dtype=np.float32)


# ── Greedy geology-constrained candidate selector ────────────────────────────
def _greedy_select(
    fixed: np.ndarray,
    candidates: List[Dict],
    target_counts: Dict[int, int],
    spacing_floors: Dict[int, float],
    inter_limit: float,
    fam_centers: np.ndarray,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Same algorithm as run_controlled_comparison_v2.py:select_geology_constrained
    but self-contained.  source_weight lower = preferred.
    """
    from exp_common import intersection_density as _id

    src_w = {"semantic": 0.0, "seq_evae": 0.35, "seq_only": 0.45,
             "evae": 0.70, "set_predictor": 0.30, "statistical": 0.90}

    fixed = np.asarray(fixed, dtype=float).reshape(-1, 4)
    fixed_fams = assign_family_ids(fixed, fam_centers)
    current = {fam: int((fixed_fams == fam).sum()) for fam in range(3)}
    for fam in range(3):
        target_counts[fam] = max(int(target_counts[fam]), int(current[fam]))

    selected, selected_meta, remaining = [], [], list(candidates)

    for relax in [1.0, 0.65, 0.35]:
        improved = True
        while (improved and sum(current.values()) < sum(target_counts.values())
               and remaining):
            improved = False
            existing = (np.vstack([fixed, *selected]) if selected
                        else fixed.copy())
            existing_fams = assign_family_ids(existing, fam_centers)
            pool = []
            for idx, cand in enumerate(remaining):
                row = np.asarray(cand["row"], dtype=float)
                fam = int(cand["family"])
                if current[fam] >= target_counts[fam] + 1:
                    continue
                dists = np.sqrt(((existing[:, 0:2] - row[0:2])**2).sum(axis=1))
                if len(dists) and dists.min() < 0.004:
                    continue
                same = existing[existing_fams == fam]
                if len(same):
                    d = float(np.sqrt(((same[:, 0:2] - row[0:2])**2).sum(axis=1)).min())
                    if d < float(spacing_floors[fam]) * relax:
                        continue
                trial = np.vstack([existing, row.reshape(1, 4)])
                if _id(trial) > inter_limit * (1.0 + (1.0 - relax)):
                    continue
                deficit = -2.0 * max(0, target_counts[fam] - current[fam])
                center_b = 0.25 * abs(float(row[0]) - 0.5)
                score = float(src_w.get(str(cand["source"]), 1.0)
                               + center_b + deficit)
                pool.append((score, idx, cand))
            if not pool:
                break
            _, idx, cand = sorted(pool, key=lambda x: x[0])[0]
            selected.append(np.asarray(cand["row"], dtype=float))
            selected_meta.append({"source": cand["source"],
                                   "family": int(cand["family"])})
            current[int(cand["family"])] += 1
            remaining.pop(idx)
            improved = True

    inferred = np.vstack(selected) if selected else np.zeros((0, 4), dtype=float)
    final    = np.vstack([fixed, inferred]) if len(inferred) else fixed.copy()
    return final.astype(np.float32), selected_meta


# ── IMPROVEMENT 2: Soft-quota selector ────────────────────────────────────────
# Hard quota forces exactly target_counts[f] lines per family.
# Soft quota: take whatever the model proposes (filtered by guardrails),
# fall back to statistical sampling ONLY if the model produced fewer than
# `min_count_floor[f]` (e.g. 50% of training mean) for a family.
def _greedy_select_soft(
    fixed: np.ndarray,
    candidates: List[Dict],
    target_counts: Dict[int, int],
    spacing_floors: Dict[int, float],
    inter_limit: float,
    fam_centers: np.ndarray,
    min_count_frac: float = 0.5,
    soft_cap_mult: float = 1.5,     # STEP 2: tighten by passing 1.2
    enforce_floor: bool = False,    # STEP 2: hard floor if True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Soft-quota variant: model decides count, hard quota only as floor.
    - Accept any model-proposed (non-statistical) candidate that passes
      spacing + intersection guardrails.
    - After exhausting model candidates, if a family count is below
      min_count_frac * target_counts[f], top up from statistical fallback
      until that floor is reached.
    """
    from exp_common import intersection_density as _id

    src_w = {"semantic": 0.0, "seq_evae": 0.20, "seq_only": 0.30,
             "evae": 0.40, "set_predictor": 0.15, "statistical": 0.95}
    is_model_src = lambda s: s != "statistical"

    fixed = np.asarray(fixed, dtype=float).reshape(-1, 4)
    fixed_fams = assign_family_ids(fixed, fam_centers)
    current = {fam: int((fixed_fams == fam).sum()) for fam in range(3)}
    # STEP 2: soft_cap_mult is configurable (default 1.5, tightened to 1.2 in v5fx)
    soft_cap = {f: max(1, int(round(target_counts[f] * soft_cap_mult))) for f in range(3)}

    selected, selected_meta = [], []

    # Phase A: greedy add ALL acceptable model candidates (no hard count limit
    #          beyond the soft cap), respecting spacing + intersection
    model_cands = [c for c in candidates if is_model_src(c["source"])]
    stat_cands  = [c for c in candidates if not is_model_src(c["source"])]
    # Sort by source preference, then proximity to centre
    model_cands.sort(key=lambda c: (
        src_w.get(str(c["source"]), 1.0),
        abs(float(np.asarray(c["row"])[0]) - 0.5),
    ))

    for cand in model_cands:
        row = np.asarray(cand["row"], dtype=float)
        fam = int(cand["family"])
        if fam > 2 or current[fam] >= soft_cap[fam]:
            continue
        existing = (np.vstack([fixed, *selected]) if selected else fixed.copy())
        existing_fams = assign_family_ids(existing, fam_centers)
        if len(existing):
            dists = np.sqrt(((existing[:, 0:2] - row[0:2])**2).sum(axis=1))
            if dists.min() < 0.004:
                continue
            same = existing[existing_fams == fam]
            if len(same):
                d = float(np.sqrt(((same[:, 0:2] - row[0:2])**2).sum(axis=1)).min())
                if d < float(spacing_floors[fam]) * 0.6:   # softer floor
                    continue
            trial = np.vstack([existing, row.reshape(1, 4)])
            if _id(trial) > inter_limit * 1.4:           # softer cap
                continue
        selected.append(row)
        selected_meta.append({"source": cand["source"], "family": fam})
        current[fam] += 1

    # Phase B: top-up below the floor using statistical fallback
    floors = {f: max(1, int(round(target_counts[f] * min_count_frac))) for f in range(3)}
    for cand in stat_cands:
        row = np.asarray(cand["row"], dtype=float)
        fam = int(cand["family"])
        if fam > 2 or current[fam] >= floors[fam]:
            continue
        existing = (np.vstack([fixed, *selected]) if selected else fixed.copy())
        existing_fams = assign_family_ids(existing, fam_centers)
        if len(existing):
            dists = np.sqrt(((existing[:, 0:2] - row[0:2])**2).sum(axis=1))
            if dists.min() < 0.004:
                continue
            same = existing[existing_fams == fam]
            if len(same):
                d = float(np.sqrt(((same[:, 0:2] - row[0:2])**2).sum(axis=1)).min())
                if d < float(spacing_floors[fam]) * 0.6:
                    continue
            trial = np.vstack([existing, row.reshape(1, 4)])
            if _id(trial) > inter_limit * 1.4:
                continue
        selected.append(row)
        selected_meta.append({"source": cand["source"], "family": fam})
        current[fam] += 1

    # STEP 2: hard lower floor — if any family is below 0.7×target after
    # greedy selection, top up from remaining candidate pool using looser
    # spacing rule.  Bounded by 10 attempts per family.
    if enforce_floor:
        floor = {f: max(1, int(round(target_counts[f] * 0.7))) for f in range(3)}
        for f in range(3):
            attempts = 0
            while current[f] < floor[f] and attempts < 10:
                attempts += 1
                existing = (np.vstack([fixed, *selected]) if selected else fixed.copy())
                pool = [c for c in candidates if int(c["family"]) == f
                         and not any(np.allclose(np.asarray(c["row"]), s) for s in selected)]
                if not pool:
                    break
                # Accept first that passes half-strength spacing + no-overlap
                added = False
                for cand in pool:
                    row = np.asarray(cand["row"], dtype=float)
                    if len(existing):
                        dists = np.sqrt(((existing[:, 0:2] - row[0:2])**2).sum(axis=1))
                        if dists.min() < 0.004:
                            continue
                        same = existing[assign_family_ids(existing, fam_centers) == f]
                        if len(same):
                            d = float(np.sqrt(((same[:, 0:2] - row[0:2])**2).sum(axis=1)).min())
                            if d < spacing_floors[f] * 0.3:   # very loose
                                continue
                    selected.append(row)
                    selected_meta.append({"source": cand["source"], "family": f})
                    current[f] += 1
                    added = True
                    break
                if not added:
                    break

    inferred = np.vstack(selected) if selected else np.zeros((0, 4), dtype=float)
    final    = np.vstack([fixed, inferred]) if len(inferred) else fixed.copy()
    return final.astype(np.float32), selected_meta


def _train_geology_stats(train_panels, fam_centers):
    counts, inter, sdist = [], [], {0: [], 1: [], 2: []}
    # Per-panel spatial stats for centre-mass and x-bin reference
    panel_cx_means = {f: [] for f in range(3)}
    panel_cy_means = {f: [] for f in range(3)}
    panel_cx_all = []  # overall (all families) cx per panel
    panel_cy_all = []
    fam_lengths = {f: [] for f in range(3)}
    XBINS_REF = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    panel_xbin_props = []
    for p in train_panels:
        arr  = np.asarray(p, dtype=float)
        fams = assign_family_ids(arr, fam_centers)
        counts.append([(fams == f).sum() for f in range(3)])
        from exp_common import intersection_density as _id
        inter.append(_id(arr))
        for f in range(3):
            sub = arr[fams == f]
            for i in range(len(sub)):
                for j in range(i+1, len(sub)):
                    sdist[f].append(float(np.linalg.norm(sub[i, 0:2]-sub[j, 0:2])))
            if len(sub) > 0:
                panel_cx_means[f].append(float(sub[:, 0].mean()))
                panel_cy_means[f].append(float(sub[:, 1].mean()))
                fam_lengths[f].extend(sub[:, 2].tolist())
        # Overall panel spatial distribution
        if len(arr) > 0:
            panel_cx_all.append(float(arr[:, 0].mean()))
            panel_cy_all.append(float(arr[:, 1].mean()))
            h = np.histogram(arr[:, 0], bins=XBINS_REF)[0].astype(float)
            s = h.sum()
            panel_xbin_props.append(h / s if s > 0 else np.ones(5) / 5.0)
        else:
            panel_xbin_props.append(np.ones(5) / 5.0)
    ca = np.array(counts, dtype=float)
    sf = {}
    for f in range(3):
        v = np.array(sdist[f], dtype=float)
        sf[f] = float(max(0.008, 0.45*np.percentile(v, 10))) if v.size else 0.012
    # Centre-mass reference (mean and std across training panels)
    cm_ref = {}
    for f in range(3):
        cx_arr = np.array(panel_cx_means[f]) if panel_cx_means[f] else np.array([0.5])
        cy_arr = np.array(panel_cy_means[f]) if panel_cy_means[f] else np.array([0.15])
        cm_ref[f] = {
            "cx_mean": float(cx_arr.mean()), "cx_std": float(max(cx_arr.std(), 0.02)),
            "cy_mean": float(cy_arr.mean()), "cy_std": float(max(cy_arr.std(), 0.02)),
        }
    # X-bin reference distribution (mean across training panels)
    xbin_ref = np.mean(panel_xbin_props, axis=0) if panel_xbin_props else np.ones(5) / 5.0
    xbin_std = np.std(panel_xbin_props, axis=0) if len(panel_xbin_props) > 1 else np.ones(5) * 0.05
    xbin_std = np.maximum(xbin_std, 0.02)
    # F2 length upper quantiles
    f2_len_ref = {}
    for f in range(3):
        larr = np.array(fam_lengths[f])
        larr = larr[larr > 1e-6] if len(larr) else np.array([0.1])
        f2_len_ref[f] = {
            "p75": float(np.percentile(larr, 75)),
            "p90": float(np.percentile(larr, 90)),
            "p95": float(np.percentile(larr, 95)),
            "mean": float(larr.mean()),
            "std": float(max(larr.std(), 0.01)),
        }
    # Intersection density stats
    inter_arr = np.array(inter) if inter else np.array([0.0])
    gstats = {
        "target_counts": {f: int(max(1, round(ca[:, f].mean()))) for f in range(3)},
        "count_std":     {f: float(max(ca[:, f].std(), 1.0)) for f in range(3)},
        "spacing_floor":  sf,
        "intersection_limit": float(np.mean(inter_arr) + 2*max(np.std(inter_arr), 1e-6)),
        "intersection_mean": float(np.mean(inter_arr)),
        "intersection_std":  float(max(np.std(inter_arr), 1e-6)),
        "centre_mass_ref":   cm_ref,
        "xbin_ref":          xbin_ref,
        "xbin_std":          xbin_std,
        "fam_length_ref":    f2_len_ref,
    }

    # Auto-calibrate scorer weights from data variance
    gstats["scorer_weights"] = _calibrate_scorer_weights(train_panels, fam_centers, gstats)
    return gstats


def _calibrate_scorer_weights(train_panels, fam_centers, gstats):
    """
    Auto-calibrate scorer weights from calibration data.

    Strategy: w_k = 1 / std(f_k) across calibration samples.
    Features with low variance (stable geological signal) get high weight.
    Features with high variance (noisy) get low weight.
    Sign encodes reward (+) vs penalty (-).

    Returns dict with 7 weights + metadata.
    """
    # Generate synthetic scoring contexts from calibration panels
    # to measure feature variance
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
        fams = assign_family_ids(arr, fam_centers)
        if len(arr) < 5:
            continue

        # Compute log-length stats
        fam_logmu = {}
        fam_logsig = {}
        for f in range(3):
            sub = arr[fams == f, 2]
            sub = sub[sub > 1e-6]
            if len(sub) > 2:
                fam_logmu[f] = float(np.log(sub).mean())
                fam_logsig[f] = float(np.log(sub).std())
            else:
                fam_logmu[f] = -2.2
                fam_logsig[f] = 0.4

        # X-bin props of this panel
        h = np.histogram(arr[:, 0], bins=XBINS)[0].astype(float)
        s = h.sum()
        cur_props = h / s if s > 0 else np.zeros(5)

        # Sample features for each fracture (as if it were a candidate)
        for idx in range(len(arr)):
            cx, cy, L, A = arr[idx]
            fam = int(fams[idx])
            if fam > 2:
                continue

            # F0: spacing
            same = arr[fams == fam]
            if len(same) > 1:
                dists = np.sqrt((same[:, 0] - cx)**2 + (same[:, 1] - cy)**2)
                dists = dists[dists > 1e-6]  # exclude self
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
            import math as _m
            cm_dev = _m.sqrt(((cx - ref_cx) / ref_cx_std)**2 +
                             ((cy - ref_cy) / ref_cy_std)**2)
            feature_samples[3].append(min(cm_dev, 4.0))

            # F4: F2 tail
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
            len_dev = abs(_m.log(max(L, 1e-8)) - fam_logmu[fam]) / max(fam_logsig[fam], 0.1)
            feature_samples[5].append(min(len_dev, 3.0))

            # F6: intersection (approximate)
            feature_samples[6].append(0.0)  # can't compute without placement context

    # Compute inverse-variance weights
    weights = np.zeros(7)
    signs = np.array([+1, +1, +1, -1, +1, -1, -1], dtype=float)  # reward/penalty
    feature_names = ["spacing", "xbin", "fam_deficit", "cm_dev",
                     "f2_tail", "len_quality", "intersection"]

    for k in range(7):
        vals = np.array(feature_samples[k])
        if len(vals) > 5 and vals.std() > 1e-6:
            # Inverse variance, scaled so mean weight ~ 2.0
            weights[k] = 1.0 / vals.std()
        else:
            weights[k] = 1.0  # fallback

    # Normalise so weights sum to ~11.5 (same total magnitude as original HC)
    target_sum = 11.5
    w_sum = weights.sum()
    if w_sum > 1e-6:
        weights = weights * (target_sum / w_sum)

    # Apply signs
    signed_weights = weights * signs

    # Physics floor constraints:
    # The DATA determines the scale (how much weight).
    # The PHYSICS determines the sign (reward/penalty) and floor (minimum importance).
    # These floors encode rock mechanics laws, not data preferences:
    #   - Stress shadow: spacing ALWAYS matters (positive, >= 1.5)
    #   - Spatial homogeneity: x-bin balance is tectonic (positive, >= 2.0)
    #   - Family identity: tectonic regime sets family counts (positive, >= 1.0)
    #   - Centre-mass: penalty for spatial drift (negative, <= -0.5)
    #   - Length quality: penalty for wrong lengths (negative, <= -0.3)
    #   - Intersection: penalty for excess crossings (negative, <= -0.3)
    physics_floors = {
        0: +1.5,   # spacing: stress shadow always applies
        1: +2.0,   # xbin: spatial homogeneity is tectonic
        2: +1.0,   # fam_deficit: family identity is tectonic
        3: -0.5,   # cm_dev: penalty floor
        4: +0.5,   # tail: persistence is geological
        5: -0.3,   # len_quality: penalty floor
        6: -0.3,   # intersection: penalty floor
    }
    for k in range(7):
        floor = physics_floors[k]
        if floor > 0:
            signed_weights[k] = max(signed_weights[k], floor)
        else:
            signed_weights[k] = min(signed_weights[k], floor)

    result = {
        "w_spacing": float(signed_weights[0]),
        "w_xbin": float(signed_weights[1]),
        "w_fam_deficit": float(signed_weights[2]),
        "w_cm_dev": float(signed_weights[3]),
        "w_f2_tail": float(signed_weights[4]),
        "w_len_quality": float(signed_weights[5]),
        "w_intersection": float(signed_weights[6]),
    }

    print(f"  [physics-constrained auto-calibrated scorer weights]")
    for k, name in enumerate(feature_names):
        floor = physics_floors[k]
        clamped = " (CLAMPED)" if (floor > 0 and weights[k]*signs[k] < floor) or \
                                   (floor < 0 and weights[k]*signs[k] > floor) else ""
        print(f"    {name:<15}: {signed_weights[k]:+.3f} "
              f"(raw={weights[k]*signs[k]:+.3f}, floor={floor:+.1f}){clamped}")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# Per-panel, per-condition, per-method generation
# ════════════════════════════════════════════════════════════════════════════════

def run_one(
    real:          np.ndarray,
    measured:      np.ndarray,
    target_arr:    np.ndarray,
    strips,
    spec:          dict,
    panel_id:      int,
    seed:          int,
    ps:            dict,
    gstats:        dict,
    dev:           torch.device,
    methods_to_run: List[str],
    train_panels:  List[np.ndarray],
) -> List[Dict]:
    """
    Run all requested generation methods for one (panel, condition, seed) triple.
    Returns list of metric dicts (one per method).
    """
    np.random.seed(seed + panel_id)
    torch.manual_seed(seed + panel_id)

    fam_centers = np.asarray(ps["fam_centers_rad"], dtype=float)
    y_max       = float(ps.get("yscale", 0.32))
    n_target    = len(target_arr)
    rows        = []

    meta = {
        "condition":    spec["condition"],
        "family":       spec["family"],
        "interval_m":   spec.get("interval_m"),
        "coverage_pct": spec["coverage_pct"],
        "panel_id":     panel_id,
        "seed":         seed,
        "n_measured":   len(measured),
        "n_target":     n_target,
    }

    def _record(method, generated):
        try:
            gen_arr = np.asarray(generated, dtype=np.float32).reshape(-1, 4)
            # Combine measured + inferred if generated is inferred-only
            # (methods that return full panel already include measured)
            m = compute_all_metrics(real, gen_arr, measured, strips, ps,
                                    label=method)
            rows.append({**meta, "method": method, **m})
        except Exception as exc:
            rows.append({**meta, "method": method, "error": str(exc)})

    # ── 1. input_only ─────────────────────────────────────────────────────────
    if "input_only" in methods_to_run:
        _record("input_only", measured)

    # ── 2. statistical_sampler ────────────────────────────────────────────────
    if "statistical_sampler" in methods_to_run:
        sampler = StatisticalSampler(ps)
        stat_gen = sampler.sample(n_target, strips=strips, seed=seed+panel_id)
        full = (np.vstack([measured, stat_gen]) if len(stat_gen)
                else measured.copy())
        _record("statistical_sampler", full)

    # ── 3. evae_oracle ────────────────────────────────────────────────────────
    if "evae_oracle" in methods_to_run:
        evae, _ps = _load_evae(dev)
        oracle = decode_evae_continuous(evae, real, n_out=len(real),
                                        ps=ps, device=dev, strips=None)
        _record("evae_oracle", oracle)

    # ── 4. evae_sparse_legacy (angle-snapped to family centre) ───────────────
    if "evae_sparse_legacy" in methods_to_run:
        evae, _ps = _load_evae(dev)
        # Replicate legacy behaviour: hard-snap angle to argmax family centre
        arr = np.asarray(measured, dtype=np.float32).reshape(-1, 4)
        if len(arr) > 0:
            n_in = min(len(arr), N_MAX_EVAE)
            C = torch.from_numpy(arr[:n_in, 0:2].reshape(1,n_in,2)).to(dev)
            Lv= torch.from_numpy(arr[:n_in, 2:3].reshape(1,n_in,1)).to(dev)
            Av= torch.from_numpy(arr[:n_in, 3:4].reshape(1,n_in,1)).to(dev)
            mk= torch.ones(1,n_in,dtype=torch.bool,device=dev)
            with torch.no_grad():
                sm,_sl,gm,_gl = evae.encode(C,Lv,Av,mk)
                c_o,l_o,ml_o,_kp,el_o = evae.decode(sm,gm,mode="B")
            ctr_np = c_o[0].cpu().numpy()
            len_np = l_o[0,:,0].cpu().numpy()
            fam_np = ml_o[0,:,:3].cpu().numpy()
            ex_np  = torch.sigmoid(el_o[0]).cpu().numpy()
            fam_pred = np.argmax(fam_np, axis=1).clip(0,2)
            ang_hard = fam_centers[:3][fam_pred]
            lmax = float(ps.get("length_max", arr[:,2].max()*2+1e-6))
            lines_hard = np.column_stack([ctr_np[:,0], ctr_np[:,1],
                                          len_np*lmax, ang_hard]).astype(np.float32)
            keep = []
            for ii in np.argsort(-ex_np):
                cx,cy,L,A = lines_hard[ii]
                if L < 1e-5: continue
                x1=cx-0.5*L*math.cos(A); x2=cx+0.5*L*math.cos(A)
                y1=cy-0.5*L*math.sin(A); y2=cy+0.5*L*math.sin(A)
                if min(x1,x2)<-0.01 or max(x1,x2)>1.01: continue
                if min(y1,y2)<-0.01 or max(y1,y2)>y_max+0.01: continue
                if any(float(lo)<=cx<=float(hi) for lo,hi in strips): continue
                keep.append(ii)
                if len(keep) >= n_target + len(measured): break
            evae_leg = lines_hard[keep] if keep else np.zeros((0,4),dtype=np.float32)
        else:
            evae_leg = np.zeros((0,4),dtype=np.float32)
        full = (np.vstack([measured, evae_leg[:n_target]]) if len(evae_leg)
                else measured.copy())
        _record("evae_sparse_legacy", full)

    # ── 5. evae_sparse_fixed (continuous angle) ───────────────────────────────
    if "evae_sparse_fixed" in methods_to_run:
        evae, _ps = _load_evae(dev)
        evae_cont = decode_evae_continuous(evae, measured, n_out=n_target+len(measured),
                                           ps=ps, device=dev, strips=strips)
        full = (np.vstack([measured, evae_cont[:n_target]]) if len(evae_cont)
                else measured.copy())
        _record("evae_sparse_fixed", full)

    # ── 6. seq_v2_global (legacy v2 checkpoint with global EVAE latent) ───────
    if "seq_v2_global" in methods_to_run and SEQ_CKPT_V2.exists():
        model6, sm6, ps6, ck6 = _load_seq(SEQ_CKPT_V2, dev)
        # Rebuild global latent from training panels
        utils_mod, _mm, _te, _sq = load_revised_modules()
        pu = utils_mod.load_labelme_data(str(DATA_DIR))
        sz6, gz6 = sm6.build_global_sz_gz(
            str(EVAE_CKPT_LEGACY), pu[:7], ps6["yscale"], dev)
        sz_np6 = sz6.squeeze(0).cpu().numpy()
        gz_np6 = gz6.squeeze(0).cpu().numpy()
        gen6 = _seq_generate(model6, sm6, measured, n_target, ps6,
                             strips, dev, sz_np=sz_np6, gz_np=gz_np6)
        full6 = (np.vstack([measured, gen6]) if len(gen6) else measured.copy())
        _record("seq_v2_global", full6)

    # ── 7. seq_no_geo_panel (no geology losses, panel-specific latent) ────────
    if "seq_no_geo_panel" in methods_to_run and SEQ_NO_GEO_CKPT.exists():
        model7, sm7, ps7, _ = _load_seq(SEQ_NO_GEO_CKPT, dev)
        evae, _ = _load_evae(dev)
        sz_np7, gz_np7 = encode_panel_specific_latent(evae, measured, dev)
        gen7 = _seq_generate(model7, sm7, measured, n_target, ps7,
                             strips, dev, sz_np=sz_np7, gz_np=gz_np7)
        full7 = (np.vstack([measured, gen7]) if len(gen7) else measured.copy())
        _record("seq_no_geo_panel", full7)

    # ── 8. seq_geo_panel (geology x1, panel-specific latent) ─────────────────
    if "seq_geo_panel" in methods_to_run and SEQ_GEO_CKPT.exists():
        model8, sm8, ps8, _ = _load_seq(SEQ_GEO_CKPT, dev)
        evae, _ = _load_evae(dev)
        sz_np8, gz_np8 = encode_panel_specific_latent(evae, measured, dev)
        gen8 = _seq_generate(model8, sm8, measured, n_target, ps8,
                             strips, dev, sz_np=sz_np8, gz_np=gz_np8)
        full8 = (np.vstack([measured, gen8]) if len(gen8) else measured.copy())
        _record("seq_geo_panel", full8)

    # ── 9. seq_high_geo_panel (geology x5, panel-specific latent) ────────────
    if "seq_high_geo_panel" in methods_to_run and SEQ_HIGHGEO_CKPT.exists():
        model9, sm9, ps9, _ = _load_seq(SEQ_HIGHGEO_CKPT, dev)
        evae, _ = _load_evae(dev)
        sz_np9, gz_np9 = encode_panel_specific_latent(evae, measured, dev)
        gen9 = _seq_generate(model9, sm9, measured, n_target, ps9,
                             strips, dev, sz_np=sz_np9, gz_np=gz_np9)
        full9 = (np.vstack([measured, gen9]) if len(gen9) else measured.copy())
        _record("seq_high_geo_panel", full9)

    # ── 10. set_predictor ─────────────────────────────────────────────────────
    if "set_predictor" in methods_to_run and NEW_SET_PRED_CKPT.exists():
        sp_model, sp_ps = _load_set_predictor(dev)
        sp_gen = sp_model.predict(measured, sp_ps, dev,
                                  exist_threshold=0.30,
                                  n_max=n_target + 20,
                                  strips=strips, y_max=y_max)
        full_sp = (np.vstack([measured, sp_gen]) if len(sp_gen)
                   else measured.copy())
        _record("set_predictor", full_sp)

    # ── 11. hybrid_no_ml (semantic rule-based only) ───────────────────────────
    # Reconstruct using only statistical-sampler as proxy for semantic output
    # (actual semantic module requires the full hybrid pipeline with Hybrid_evae_sequential)
    if "hybrid_no_ml" in methods_to_run:
        sampler = StatisticalSampler(ps)
        candidates = []
        for _ in range(3):  # 3 random realisations → take best
            s = sampler.sample(n_target * 2, strips=strips,
                               seed=seed + panel_id + _)
            fams_s = assign_family_ids(s, fam_centers)
            for row, fam in zip(s, fams_s):
                if fam <= 2:
                    candidates.append({"row": row, "family": int(fam),
                                       "source": "statistical"})
        final_noml, _ = _greedy_select(
            measured, candidates,
            dict(gstats["target_counts"]),
            dict(gstats["spacing_floor"]),
            float(gstats["intersection_limit"]),
            fam_centers,
        )
        _record("hybrid_no_ml", final_noml)

    # ── 12. hybrid_ml_improved (all improved ML sources + greedy selector) ────
    if "hybrid_ml_improved" in methods_to_run:
        evae, _ps = _load_evae(dev)
        candidates_all = []

        # EVAE continuous
        evae_cont2 = decode_evae_continuous(evae, measured,
                                            n_out=n_target+len(measured),
                                            ps=ps, device=dev, strips=strips)
        fams_ec = assign_family_ids(evae_cont2, fam_centers)
        for row, fam in zip(evae_cont2, fams_ec):
            if fam <= 2:
                candidates_all.append({"row": row, "family": int(fam),
                                       "source": "evae"})

        # SetPredictor
        if NEW_SET_PRED_CKPT.exists():
            sp_m, sp_ps2 = _load_set_predictor(dev)
            sp_cands = sp_m.predict(measured, sp_ps2, dev,
                                    exist_threshold=0.25,
                                    n_max=n_target + 30,
                                    strips=strips, y_max=y_max)
            fams_sp = assign_family_ids(sp_cands, fam_centers)
            for row, fam in zip(sp_cands, fams_sp):
                if fam <= 2:
                    candidates_all.append({"row": row, "family": int(fam),
                                           "source": "set_predictor"})

        # Seq geo panel-specific
        if SEQ_GEO_CKPT.exists():
            m8, sm8_2, ps8_2, _ = _load_seq(SEQ_GEO_CKPT, dev)
            sz_8, gz_8 = encode_panel_specific_latent(evae, measured, dev)
            gen8_2 = _seq_generate(m8, sm8_2, measured, n_target, ps8_2,
                                   strips, dev, sz_np=sz_8, gz_np=gz_8)
            fams_g8 = assign_family_ids(gen8_2, fam_centers)
            for row, fam in zip(gen8_2, fams_g8):
                if fam <= 2:
                    candidates_all.append({"row": row, "family": int(fam),
                                           "source": "seq_evae"})

        # Statistical fallback
        sampler2 = StatisticalSampler(ps)
        stat2    = sampler2.sample(n_target * 2, strips=strips,
                                   seed=seed + panel_id + 999)
        fams_st2 = assign_family_ids(stat2, fam_centers)
        for row, fam in zip(stat2, fams_st2):
            if fam <= 2:
                candidates_all.append({"row": row, "family": int(fam),
                                       "source": "statistical"})

        final_imp, _ = _greedy_select(
            measured, candidates_all,
            dict(gstats["target_counts"]),
            dict(gstats["spacing_floor"]),
            float(gstats["intersection_limit"]),
            fam_centers,
        )
        _record("hybrid_ml_improved", final_imp)

    # ── 13. hybrid_v4 (LIFTED tokens + LOW set-pred threshold + SOFT quota) ──
    if "hybrid_v4" in methods_to_run:
        from model_v3 import (decode_evae_continuous_lifted,
                               encode_panel_specific_latent_lifted,
                               lift_fragments_to_full_lines)
        evae, _ = _load_evae(dev)
        candidates_v4 = []

        # 1) Lifted EVAE  ── feed lifted full-length tokens, not raw fragments
        evae_lifted = decode_evae_continuous_lifted(
            evae, measured,
            n_out=int(n_target * 1.5) + len(measured),
            ps=ps, device=dev, strips=strips,
        )
        for row, fam in zip(evae_lifted,
                             assign_family_ids(evae_lifted, fam_centers)):
            if fam <= 2:
                candidates_v4.append({"row": row, "family": int(fam),
                                       "source": "evae"})

        # 2) SetPredictor with LOWERED threshold (0.10 instead of 0.25)
        if NEW_SET_PRED_CKPT.exists():
            sp_m, sp_ps = _load_set_predictor(dev)
            sp_low = sp_m.predict(measured, sp_ps, dev,
                                   exist_threshold=0.10,         # LOWERED
                                   n_max=int(n_target * 1.5) + 30,
                                   strips=strips, y_max=y_max)
            for row, fam in zip(sp_low,
                                 assign_family_ids(sp_low, fam_centers)):
                if fam <= 2:
                    candidates_v4.append({"row": row, "family": int(fam),
                                           "source": "set_predictor"})

        # 3) Sequential transformer with LIFTED panel-specific latent
        if SEQ_GEO_CKPT.exists():
            m_sq, sm_sq, ps_sq, _ = _load_seq(SEQ_GEO_CKPT, dev)
            sz_lf, gz_lf = encode_panel_specific_latent_lifted(
                evae, measured, ps, dev)
            # Also pass lifted measured tokens as the seq context
            lifted_ctx = lift_fragments_to_full_lines(measured, ps)
            sq_lf = _seq_generate(m_sq, sm_sq, lifted_ctx,
                                   int(n_target * 1.3), ps_sq,
                                   strips, dev, sz_np=sz_lf, gz_np=gz_lf)
            for row, fam in zip(sq_lf,
                                 assign_family_ids(sq_lf, fam_centers)):
                if fam <= 2:
                    candidates_v4.append({"row": row, "family": int(fam),
                                           "source": "seq_evae"})

        # 4) Statistical fallback (only used by soft-quota when below floor)
        sampler_v4 = StatisticalSampler(ps)
        st_v4 = sampler_v4.sample(n_target * 2, strips=strips,
                                   seed=seed + panel_id + 7777)
        for row, fam in zip(st_v4,
                             assign_family_ids(st_v4, fam_centers)):
            if fam <= 2:
                candidates_v4.append({"row": row, "family": int(fam),
                                       "source": "statistical"})

        # SOFT quota: model decides count, statistical only as floor top-up
        final_v4, meta_v4 = _greedy_select_soft(
            measured, candidates_v4,
            dict(gstats["target_counts"]),
            dict(gstats["spacing_floor"]),
            float(gstats["intersection_limit"]),
            fam_centers,
            min_count_frac=0.5,
        )
        _record("hybrid_v4", final_v4)

    return rows


# ════════════════════════════════════════════════════════════════════════════════
# Main experiment loop
# ════════════════════════════════════════════════════════════════════════════════

ALL_METHODS = [
    "input_only",
    "statistical_sampler",
    "evae_oracle",
    "evae_sparse_legacy",
    "evae_sparse_fixed",
    "seq_v2_global",
    "seq_no_geo_panel",
    "seq_geo_panel",
    "seq_high_geo_panel",
    "set_predictor",
    "hybrid_no_ml",
    "hybrid_ml_improved",
    "hybrid_v4",
]


def run_all(
    methods:   Optional[List[str]] = None,
    seeds:     Optional[List[int]] = None,
    panels_to_test: Optional[List[int]] = None,
    conditions_filter: Optional[List[str]] = None,
    append: bool = False,
) -> pd.DataFrame:
    ensure_dirs()
    dev     = get_device()
    panels  = load_panels_raw()
    ps      = get_physics_stats()
    train_p = panels[:7]
    specs   = condition_specs()
    if conditions_filter:
        specs = [s for s in specs if s["condition"] in conditions_filter]
        if not specs:
            raise ValueError(f"No conditions match filter: {conditions_filter}")
    fam_centers = np.asarray(ps["fam_centers_rad"], dtype=float)
    gstats  = _train_geology_stats(train_p, fam_centers)

    methods_run = methods or ALL_METHODS
    seeds_run   = seeds   or CONTROL_SEEDS
    test_panels = panels_to_test or VALIDATION_PANELS

    print(f"Methods : {methods_run}")
    print(f"Panels  : {test_panels}")
    print(f"Seeds   : {seeds_run}")
    print(f"Conditions: {[s['condition'] for s in specs]}")
    print(f"Device  : {dev}")
    print()

    all_rows = []
    t0 = time.time()
    total_runs = len(specs) * len(test_panels) * len(seeds_run)
    done = 0

    for spec in specs:
        strips = spec["strips"]
        for pid in test_panels:
            real = np.asarray(panels[pid], dtype=np.float32)
            measured, target_arr = split_panel(real, strips)
            for seed in seeds_run:
                t1 = time.time()
                rows = run_one(
                    real=real,
                    measured=measured,
                    target_arr=target_arr,
                    strips=strips,
                    spec=spec,
                    panel_id=pid,
                    seed=seed,
                    ps=ps,
                    gstats=gstats,
                    dev=dev,
                    methods_to_run=methods_run,
                    train_panels=train_p,
                )
                all_rows.extend(rows)
                done += 1
                elapsed = time.time() - t1
                print(f"  [{done}/{total_runs}] {spec['condition']} "
                      f"panel={pid} seed={seed}  "
                      f"{len(rows)} method rows  {elapsed:.1f}s")

                # Periodic checkpoint save
                if done % 5 == 0:
                    df_tmp = pd.DataFrame(all_rows)
                    df_tmp.to_csv(TABLE_DIR / "ablation_v3_partial.csv",
                                  index=False)

    df = pd.DataFrame(all_rows)
    out_path = TABLE_DIR / "ablation_v3_full_results.csv"
    if append and out_path.exists():
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)
        print(f"\nAppended {len(all_rows)} new rows to {len(existing)} existing "
              f"-> {len(df)} total rows -> {out_path}  (total {time.time()-t0:.1f}s)")
    else:
        print(f"\nSaved {len(df)} rows -> {out_path}  (total {time.time()-t0:.1f}s)")
    df.to_csv(out_path, index=False)
    return df


# ════════════════════════════════════════════════════════════════════════════════
# Summary aggregation
# ════════════════════════════════════════════════════════════════════════════════

SUMMARY_METRICS = [
    "count_error_pct", "p20_error_pct", "p21_error_pct",
    "family_prop_l1", "angle_sim", "length_sim",
    "mean_length_err", "mean_angle_err_deg", "mean_pos_err",
    "spatial_sim", "int_density_err", "spacing_err",
    "measured_preservation_rate",
]


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run results: mean ± std by (condition_family, method)."""
    numeric_cols = [c for c in SUMMARY_METRICS if c in df.columns]
    if not numeric_cols:
        return df

    rows = []
    for (fam, method), g in df.groupby(["family", "method"]):
        row = {"condition_family": fam, "method": method, "n_runs": len(g)}
        for col in numeric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            row[f"{col}_std"]  = round(float(vals.std()),  4) if len(vals) else None
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "ablation_v3_summary.csv", index=False)

    # Also a pivot table for quick comparison
    pivot_cols = ["count_error_pct_mean", "mean_angle_err_deg_mean",
                  "mean_pos_err_mean", "family_prop_l1_mean",
                  "measured_preservation_rate_mean"]
    pivot_cols = [c for c in pivot_cols if c in summary.columns]
    if pivot_cols:
        pivot = summary.pivot_table(
            index="method", columns="condition_family",
            values=pivot_cols, aggfunc="first",
        )
        pivot.to_csv(TABLE_DIR / "ablation_v3_pivot.csv")

    print(f"Summary saved -> {TABLE_DIR / 'ablation_v3_summary.csv'}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run v3 ablation experiments.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help=f"Methods to run. Default: all. Options: {ALL_METHODS}")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="Random seeds. Default: 42001 42002 42003")
    ap.add_argument("--panels", nargs="*", type=int, default=None,
                    help="Panel IDs to test. Default: 7 8")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Condition names to run (subset). E.g. sparse_10cm_05m")
    ap.add_argument("--append", action="store_true",
                    help="Append new results to existing CSV instead of overwriting.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Only compute summary from existing CSV, skip generation.")
    args = ap.parse_args()

    if args.summary_only:
        full_csv = TABLE_DIR / "ablation_v3_full_results.csv"
        if full_csv.exists():
            df = pd.read_csv(full_csv)
            make_summary(df)
        else:
            print(f"No results found at {full_csv}. Run without --summary-only first.")
    else:
        df = run_all(
            methods=args.methods,
            seeds=[int(s) for s in args.seeds] if args.seeds else None,
            panels_to_test=[int(p) for p in args.panels] if args.panels else None,
            conditions_filter=args.conditions if args.conditions else None,
            append=args.append,
        )
        make_summary(df)
