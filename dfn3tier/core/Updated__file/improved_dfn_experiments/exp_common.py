"""
exp_common.py  –  Shared config, data loading, and utilities for the
                   improved DFN experiment suite (v3).

All new experiment scripts import from here.  Nothing is overwritten in
the legacy folders.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT        = Path((__import__('os').environ['DFN3_CORE'] + r"".replace(chr(92), '/')))
UPDATED_DIR = ROOT / "Updated__file"
REVISED_DIR = ROOT / "Evae_sequential_hybrid_revised"
HYBRID_DIR  = ROOT / "Hybrid_evae_sequential"
DATA_DIR    = UPDATED_DIR / "data" / "9_original"
LEGACY_WORK = UPDATED_DIR / "borehole_10cm_joint_training"
LEGACY_MODELS = LEGACY_WORK / "outputs" / "models"

EXP_DIR    = UPDATED_DIR / "improved_dfn_experiments"
OUT_DIR    = EXP_DIR / "outputs"
MODEL_DIR  = OUT_DIR / "models"
TABLE_DIR  = OUT_DIR / "tables"
FIG_DIR    = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"

# ── Dataset constants ────────────────────────────────────────────────────────
TOTAL_WIDTH_M      = 36.0
BOREHOLE_WIDTH_M   = 0.10
BOREHOLE_FRACTION  = BOREHOLE_WIDTH_M / TOTAL_WIDTH_M   # ≈ 0.002778
SPARSE_INTERVALS_M = [3.0, 5.0, 7.0, 9.0, 11.0]
STRIP_WIDTHS_M     = [0.05, 0.10, 0.20]          # for sensitivity test
VALIDATION_PANELS  = [7, 8]
TRAIN_PANELS       = list(range(7))               # panels 0-6
ALL_PANELS         = list(range(9))
CONTROL_SEEDS      = [42001, 42002, 42003]
N_MAX_EVAE         = 120    # max fractures in EVAE / SetPredictor
MAX_CTX            = 40     # max context tokens in sequential model
MAX_GEN            = 80     # max generation tokens in sequential model

# ── Default checkpoint paths ─────────────────────────────────────────────────
EVAE_CKPT_LEGACY  = LEGACY_MODELS / "evae_10cm_finetuned.pt"
SEQ_CKPT_V2       = (UPDATED_DIR / "borehole_10cm_hybrid_v2" /
                     "outputs" / "models" / "seq_clip_10cm_hybrid_v2_geology.pth")

NEW_SET_PRED_CKPT = MODEL_DIR / "set_predictor_dfn_v3.pt"
NEW_SEQ_PANEL_CKPT = MODEL_DIR / "seq_panel_specific_v3.pth"
NEW_SEQ_HI_GEO_CKPT = MODEL_DIR / "seq_high_geology_v3.pth"


def ensure_dirs() -> None:
    for p in [OUT_DIR, MODEL_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


# ── Module loader (reuses Evae_sequential_hybrid_revised) ───────────────────
def load_revised_modules():
    """Load utils, model, train_evae, sequential_transformer from REVISED_DIR."""
    if str(REVISED_DIR) not in sys.path:
        sys.path.insert(0, str(REVISED_DIR))
    for name in ["utils", "model", "train_evae", "sequential_transformer"]:
        mod = sys.modules.get(name)
        if mod is not None:
            mod_path = Path(getattr(mod, "__file__", ""))
            if REVISED_DIR not in mod_path.parents:
                sys.modules.pop(name, None)
    utils_mod  = importlib.import_module("utils")
    model_mod  = importlib.import_module("model")
    te_mod     = importlib.import_module("train_evae")
    seq_mod    = importlib.import_module("sequential_transformer")
    return utils_mod, model_mod, te_mod, seq_mod


# ── Data loading ─────────────────────────────────────────────────────────────
def json_files() -> List[Path]:
    files = sorted(DATA_DIR.glob("*.json"))
    if len(files) != 9:
        raise RuntimeError(f"Expected 9 JSON files, found {len(files)} in {DATA_DIR}")
    return files


def load_panels_raw() -> List[np.ndarray]:
    """Load all 9 panels as (N,4) arrays [cx,cy,L,A_rad], normalised to [0,1]."""
    panels = []
    for fp in json_files():
        data = json.loads(fp.read_text(encoding="utf-8"))
        W    = float(data.get("imageWidth",  5280))
        H    = float(data.get("imageHeight", 1600))
        scale = max(W, H)
        rows = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "line":
                continue
            pts = np.array(shape["points"], dtype=np.float32)
            if len(pts) < 2:
                continue
            cx_px, cy_px = pts.mean(axis=0)
            dx, dy = pts[1] - pts[0]
            L_px = float(np.hypot(dx, dy))
            A    = float(np.arctan2(dy, dx) % np.pi)
            rows.append([cx_px / scale, cy_px / scale, L_px / scale, A])
        if rows:
            panels.append(np.array(rows, dtype=np.float32))
    if len(panels) != 9:
        raise RuntimeError(f"Expected 9 panels, got {len(panels)}")
    return panels


def get_train_val_panels() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    panels = load_panels_raw()
    return panels[:7], panels[7:]


# ── Physics stats ─────────────────────────────────────────────────────────────
def get_physics_stats(train_panels_raw: Optional[List[np.ndarray]] = None) -> dict:
    """Compute physics_stats dict from training panels using the legacy utils."""
    utils_mod, *_ = load_revised_modules()
    # utils_mod.load_labelme_data returns list of dicts with 'lines' key
    panels_utils = utils_mod.load_labelme_data(str(DATA_DIR))
    train_utils  = panels_utils[:7]
    return utils_mod.build_physics_stats(train_utils)


# ── Strip / borehole geometry ─────────────────────────────────────────────────
def borehole_centers_norm(interval_m: float) -> List[float]:
    c, centers = 0.5 * interval_m, []
    while c <= TOTAL_WIDTH_M - 0.5 * interval_m + 1e-9:
        centers.append(c / TOTAL_WIDTH_M)
        c += interval_m
    return centers


def strips_for_interval(interval_m: float,
                        width_m: float = BOREHOLE_WIDTH_M) -> List[Tuple[float, float]]:
    half = 0.5 * width_m / TOTAL_WIDTH_M
    return [(max(0.0, c - half), min(1.0, c + half))
            for c in borehole_centers_norm(interval_m)]


def central_strip(width_m: float = 1.80) -> List[Tuple[float, float]]:
    """5 % centre strip used as the dense-observation baseline."""
    half = 0.5 * width_m / TOTAL_WIDTH_M
    return [(0.5 - half, 0.5 + half)]


# ── Line clipping ─────────────────────────────────────────────────────────────
def _endpoints(cx: float, cy: float, L: float, A: float):
    return (cx - 0.5*L*math.cos(A), cy - 0.5*L*math.sin(A),
            cx + 0.5*L*math.cos(A), cy + 0.5*L*math.sin(A))


def split_line_by_strips(line: Sequence[float],
                         strips: Sequence[Tuple[float, float]]
                         ) -> Tuple[List[List[float]], List[List[float]]]:
    """Clip one fracture line into measured (within-strip) and target segments.
    Membership is by midpoint of each clipped segment."""
    cx, cy, L, A = [float(v) for v in line]
    x1, y1, x2, y2 = _endpoints(cx, cy, L, A)
    dx, dy = x2 - x1, y2 - y1
    t_vals = [0.0, 1.0]
    if abs(dx) > 1e-10:
        for lo, hi in strips:
            for xb in [float(lo), float(hi)]:
                t = (xb - x1) / dx
                if 1e-9 < t < 1.0 - 1e-9:
                    t_vals.append(t)
    t_vals = sorted(set(round(t, 12) for t in t_vals))
    measured, target = [], []
    for ta, tb in zip(t_vals[:-1], t_vals[1:]):
        if tb - ta <= 1e-9:
            continue
        xa = x1 + ta*dx;  ya = y1 + ta*dy
        xb = x1 + tb*dx;  yb = y1 + tb*dy
        seg_cx = 0.5*(xa+xb);  seg_cy = 0.5*(ya+yb)
        seg_L  = float(math.hypot(xb-xa, yb-ya))
        seg_A  = float(math.atan2(yb-ya, xb-xa) % math.pi)
        if seg_L <= 1e-6:
            continue
        midx = 0.5*(xa+xb)
        if any(float(lo) <= midx <= float(hi) for lo, hi in strips):
            measured.append([seg_cx, seg_cy, seg_L, seg_A])
        else:
            target.append([seg_cx, seg_cy, seg_L, seg_A])
    return measured, target


def split_panel(panel: np.ndarray,
                strips: Sequence[Tuple[float, float]]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (measured_arr, target_arr) for a whole panel."""
    measured, target = [], []
    for line in np.asarray(panel, dtype=float):
        m, t = split_line_by_strips(line, strips)
        measured.extend(m)
        target.extend(t)
    measured_arr = (np.array(measured, dtype=np.float32)
                    if measured else np.zeros((0, 4), dtype=np.float32))
    target_arr   = (np.array(target,  dtype=np.float32)
                    if target  else np.zeros((0, 4), dtype=np.float32))
    return measured_arr, target_arr


# ── Family assignment ─────────────────────────────────────────────────────────
def assign_family_ids(lines: np.ndarray,
                      family_centers_rad: Sequence[float],
                      threshold_deg: float = 20.0) -> np.ndarray:
    lines   = np.asarray(lines, dtype=float).reshape(-1, 4)
    centers = np.asarray(family_centers_rad, dtype=float)
    thresh  = math.radians(threshold_deg)
    fids    = []
    for A in lines[:, 3] if len(lines) else []:
        diffs = np.array([min(abs(float(A) - c), np.pi - abs(float(A) - c))
                          for c in centers])
        best  = int(np.argmin(diffs))
        fids.append(best if diffs[best] <= thresh else 3)
    return np.array(fids, dtype=int)


# ── Condition specifications ──────────────────────────────────────────────────
def condition_specs() -> List[Dict]:
    specs = [{
        "condition": "single_5pct_center",
        "family":    "baseline_5pct",
        "interval_m": None,
        "width_m":    1.80,
        "strips":     central_strip(1.80),
        "coverage_pct": 5.0,
    }]
    for iv in SPARSE_INTERVALS_M:
        stps = strips_for_interval(iv)
        specs.append({
            "condition":    f"sparse_10cm_{int(iv):02d}m",
            "family":       "sparse_10cm",
            "interval_m":   iv,
            "width_m":      BOREHOLE_WIDTH_M,
            "strips":       stps,
            "coverage_pct": 100.0 * sum(hi - lo for lo, hi in stps),
        })
    return specs


# ── Token helpers (mirror sequential_transformer.py) ──────────────────────────
def tok(cx: float, cy: float, L: float, A_rad: float) -> List[float]:
    return [cx, cy, math.log(max(L, 1e-8)),
            math.sin(A_rad), math.cos(A_rad)]


def tok_array(lines: np.ndarray) -> np.ndarray:
    """Convert (N,4) [cx,cy,L,A] → (N,5) tokens."""
    lines = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
    logL  = np.log(np.maximum(lines[:, 2], 1e-8))
    s     = np.sin(lines[:, 3])
    c     = np.cos(lines[:, 3])
    return np.column_stack([lines[:, 0], lines[:, 1], logL, s, c]).astype(np.float32)


def detok_array(tokens: np.ndarray) -> np.ndarray:
    """Convert (N,5) tokens → (N,4) [cx,cy,L,A]."""
    tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, 5)
    cx   = tokens[:, 0]
    cy   = tokens[:, 1]
    L    = np.exp(tokens[:, 2])
    A    = np.arctan2(tokens[:, 3], tokens[:, 4]) % np.pi
    return np.column_stack([cx, cy, L, A]).astype(np.float32)


# ── Geometry helpers ─────────────────────────────────────────────────────────
def line_endpoints(row: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy, L, A = [float(v) for v in row]
    x1, y1, x2, y2 = _endpoints(cx, cy, L, A)
    return np.array([x1, y1]), np.array([x2, y2])


def segments_intersect(a: Sequence[float], b: Sequence[float]) -> bool:
    p1, p2 = line_endpoints(a)
    p3, p4 = line_endpoints(b)
    def orient(p, q, r):
        return float((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))
    o1, o2 = orient(p1,p2,p3), orient(p1,p2,p4)
    o3, o4 = orient(p3,p4,p1), orient(p3,p4,p2)
    return bool(o1*o2 < 0.0 and o3*o4 < 0.0)


def intersection_density(lines: np.ndarray) -> float:
    lines = np.asarray(lines, dtype=float).reshape(-1, 4)
    if len(lines) < 2:
        return 0.0
    count = sum(1 for i in range(len(lines))
                for j in range(i+1, len(lines))
                if segments_intersect(lines[i], lines[j]))
    return count / max(len(lines), 1)


# ── Save helpers ─────────────────────────────────────────────────────────────
def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    ensure_dirs()
    panels = load_panels_raw()
    ps = get_physics_stats()
    print(f"Loaded {len(panels)} panels.")
    print(f"Physics stats keys: {list(ps.keys())}")
    for sp in condition_specs():
        n_strips = len(sp["strips"])
        pct      = sp["coverage_pct"]
        print(f"  {sp['condition']}: {n_strips} strip(s), {pct:.2f}% coverage")
