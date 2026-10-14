# -*- coding: utf-8 -*-
"""Every path in this repository, resolved relative to the repository root.

The root is the directory holding the marker file `.repo-root`, found by walking up from this file, so the
code runs from any working directory and on any machine. A location can be overridden with an environment
variable if you keep the outputs elsewhere.
"""
import os
from pathlib import Path


def _find_root(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / ".repo-root").exists():
            return d
    raise RuntimeError("repository root not found: no .repo-root marker above " + str(start))


def _env(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


REPO_ROOT = _find_root(Path(__file__).resolve().parent)

DATA_DIR = REPO_ROOT / "data"
BOX_LIBRARY_DIR = DATA_DIR / "box_library"
SPLITS_DIR = DATA_DIR / "splits"
CLUSTERS_DIR = DATA_DIR / "orientation_clusters"
BASELINE_FITS_DIR = DATA_DIR / "baseline_fits"
# the box files are LabelMe JSON; the EVAE training module reads its default data path from here
MAPPED_TRACES_DIR = BOX_LIBRARY_DIR / "boxes"

SRC_DIR = REPO_ROOT / "src"
MODEL_DIR = SRC_DIR / "model"
EVALUATION_DIR = SRC_DIR / "evaluation"
BASELINES_DIR = SRC_DIR / "baselines"

MODELS_DIR = _env("S1_MODELS_DIR", REPO_ROOT / "models" / "evae")
NETWORKS_DIR = _env("S1_NETWORKS_DIR", REPO_ROOT / "networks")
SEEDS_DIR = REPO_ROOT / "seeds"
FIGURES_DIR = _env("S1_FIGURES_DIR", REPO_ROOT / "figures")
TABLES_DIR = _env("S1_TABLES_DIR", REPO_ROOT / "tables")

# Scratch space for reruns. Nothing here is tracked by git.
WORK_DIR = _env("S1_WORK_DIR", REPO_ROOT / "work")

# ADFNE 1.5 is third-party software and is NOT redistributed here. Download it separately and either place
# it at src/baselines/adfne_octave/ADFNE1.5 or set ADFNE_ROOT to a folder holding adfne_init.m and ADFNE1.5/.
ADFNE_ROOT = _env("ADFNE_ROOT", BASELINES_DIR / "adfne_octave")

__all__ = [n for n in dir() if n.isupper()]
