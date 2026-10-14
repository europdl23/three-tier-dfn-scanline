"""Generation arms per the frozen protocol Sections 5-6.

EVAE: Route B revised-branch checkpoint, bare decoder output (no NMS/rerank/refine/quota).
ADFNE 1.5: unmodified, under Octave, per-set parameters fitted on fitting windows only.
KDE: per-set gaussian_kde centres + empirical length/angle resampling.
All counts for ADFNE/KDE: protocol Section 6 rule (imposed fitted parameter, disclosed).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import common

HERE = Path(__file__).resolve().parent
from repo_paths import MODEL_DIR as _MODEL_DIR, ADFNE_ROOT as _ADFNE_ROOT, WORK_DIR as _WORK_DIR
BRANCH = _MODEL_DIR            # released: src/model (the EVAE implementation)
ADFNE_ROOT = _ADFNE_ROOT       # released: src/baselines/adfne_octave, override with ADFNE_ROOT
OCT_WORK = Path(os.environ.get("ADFNE_WORK", str(_WORK_DIR / "adfne_work")))


# --------------------------------------------------------------- fits (Section 6)

def fit_baseline_stats(fit_panels, rule):
    """Per-set ADFNE/KDE parameters from the fitting windows only."""
    lines = np.vstack([p["lines"] for p in fit_panels])
    _, lens, angs = common.geom(lines)
    lab = common.assign_sets(angs, rule)
    mean_ymax = float(np.mean([p["y_max"] for p in fit_panels]))
    sets = []
    for k in range(len(rule["mu2"])):
        use = lab == k
        counts = []
        for p in fit_panels:
            _, _, a = common.geom(p["lines"])
            counts.append(int((common.assign_sets(a, rule) == k).sum()))
        centres = common.geom(lines)[0][use]
        sets.append({
            "set": k,
            "centre_deg": math.degrees(rule["centres_theta"][k]),
            "kappa_dir": common.kappa_directional_from_axial(angs[use]),
            "length_fit": common.trunc_exp_fit(lens[use]),      # (low, mu, high)
            "mean_count": float(np.mean(counts)),
            "angles": angs[use], "lengths": lens[use], "centres": centres,
        })
    return {"sets": sets, "mean_ymax": mean_ymax}


def quota(stats_set, mean_ymax, y_max_target):
    """Protocol Section 6 count rule."""
    return max(1, int(round(stats_set["mean_count"] / mean_ymax * y_max_target)))


# --------------------------------------------------------------- EVAE (Section 5)

_training_mod = None


def _training():
    global _training_mod
    if _training_mod is None:
        os.environ["ROUTEB_ISO"] = "1"
        sys.path.insert(0, str(BRANCH))
        import training as t  # noqa
        assert t.ROUTEB_ISO is True
        _training_mod = t
    return _training_mod


def load_evae(ckpt_path):
    t = _training()
    import torch
    model = t.DualLatentVAE_Updated(spatial_dim=24, geometry_dim=32, K_angle=3)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev).eval(), dev


def generate_evae(model, dev, y_max, seed_gen):
    """Protocol Section 5: z ~ N(0,I); exist >= 0.5 (top-10 fallback if < 2); angle
    sampled from the slot's own axial vM mixture (2theta ~ vM(2mu_k, kappa_k)); isotropic
    endpoints; Liang-Barsky clip. Temperature 1.0."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed_gen)
    rng = np.random.default_rng(seed_gen)
    with torch.no_grad():
        sz = torch.randn(1, 24, device=dev)
        gz = torch.randn(1, 32, device=dev)
        centers = torch.sigmoid(model.spatial_decoder(sz)).view(-1, model.max_lines, 2)[0]
        lengths = torch.sigmoid(model.length_decoder(gz)).view(-1, model.max_lines)[0]
        logits = model.angle_logits_head(gz).view(-1, model.max_lines, model.K_angle)[0]
        mu_raw = model.angle_mu_head(gz).view(-1, model.max_lines, model.K_angle)[0]
        kappa_raw = model.angle_kappa_head(gz).view(-1, model.max_lines, model.K_angle)[0]
        alpha = F.softmax(logits, dim=-1).cpu().numpy()
        mu = (math.pi * torch.tanh(mu_raw)).cpu().numpy()
        kappa = (F.softplus(kappa_raw) + 1e-4).cpu().numpy()
        exist = model.existence_head(torch.cat([sz, gz], 1)).view(-1, model.max_lines)[0]
        keepmask = (torch.sigmoid(exist) >= 0.5).cpu().numpy()
        if keepmask.sum() < 2:
            top = torch.topk(exist, 10).indices.cpu().numpy()
            keepmask = np.zeros(model.max_lines, bool)
            keepmask[top] = True
        c = centers.cpu().numpy()[keepmask]
        L = lengths.cpu().numpy()[keepmask]
        a_alpha, a_mu, a_kappa = alpha[keepmask], mu[keepmask], kappa[keepmask]
    thetas = []
    for j in range(len(c)):
        comp = rng.choice(model.K_angle, p=a_alpha[j] / a_alpha[j].sum())
        z2 = rng.vonmises(2.0 * a_mu[j, comp], max(a_kappa[j, comp], 1e-4))
        thetas.append((z2 / 2.0) % math.pi)
    thetas = np.asarray(thetas)
    dx = np.cos(thetas) * L / 2
    dy = np.sin(thetas) * L / 2
    segs = np.column_stack([c[:, 0] - dx, c[:, 1] - dy, c[:, 0] + dx, c[:, 1] + dy])
    return common.clip_lines(segs, y_max)


# --------------------------------------------------------------- ADFNE (Section 6)

def generate_adfne(stats, y_max, seed_gen, tag):
    """One network. Writes and runs an Octave script (rng = seed_gen mod 2^31), then
    filters and subsamples host-side with the same derived seed."""
    OCT_WORK.mkdir(parents=True, exist_ok=True)
    m_path = OCT_WORK / f"adfne_{tag}.m"
    init_path = str(ADFNE_ROOT / "adfne_init.m").replace("\\", "/")
    rows = [f"run('{init_path}');", f"rng({seed_gen % (2**31)});"]
    out_paths = []
    for s in stats["sets"]:
        q = quota(s, stats["mean_ymax"], y_max)
        requested = max(100, int(math.ceil(q / max(y_max, 1e-4) * 8)))
        low, mu, high = s["length_fit"]
        out_path = OCT_WORK / f"adfne_{tag}_set{s['set']}.csv"
        out_paths.append((out_path, q))
        op = str(out_path).replace("\\", "/")
        rows += [
            f"o = DFN('dim',2,'n',{requested},'dir',{s['centre_deg']:.10f},"
            f"'ddir',{-s['kappa_dir']:.10f},'minl',{low:.10f},'mu',{mu:.10f},"
            f"'maxl',{high:.10f},'bbx',[0,0,1,1]);",
            "allx=o.Orig;",
            f"dlmwrite('{op}',allx,',');",
        ]
    m_path.write_text("\n".join(rows), encoding="ascii")
    cmd = [str(ADFNE_ROOT / "run_adfne.bat"), str(m_path)]
    done = subprocess.run(cmd, cwd=str(ADFNE_ROOT), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=180)
    if done.returncode != 0 or any(not p.exists() for p, _ in out_paths):
        raise RuntimeError(f"ADFNE failed for {tag}:\n{done.stdout[-2000:]}")
    rng = np.random.default_rng(seed_gen)
    selected = []
    for out_path, q in out_paths:
        raw = np.atleast_2d(np.loadtxt(out_path, delimiter=","))
        ctr_y = (raw[:, 1] + raw[:, 3]) / 2
        raw = raw[(ctr_y >= 0) & (ctr_y <= y_max)]      # filter FIRST (protocol order)
        if len(raw) == 0:
            continue
        picks = rng.choice(len(raw), size=q, replace=len(raw) < q)
        selected.extend(raw[picks])
    for p, _ in out_paths:
        p.unlink(missing_ok=True)
    m_path.unlink(missing_ok=True)
    return common.clip_lines(selected, y_max)


# --------------------------------------------------------------- KDE (Section 6)

def generate_kde(stats, y_max, seed_gen):
    rng = np.random.default_rng(seed_gen)
    lines = []
    for s in stats["sets"]:
        q = quota(s, stats["mean_ymax"], y_max)
        samples = []
        try:
            density = gaussian_kde(s["centres"].T)
            attempts = 0
            while len(samples) < q and attempts < q * 40:
                pt = density.resample(1, seed=rng).reshape(2)
                if 0 <= pt[0] <= 1 and 0 <= pt[1] <= y_max:
                    samples.append(pt)
                attempts += 1
        except Exception:
            samples = [s["centres"][rng.integers(len(s["centres"]))] for _ in range(q)]
        while len(samples) < q:
            samples.append(np.asarray([rng.uniform(), rng.uniform(0, y_max)]))
        lens = rng.choice(s["lengths"], size=q, replace=True)
        angs = rng.choice(s["angles"], size=q, replace=True)
        for pt, ln, an in zip(samples, lens, angs):
            dx, dy = math.cos(an) * ln / 2, math.sin(an) * ln / 2
            lines.append([pt[0] - dx, pt[1] - dy, pt[0] + dx, pt[1] + dy])
    return common.clip_lines(lines, y_max)
