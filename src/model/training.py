import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from utils import *
from utils_additions import *
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml
from datetime import datetime

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# released: repo-relative. The training driver overrides this before use.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from repo_paths import MAPPED_TRACES_DIR as _MTD, WORK_DIR as _WD
DATA_PATH = str(_MTD)
BASE_SAVE_DIR = str(_WD / "training_side_outputs")

# ---------------------------------------------------------------------------------------------
# REVISED METHOD, 2026-09-03. Single change against the frozen baseline: a per-branch free-bits
# floor on the KL objective. LAMBDA_FREE = 0.0 makes this an exact no-op, verified bitwise.
# Derivation of the value and full rationale: 07_revised_evae_method/FREE_BITS_REPAIR_PROTOCOL.md
LAMBDA_FREE = float(os.environ.get("LAMBDA_FREE", "0.0"))
# ---------------------------------------------------------------------------------------------

# ROUTE B PATCH 2026-09-07 (Gate 1 Section 8 item 2, T16/T12/T13/T14/T17a-T17d sites).
# Under the isotropic convention every evaluation/generation constant becomes (5280, 5280),
# which makes T09's atan2(dy*H, dx*W) the identity and the T13 snap radius circular.
# ROUTEB_ISO=0 restores the source branch behaviour bitwise (Section 8 item 6 no-op check).
ROUTEB_ISO = os.environ.get("ROUTEB_ISO", "1") == "1"
EVAL_W, EVAL_H = (5280.0, 5280.0) if ROUTEB_ISO else (5280, 1600)
# ---------------------------------------------------------------------------------------------

USE_MVM_ANGLE = True
USE_ANGLE_OT = False
ANGLE_OT_WEIGHT = 0.002
USE_HUNGARIAN = True
USE_DUAL_ENCODER = True
AUG_TRAIN = True

# released: these were created at import time, relative to the current working directory, which
# scattered empty folders wherever the module was imported from. The training driver creates them
# inside the run directory before training starts, and the checkpoint save below creates its own.
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

G1_CONFIG = {"N": 800, "T": 1.05, "q": 0.55, "alpha": 0.03, "K": 20, "angle_weight": 0.6, "nms_dist": 0.02}

def A1_inverse(R):
    if R < 0.53:
        return 2 * R + R ** 3 + (5 * R ** 5) / 6
    elif R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        return 1 / (R ** 3 - 4 * R ** 2 + 3 * R)

def fit_vm_mixture_em(angles, K, max_iter=20):
    from scipy.special import i0e
    N = len(angles)
    if N == 0:
        return None
    angles = np.array(angles)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    points = np.stack([cos_a, sin_a], axis=1)
    if K == 1:
        mu = np.arctan2(sin_a.mean(), cos_a.mean())
        R = np.sqrt((cos_a.mean()) ** 2 + (sin_a.mean()) ** 2)
        kappa = A1_inverse(R)
        return {'K': 1, 'alpha': np.array([1.0]), 'mu': np.array([mu]), 'kappa': np.array([kappa])}
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=1337)
    labels = kmeans.fit_predict(points)
    alpha = np.zeros(K)
    mu = np.zeros(K)
    kappa = np.zeros(K)
    for k in range(K):
        mask = (labels == k)
        if mask.sum() == 0:
            alpha[k] = 1.0 / K
            mu[k] = 0.0
            kappa[k] = 1.0
            continue
        alpha[k] = mask.sum() / N
        mu[k] = np.arctan2(sin_a[mask].mean(), cos_a[mask].mean())
        R_k = np.sqrt((cos_a[mask].mean()) ** 2 + (sin_a[mask].mean()) ** 2)
        kappa[k] = A1_inverse(np.clip(R_k, 0.01, 0.99))
    for iteration in range(max_iter):
        resp = np.zeros((N, K))
        for k in range(K):
            log_vm = kappa[k] * np.cos(angles - mu[k]) - np.log(2 * np.pi) - (np.log(i0e(kappa[k]) + 1e-12) + kappa[k])
            resp[:, k] = alpha[k] * np.exp(log_vm)
        resp_sum = resp.sum(axis=1, keepdims=True) + 1e-10
        resp = resp / resp_sum
        for k in range(K):
            r_k = resp[:, k]
            alpha[k] = r_k.mean()
            cos_mean = (r_k * cos_a).sum() / (r_k.sum() + 1e-10)
            sin_mean = (r_k * sin_a).sum() / (r_k.sum() + 1e-10)
            mu[k] = np.arctan2(sin_mean, cos_mean)
            R_k = np.sqrt(cos_mean ** 2 + sin_mean ** 2)
            kappa[k] = A1_inverse(np.clip(R_k, 0.01, 0.99))
    return {'K': K, 'alpha': alpha, 'mu': mu, 'kappa': kappa}

def compute_bic_vm_mixture(angles, params):
    from scipy.special import i0e
    N = len(angles)
    K = params['K']
    alpha = params['alpha']
    mu = params['mu']
    kappa = params['kappa']
    log_lik = 0.0
    for theta in angles:
        mix_prob = 0.0
        for k in range(K):
            log_vm = kappa[k] * np.cos(theta - mu[k]) - np.log(2 * np.pi) - (np.log(i0e(kappa[k]) + 1e-12) + kappa[k])
            mix_prob += alpha[k] * np.exp(log_vm)
        log_lik += np.log(mix_prob + 1e-10)
    num_params = 3 * K - 1
    bic = -2 * log_lik + num_params * np.log(N)
    return bic

def select_K_via_bic(angles, K_range=[1, 2, 3]):
    best_K = 1
    best_bic = float('inf')
    best_params = None
    for K in K_range:
        params = fit_vm_mixture_em(angles, K)
        if params is None:
            continue
        bic = compute_bic_vm_mixture(angles, params)
        if bic < best_bic:
            best_bic = bic
            best_K = K
            best_params = params
    return best_K, best_params

class DualLatentVAE_Updated(nn.Module):
    def __init__(self, spatial_dim=24, geometry_dim=32, max_lines=120, K_angle=1):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.geometry_dim = geometry_dim
        self.max_lines = max_lines
        self.K_angle = K_angle
        self.spatial_encoder = nn.Sequential(
            nn.Linear(max_lines * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, spatial_dim * 2)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(max_lines * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, geometry_dim * 2)
        )
        self.spatial_decoder = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_lines * 2)
        )
        self.length_decoder = nn.Sequential(
            nn.Linear(geometry_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_lines)
        )
        if USE_MVM_ANGLE:
            self.angle_logits_head = nn.Sequential(
                nn.Linear(geometry_dim, 128),
                nn.ReLU(),
                nn.Linear(128, max_lines * K_angle)
            )
            self.angle_mu_head = nn.Sequential(
                nn.Linear(geometry_dim, 128),
                nn.ReLU(),
                nn.Linear(128, max_lines * K_angle)
            )
            self.angle_kappa_head = nn.Sequential(
                nn.Linear(geometry_dim, 128),
                nn.ReLU(),
                nn.Linear(128, max_lines * K_angle)
            )
        else:
            self.angle_decoder = nn.Sequential(
                nn.Linear(geometry_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, max_lines)
            )
        self.existence_head = nn.Sequential(
            nn.Linear(spatial_dim + geometry_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_lines)
        )

    def encode(self, centers, lengths_angles):
        B = centers.shape[0]
        centers_flat = centers.view(B, -1)
        la_flat = lengths_angles.view(B, -1)
        spatial_params = self.spatial_encoder(centers_flat)
        geometry_params = self.geometry_encoder(la_flat)
        smu, slog = spatial_params.chunk(2, dim=1)
        gmu, glog = geometry_params.chunk(2, dim=1)
        return smu, slog, gmu, glog

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def decode(self, z_spatial, z_geometry):
        centers = torch.sigmoid(self.spatial_decoder(z_spatial)).view(-1, self.max_lines, 2)[0]
        lengths = torch.sigmoid(self.length_decoder(z_geometry)).view(-1, self.max_lines)[0]
        if USE_MVM_ANGLE:
            logits = self.angle_logits_head(z_geometry).view(-1, self.max_lines, self.K_angle)[0]
            mu_raw = self.angle_mu_head(z_geometry).view(-1, self.max_lines, self.K_angle)[0]
            kappa_raw = self.angle_kappa_head(z_geometry).view(-1, self.max_lines, self.K_angle)[0]
            alpha = F.softmax(logits, dim=-1)
            mu = math.pi * torch.tanh(mu_raw)
            kappa = F.softplus(kappa_raw) + 1e-4
            sin_exp = (alpha * torch.sin(mu)).sum(dim=-1)
            cos_exp = (alpha * torch.cos(mu)).sum(dim=-1)
            angles = torch.atan2(sin_exp, cos_exp) % math.pi
        else:
            angles = torch.sigmoid(self.angle_decoder(z_geometry)).view(-1, self.max_lines)[0] * math.pi
        zc = torch.cat([z_spatial, z_geometry], dim=1)
        exist_logits = self.existence_head(zc).view(-1, self.max_lines)[0]
        return centers, lengths, angles, exist_logits

    def forward(self, lines_norm, valid_mask, img_size):
        centers, lengths, angles = to_centers_lengths_angles(lines_norm)
        lengths_angles = torch.stack([lengths, angles], dim=-1)
        smu, slog, gmu, glog = self.encode(centers, lengths_angles)
        z_spatial = self.reparameterize(smu, slog)
        z_geometry = self.reparameterize(gmu, glog)
        pred_centers = torch.sigmoid(self.spatial_decoder(z_spatial)).view(-1, self.max_lines, 2)
        pred_lengths = torch.sigmoid(self.length_decoder(z_geometry))
        if USE_MVM_ANGLE:
            logits = self.angle_logits_head(z_geometry).view(-1, self.max_lines, self.K_angle)
            mu_raw = self.angle_mu_head(z_geometry).view(-1, self.max_lines, self.K_angle)
            kappa_raw = self.angle_kappa_head(z_geometry).view(-1, self.max_lines, self.K_angle)
            alpha = F.softmax(logits, dim=-1)
            mu = math.pi * torch.tanh(mu_raw)
            kappa = F.softplus(kappa_raw) + 1e-4
            angle_params = {'alpha': alpha, 'mu': mu, 'kappa': kappa}
        else:
            pred_angles = torch.sigmoid(self.angle_decoder(z_geometry)) * math.pi
            angle_params = {'angles': pred_angles}
        z_combined = torch.cat([z_spatial, z_geometry], dim=1)
        existence_logits = self.existence_head(z_combined)
        return {
            'centers': pred_centers,
            'lengths': pred_lengths,
            'angle_params': angle_params,
            'existence_logits': existence_logits,
            'smu': smu, 'slog': slog,
            'gmu': gmu, 'glog': glog,
            'z_spatial': z_spatial,
            'z_geometry': z_geometry
        }

def vm_log_prob(theta, mu, kappa):
    two_pi = theta.new_tensor(2.0 * math.pi)
    try:
        log_i0 = torch.log(torch.special.i0e(kappa) + 1e-10) + kappa
    except AttributeError:
        log_i0 = torch.log(torch.i0(kappa) + 1e-10)
    return kappa * torch.cos(theta - mu) - torch.log(two_pi) - log_i0


def mixture_vm_nll(theta, alpha, mu, kappa):
    B, N = theta.shape
    K = alpha.shape[-1]
    theta_exp = theta.unsqueeze(-1).expand(B, N, K)
    mu_exp = mu.expand(B, N, K)
    kappa_exp = kappa.expand(B, N, K)
    log_probs = vm_log_prob(theta_exp, mu_exp, kappa_exp)
    log_weighted = torch.log(alpha + 1e-10) + log_probs
    log_mixture = torch.logsumexp(log_weighted, dim=-1)
    nll = -log_mixture.mean()
    return nll

def generated_direction_w1(angle_params, existence_logits, gt_angles, valid_mask, nbins=36, kappa_t=20.0):
    """FAMILY FIX A (30 Sep 2026, cv_study PROTOCOL_FAMILY_FIX.md). Circular W1 between the direction distribution the
    model would GENERATE for each window (each slot's axial von Mises mixture, components weighted by alpha exactly as
    generation samples them, slots weighted by existence probability) and the window's REAL directions (von Mises
    kernel, concentration kappa_t on 2-theta). Evaluated on nbins axial bins over [0, pi). Returns radians of theta."""
    dev = gt_angles.device
    grid = (torch.arange(nbins, device=dev, dtype=gt_angles.dtype) + 0.5) * (math.pi / nbins)          # theta
    a, mu, k = angle_params['alpha'], angle_params['mu'], angle_params['kappa']                         # B,N,K
    # slot density on the grid: sum_k alpha_k vM(2 theta; 2 mu_k, kappa_k), normalised per slot over the grid
    arg = 2.0 * grid.view(1, 1, 1, -1) - 2.0 * mu.unsqueeze(-1)
    logd = k.unsqueeze(-1) * (torch.cos(arg) - 1.0)                                                    # stable (divide by exp(kappa))
    d = (a.unsqueeze(-1) * torch.exp(logd) / (torch.exp(logd).sum(-1, keepdim=True) + 1e-12)).sum(2)  # B,N,nbins
    w = torch.sigmoid(existence_logits).unsqueeze(-1)
    G = (w * d).sum(1); G = G / (G.sum(-1, keepdim=True) + 1e-12)                                      # B,nbins
    R = torch.exp(kappa_t * (torch.cos(2.0 * grid.view(1, 1, -1) - 2.0 * gt_angles.unsqueeze(-1)) - 1.0))
    R = R / (R.sum(-1, keepdim=True) + 1e-12)
    R = (R * valid_mask.unsqueeze(-1)).sum(1); R = R / (R.sum(-1, keepdim=True) + 1e-12)              # B,nbins
    D = torch.cumsum(G - R, dim=-1)
    t = torch.median(D, dim=-1, keepdim=True).values
    w1 = (D - t).abs().mean(-1) * math.pi                                                              # circular W1 (radians)
    has = (valid_mask.sum(-1) > 0).to(w1.dtype)
    return (w1 * has).sum() / has.sum().clamp_min(1.0)

_EVBG_RB = None


def _evbg_bin_resp(nbins, dev, dtype):
    """responsibilities (set 1..K, background) of the learned set model at the nbins axial bin centres (EVBG_MODEL json)"""
    global _EVBG_RB
    if _EVBG_RB is None:
        m = json.load(open(os.environ["EVBG_MODEL"]))
        grid = (np.arange(nbins) + 0.5) * math.pi / nbins; z = 2.0 * grid
        from scipy.special import i0e
        cols = [w * np.exp(k * (np.cos(z - mu) - 1.0)) / (2 * math.pi * i0e(k)) for w, mu, k in zip(m["w_sets"], m["mu2"], m["kappa2"])]
        cols.append(np.full(nbins, m["w_bg"] / (2 * math.pi)))
        D = np.stack(cols, 1); _EVBG_RB = D / D.sum(1, keepdims=True)
    return torch.as_tensor(_EVBG_RB, device=dev, dtype=dtype)


def generated_class_share_l1(angle_params, existence_logits, gt_angles, valid_mask, nbins=36, kappa_t=20.0):
    """SETS + BACKGROUND GUIDANCE (1 Oct 2026, PROTOCOL_SETS_BG.md). L1 difference between the class shares (each learned
    set and background) of the directions the model would GENERATE for a window and those of the window's REAL traces,
    both under the fold's learned set model (EVBG_MODEL). Generated directions: each slot's axial von Mises mixture,
    slots weighted by existence probability (as in generated_direction_w1)."""
    dev = gt_angles.device
    grid = (torch.arange(nbins, device=dev, dtype=gt_angles.dtype) + 0.5) * (math.pi / nbins)
    a, mu, k = angle_params['alpha'], angle_params['mu'], angle_params['kappa']
    arg = 2.0 * grid.view(1, 1, 1, -1) - 2.0 * mu.unsqueeze(-1)
    logd = k.unsqueeze(-1) * (torch.cos(arg) - 1.0)
    d = (a.unsqueeze(-1) * torch.exp(logd) / (torch.exp(logd).sum(-1, keepdim=True) + 1e-12)).sum(2)
    w = torch.sigmoid(existence_logits).unsqueeze(-1)
    G = (w * d).sum(1); G = G / (G.sum(-1, keepdim=True) + 1e-12)
    R = torch.exp(kappa_t * (torch.cos(2.0 * grid.view(1, 1, -1) - 2.0 * gt_angles.unsqueeze(-1)) - 1.0))
    R = R / (R.sum(-1, keepdim=True) + 1e-12)
    R = (R * valid_mask.unsqueeze(-1)).sum(1); R = R / (R.sum(-1, keepdim=True) + 1e-12)
    Rb = _evbg_bin_resp(nbins, dev, G.dtype)
    l1 = ((G @ Rb) - (R @ Rb)).abs().sum(-1)
    has = (valid_mask.sum(-1) > 0).to(l1.dtype)
    return (l1 * has).sum() / has.sum().clamp_min(1.0)


def hungarian_assignment_loss_updated(pred_centers, pred_lengths, angle_params, gt_centers, gt_lengths, gt_angles, valid_mask):
    B = pred_centers.shape[0]
    total_loss = 0.0
    for b in range(B):
        n_valid = int(valid_mask[b].sum().item())
        if n_valid == 0:
            continue
        pred_c = pred_centers[b]
        pred_l = pred_lengths[b]
        gt_c = gt_centers[b][:n_valid]
        gt_l = gt_lengths[b][:n_valid]
        gt_a = gt_angles[b][:n_valid]
        cost_center = torch.cdist(pred_c, gt_c, p=2)
        cost_length = torch.cdist(pred_l.unsqueeze(1), gt_l.unsqueeze(1), p=1).squeeze(-1)
        if USE_MVM_ANGLE:
            alpha = angle_params['alpha'][b]
            mu = angle_params['mu'][b]
            kappa = angle_params['kappa'][b]
            diff2 = torch.abs(2 * mu.unsqueeze(1) - 2 * gt_a.unsqueeze(0).unsqueeze(-1))
            cost_angle, _ = torch.minimum(diff2, 2 * math.pi - diff2).min(dim=-1)
        else:
            pred_a = angle_params['angles'][b]
            cost_angle = torch.abs(torch.sin(pred_a.unsqueeze(1) - gt_a.unsqueeze(0)))
        cost_matrix = 1.0 * cost_center + 1.0 * cost_length + 1.0 * cost_angle
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        row_ind = torch.tensor(row_ind, dtype=torch.long, device=pred_c.device)
        col_ind = torch.tensor(col_ind, dtype=torch.long, device=pred_c.device)
        center_loss = F.mse_loss(pred_c[row_ind], gt_c[col_ind])
        length_loss = F.mse_loss(pred_l[row_ind], gt_l[col_ind])
        if USE_MVM_ANGLE:
            matched_angles = gt_a[col_ind]
            alpha_matched = angle_params['alpha'][b][row_ind]
            mu_matched = angle_params['mu'][b][row_ind]
            kappa_matched = angle_params['kappa'][b][row_ind]
            angle_loss = mixture_vm_nll(2 * matched_angles.unsqueeze(0), alpha_matched.unsqueeze(0), 2 * mu_matched.unsqueeze(0), kappa_matched.unsqueeze(0))
        else:
            pred_a = angle_params['angles'][b]
            angle_loss = torch.mean(1.0 - torch.cos(pred_a[row_ind] - gt_a[col_ind]))
        total_loss += center_loss + length_loss + angle_loss
    return total_loss / B

def naive_assignment_loss(pred_centers, pred_lengths, angle_params, gt_centers, gt_lengths, gt_angles, valid_mask):
    B = pred_centers.shape[0]
    total_loss = 0.0
    for b in range(B):
        n_valid = int(valid_mask[b].sum().item())
        if n_valid == 0:
            continue
        pred_c = pred_centers[b][:n_valid]
        pred_l = pred_lengths[b][:n_valid]
        gt_c = gt_centers[b][:n_valid]
        gt_l = gt_lengths[b][:n_valid]
        gt_a = gt_angles[b][:n_valid]
        center_loss = F.mse_loss(pred_c, gt_c)
        length_loss = F.mse_loss(pred_l, gt_l)
        if USE_MVM_ANGLE:
            alpha_matched = angle_params['alpha'][b][:n_valid]
            mu_matched = angle_params['mu'][b][:n_valid]
            kappa_matched = angle_params['kappa'][b][:n_valid]
            angle_loss = mixture_vm_nll(2 * gt_a.unsqueeze(0), alpha_matched.unsqueeze(0), 2 * mu_matched.unsqueeze(0), kappa_matched.unsqueeze(0))
        else:
            pred_a = angle_params['angles'][b][:n_valid]
            angle_loss = torch.mean(1.0 - torch.cos(pred_a - gt_a))
        total_loss += center_loss + length_loss + angle_loss
    return total_loss / B

def length_constraint_loss_data(pred_lengths, max_length):
    return F.relu(pred_lengths - max_length).mean()

def intersection_penalty_loss_data(pred_centers, pred_lengths, pred_angles, exist_logits, target_rate):
    endpoints = centers_lengths_angles_to_endpoints(pred_centers, pred_lengths.view(-1, pred_centers.size(1)), pred_angles)
    exist_probs = torch.sigmoid(exist_logits)
    s = soft_intersection_density(endpoints, exist_probs)
    t = s.new_full(s.shape, float(target_rate))
    return F.mse_loss(s, t)

def local_spacing_loss_data(pred_centers, exist_logits, target_min_dist):
    B = pred_centers.shape[0]
    p = torch.sigmoid(exist_logits)
    loss = pred_centers.new_tensor(0.0)
    for b in range(B):
        mask = p[b] > 0.5
        if mask.sum() < 2:
            continue
        c = pred_centers[b][mask]
        D = torch.cdist(c, c, p=2) + torch.eye(c.size(0), device=c.device) * 10.0
        nn = D.min(dim=1)[0]
        loss = loss + F.relu(target_min_dist - nn).mean()
    return loss / B

def compute_histogram_metrics_console(lines, priors, W, H):
    if len(lines) == 0:
        return 0.0, 0.0

    angles = []
    lengths = []

    for line in lines:
        if isinstance(line, dict):
            x1, y1 = line['start']
            x2, y2 = line['end']
        else:
            x1, y1, x2, y2 = line

        dx_pix = (x2 - x1)
        dy_pix = (y2 - y1)
        angle = np.degrees(np.arctan2(dy_pix, dx_pix)) % 180.0

        dx = dx_pix / W
        dy = dy_pix / H
        length = np.sqrt(dx * dx + dy * dy)

        angles.append(angle)
        lengths.append(length)

    angle_hist, _ = np.histogram(angles, bins=18, range=(0, 180))
    angle_hist = angle_hist / (angle_hist.sum() + 1e-8)

    length_edges = np.array(priors['length_edges'])
    length_hist, _ = np.histogram(np.clip(lengths, length_edges[0], length_edges[-1]), bins=length_edges)
    length_hist = length_hist / (length_hist.sum() + 1e-8)

    target_angle = np.array(priors['global_angle_hist'])
    target_length = np.array(priors['global_length_hist'])

    angle_sim = 1.0 - 0.5 * np.sum(np.abs(angle_hist - target_angle))
    length_sim = 1.0 - 0.5 * np.sum(np.abs(length_hist - target_length))

    return angle_sim, length_sim

def pixel_aware_endpoints_from_center_length_angle(centers, lengths, angles, W, H):
    centers = centers.detach()
    lengths = lengths.detach()
    angles  = angles.detach()
    device  = centers.device
    dtype   = centers.dtype

    Wt = torch.as_tensor(W, dtype=dtype, device=device)
    Ht = torch.as_tensor(H, dtype=dtype, device=device)

    vx = torch.cos(angles) / (Wt + 1e-8)
    vy = torch.sin(angles) / (Ht + 1e-8)
    norm = torch.sqrt(vx * vx + vy * vy) + 1e-8
    vx = vx / norm
    vy = vy / norm

    dx = 0.5 * lengths * vx
    dy = 0.5 * lengths * vy

    x1 = torch.clamp(centers[:, 0] - dx, 0, 1) * Wt
    y1 = torch.clamp(centers[:, 1] - dy, 0, 1) * Ht
    x2 = torch.clamp(centers[:, 0] + dx, 0, 1) * Wt
    y2 = torch.clamp(centers[:, 1] + dy, 0, 1) * Ht

    results = []
    for i in range(centers.shape[0]):
        results.append({
            'start': [float(x1[i].detach().cpu().item()), float(y1[i].detach().cpu().item())],
            'end':   [float(x2[i].detach().cpu().item()), float(y2[i].detach().cpu().item())]
        })
    return results

def connectivity_refine(lines, priors, W, H):
    if len(lines) == 0:
        return lines
    ts = float(priors.get('constraints', {}).get('target_spacing', 0.05))
    eps = max(1e-4, 0.25 * ts)
    P = []
    is_dict = []
    for ln in lines:
        if isinstance(ln, dict):
            x1, y1 = ln['start']
            x2, y2 = ln['end']
            is_dict.append(True)
        else:
            x1, y1, x2, y2 = ln
            is_dict.append(False)
        P.append([x1 / W, y1 / H])
        P.append([x2 / W, y2 / H])
    P = np.array(P, dtype=np.float64)
    for _ in range(2):
        D = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=-1)
        np.fill_diagonal(D, 1e9)
        pairs = np.argwhere(D < eps)
        if pairs.size == 0:
            break
        seen = set()
        newP = P.copy()
        for i, j in pairs:
            if i in seen or j in seen:
                continue
            m = 0.5 * (P[i] + P[j])
            newP[i] = m
            newP[j] = m
            seen.add(i)
            seen.add(j)
        P = np.clip(newP, 0.0, 1.0)
    out = []
    for i in range(0, len(P), 2):
        x1 = float(P[i, 0] * W)
        y1 = float(P[i, 1] * H)
        x2 = float(P[i + 1, 0] * W)
        y2 = float(P[i + 1, 1] * H)
        if is_dict[i // 2]:
            out.append({'start': [x1, y1], 'end': [x2, y2]})
        else:
            out.append([x1, y1, x2, y2])
    return out


def two_stage_rerank_score(lines, priors, W, H, stage='A'):
    X, Y = compute_coverage(lines, W, H)
    ang, length = compute_histogram_metrics_console(lines, priors, W, H)
    if len(lines) > 1:
        centers = []
        for line in lines:
            if isinstance(line, dict):
                x1, y1 = line['start']
                x2, y2 = line['end']
            else:
                x1, y1, x2, y2 = line
            cx = ((x1 + x2) * 0.5) / W
            cy = ((y1 + y2) * 0.5) / H
            centers.append([cx, cy])
        centers = torch.tensor(centers, dtype=torch.float32)
        conn = connectivity_from_centers(centers)
    else:
        conn = 0.0
    target_conn = float(priors.get('constraints', {}).get('target_intersection_rate', 0.05))
    conn_match = 1.0 - abs(conn - target_conn)
    if stage == 'A':
        score = 0.60 * ang + 0.20 * length + 0.10 * min(X, Y) + 0.10 * conn_match
    else:
        score = 0.75 * ang + 0.10 * length + 0.05 * min(X, Y) + 0.10 * conn_match
    geoq = compute_geoq(X, Y, ang, length)
    return score, geoq, X, Y, ang, length

@torch.no_grad()
def generate_with_two_stage_rerank(model, priors, device, n_samples=3, N=1200, T=1.05, K=30, nms_dist=0.02, q=0.55, alpha=0.03, angle_weight=0.6, W=None, H=None):
    W = EVAL_W if W is None else W   # ROUTE B (T12/T14)
    H = EVAL_H if H is None else H
    model.eval()
    results = []
    with torch.no_grad():
        for si in range(n_samples):
            candidates_A = []
            for _ in range(N):
                z_spatial = torch.randn(1, 24, device=device) * T
                z_geometry = torch.randn(1, 32, device=device) * T
                centers, lengths, angles, exist_logits = model.decode(z_spatial, z_geometry)
                k_target = sample_k(priors)
                sel_c, sel_l, sel_a = guided_inference_selection(centers, lengths, angles, exist_logits, vm_mu=priors['vm_mu'], k_target=k_target, T=T, q=q, alpha=alpha, K=K, angle_weight=angle_weight, nms_radius=nms_dist)
                lines = pixel_aware_endpoints_from_center_length_angle(sel_c, sel_l, sel_a, W, H)
                lines = connectivity_refine(lines, priors, W, H)
                score_A, geoq, X, Y, ang, length = two_stage_rerank_score(lines, priors, W, H, stage='A')
                candidates_A.append((score_A, geoq, X, Y, ang, length, lines))
            candidates_A.sort(key=lambda t: t[0], reverse=True)
            topK_A = candidates_A[:K]
            candidates_B = []
            for cand in topK_A:
                score_B, geoq, X, Y, ang, length = two_stage_rerank_score(cand[-1], priors, W, H, stage='B')
                candidates_B.append((score_B, geoq, X, Y, ang, length, cand[-1]))
            candidates_B.sort(key=lambda t: t[0], reverse=True)
            best_B = candidates_B[0]
            best_A = topK_A[0]
            if best_B[2] >= 0.95 and best_B[3] >= 0.90:
                chosen = best_B
            else:
                chosen = best_A
            results.append(chosen[-1])
    return results

def evaluate_sample_console(lines, priors, W, H, method, sample_id):
    if len(lines) == 0:
        return {'method': method, 'sample_id': sample_id, 'k': 0, 'X': 0.0, 'Y': 0.0, 'AngleSim': 0.0, 'LenSim': 0.0, 'GeoQ': 0.0, 'Conn': 0.0}
    X, Y = compute_coverage(lines, W, H)
    angle_sim, length_sim = compute_histogram_metrics_console(lines, priors, W, H)
    geoq = compute_geoq(X, Y, angle_sim, length_sim)
    count = len(lines)
    if len(lines) > 1:
        centers = []
        for line in lines:
            if isinstance(line, dict):
                x1, y1 = line['start']
                x2, y2 = line['end']
            else:
                x1, y1, x2, y2 = line
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            centers.append([cx, cy])
        centers = torch.tensor(centers, dtype=torch.float32)
        connectivity = connectivity_from_centers(centers)
    else:
        connectivity = 0.0
    return {'method': method, 'sample_id': sample_id, 'k': count, 'X': X, 'Y': Y, 'AngleSim': angle_sim, 'LenSim': length_sim, 'GeoQ': geoq, 'Conn': connectivity}

def count_intersections_local(lines_norm):
    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    def intersect(a1, a2, b1, b2):
        o1 = orient(a1[0], a1[1], a2[0], a2[1], b1[0], b1[1])
        o2 = orient(a1[0], a1[1], a2[0], a2[1], b2[0], b2[1])
        o3 = orient(b1[0], b1[1], b2[0], b2[1], a1[0], a1[1])
        o4 = orient(b1[0], b1[1], b2[0], b2[1], a2[0], a2[1])
        return (o1 * o2 < 0) and (o3 * o4 < 0)
    n = len(lines_norm)
    cnt = 0
    for i in range(n):
        x1, y1, x2, y2 = lines_norm[i]
        a1 = (x1, y1)
        a2 = (x2, y2)
        for j in range(i + 1, n):
            u1, v1, u2, v2 = lines_norm[j]
            b1 = (u1, v1)
            b2 = (u2, v2)
            if intersect(a1, a2, b1, b2):
                cnt += 1
    return cnt

def compute_data_driven_constraints_local(samples):
    """BUGFIX 2026-09-10 (B1): normalise in the same frame the model predicts in.

    FILE: 01_fixed_branch/evae_routeb_fixed_2026-09-10/training.py
    FUNCTION: compute_data_driven_constraints_local
    BEFORE (frozen branch training.py:628-630):
        W = s['width']
        H = s['height']
        L = [[x1 / W, y1 / H, x2 / W, y2 / H] for x1, y1, x2, y2 in s['lines']]
    BEFORE (frozen branch training.py:648):
        ml = float(np.percentile(np.array(all_lengths, dtype=np.float32), 95))

    The ROUTE B PATCH at training.py:44-45 makes every other evaluation and generation
    constant isotropic, scale = max(W, H) on both axes. This function was never reached by
    that patch, so it returned constants in the anisotropic frame while the model's
    predictions live in the isotropic one. The result was max_length 0.9369 (A) / 0.8593 (B)
    against a longest real trace of 0.4195 / 0.3738, so relu(pred - max_length) was
    identically zero for every epoch of every run, and target_spacing was inflated by 1.76.

    AFTER: the frame follows ROUTEB_ISO exactly as EVAL_W/EVAL_H do, and the length ceiling
    is the longest trace actually present in the fitting windows rather than a percentile
    that would penalise real rock. Both are derived from `samples` at run time; no constant
    is introduced.
    """
    all_lengths = []
    all_nn = []
    all_inter_rates = []
    for s in samples:
        W = float(s['width'])
        H = float(s['height'])
        if ROUTEB_ISO:
            W = H = max(W, H)
        L = [[x1 / W, y1 / H, x2 / W, y2 / H] for x1, y1, x2, y2 in s['lines']]
        if len(L) == 0:
            continue
        ln = [math.hypot(l[2] - l[0], l[3] - l[1]) for l in L]
        all_lengths.extend(ln)
        C = [((l[0] + l[2]) * 0.5, (l[1] + l[3]) * 0.5) for l in L]
        if len(C) > 1:
            for i, c in enumerate(C):
                d = [math.hypot(c[0] - C[j][0], c[1] - C[j][1]) for j in range(len(C)) if j != i]
                if len(d) > 0:
                    all_nn.append(min(d))
        inter = count_intersections_local(L)
        pairs = len(L) * (len(L) - 1) / 2.0
        rate = inter / max(pairs, 1.0)
        all_inter_rates.append(rate)
    if len(all_lengths) == 0:
        ml = 0.35
    else:
        ml = float(np.max(np.array(all_lengths, dtype=np.float32)))
    if len(all_nn) == 0:
        ts = 0.05
    else:
        ts = float(np.percentile(np.array(all_nn, dtype=np.float32), 60))
    if len(all_inter_rates) == 0:
        ir = 0.05
    else:
        ir = float(np.mean(np.array(all_inter_rates, dtype=np.float32)))
    return {"max_length": ml, "target_spacing": ts, "target_intersection_rate": ir}

def save_generation_outputs(category, run_name, samples_lines, priors, g1_cfg, train_cfg, W, H):
    run_dir = os.path.join(BASE_SAVE_DIR, category, run_name)
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    for i, lines in enumerate(samples_lines):
        sample_dir = os.path.join(run_dir, f"sample_{i+1:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        img_name = f"image_N={g1_cfg['N']}_T={g1_cfg['T']}_q={g1_cfg['q']}_K={g1_cfg['K']}_aw={g1_cfg['angle_weight']}_nms={g1_cfg['nms_dist']}.png"
        img_path = os.path.join(sample_dir, img_name)
        save_image(lines, W, H, img_path)
        metrics = evaluate_sample_console(lines, priors, W, H, run_name, i + 1)
        with open(os.path.join(sample_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(sample_dir, "lines.json"), "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)
        rows.append(metrics)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "run_summary.csv"), index=False)
    cfg = {"run_name": run_name, "category": category, "g1_config": g1_cfg, "training_config": train_cfg, "timestamp": datetime.now().isoformat()}
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def train_model_upgraded(dataset, val_dataset, priors, per, device, epochs=100, K_angle=1, run_name='R1'):
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None
    model = DualLatentVAE_Updated(spatial_dim=24, geometry_dim=32, K_angle=K_angle).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 100], gamma=0.5)
    normalizer = EMAConstraintNormalizer(
        ['ang_hist', 'len_hist', 'spacing', 'pairwise', 'kde', 'conn', 'len_mean', 'ang_vm'])
    best_val = float('inf')
    epochs_no_improve = 0
    patience = 20
    train_history = []
    if 'constraints' in priors and isinstance(priors['constraints'], dict):
        max_length_thr = float(priors['constraints'].get('max_length', 0.35))
        target_spacing_thr = float(priors['constraints'].get('target_spacing', 0.05))
        target_intersection_rate = float(priors['constraints'].get('target_intersection_rate', 0.05))
    else:
        max_length_thr = 0.35
        target_spacing_thr = 0.05
        target_intersection_rate = 0.05
    w_len = float(os.environ.get("EVFIX_WLEN", "0.5"))  # EVFIX 2026-09-25, default = original
    w_int = 0.3
    w_sp = 0.2
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'recon': 0, 'exist': 0, 'kl': 0, 'physics': 0, 'length_pen': 0, 'intersect_pen': 0, 'spacing_pen': 0, 'total': 0}
        _bmax = float(os.environ.get("EVFIX_BETA_MAX", "0.1"))  # EVFIX 2026-09-25
        beta = min(_bmax, _bmax * 0.1 * (epoch + 1)) if _bmax != 0.1 else min(0.1, 0.01 * (epoch + 1))
        for batch in train_dataloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            output = model(batch['lines_norm'], batch['valid_mask'], batch['img_size'])
            centers, lengths, angles = to_centers_lengths_angles(batch['lines_norm'])
            angles = angles % math.pi
            if USE_HUNGARIAN:
                recon_loss = hungarian_assignment_loss_updated(output['centers'], output['lengths'], output['angle_params'], centers, lengths, angles, batch['valid_mask'])
            else:
                recon_loss = naive_assignment_loss(output['centers'], output['lengths'], output['angle_params'], centers, lengths, angles, batch['valid_mask'])
            exist_loss = F.binary_cross_entropy_with_logits(output['existence_logits'], batch['valid_mask'])
            kl_spatial = 0.5 * (output['smu'].pow(2) + output['slog'].exp() - output['slog'] - 1).mean()
            kl_geometry = 0.5 * (output['gmu'].pow(2) + output['glog'].exp() - output['glog'] - 1).mean()
            # REVISED: per-branch free-bits floor. torch.clamp_min, NOT builtin max, which would
            # return a Python float when the floor wins and break kl_loss.item() downstream.
            # Below the floor the gradient into that branch is exactly zero; above it the gradient
            # is bitwise identical to the baseline. At LAMBDA_FREE = 0.0 this is an exact no-op,
            # because every element of the KL is non-negative.
            kl_loss = (torch.clamp_min(kl_spatial, LAMBDA_FREE)
                       + torch.clamp_min(kl_geometry, LAMBDA_FREE))
            if USE_MVM_ANGLE:
                angles_hat = torch.atan2((output['angle_params']['alpha'] * torch.sin(output['angle_params']['mu'])).sum(dim=-1), (output['angle_params']['alpha'] * torch.cos(output['angle_params']['mu'])).sum(dim=-1)) % math.pi
            else:
                angles_hat = output['angle_params']['angles'] % math.pi
            recon_lines = centers_lengths_angles_to_endpoints(output['centers'], output['lengths'].view(-1, model.max_lines), angles_hat)
            targets = build_targets_batch(batch['sample_id'].cpu().numpy(), per)
            physics_losses = compute_physics_losses(recon_lines, output['existence_logits'], batch['valid_mask'],
                                                    batch['img_size'], targets, priors)
            L_ang_hist = 1.25 * normalizer.norm('ang_hist', physics_losses[0])
            L_len_hist = normalizer.norm('len_hist', physics_losses[1])
            L_spacing = normalizer.norm('spacing', physics_losses[2])
            L_pairwise = normalizer.norm('pairwise', physics_losses[3])
            L_kde = normalizer.norm('kde', physics_losses[4])
            L_conn = normalizer.norm('conn', physics_losses[5])
            L_len_mean = normalizer.norm('len_mean', physics_losses[6])
            L_ang_vm = 1.25 * normalizer.norm('ang_vm', physics_losses[7])
            physics_loss = L_ang_hist + L_len_hist + L_spacing + L_pairwise + L_kde + L_conn + L_len_mean + L_ang_vm

            length_penalty = length_constraint_loss_data(output['lengths'].view(-1, model.max_lines), max_length_thr)
            intersect_penalty = intersection_penalty_loss_data(output['centers'], output['lengths'].view(-1, model.max_lines), angles_hat, output['existence_logits'], target_intersection_rate)
            spacing_penalty = local_spacing_loss_data(output['centers'], output['existence_logits'], target_spacing_thr)
            loss = recon_loss + exist_loss + beta * kl_loss + 0.005 * physics_loss + w_len * length_penalty + w_int * intersect_penalty + w_sp * spacing_penalty
            _wq = float(os.environ.get("EVFIX_WQ", "0"))  # EVFIX 2026-09-25: sorted log-length W1, 0 = off
            if _wq > 0:
                _pl = output['lengths'].view(-1, model.max_lines)
                _ql = 0.0
                for _b in range(_pl.shape[0]):
                    _vm = batch['valid_mask'][_b] > 0.5
                    _n = int(_vm.sum())
                    if _n < 2:
                        continue
                    _idx = torch.topk(output['existence_logits'][_b], _n).indices
                    _p = torch.sort(torch.log(_pl[_b, _idx].clamp_min(1e-4)))[0]
                    _t = torch.sort(torch.log(lengths[_b][_vm].clamp_min(1e-4)))[0]
                    _ql = _ql + (_p - _t).abs().mean()
                loss = loss + _wq * _ql / _pl.shape[0]
            _wg = float(os.environ.get("EVGEO_W", "0"))  # FAMILY FIX A: 0 = off (identical to the release)
            if _wg > 0 and USE_MVM_ANGLE:
                loss = loss + _wg * generated_direction_w1(output['angle_params'], output['existence_logits'], angles, batch['valid_mask'])
            _wb = float(os.environ.get("EVBG_W", "0"))  # SETS + BACKGROUND guidance: 0 = off
            if _wb > 0 and USE_MVM_ANGLE:
                loss = loss + _wb * generated_class_share_l1(output['angle_params'], output['existence_logits'], angles, batch['valid_mask'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['exist'] += exist_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['physics'] += physics_loss.item()
            epoch_losses['length_pen'] += length_penalty.item()
            epoch_losses['intersect_pen'] += intersect_penalty.item()
            epoch_losses['spacing_pen'] += spacing_penalty.item()
            epoch_losses['total'] += loss.item()
        for k in epoch_losses:
            epoch_losses[k] /= len(train_dataloader)
        scheduler.step()
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(device)
                    output = model(batch['lines_norm'], batch['valid_mask'], batch['img_size'])
                    centers, lengths, angles = to_centers_lengths_angles(batch['lines_norm'])
                    angles = angles % math.pi
                    if USE_HUNGARIAN:
                        recon_loss = hungarian_assignment_loss_updated(output['centers'], output['lengths'], output['angle_params'], centers, lengths, angles, batch['valid_mask'])
                    else:
                        recon_loss = naive_assignment_loss(output['centers'], output['lengths'], output['angle_params'], centers, lengths, angles, batch['valid_mask'])
                    exist_loss = F.binary_cross_entropy_with_logits(output['existence_logits'], batch['valid_mask'])
                    kl_loss = 0.5 * (output['smu'].pow(2) + output['slog'].exp() - output['slog'] - 1).mean()
                    kl_loss += 0.5 * (output['gmu'].pow(2) + output['glog'].exp() - output['glog'] - 1).mean()
                    val_loss += (recon_loss + exist_loss + beta * kl_loss).item()
            val_loss /= len(val_dataloader)
        else:
            val_loss = epoch_losses['total']
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_losses['total'],
            'val_loss': val_loss,
            'recon': epoch_losses['recon'],
            'exist': epoch_losses['exist'],
            'kl': epoch_losses['kl'],
            'physics': epoch_losses['physics'],
            'length_pen': epoch_losses['length_pen'],
            'intersect_pen': epoch_losses['intersect_pen'],
            'spacing_pen': epoch_losses['spacing_pen'],
            'beta': beta,
            'lr': optimizer.param_groups[0]['lr']
        })
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(best_state, f"checkpoints/evai_best_{run_name}.pt")
            print(f"[CHECKPOINT] val_loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] {patience} epochs no improvement at epoch {epoch + 1}")
                break
    model.load_state_dict(torch.load(f"checkpoints/evai_best_{run_name}.pt"))
    model = model.to(device)
    pd.DataFrame(train_history).to_csv(f"results/train_history_{run_name}.csv", index=False)
    return model, train_history

def finetune_angle_heads(model, dataset, priors, per, device, epochs=5):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.angle_logits_head.parameters():
        param.requires_grad = True
    for param in model.angle_mu_head.parameters():
        param.requires_grad = True
    for param in model.angle_kappa_head.parameters():
        param.requires_grad = True
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    normalizer = EMAConstraintNormalizer(['ang_hist', 'len_hist', 'spacing', 'kde', 'conn', 'len_mean', 'ang_vm'])
    print(f"--- Fine-tuning angle heads only for {epochs} epochs ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            output = model(batch['lines_norm'], batch['valid_mask'], batch['img_size'])
            centers, lengths, angles = to_centers_lengths_angles(batch['lines_norm'])
            angles = angles % math.pi
            recon_loss = hungarian_assignment_loss_updated(output['centers'], output['lengths'], output['angle_params'], centers, lengths, angles, batch['valid_mask'])
            loss = 2.0 * recon_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}: angle_loss={epoch_loss:.4f}")
    for param in model.parameters():
        param.requires_grad = True
    return model

def generate_stochastic_recheck(priors, n_samples=3):
    W, H = EVAL_W, EVAL_H   # ROUTE B (T16)
    results = stochastic_baseline_sampler(priors, W, H, n_samples)
    return results

def save_tables_and_figures(all_results, train_histories):
    W, H = EVAL_W, EVAL_H   # ROUTE B (T16)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results/all_evaluation_results.csv", index=False)
    method_summary = df_results.groupby('method').agg({'X': 'mean', 'Y': 'mean', 'AngleSim': 'mean', 'LenSim': 'mean', 'GeoQ': 'mean', 'k': 'mean'}).reset_index()
    method_summary.to_csv("results/T4_method_comparison.csv", index=False)
    with open("results/T4_method_comparison.tex", "w") as f:
        f.write(method_summary.to_latex(index=False, float_format="%.3f"))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(method_summary))
    width = 0.15
    metrics = ['X', 'Y', 'AngleSim', 'LenSim', 'GeoQ']
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, method_summary[metric], width, label=metric)
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(method_summary['method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/F3_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    all_train_history = []
    for run_name, history in train_histories.items():
        for row in history:
            row['run'] = run_name
            all_train_history.append(row)
    df_train = pd.DataFrame(all_train_history)
    df_train.to_csv("results/T1_training_sensitivity.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for run_name in train_histories.keys():
        run_data = df_train[df_train['run'] == run_name]
        axes[0, 0].plot(run_data['epoch'], run_data['train_loss'], label=f'{run_name} train', marker='o')
        axes[0, 0].plot(run_data['epoch'], run_data['val_loss'], label=f'{run_name} val', marker='s', linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    for run_name in train_histories.keys():
        run_data = df_train[df_train['run'] == run_name]
        axes[0, 1].plot(run_data['epoch'], run_data['recon'], label=f'{run_name}', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    for run_name in train_histories.keys():
        run_data = df_train[df_train['run'] == run_name]
        axes[1, 0].plot(run_data['epoch'], run_data['kl'], label=f'{run_name}', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].set_title('KL Divergence Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    for run_name in train_histories.keys():
        run_data = df_train[df_train['run'] == run_name]
        axes[1, 1].plot(run_data['epoch'], run_data['physics'], label=f'{run_name}', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Physics Loss')
    axes[1, 1].set_title('Physics Constraints Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/F1_train_curves.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nTables and figures saved:")
    print("  - results/T4_method_comparison.csv/.tex")
    print("  - results/T1_training_sensitivity.csv")
    print("  - figures/F3_comparison.png")
    print("  - figures/F1_train_curves.png")

def main():
    print("=== EVAE EXTENDED TRAINING — 100-200 EPOCHS WITH FULL EVALUATION ===\n")
    device = device_select()
    print(f"Device: {device}\n")
    print("--- A. DATA LOADING & BIC K-SELECTION ---")
    samples = load_labelme(DATA_PATH)
    print(f"JSONs found: {len(samples)}, total_lines: {sum(len(s['lines']) for s in samples)}")
    print("--- B. SELECT K VIA BIC ---")
    all_train_angles = []
    for sample in samples:
        for ln in sample['lines']:
            if isinstance(ln, dict) and 'start' in ln and 'end' in ln:
                (x1, y1), (x2, y2) = ln['start'], ln['end']
            else:
                x1, y1, x2, y2 = ln
            angle = np.arctan2(y2 - y1, x2 - x1)
            angle = angle % np.pi
            all_train_angles.append(angle)
    if len(all_train_angles) == 0:
        K_selected = 1
        vm_params = {'K': 1, 'alpha': np.array([1.0]), 'mu': np.array([0.0]), 'kappa': np.array([1.0])}
        print("No training angles found, using K=1 default")
    else:
        K_selected, vm_params = select_K_via_bic(all_train_angles, K_range=[1, 2, 3])
        if K_selected > 1:
            mu_deg = np.degrees(vm_params['mu'])
            mu_diff = np.abs(mu_deg[0] - mu_deg[1]) if K_selected >= 2 else 180
            if mu_diff < 20:
                print(f"K={K_selected} selected but modes too close ({mu_diff:.1f}°), forcing K=1")
                K_selected = 1
                vm_params = fit_vm_mixture_em(all_train_angles, K=1)
        print(f"Selected K={K_selected} via BIC")
        print(f"VM Mixture params: alpha={vm_params['alpha']}, mu(deg)={np.degrees(vm_params['mu'])}, kappa={vm_params['kappa']}\n")
    assert K_selected in [1, 2, 3], f"K_selected must be 1, 2, or 3, got {K_selected}"
    K_selected = 3
    print(f"Forcing K={K_selected} as requested\n")
    print("--- C. TRAIN/VAL SPLIT ---")
    num_val = 1
    train_samples = samples[:-num_val]
    val_samples = samples[-num_val:]
    global per
    per, priors = extract_targets_and_priors(train_samples)
    constraints_local = compute_data_driven_constraints_local(train_samples)
    priors['constraints'] = constraints_local
    for i, s in enumerate(train_samples):
        s['sample_id'] = i
    for i, s in enumerate(val_samples):
        s['sample_id'] = i
    vm_mu_deg = [np.degrees(mu) for mu in priors['vm_mu']]
    print(f"VM modes (deg): {[f'{m:.1f}' for m in vm_mu_deg]}")
    print(f"Count stats: min={priors['count_stats']['min']}, max={priors['count_stats']['max']}, mean={priors['count_stats']['mean']:.1f}")
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    print(f"Per dictionary has {len(per)} entries with keys: {list(per.keys())}\n")
    train_dataset = EnhancedDataset(train_samples, augment=False)
    val_dataset = EnhancedDataset(val_samples, augment=False)
    W, H = EVAL_W, EVAL_H   # ROUTE B (T16)
    all_results = []
    train_histories = {}
    TRAINING_CFG = {
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'milestones': [20, 60, 100],
        'gamma': 0.5,
        'beta_warmup': '0->0.1 over 10 epochs',
        'lambda_physics': 0.005,
        'patience': 20,
        'grad_clip': 5.0
    }
    print("--- D. BASELINE 20-EPOCH TRAINING ---")
    baseline_dataset = EnhancedDataset(train_samples, augment=False)
    model_baseline, history_baseline = train_model_upgraded(baseline_dataset, val_dataset, priors, per, device, epochs=20, K_angle=K_selected, run_name='baseline_20e')
    train_histories['baseline_20e'] = history_baseline
    print("\n--- Evaluating Baseline 20e with G1 ---")
    baseline_samples = generate_with_two_stage_rerank(model_baseline, priors, device, n_samples=3, N=G1_CONFIG['N'], T=G1_CONFIG['T'], K=G1_CONFIG['K'], nms_dist=G1_CONFIG['nms_dist'], q=G1_CONFIG['q'], alpha=G1_CONFIG['alpha'], angle_weight=G1_CONFIG['angle_weight'], W=W, H=H)
    for i, lines in enumerate(baseline_samples):
        result = evaluate_sample_console(lines, priors, W, H, "Baseline_20e", i + 1)
        all_results.append(result)
    baseline_metrics = {
        'X': np.mean([r['X'] for r in all_results if r['method'] == 'Baseline_20e']),
        'Y': np.mean([r['Y'] for r in all_results if r['method'] == 'Baseline_20e']),
        'AngleSim': np.mean([r['AngleSim'] for r in all_results if r['method'] == 'Baseline_20e']),
        'LenSim': np.mean([r['LenSim'] for r in all_results if r['method'] == 'Baseline_20e']),
        'GeoQ': np.mean([r['GeoQ'] for r in all_results if r['method'] == 'Baseline_20e'])
    }
    save_generation_outputs("sensitivity", "baseline_20e", baseline_samples, priors, G1_CONFIG, TRAINING_CFG, W, H)
    print(f"Baseline 20e: GeoQ={baseline_metrics['GeoQ']:.3f}, AngleSim={baseline_metrics['AngleSim']:.3f}, X={baseline_metrics['X']:.3f}, Y={baseline_metrics['Y']:.3f}\n")
    print("--- E. R1: 100-EPOCH TRAINING ---")
    model_r1, history_r1 = train_model_upgraded(train_dataset, val_dataset, priors, per, device, epochs=100, K_angle=K_selected, run_name='R1')
    train_histories['R1'] = history_r1
    print("\n--- Fine-tuning R1 angle heads ---")
    model_r1 = finetune_angle_heads(model_r1, train_dataset, priors, per, device, epochs=5)
    torch.save(model_r1.state_dict(), "checkpoints/evai_best_R1_finetuned.pt")
    print("\n--- Evaluating R1 with G1 ---")
    r1_samples = generate_with_two_stage_rerank(model_r1, priors, device, n_samples=3, N=G1_CONFIG['N'], T=G1_CONFIG['T'], K=G1_CONFIG['K'], nms_dist=G1_CONFIG['nms_dist'], q=G1_CONFIG['q'], alpha=G1_CONFIG['alpha'], angle_weight=G1_CONFIG['angle_weight'], W=W, H=H)
    for i, lines in enumerate(r1_samples):
        result = evaluate_sample_console(lines, priors, W, H, "R1", i + 1)
        all_results.append(result)
    r1_metrics = {
        'X': np.mean([r['X'] for r in all_results if r['method'] == 'R1']),
        'Y': np.mean([r['Y'] for r in all_results if r['method'] == 'R1']),
        'AngleSim': np.mean([r['AngleSim'] for r in all_results if r['method'] == 'R1']),
        'LenSim': np.mean([r['LenSim'] for r in all_results if r['method'] == 'R1']),
        'GeoQ': np.mean([r['GeoQ'] for r in all_results if r['method'] == 'R1'])
    }
    save_generation_outputs("sensitivity", "R1", r1_samples, priors, G1_CONFIG, TRAINING_CFG, W, H)
    print(f"R1: GeoQ={r1_metrics['GeoQ']:.3f}, AngleSim={r1_metrics['AngleSim']:.3f}, X={r1_metrics['X']:.3f}, Y={r1_metrics['Y']:.3f}\n")
    print("--- F. R2: 200-EPOCH TRAINING ---")
    model_r2, history_r2 = train_model_upgraded(train_dataset, val_dataset, priors, per, device, epochs=200, K_angle=K_selected, run_name='R2')
    train_histories['R2'] = history_r2
    print("\n--- Fine-tuning R2 angle heads ---")
    model_r2 = finetune_angle_heads(model_r2, train_dataset, priors, per, device, epochs=5)
    torch.save(model_r2.state_dict(), "checkpoints/evai_best_R2_finetuned.pt")
    print("\n--- Evaluating R2 with G1 ---")
    r2_samples = generate_with_two_stage_rerank(model_r2, priors, device, n_samples=3, N=G1_CONFIG['N'], T=G1_CONFIG['T'], K=G1_CONFIG['K'], nms_dist=G1_CONFIG['nms_dist'], q=G1_CONFIG['q'], alpha=G1_CONFIG['alpha'], angle_weight=G1_CONFIG['angle_weight'], W=W, H=H)
    for i, lines in enumerate(r2_samples):
        result = evaluate_sample_console(lines, priors, W, H, "R2", i + 1)
        all_results.append(result)
    r2_metrics = {
        'X': np.mean([r['X'] for r in all_results if r['method'] == 'R2']),
        'Y': np.mean([r['Y'] for r in all_results if r['method'] == 'R2']),
        'AngleSim': np.mean([r['AngleSim'] for r in all_results if r['method'] == 'R2']),
        'LenSim': np.mean([r['LenSim'] for r in all_results if r['method'] == 'R2']),
        'GeoQ': np.mean([r['GeoQ'] for r in all_results if r['method'] == 'R2'])
    }
    save_generation_outputs("sensitivity", "R2", r2_samples, priors, G1_CONFIG, TRAINING_CFG, W, H)
    print(f"R2: GeoQ={r2_metrics['GeoQ']:.3f}, AngleSim={r2_metrics['AngleSim']:.3f}, X={r2_metrics['X']:.3f}, Y={r2_metrics['Y']:.3f}\n")
    print("--- G. STOCHASTIC BASELINE ---")
    stoch_samples = generate_stochastic_recheck(priors, n_samples=3)
    for i, lines in enumerate(stoch_samples):
        result = evaluate_sample_console(lines, priors, W, H, "Stochastic", i + 1)
        all_results.append(result)
    stoch_metrics = {
        'X': np.mean([r['X'] for r in all_results if r['method'] == 'Stochastic']),
        'Y': np.mean([r['Y'] for r in all_results if r['method'] == 'Stochastic']),
        'AngleSim': np.mean([r['AngleSim'] for r in all_results if r['method'] == 'Stochastic']),
        'LenSim': np.mean([r['LenSim'] for r in all_results if r['method'] == 'Stochastic']),
        'GeoQ': np.mean([r['GeoQ'] for r in all_results if r['method'] == 'Stochastic'])
    }
    save_generation_outputs("sensitivity", "Stochastic", stoch_samples, priors, G1_CONFIG, TRAINING_CFG, W, H)
    print(f"Stochastic: GeoQ={stoch_metrics['GeoQ']:.3f}, AngleSim={stoch_metrics['AngleSim']:.3f}, X={stoch_metrics['X']:.3f}, Y={stoch_metrics['Y']:.3f}\n")
    print("--- H. ABLATION STUDIES ---")
    print("\n[Ablation 1: No Hungarian Matching]")
    global USE_HUNGARIAN
    USE_HUNGARIAN = False
    ablation_dataset_1 = EnhancedDataset(train_samples, augment=False)
    model_ablation_1, history_ablation_1 = train_model_upgraded(ablation_dataset_1, val_dataset, priors, per, device, epochs=15, K_angle=K_selected, run_name='ablation_no_hungarian')
    train_histories['ablation_no_hungarian'] = history_ablation_1
    ablation_1_samples = generate_with_two_stage_rerank(model_ablation_1, priors, device, n_samples=3, N=G1_CONFIG['N'], T=G1_CONFIG['T'], K=G1_CONFIG['K'], nms_dist=G1_CONFIG['nms_dist'], q=G1_CONFIG['q'], alpha=G1_CONFIG['alpha'], angle_weight=G1_CONFIG['angle_weight'], W=W, H=H)
    for i, lines in enumerate(ablation_1_samples):
        result = evaluate_sample_console(lines, priors, W, H, "NoHungarian", i + 1)
        all_results.append(result)
    save_generation_outputs("ablation", "NoHungarian", ablation_1_samples, priors, G1_CONFIG, TRAINING_CFG, W, H)
    USE_HUNGARIAN = True
    print("\n--- I. SAVING RESULTS ---")
    save_tables_and_figures(all_results, train_histories)
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'K_selected': int(K_selected),
        'seeds': 1337,
        'baseline_20e': baseline_metrics,
        'R1_100e': r1_metrics,
        'R2_200e': r2_metrics,
        'stochastic': stoch_metrics,
        'G1_config': G1_CONFIG,
        'training_config': TRAINING_CFG,
        'constraints': priors.get('constraints', {})
    }
    with open("results/summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    with open("results/runlog.txt", "w", encoding="utf-8") as f:
        f.write("=== EVAE EXTENDED TRAINING RUN LOG ===\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Device: {summary['device']}\n")
        f.write(f"Seeds: {summary['seeds']}\n")
        f.write(f"K (VM components): {summary['K_selected']}\n")
        f.write(f"Training samples: {len(train_samples)}\n")
        f.write(f"Validation samples: {len(val_samples)}\n\n")
        f.write("--- TRAINING CONFIG ---\n")
        for key, val in summary['training_config'].items():
            f.write(f"{key}: {val}\n")
        f.write("\n--- G1 CONFIG ---\n")
        for key, val in summary['G1_config'].items():
            f.write(f"{key}: {val}\n")
        f.write("\n--- CONSTRAINTS (DATA-DRIVEN) ---\n")
        for key, val in summary['constraints'].items():
            f.write(f"{key}: {val}\n")
        f.write("\n--- RESULTS SUMMARY ---\n")
        f.write(f"Baseline 20e: GeoQ={baseline_metrics['GeoQ']:.4f}, AngleSim={baseline_metrics['AngleSim']:.4f}\n")
        f.write(f"R1 100e:      GeoQ={r1_metrics['GeoQ']:.4f}, AngleSim={r1_metrics['AngleSim']:.4f}\n")
        f.write(f"R2 200e:      GeoQ={r2_metrics['GeoQ']:.4f}, AngleSim={r2_metrics['AngleSim']:.4f}\n")
        f.write(f"Stochastic:   GeoQ={stoch_metrics['GeoQ']:.4f}, AngleSim={stoch_metrics['AngleSim']:.4f}\n")
        f.write("\n--- IMPROVEMENTS vs BASELINE ---\n")
        f.write(f"R1 vs Baseline: Delta_GeoQ={r1_metrics['GeoQ'] - baseline_metrics['GeoQ']:+.4f}, Delta_AngleSim={r1_metrics['AngleSim'] - baseline_metrics['AngleSim']:+.4f}\n")
        f.write(f"R2 vs Baseline: Delta_GeoQ={r2_metrics['GeoQ'] - baseline_metrics['GeoQ']:+.4f}, Delta_AngleSim={r2_metrics['AngleSim'] - baseline_metrics['AngleSim']:+.4f}\n")
        f.write("\n--- IMPROVEMENTS vs STOCHASTIC ---\n")
        f.write(f"R1 vs Stoch: Delta_GeoQ={r1_metrics['GeoQ'] - stoch_metrics['GeoQ']:+.4f}, Delta_AngleSim={r1_metrics['AngleSim'] - stoch_metrics['AngleSim']:+.4f}\n")
        f.write(f"R2 vs Stoch: Delta_GeoQ={r2_metrics['GeoQ'] - stoch_metrics['GeoQ']:+.4f}, Delta_AngleSim={r2_metrics['AngleSim'] - stoch_metrics['AngleSim']:+.4f}\n")
    print("\n=== J. FINAL REPORT ===")
    print("\n--- HEADLINE NUMBERS (for paper) ---")
    print(f"R1 (100 epochs): X={r1_metrics['X']:.3f}, Y={r1_metrics['Y']:.3f}, AngleSim={r1_metrics['AngleSim']:.3f}, LenSim={r1_metrics['LenSim']:.3f}, GeoQ={r1_metrics['GeoQ']:.3f}")
    print(f"R2 (200 epochs): X={r2_metrics['X']:.3f}, Y={r2_metrics['Y']:.3f}, AngleSim={r2_metrics['AngleSim']:.3f}, LenSim={r2_metrics['LenSim']:.3f}, GeoQ={r2_metrics['GeoQ']:.3f}")
    print("\n--- METHODS TEXT (for paper §2-3) ---")
    print("Training: Physics losses use Sinkhorn-regularized EMD (circular for angles; linear for lengths). Evaluation uses L1 histogram similarity and span-based X/Y coverage.")
    print(f"Optimizer: Adam (lr=1e-3, weight_decay=1e-5) with step LR decay at epochs [20, 60, 100] (γ=0.5); β annealed 0→0.1 over 10 epochs; λ_physics=0.005; batch=4; patience=20; gradient clipping max_norm=5.0.")
    print(f"Inference (G1): N={G1_CONFIG['N']}, T={G1_CONFIG['T']}, q={G1_CONFIG['q']}, α={G1_CONFIG['alpha']}, K={G1_CONFIG['K']}, angle_weight={G1_CONFIG['angle_weight']}, nms_dist={G1_CONFIG['nms_dist']}; three realizations per map, metrics averaged.")
    print("\n--- DELIVERABLES ---")
    print("Checkpoints:")
    print("  - checkpoints/evai_best_baseline_20e.pt")
    print("  - checkpoints/evai_best_R1.pt")
    print("  - checkpoints/evai_best_R1_finetuned.pt")
    print("  - checkpoints/evai_best_R2.pt")
    print("  - checkpoints/evai_best_R2_finetuned.pt")
    print("\nTables & Figures:")
    print("  - results/T1_training_sensitivity.csv")
    print("  - results/T4_method_comparison.csv/.tex")
    print("  - results/all_evaluation_results.csv")
    print("  - figures/F1_train_curves.png")
    print("  - figures/F3_comparison.png")
    print("\nLogs:")
    print("  - results/runlog.txt")
    print("  - results/summary.yaml")
    print("\n=== EXECUTION COMPLETE ===")

if __name__ == "__main__":
    main()
