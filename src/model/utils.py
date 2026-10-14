import os
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans

# ROUTE B PATCH 2026-09-07 (Gate 1 Section 8 item 2; author ROUTE_DECISION.md 2026-09-07).
# One isotropic scale s = max(W, H) = 5280 replaces the anisotropic per-window (x/W, y/H)
# normalisation, so x is in [0, 1] and y is in [0, H/s] and the image-plane direction is
# preserved exactly (Gate 1 T18: max |theta_raw - theta_iso| = 2.8e-14 deg).
# ROUTEB_ISO=0 restores the source branch behaviour bitwise for the Section 8 item 6 no-op check.
ROUTEB_ISO = os.environ.get("ROUTEB_ISO", "1") == "1"

CONFIG = {
    "data": {
        "IMG_WIDTH": 5280,
        "IMG_HEIGHT": 1600,
        "NMAX": 120,
        "ANGLE_BINS": 18,
        "LEN_BINS": 18,
        "SPACING_BINS": 20,
        "GRID_W": 12,
        "GRID_H": 8,
        "PMIX": 0.3
    },
    "model": {
        "spatial_dim": 24,
        "geometry_dim": 32,
        "max_lines": 120
    },
    "inference": {
        "temperature": 1.05,
        "N_samples": 800,
        "q_threshold": 0.55,
        "alpha_nudge": 0.03,
        "nms_radius": 0.02,
        "shortlist_K": 20,
        "angle_weight": 0.6
    },
    "physics": {
        "SIGMA_ANGLE_DEG": 9.0,
        "SIGMA_LEN": 0.02,
        "SIGMA_SPACING": 0.05,
        "SIGMA_KDE": 0.1,
        "SIGMA_CONN": 0.04
    },
    "reranking": {
        "stageA_weights": [0.70, 0.20, 0.10],
        "stageB_weights": [0.85, 0.10, 0.05],
        "stageB_trigger": {"x_cov_min": 0.95, "y_cov_min": 0.90}
    },
    "training": {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "milestones": [20, 60, 100],
        "gamma": 0.5,
        "lambda_physics": 0.005,
        "patience": 20,
        "grad_clip": 5.0,
        "batch_size": 4
    }
}

NMAX = CONFIG["data"]["NMAX"]
ANGLE_BINS = CONFIG["data"]["ANGLE_BINS"]
LEN_BINS = CONFIG["data"]["LEN_BINS"]
SPACING_BINS = CONFIG["data"]["SPACING_BINS"]
GRID_W = CONFIG["data"]["GRID_W"]
GRID_H = CONFIG["data"]["GRID_H"]
SIGMA_ANGLE_DEG = CONFIG["physics"]["SIGMA_ANGLE_DEG"]
SIGMA_LEN = CONFIG["physics"]["SIGMA_LEN"]
SIGMA_SPACING = CONFIG["physics"]["SIGMA_SPACING"]
SIGMA_KDE = CONFIG["physics"]["SIGMA_KDE"]
SIGMA_CONN = CONFIG["physics"]["SIGMA_CONN"]
PMIX = CONFIG["data"]["PMIX"]

AUG_TRAIN = True

def torch_i0(x):
    return torch.special.i0(x) if hasattr(torch.special, "i0") else torch.i0(x)

def device_select():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_labelme(path):
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    data = []
    sid = 0
    for f in files:
        with open(os.path.join(path, f), 'r') as fh:
            d = json.load(fh)
        lines = []
        for s in d.get('shapes', []):
            if s.get('shape_type') == 'line' and len(s.get('points', [])) == 2:
                p0 = s['points'][0]
                p1 = s['points'][1]
                lines.append([p0[0], p0[1], p1[0], p1[1]])
        if len(lines) > 0:
            data.append({
                'sample_id': sid,
                'file': f,
                'lines': lines,
                'width': d['imageWidth'],
                'height': d['imageHeight'],
                'shapes': d.get('shapes', [])
            })
            sid += 1
    return data

def _a1_inverse(R):
    if R < 0.53:
        return 2 * R + R ** 3 + (5 * R ** 5) / 6
    elif R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        return 1 / (R ** 3 - 4 * R ** 2 + 3 * R)

def _fit_vm_mixture_em_axial(angles_rad, K, max_iter=20):
    from scipy.special import i0e
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    if angles_rad.size == 0:
        return None
    ang2 = 2.0 * angles_rad
    cos_a = np.cos(ang2)
    sin_a = np.sin(ang2)
    X = np.stack([cos_a, sin_a], axis=1)
    labels = KMeans(n_clusters=K, n_init=10, random_state=1337).fit_predict(X)
    alpha = np.zeros(K, dtype=np.float64)
    mu2 = np.zeros(K, dtype=np.float64)
    kappa = np.zeros(K, dtype=np.float64)
    for k in range(K):
        m = labels == k
        if not np.any(m):
            alpha[k] = 1.0 / K
            mu2[k] = 0.0
            kappa[k] = 1.0
            continue
        alpha[k] = float(m.mean())
        c = float(cos_a[m].mean())
        s = float(sin_a[m].mean())
        mu2[k] = math.atan2(s, c)
        Rk = math.hypot(c, s)
        Rk = float(np.clip(Rk, 1e-4, 0.9999))
        kappa[k] = float(_a1_inverse(Rk))
    N = ang2.shape[0]
    for _ in range(max_iter):
        log_probs = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            log_vm = kappa[k] * np.cos(ang2 - mu2[k]) - math.log(2 * math.pi) - (np.log(i0e(kappa[k]) + 1e-12) + kappa[k])
            log_probs[:, k] = np.log(alpha[k] + 1e-12) + log_vm
        mlogsum = log_probs.max(axis=1, keepdims=True)
        resp = np.exp(log_probs - mlogsum)
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-12)
        rk = resp.sum(axis=0) + 1e-12
        alpha = rk / N
        c = (resp * cos_a[:, None]).sum(axis=0) / rk
        s = (resp * sin_a[:, None]).sum(axis=0) / rk
        mu2 = np.arctan2(s, c)
        Rk = np.hypot(c, s)
        Rk = np.clip(Rk, 1e-4, 0.9999)
        kappa = np.array([_a1_inverse(r) for r in Rk], dtype=np.float64)
    mu = (np.mod(mu2, 2 * math.pi)) / 2.0
    return {"K": int(K), "alpha": alpha.astype(np.float32), "mu": mu.astype(np.float32), "kappa": kappa.astype(np.float32)}

def _bic_vm_mixture_axial(angles_rad, params):
    from scipy.special import i0e
    ang2 = 2.0 * np.asarray(angles_rad, dtype=np.float64)
    K = int(params["K"])
    alpha = params["alpha"].astype(np.float64)
    mu2 = params["mu"].astype(np.float64) * 2.0
    kappa = params["kappa"].astype(np.float64)
    N = ang2.shape[0]
    mix = np.zeros(N, dtype=np.float64)
    for k in range(K):
        log_vm = kappa[k] * np.cos(ang2 - mu2[k]) - math.log(2 * math.pi) - (np.log(i0e(kappa[k]) + 1e-12) + kappa[k])
        mix += np.exp(np.log(alpha[k] + 1e-12) + log_vm)
    log_lik = np.log(mix + 1e-12).sum()
    num_params = 3 * K - 1
    return -2.0 * log_lik + num_params * math.log(max(N, 1))

def select_angle_families_axial(angles_rad, K_candidates=(1, 2, 3, 4)):
    best = None
    best_bic = float("inf")
    for K in K_candidates:
        p = _fit_vm_mixture_em_axial(angles_rad, K)
        if p is None:
            continue
        bic = _bic_vm_mixture_axial(angles_rad, p)
        if bic < best_bic:
            best_bic = bic
            best = p
    if best is None:
        best = {"K": 1, "alpha": np.array([1.0], dtype=np.float32), "mu": np.array([0.0], dtype=np.float32), "kappa": np.array([1.0], dtype=np.float32)}
    return best

class EnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, samples, Nmax=None, augment=False):
        self.S = samples
        self.Nmax = Nmax if Nmax is not None else CONFIG["data"]["NMAX"]
        self.augment = augment

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        s = self.S[idx]
        if ROUTEB_ISO:
            # ROUTE B (T04): one isotropic scale per window, s = max(W, H)
            W = float(max(s['width'], s['height']))
            H = W
        else:
            W = s['width']
            H = s['height']
        L = s['lines']
        n = min(len(L), self.Nmax)
        P = np.zeros((self.Nmax, 4), dtype=np.float32)
        M = np.zeros((self.Nmax,), dtype=np.float32)
        for i in range(n):
            x1, y1, x2, y2 = L[i]
            P[i] = [x1 / W, y1 / H, x2 / W, y2 / H]
            M[i] = 1.0
        if self.augment:
            drop_prob = 0.08
            drop_mask = (np.random.rand(self.Nmax) > drop_prob).astype(np.float32)
            M = M * drop_mask
            jitter = np.random.uniform(-0.01, 0.01, size=(self.Nmax, 2)).astype(np.float32)
            P[:, 0:2] += jitter
            P[:, 2:4] += jitter
            P[:, 0:4] = np.clip(P[:, 0:4], 0.0, 1.0)
        return {
            'lines_norm': torch.from_numpy(P),
            'valid_mask': torch.from_numpy(M),
            'img_size': torch.tensor([W, H], dtype=torch.float32),
            'sample_id': int(s['sample_id'])
        }

def to_centers_lengths_angles(L):
    c = torch.stack([(L[..., 0] + L[..., 2]) * 0.5, (L[..., 1] + L[..., 3]) * 0.5], -1)
    dl = L[..., 2] - L[..., 0]
    dk = L[..., 3] - L[..., 1]
    lengths = torch.sqrt(dl * dl + dk * dk)
    ang = torch.atan2(dk, dl)
    return c, lengths, ang

def centers_lengths_angles_to_endpoints(centers, lengths_norm, angles):
    dx = 0.5 * lengths_norm * torch.cos(angles)
    dy = 0.5 * lengths_norm * torch.sin(angles)
    x1 = torch.clamp(centers[..., 0] - dx, 0, 1)
    y1 = torch.clamp(centers[..., 1] - dy, 0, 1)
    x2 = torch.clamp(centers[..., 0] + dx, 0, 1)
    y2 = torch.clamp(centers[..., 1] + dy, 0, 1)
    return torch.stack([x1, y1, x2, y2], -1)

def wrapped_axial_delta_deg(a_deg, b_deg):
    d = torch.abs(a_deg - b_deg)
    return torch.minimum(d, 180.0 - d)

def soft_histogram1d_angles(angles_deg, weights, bin_centers_deg, sigma_degrees):
    angles_expanded = angles_deg.unsqueeze(-1)
    bin_centers_expanded = bin_centers_deg.unsqueeze(0)
    weights_expanded = weights.unsqueeze(-1)
    dist = wrapped_axial_delta_deg(angles_expanded, bin_centers_expanded)
    gaussian_weights = torch.exp(-0.5 * (dist / sigma_degrees) ** 2)
    weighted_contributions = gaussian_weights * weights_expanded
    histogram = torch.sum(weighted_contributions, dim=0)
    histogram_normalized = histogram / (torch.sum(histogram) + 1e-8)
    return histogram_normalized

def soft_histogram1d_lengths(lengths, weights, bin_centers, sigma):
    lengths_expanded = lengths.unsqueeze(-1)
    bin_centers_expanded = bin_centers.unsqueeze(0)
    weights_expanded = weights.unsqueeze(-1)
    distances_squared = ((lengths_expanded - bin_centers_expanded) / sigma) ** 2
    gaussian_weights = torch.exp(-0.5 * distances_squared)
    weighted_contributions = gaussian_weights * weights_expanded
    histogram = torch.sum(weighted_contributions, dim=0)
    histogram_normalized = histogram / (torch.sum(histogram) + 1e-8)
    return histogram_normalized

def soft_hist_angles(axial_deg, w, edges_deg, sigma_deg):
    if not torch.is_tensor(edges_deg):
        edges_deg = torch.as_tensor(edges_deg, device=axial_deg.device, dtype=axial_deg.dtype)
    bc = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    d_raw = torch.abs(axial_deg.unsqueeze(-1) - bc)
    d = torch.minimum(d_raw, 180.0 - d_raw)
    sigma = torch.as_tensor(sigma_deg, device=axial_deg.device, dtype=axial_deg.dtype)
    g = torch.exp(-0.5 * (d / (sigma + 1e-8)) ** 2)
    h = (g * w.unsqueeze(-1)).sum(dim=1)
    h = h / (h.sum(dim=1, keepdim=True) + 1e-8)
    return h

def soft_hist_scalar(vals, w, edges, sigma):
    bc = 0.5 * (edges[:-1] + edges[1:])
    d = (vals.unsqueeze(-1) - bc) / (sigma + 1e-8)
    g = torch.exp(-0.5 * (d ** 2))
    h = (g * w.unsqueeze(-1)).sum(dim=1)
    h = h / (h.sum(dim=1, keepdim=True) + 1e-8)
    return h

def spacing_histogram(centers, w, edges):
    B, N, _ = centers.shape
    out = []
    for b in range(B):
        C = centers[b]
        D = torch.cdist(C, C, p=2.0)
        D = D + torch.eye(N, device=D.device) * 1e6
        nn, _ = D.min(dim=1)
        vals = torch.clamp(nn / math.sqrt(2.0), 0, 1)
        out.append(soft_hist_scalar(vals.unsqueeze(0), w[b].unsqueeze(0), edges, SIGMA_SPACING)[0])
    return torch.stack(out, 0)

def kde2d(centers, w, grid_w, grid_h, sigma):
    B, N, _ = centers.shape
    xs = torch.linspace(0.0, 1.0, grid_w, device=centers.device)
    ys = torch.linspace(0.0, 1.0, grid_h, device=centers.device)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    G = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    out = []
    for b in range(B):
        C = centers[b]
        D = torch.cdist(C, G, p=2.0)
        K = torch.exp(-0.5 * (D / (sigma + 1e-8)) ** 2)
        Z = (K * w[b].unsqueeze(-1)).sum(dim=0).view(grid_w, grid_h)
        Z = Z / (Z.sum() + 1e-8)
        out.append(Z)
    return torch.stack(out, 0)

def repulsion(centers, w, tau=0.1):
    B, N, _ = centers.shape
    out = []
    for b in range(B):
        C = centers[b]
        D = torch.cdist(C, C, p=2.0)
        R = torch.exp(-D / (tau + 1e-8))
        R = R * (1 - torch.eye(N, device=C.device))
        WM = w[b].unsqueeze(1) * w[b].unsqueeze(0)
        num = (R * WM).sum()
        den = (WM.sum() + 1e-8)
        out.append(num / (den + 1e-8))
    return torch.stack(out, 0).mean()

def soft_intersection_density(endpoints, w, sigma=None):
    if sigma is None:
        sigma = CONFIG["physics"]["SIGMA_CONN"]
    B, N, _ = endpoints.shape
    out = []
    for b in range(B):
        L = endpoints[b]
        M = torch.stack([(L[:, 0] + L[:, 2]) * 0.5, (L[:, 1] + L[:, 3]) * 0.5], -1)
        D = torch.cdist(M, M, p=2.0)
        K = torch.exp(-D / (sigma + 1e-8))
        K = K * (1 - torch.eye(N, device=L.device))
        WM = w[b].unsqueeze(1) * w[b].unsqueeze(0)
        upper = torch.triu(K * WM, 1).sum()
        s = w[b].sum()
        den = (s * (s - 1) / 2.0) + 1e-8
        out.append(upper / den)
    return torch.stack(out, 0)

def soft_degree_density(endpoints, w, sigma=None):
    if sigma is None:
        sigma = CONFIG["physics"]["SIGMA_CONN"]
    B, N, _ = endpoints.shape
    out = []
    for b in range(B):
        L = endpoints[b]
        C = torch.stack([(L[:, 0] + L[:, 2]) * 0.5, (L[:, 1] + L[:, 3]) * 0.5], -1)
        D = torch.cdist(C, C, p=2.0)
        K = torch.exp(-D / (sigma + 1e-8))
        K = K * (1 - torch.eye(N, device=L.device))
        wb = w[b].clamp(min=0.0, max=1.0)
        num_i = (K * wb.unsqueeze(0)).sum(dim=1)
        den_i = (wb.sum() - wb).clamp(min=1e-8)
        deg_i = num_i / den_i
        avg_deg = (deg_i * wb).sum() / (wb.sum() + 1e-8)
        out.append(avg_deg)
    return torch.stack(out, 0)

def soft_mean_degree(endpoints, w, sigma=None):
    if sigma is None:
        sigma = CONFIG["physics"]["SIGMA_CONN"]
    B, N, _ = endpoints.shape
    out = []
    for b in range(B):
        L = endpoints[b]
        M = torch.stack([(L[:, 0] + L[:, 2]) * 0.5, (L[:, 1] + L[:, 3]) * 0.5], -1)
        D = torch.cdist(M, M, p=2.0)
        K = torch.exp(-D / (sigma + 1e-8))
        K = K * (1 - torch.eye(N, device=L.device, dtype=L.dtype))
        wb = w[b]
        deg = (K * wb.unsqueeze(0)).sum(dim=1)
        s = wb.sum()
        mean_deg = (deg * wb).sum() / (s + 1e-8)
        norm = torch.clamp(s - 1.0, min=1.0)
        out.append(mean_deg / norm)
    return torch.stack(out, 0)

def sinkhorn_emd_linear(h1, h2, reg=0.1, iters=80):
    h1 = h1.reshape(-1).to(dtype=torch.float32)
    h2 = h2.reshape(-1).to(dtype=torch.float32)
    if h1.numel() != h2.numel():
        raise ValueError(f"Histogram bin mismatch: {h1.numel()} vs {h2.numel()}")
    device = h1.device
    K = h1.numel()
    a = h1.clamp(min=0)
    b = h2.clamp(min=0)
    a = a / (a.sum() + 1e-8)
    b = b / (b.sum() + 1e-8)
    pos = torch.linspace(0.0, float(K - 1), K, device=device, dtype=torch.float32).unsqueeze(1)
    C = torch.cdist(pos, pos, p=1)
    Kmat = torch.exp(-C / max(reg, 1e-6))
    u = torch.full((K,), 1.0 / K, device=device, dtype=torch.float32)
    v = torch.full((K,), 1.0 / K, device=device, dtype=torch.float32)
    for _ in range(iters):
        Kv = Kmat @ v + 1e-8
        u = a / Kv
        KTu = Kmat.t() @ u + 1e-8
        v = b / KTu
    P = (u.unsqueeze(1) * Kmat) * v.unsqueeze(0)
    return (P * C).sum()

def sinkhorn_emd_circular(p, q, reg=0.1, iters=100):
    n = p.shape[-1]
    dev = p.device
    i = torch.arange(n, device=dev, dtype=torch.float32).unsqueeze(1)
    j = torch.arange(n, device=dev, dtype=torch.float32).unsqueeze(0)
    d = torch.abs(i - j)
    C = torch.minimum(d, torch.tensor(float(n), device=dev) - d)
    K = torch.exp(-C / (reg + 1e-8))
    u = torch.ones(n, device=dev) / n
    v = torch.ones(n, device=dev) / n
    for _ in range(iters):
        u = p / (K @ (v) + 1e-8)
        v = q / (K.t() @ (u) + 1e-8)
    P = torch.diag(u) @ K @ torch.diag(v)
    return (P * C).sum()

def vm_mixture_nll(theta, w_exist, mu_vec, kappa_vec, pi_vec):
    B, N = theta.shape
    K = mu_vec.shape[0]
    T = theta.unsqueeze(-1).expand(B, N, K)
    MU = mu_vec.view(1, 1, K)
    KP = kappa_vec.view(1, 1, K)
    logcomp = KP * torch.cos(T - MU) - math.log(2 * math.pi) - torch.log(torch_i0(KP))
    logpi = torch.log(pi_vec.view(1, 1, K) + 1e-8)
    ll = torch.logsumexp(logpi + logcomp, dim=-1)
    num = (ll) * w_exist
    den = (w_exist.sum(dim=1, keepdim=True) + 1e-8)
    return -(num.sum(dim=1) / den.squeeze(1)).mean()

def compute_physics_losses(recon_lines, exist_logits, valid_mask, img_size, targets, priors,
                           use_angle_physics=True, use_length_physics=True, use_spacing_physics=True,
                           use_connectivity=True, use_hard_histogram=False, pixel_aware=True):
    dev = recon_lines.device
    B = recon_lines.shape[0]
    L_ang_hist = torch.tensor(0.0, device=dev)
    L_len_hist = torch.tensor(0.0, device=dev)
    L_spacing = torch.tensor(0.0, device=dev)
    L_pairwise = torch.tensor(0.0, device=dev)
    L_kde = torch.tensor(0.0, device=dev)
    L_conn = torch.tensor(0.0, device=dev)
    L_len_mean = torch.tensor(0.0, device=dev)
    L_ang_vm = torch.tensor(0.0, device=dev)
    if not any([use_angle_physics, use_length_physics, use_spacing_physics, use_connectivity]):
        return L_ang_hist, L_len_hist, L_spacing, L_pairwise, L_kde, L_conn, L_len_mean, L_ang_vm
    centers, lengths, angles = to_centers_lengths_angles(recon_lines)
    exist_probs = torch.sigmoid(exist_logits).to(dtype=recon_lines.dtype)
    if pixel_aware:
        H = img_size[:, 1].unsqueeze(1)
        W = img_size[:, 0].unsqueeze(1)
        dy = recon_lines[..., 3] - recon_lines[..., 1]
        dx = recon_lines[..., 2] - recon_lines[..., 0]
        angles = torch.atan2(dy * H, dx * W)
    angles_deg = (torch.rad2deg(angles) % 180.0)
    if use_angle_physics:
        angle_edges = torch.tensor(priors['angle_edges'], device=dev, dtype=recon_lines.dtype)
        for i, target in enumerate(targets):
            target_hist = torch.tensor(target['angle_hist'], device=dev, dtype=recon_lines.dtype)
            if use_hard_histogram:
                pred_hist, _ = torch.histogram(angles_deg[i], bins=angle_edges, weight=exist_probs[i])
                pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
                L_ang_hist += F.mse_loss(pred_hist, target_hist)
            else:
                angles_axial = (angles_deg[i] % 180.0)
                bin_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
                pred_hist = soft_histogram1d_angles(angles_axial, exist_probs[i], bin_centers, SIGMA_ANGLE_DEG)
                L_ang_hist += sinkhorn_emd_circular(pred_hist, target_hist)
        L_ang_hist = L_ang_hist / B
        vm_mu = torch.tensor(priors['vm_mu'], device=dev, dtype=recon_lines.dtype)
        vm_kappa = torch.tensor(priors['vm_kappa'], device=dev, dtype=recon_lines.dtype)
        vm_pi = torch.tensor(priors['vm_pi'], device=dev, dtype=recon_lines.dtype)
        angles_rad = torch.deg2rad(angles_deg)
        L_ang_vm = vm_mixture_nll_axial(angles_rad, exist_probs, vm_mu, vm_kappa, vm_pi)
    if use_length_physics:
        length_edges = torch.tensor(priors['length_edges'], device=dev, dtype=recon_lines.dtype)
        for i, target in enumerate(targets):
            target_hist = torch.tensor(target['length_hist'], device=dev, dtype=recon_lines.dtype)
            bin_centers = 0.5 * (length_edges[:-1] + length_edges[1:])
            pred_hist = soft_histogram1d_lengths(lengths[i], exist_probs[i], bin_centers, SIGMA_LEN)
            L_len_hist += sinkhorn_emd_linear(pred_hist, target_hist)
        L_len_hist = L_len_hist / B
        target_mean = priors['global_length_stats']['mean']
        pred_mean = (lengths * exist_probs).sum(dim=1) / (exist_probs.sum(dim=1) + 1e-8)
        L_len_mean = F.mse_loss(pred_mean, torch.full_like(pred_mean, target_mean))
    if use_spacing_physics:
        spacing_edges = torch.tensor(priors['spacing_edges'], device=dev, dtype=recon_lines.dtype)
        pred_spacing_hist = spacing_histogram(centers, exist_probs, spacing_edges)
        for i, target in enumerate(targets):
            target_hist = torch.tensor(target['spacing_hist'], device=dev, dtype=recon_lines.dtype)
            L_spacing += sinkhorn_emd_linear(pred_spacing_hist[i], target_hist)
        L_spacing = L_spacing / B
        kde_pred = kde2d(centers, exist_probs, GRID_W, GRID_H, SIGMA_KDE)
        for i, target in enumerate(targets):
            target_kde = torch.tensor(target['kde'], device=dev, dtype=recon_lines.dtype)
            over = torch.relu(kde_pred[i] - target_kde)
            L_kde += (over.pow(2)).mean()
        L_kde = L_kde / B
        if 'pairwise_edges' in priors and 'global_pairwise_hist' in priors:
            pair_edges = torch.tensor(priors['pairwise_edges'], device=dev, dtype=recon_lines.dtype)
            global_pair_tgt = torch.tensor(priors['global_pairwise_hist'], device=dev, dtype=recon_lines.dtype)
            pred_pair_hist = pairwise_distance_histogram(centers, exist_probs, pair_edges)
            for i in range(B):
                L_pairwise += sinkhorn_emd_linear(pred_pair_hist[i], global_pair_tgt)
            L_pairwise = L_pairwise / B
    if use_connectivity:
        conn_pred = soft_intersection_density(recon_lines, exist_probs)
        conn_targets = torch.tensor([t['connectivity_target'] for t in targets], device=dev, dtype=recon_lines.dtype)
        L_conn = F.mse_loss(conn_pred, conn_targets)
    return L_ang_hist, L_len_hist, L_spacing, L_pairwise, L_kde, L_conn, L_len_mean, L_ang_vm

class EMAConstraintNormalizer:
    def __init__(self, keys, alpha=0.95):
        self.alpha = alpha
        self.state = {k: 1.0 for k in keys}

    def norm(self, k, val):
        if k not in self.state:
            self.state[k] = 1.0
        v = float(val.detach().cpu().item())
        self.state[k] = self.alpha * self.state[k] + (1 - self.alpha) * max(v, 1e-8)
        return val / (self.state[k] + 1e-8)

def vm_mixture_nll_axial(theta, w_exist, mu_vec, kappa_vec, pi_vec):
    B, N = theta.shape
    K = mu_vec.shape[0]
    T2 = (2.0 * theta).unsqueeze(-1).expand(B, N, K)
    MU2 = (2.0 * mu_vec).view(1, 1, K)
    KP = kappa_vec.view(1, 1, K)
    logcomp = KP * torch.cos(T2 - MU2) - math.log(2 * math.pi) - torch.log(torch_i0(KP))
    logpi = torch.log(pi_vec.view(1, 1, K) + 1e-8)
    ll = torch.logsumexp(logpi + logcomp, dim=-1)
    num = ll * w_exist
    den = (w_exist.sum(dim=1, keepdim=True) + 1e-8)
    return -(num.sum(dim=1) / den.squeeze(1)).mean()

def extract_targets_and_priors(samples):
    all_counts = []
    all_len = []
    all_nn = []
    inter_rates = []
    all_angles_deg = []
    pairwise_edges = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    global_pair_counts = np.zeros(pairwise_edges.shape[0] - 1, dtype=np.float64)
    for s in samples:
        if ROUTEB_ISO:
            W = float(max(s["width"], s["height"]))   # ROUTE B (T07)
            H = W
        else:
            W = s["width"]
            H = s["height"]
        Pn = []
        for x1, y1, x2, y2 in s["lines"]:
            x1n, y1n, x2n, y2n = x1 / W, y1 / H, x2 / W, y2 / H
            Pn.append([x1n, y1n, x2n, y2n])
            all_len.append(math.hypot(x2n - x1n, y2n - y1n))
            ang_deg = float((math.degrees(math.atan2(y2n - y1n, x2n - x1n)) % 180.0))
            all_angles_deg.append(ang_deg)
        all_counts.append(len(s["lines"]))
        if len(Pn) >= 2:
            Cnp = np.stack([((np.array(Pn)[:, 0] + np.array(Pn)[:, 2]) * 0.5), ((np.array(Pn)[:, 1] + np.array(Pn)[:, 3]) * 0.5)], axis=1)
            D = np.sqrt(((Cnp[None, :, :] - Cnp[:, None, :]) ** 2).sum(-1)) + np.eye(len(Cnp)) * 1e6
            all_nn.extend(D.min(axis=1).tolist())
            inter = count_intersections_norm(Pn)
            total_pairs = len(Pn) * (len(Pn) - 1) / 2
            inter_rates.append(float(inter / max(total_pairs, 1)))
            iu = np.triu_indices(len(Cnp), k=1)
            pd = np.clip(np.sqrt(((Cnp[None, :, :] - Cnp[:, None, :]) ** 2).sum(-1))[iu] / np.sqrt(2.0), 0, 1)
            h_pair, _ = np.histogram(pd, bins=pairwise_edges)
            global_pair_counts += h_pair.astype(np.float64)
        else:
            inter_rates.append(0.0)
    if len(all_len) > 0:
        p05 = float(np.percentile(all_len, 5))
        p95 = float(np.percentile(all_len, 95))
        lmean = float(np.mean(all_len))
    else:
        p05 = 0.02
        p95 = 0.30
        lmean = (p05 + p95) / 2.0
    length_edges = np.linspace(p05, p95, LEN_BINS + 1, dtype=np.float32)
    angle_edges = np.linspace(0.0, 180.0, ANGLE_BINS + 1, dtype=np.float32)
    spacing_edges = np.linspace(0.0, 1.0, SPACING_BINS + 1, dtype=np.float32)
    per = {}
    global_ang_hist = np.zeros(ANGLE_BINS, dtype=np.float64)
    global_len_hist = np.zeros(LEN_BINS, dtype=np.float64)
    if len(all_nn) > 0:
        target_spacing = float(np.percentile(all_nn, 60))
    else:
        target_spacing = 0.05
    sigma_conn = 0.5 * target_spacing
    for s in samples:
        if ROUTEB_ISO:
            W = float(max(s["width"], s["height"]))   # ROUTE B (T07)
            H = W
        else:
            W = s["width"]
            H = s["height"]
        P = []
        for x1, y1, x2, y2 in s["lines"]:
            P.append([x1 / W, y1 / H, x2 / W, y2 / H])
        P = np.array(P, dtype=np.float32)
        if P.shape[0] == 0:
            continue
        Cnp = np.stack([(P[:, 0] + P[:, 2]) * 0.5, (P[:, 1] + P[:, 3]) * 0.5], axis=1)
        lengths = np.sqrt((P[:, 2] - P[:, 0]) ** 2 + (P[:, 3] - P[:, 1]) ** 2)
        ang = np.degrees(np.mod(np.arctan2(P[:, 3] - P[:, 1], P[:, 2] - P[:, 0]), np.pi))
        h_ang, _ = np.histogram(ang, bins=angle_edges)
        h_len, _ = np.histogram(np.clip(lengths, p05, p95), bins=length_edges)
        global_ang_hist += h_ang
        global_len_hist += h_len
        D = np.sqrt(((Cnp[None, :, :] - Cnp[:, None, :]) ** 2).sum(-1)) + np.eye(len(Cnp)) * 1e6
        nn = D.min(axis=1)
        spacing = np.clip(nn / np.sqrt(2.0), 0, 1)
        h_sp, _ = np.histogram(spacing, bins=spacing_edges)
        if Cnp.shape[0] >= 2:
            Dp = np.sqrt(((Cnp[None, :, :] - Cnp[:, None, :]) ** 2).sum(-1))
            iu = np.triu_indices(Dp.shape[0], k=1)
            pdist = np.clip(Dp[iu] / np.sqrt(2.0), 0, 1)
            h_pair, _ = np.histogram(pdist, bins=pairwise_edges)
            pair_norm = (h_pair / np.maximum(h_pair.sum(), 1e-8)).astype(np.float32)
        else:
            pair_norm = np.zeros(pairwise_edges.shape[0] - 1, dtype=np.float32)
        xs = np.concatenate([P[:, 0], P[:, 2]])
        ys = np.concatenate([P[:, 1], P[:, 3]])
        xr = xs.max() - xs.min()
        yr = ys.max() - ys.min()
        bcount = 0
        for i in range(P.shape[0]):
            x1, y1, x2, y2 = P[i]
            if (min(x1, x2) < 0.05 or max(x1, x2) > 0.95 or min(y1, y2) < 0.05 or max(y1, y2) > 0.95):
                bcount += 1
        br = bcount / max(1, len(P))
        pairs = P.shape[0] * (P.shape[0] - 1) / 2.0
        if pairs > 0:
            hard_conn = float(count_intersections_norm(P.tolist())) / float(pairs)
        else:
            hard_conn = 0.0
        gx = np.linspace(0.0, 1.0, GRID_W)
        gy = np.linspace(0.0, 1.0, GRID_H)
        kde = np.zeros((GRID_W, GRID_H), dtype=np.float64)
        for i, gxi in enumerate(gx):
            for j, gyj in enumerate(gy):
                d = np.sqrt(((Cnp[:, 0] - gxi) ** 2 + (Cnp[:, 1] - gyj) ** 2))
                kde[i, j] = np.exp(-0.5 * (d / (SIGMA_KDE + 1e-8)) ** 2).sum()
        kde = kde / (kde.sum() + 1e-8)
        per[s["sample_id"]] = {
            "angle_hist": (h_ang / np.maximum(h_ang.sum(), 1e-8)).astype(np.float32).tolist(),
            "length_hist": (h_len / np.maximum(h_len.sum(), 1e-8)).astype(np.float32).tolist(),
            "spacing_hist": (h_sp / np.maximum(h_sp.sum(), 1e-8)).astype(np.float32).tolist(),
            "coverage_vec": [float(xr), float(yr), float(br)],
            "connectivity_target": float(hard_conn),
            "kde": kde.astype(np.float32).tolist(),
            "pairwise_dist_hist": pair_norm.tolist()
        }
    fam = select_angle_families_axial(np.radians(all_angles_deg).astype(np.float64), K_candidates=(1, 2, 3, 4))
    vm_mu = fam["mu"].astype(np.float32)
    vm_kappa = fam["kappa"].astype(np.float32)
    vm_pi = fam["alpha"].astype(np.float32)
    shallow_share_list = []
    if fam["K"] >= 2:
        centers_deg = np.degrees(vm_mu)
        idx_sort = np.argsort(centers_deg)
        centers_deg = centers_deg[idx_sort]
        for s in samples:
            if ROUTEB_ISO:
                W = float(max(s["width"], s["height"]))   # ROUTE B (T08)
                H = W
            else:
                W = s["width"]
                H = s["height"]
            angs = []
            for x1, y1, x2, y2 in s["lines"]:
                a = float((math.degrees(math.atan2((y2 - y1) / H, (x2 - x1) / W)) % 180.0))
                angs.append(a)
            if len(angs) == 0:
                shallow_share_list.append(0.0)
                continue
            A = np.array(angs, dtype=np.float32)
            dists = np.minimum(np.abs(A[:, None] - centers_deg[None, :]), 180.0 - np.abs(A[:, None] - centers_deg[None, :]))
            lab = np.argmin(dists, axis=1)
            shallow_share_list.append(float((lab == 0).mean()))
    if len(shallow_share_list) > 0:
        sh_med = float(np.median(shallow_share_list))
        sh_p10 = float(np.percentile(shallow_share_list, 10))
        sh_p90 = float(np.percentile(shallow_share_list, 90))
    else:
        sh_med, sh_p10, sh_p90 = 0.5, 0.4, 0.6
    constraints = {
        "max_length": float(np.percentile(all_len, 95)) if len(all_len) > 0 else 0.35,
        "target_spacing": target_spacing,
        "nms_radius": target_spacing,
        "target_intersection_rate": float(np.mean(inter_rates)) if len(inter_rates) > 0 else 0.0,
        "sigma_conn": float(sigma_conn)
    }
    global_pair_hist = (global_pair_counts / np.maximum(global_pair_counts.sum(), 1e-8)).astype(np.float32)
    priors = {
        "angle_edges": angle_edges.tolist(),
        "length_edges": length_edges.tolist(),
        "spacing_edges": spacing_edges.tolist(),
        "vm_mu": vm_mu.tolist(),
        "vm_kappa": vm_kappa.tolist(),
        "vm_pi": vm_pi.tolist(),
        "global_length_stats": {"p05": p05, "p95": p95, "mean": lmean},
        "count_stats": {
            "min": int(np.min(all_counts)) if len(all_counts) > 0 else 1,
            "max": int(np.max(all_counts)) if len(all_counts) > 0 else 1,
            "mean": float(np.mean(all_counts)) if len(all_counts) > 0 else 1.0,
            "std": float(np.std(all_counts)) if len(all_counts) > 0 else 0.0
        },
        "global_angle_hist": (global_ang_hist / np.maximum(global_ang_hist.sum(), 1e-8)).astype(np.float32).tolist(),
        "global_length_hist": (global_len_hist / np.maximum(global_len_hist.sum(), 1e-8)).astype(np.float32).tolist(),
        "constraints": constraints,
        "pairwise_edges": pairwise_edges.tolist(),
        "global_pairwise_hist": global_pair_hist.tolist(),
        "family_quota": {"median": sh_med, "p10": sh_p10, "p90": sh_p90}
    }
    return per, priors

def save_priors(priors, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(priors, f)

def load_priors(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_targets_batch(sample_ids, per):
    out = []
    keys = list(per.keys())
    for sid in sample_ids:
        t0 = per[int(sid)]
        if random.random() < PMIX and len(keys) > 1:
            sid2 = random.choice(keys)
            for _ in range(10):
                if int(sid2) != int(sid):
                    break
                sid2 = random.choice(keys)
            a = random.uniform(0.3, 0.7)
            t1 = per[int(sid2)]
            mix = {
                'angle_hist': (a * np.array(t0['angle_hist']) + (1 - a) * np.array(t1['angle_hist'])).astype(np.float32).tolist(),
                'length_hist': (a * np.array(t0['length_hist']) + (1 - a) * np.array(t1['length_hist'])).astype(np.float32).tolist(),
                'spacing_hist': (a * np.array(t0['spacing_hist']) + (1 - a) * np.array(t1['spacing_hist'])).astype(np.float32).tolist(),
                'coverage_vec': (a * np.array(t0['coverage_vec']) + (1 - a) * np.array(t1['coverage_vec'])).astype(np.float32).tolist(),
                'connectivity_target': float(a * t0['connectivity_target'] + (1 - a) * t1['connectivity_target']),
                'kde': (a * np.array(t0['kde']) + (1 - a) * np.array(t1['kde'])).astype(np.float32).tolist(),
                'pairwise_dist_hist': (a * np.array(t0['pairwise_dist_hist']) + (1 - a) * np.array(t1['pairwise_dist_hist'])).astype(np.float32).tolist()
            }
            out.append(mix)
        else:
            out.append(t0)
    return out

def sample_k(priors):
    mu = priors['count_stats']['mean']
    sd = max(1.0, priors['count_stats']['std'])
    mn = priors['count_stats']['min']
    mx = priors['count_stats']['max']
    val = int(round(np.random.normal(mu, sd)))
    return int(max(mn, min(mx, val)))

def postprocess(centers, lengths, angles, exist, priors, W, H):
    k = sample_k(priors)
    dev = angles.device
    vm_mu = torch.tensor(priors['vm_mu'], device=dev, dtype=angles.dtype)
    vm_k = torch.tensor(priors['vm_kappa'], device=dev, dtype=angles.dtype)
    vm_pi = torch.tensor(priors['vm_pi'], device=dev, dtype=angles.dtype)
    T = angles.unsqueeze(-1)
    logcomp = vm_k.view(1, 1, -1) * torch.cos(T - vm_mu.view(1, 1, -1)) - math.log(2 * math.pi) - torch.log(torch_i0(vm_k.view(1, 1, -1)))
    ll = torch.logsumexp(torch.log(vm_pi.view(1, 1, -1) + 1e-8) + logcomp, dim=-1).squeeze(0)
    ll = (ll - ll.min()) / (ll.max() - ll.min() + 1e-8)
    lmean = priors['global_length_stats']['mean']
    s_len = torch.exp(-((lengths[0] - lmean) / (0.1 + 1e-8)) ** 2)
    s_exist = exist[0]
    S = 0.8 * s_exist + 0.1 * ll + 0.1 * s_len
    idx = torch.argsort(S, descending=True).tolist()
    chosen = []
    nms_r = float(priors.get('constraints', {}).get('nms_radius', 0.02))
    for i in idx:
        if len(chosen) >= k:
            break
        ok = True
        for j in chosen:
            if torch.norm(centers[0, i] - centers[0, j], p=2) < nms_r:
                ok = False
                break
        if ok:
            chosen.append(i)
    C = centers[0, chosen]
    L = lengths[0, chosen]
    A = angles[0, chosen]
    E = centers_lengths_angles_to_endpoints(C.unsqueeze(0), L.unsqueeze(0), A.unsqueeze(0))[0]
    out = []
    for i in range(E.shape[0]):
        x1 = float(E[i, 0] * W)
        y1 = float(E[i, 1] * H)
        x2 = float(E[i, 2] * W)
        y2 = float(E[i, 3] * H)
        if math.hypot(x2 - x1, y2 - y1) >= 1.0:
            out.append({'start': [x1, y1], 'end': [x2, y2]})
    return out

def save_image(lines, W, H, fn):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_facecolor('white')
    for ln in lines:
        s = ln['start']
        e = ln['end']
        ax.plot([s[0], e[0]], [s[1], e[1]], linewidth=2)
    ax.set_title(f'Generated ({len(lines)} lines)')
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def analyze(lines, W, H, priors):
    xs = []
    ys = []
    angs = []
    bcount = 0
    lensn = []
    for ln in lines:
        s = ln['start']
        e = ln['end']
        xs += [s[0], e[0]]
        ys += [s[1], e[1]]
        dx = (e[0] - s[0]) / W
        dy = (e[1] - s[1]) / H
        lensn.append(math.hypot(dx, dy))
        a = math.degrees(math.atan2((e[1] - s[1]) / H, (e[0] - s[0]) / W)) % 180.0
        angs.append(a)
        if (min(s[0], e[0]) < 0.05 * W or max(s[0], e[0]) > 0.95 * W or min(s[1], e[1]) < 0.05 * H or max(s[1], e[1]) > 0.95 * H):
            bcount += 1
    if len(xs) > 0:
        xcov = (max(xs) - min(xs)) / W
        ycov = (max(ys) - min(ys)) / H
        br = bcount / max(1, len(lines))
    else:
        xcov = 0
        ycov = 0
        br = 0
    a_hist, _ = np.histogram(angs, bins=ANGLE_BINS, range=(0, 180))
    l_edges = np.array(priors['length_edges'])
    l_hist, _ = np.histogram(np.clip(lensn, l_edges[0], l_edges[-1]), bins=l_edges)
    a_hist = a_hist / np.maximum(a_hist.sum(), 1e-8)
    l_hist = l_hist / np.maximum(l_hist.sum(), 1e-8)
    target_a = np.array(priors['global_angle_hist'])
    target_l = np.array(priors['global_length_hist'])
    angle_emd = float(np.sum(np.abs(a_hist - target_a)))
    length_emd = float(np.sum(np.abs(l_hist - target_l)))
    return {
        'count': len(lines),
        'x_coverage': xcov,
        'y_coverage': ycov,
        'boundary_ratio': br,
        'angle_emd': angle_emd,
        'length_emd': length_emd
    }

def centers_lengths_angles_to_endpoints_pixelaware(centers, lengths_norm, angles, img_size):
    assert centers.dim() == 3 and centers.size(-1) == 2
    assert lengths_norm.shape[:2] == centers.shape[:2]
    assert angles.shape[:2] == centers.shape[:2]
    assert img_size.dim() == 2 and img_size.size(1) == 2
    B, N, _ = centers.shape
    W = img_size[:, 0].view(B, 1).to(dtype=centers.dtype)
    H = img_size[:, 1].view(B, 1).to(dtype=centers.dtype)
    vx = torch.cos(angles) / (W + 1e-8)
    vy = torch.sin(angles) / (H + 1e-8)
    norm = torch.sqrt(vx * vx + vy * vy) + 1e-8
    vx, vy = vx / norm, vy / norm
    dx = 0.5 * lengths_norm * vx
    dy = 0.5 * lengths_norm * vy
    x1 = torch.clamp(centers[..., 0] - dx, 0, 1)
    y1 = torch.clamp(centers[..., 1] - dy, 0, 1)
    x2 = torch.clamp(centers[..., 0] + dx, 0, 1)
    y2 = torch.clamp(centers[..., 1] + dy, 0, 1)
    return torch.stack([x1, y1, x2, y2], -1)

def _soft_intersection_density_numpy(lines, W, H, sigma):
    if len(lines) < 2:
        return 0.0
    P = []
    for x1, y1, x2, y2 in lines:
        cx = (x1 + x2) * 0.5 / W
        cy = (y1 + y2) * 0.5 / H
        P.append([cx, cy])
    P = np.array(P, dtype=np.float32)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    K = np.exp(-D / (sigma + 1e-8))
    np.fill_diagonal(K, 0.0)
    n = P.shape[0]
    den = n * (n - 1) / 2.0 + 1e-8
    return np.triu(K, 1).sum() / den

def pairwise_distance_histogram(centers, w, edges, sigma=None):
    if sigma is None:
        sigma = CONFIG["physics"]["SIGMA_SPACING"]
    B, N, _ = centers.shape
    out = []
    rt2 = math.sqrt(2.0)
    for b in range(B):
        C = centers[b]
        D = torch.cdist(C, C, p=2.0)
        iu, ju = torch.triu_indices(N, N, offset=1, device=C.device)
        if iu.numel() == 0:
            K = edges.numel() - 1
            out.append(torch.zeros(K, device=C.device, dtype=C.dtype))
            continue
        dist = torch.clamp(D[iu, ju] / rt2, 0.0, 1.0)
        wb = w[b]
        Wpair = (wb[iu] * wb[ju]).unsqueeze(0)
        hist = soft_hist_scalar(dist.unsqueeze(0), Wpair, edges, sigma)[0]
        out.append(hist)
    return torch.stack(out, 0)

def diagnostic_latent_utilization(model, dataloader, device):
    model.eval()
    all_smu, all_slog, all_gmu, all_glog = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch['lines_norm'], batch['valid_mask'], batch['img_size'])
            all_smu.append(out['smu'].cpu())
            all_slog.append(out['slog'].cpu())
            all_gmu.append(out['gmu'].cpu())
            all_glog.append(out['glog'].cpu())
    if len(all_smu) == 0:
        return 0.0, 0.0, 0.0, 0.0
    smu = torch.cat(all_smu, 0)
    slog = torch.cat(all_slog, 0)
    gmu = torch.cat(all_gmu, 0)
    glog = torch.cat(all_glog, 0)
    skl = 0.5 * (smu.pow(2) + slog.exp() - slog - 1)
    gkl = 0.5 * (gmu.pow(2) + glog.exp() - glog - 1)
    def analyze_dims(kl_vals: torch.Tensor, mu_vals: torch.Tensor, name: str):
        d = kl_vals.shape[1]
        kl_mean = kl_vals.mean(dim=0)
        kl_median = kl_vals.median(dim=0).values
        kl_p90 = kl_vals.quantile(0.9, dim=0)
        mu_var = mu_vals.var(dim=0, unbiased=False)
        active_01 = (kl_mean >= 0.01).float().mean() * 100
        active_05 = (kl_mean >= 0.05).float().mean() * 100
        mu_var_active = (mu_var >= 1e-3).float().mean() * 100
        top5_idx = kl_mean.argsort(descending=True)[:5]
        top5_kl = kl_mean[top5_idx]
        print(f"UTILIZATION_{name}")
        print(f"d={d} KL_mean={kl_mean.mean():.6f} KL_median={kl_median.mean():.6f} KL_p90={kl_p90.mean():.6f} mu_var_mean={mu_var.mean():.6f}")
        print(f"KL>=0.01 %={active_01:.2f} KL>=0.05 %={active_05:.2f} mu_var>=1e-3 %={mu_var_active:.2f}")
        print(f"top5_dims_by_KL={top5_idx.tolist()} top5_KL={[float(v) for v in top5_kl]}")
        return float(active_01), float(active_05)
    s_act01, s_act05 = analyze_dims(skl, smu, "SPATIAL")
    g_act01, g_act05 = analyze_dims(gkl, gmu, "GEOMETRY")
    return s_act01, s_act05, g_act01, g_act05

def estimate_posterior_statistics(model, dataloader, device):
    model.eval()
    all_z_spatial = []
    all_z_geometry = []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch['lines_norm'], batch['valid_mask'], batch['img_size'])
            all_z_spatial.append(out['z_spatial'].detach().cpu())
            all_z_geometry.append(out['z_geometry'].detach().cpu())
    if len(all_z_spatial) == 0 or len(all_z_geometry) == 0:
        spatial_mean = torch.zeros(1, int(getattr(model, 'spatial_dim')), device=device, dtype=torch.float32)
        spatial_std = torch.ones(1, int(getattr(model, 'spatial_dim')), device=device, dtype=torch.float32)
        geometry_mean = torch.zeros(1, int(getattr(model, 'geometry_dim')), device=device, dtype=torch.float32)
        geometry_std = torch.ones(1, int(getattr(model, 'geometry_dim')), device=device, dtype=torch.float32)
        return {
            'spatial_mean': spatial_mean,
            'spatial_std': spatial_std,
            'geometry_mean': geometry_mean,
            'geometry_std': geometry_std
        }
    z_spatial = torch.cat(all_z_spatial, dim=0).to(dtype=torch.float32)
    z_geometry = torch.cat(all_z_geometry, dim=0).to(dtype=torch.float32)
    spatial_mean = z_spatial.mean(dim=0, keepdim=True).to(device)
    spatial_std = z_spatial.std(dim=0, keepdim=True).to(device)
    geometry_mean = z_geometry.mean(dim=0, keepdim=True).to(device)
    geometry_std = z_geometry.std(dim=0, keepdim=True).to(device)
    return {
        'spatial_mean': spatial_mean,
        'spatial_std': spatial_std,
        'geometry_mean': geometry_mean,
        'geometry_std': geometry_std
    }

def count_intersections_norm(lines_norm):
    n = len(lines_norm)
    if n < 2:
        return 0
    def seg_inter(a):
        x1,y1,x2,y2 = a
        return x1,y1,x2,y2
    def ccw(ax,ay,bx,by,cx,cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    cnt = 0
    for i in range(n):
        a = seg_inter(lines_norm[i])
        for j in range(i+1, n):
            b = seg_inter(lines_norm[j])
            a1x,a1y,a2x,a2y = a
            b1x,b1y,b2x,b2y = b
            c1 = ccw(a1x,a1y,b1x,b1y,b2x,b2y) != ccw(a2x,a2y,b1x,b1y,b2x,b2y)
            c2 = ccw(a1x,a1y,a2x,a2y,b1x,b1y) != ccw(a1x,a1y,a2x,a2y,b2x,b2y)
            if c1 and c2:
                cnt += 1
    return cnt
