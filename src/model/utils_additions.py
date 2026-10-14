import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import to_centers_lengths_angles, centers_lengths_angles_to_endpoints, build_targets_batch, \
    compute_physics_losses, EMAConstraintNormalizer, sample_k, CONFIG


class DualLatentVAE(nn.Module):
    def __init__(self, spatial_dim=None, geometry_dim=None, max_lines=None):
        super().__init__()
        self.spatial_dim = spatial_dim if spatial_dim is not None else CONFIG["model"]["spatial_dim"]
        self.geometry_dim = geometry_dim if geometry_dim is not None else CONFIG["model"]["geometry_dim"]
        self.max_lines = max_lines if max_lines is not None else CONFIG["model"]["max_lines"]

        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.max_lines * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.spatial_dim * 2)
        )

        self.geometry_encoder = nn.Sequential(
            nn.Linear(self.max_lines * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.geometry_dim * 2)
        )

        self.spatial_decoder = nn.Sequential(
            nn.Linear(self.spatial_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_lines * 2)
        )

        self.geometry_decoder = nn.Sequential(
            nn.Linear(self.geometry_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_lines * 2)
        )

        self.existence_head = nn.Sequential(
            nn.Linear(self.spatial_dim + self.geometry_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.max_lines)
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
        geom = self.geometry_decoder(z_geometry).view(-1, self.max_lines, 2)[0]
        lengths = torch.sigmoid(geom[:, 0])
        angles = torch.sigmoid(geom[:, 1]) * math.pi
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
        pred_geom = self.geometry_decoder(z_geometry).view(-1, self.max_lines, 2)
        pred_lengths = torch.sigmoid(pred_geom[:, :, 0])
        pred_angles = torch.sigmoid(pred_geom[:, :, 1]) * math.pi

        z_combined = torch.cat([z_spatial, z_geometry], dim=1)
        existence_logits = self.existence_head(z_combined)

        return {
            'centers': pred_centers,
            'lengths': pred_lengths,
            'angles': pred_angles,
            'existence_logits': existence_logits,
            'smu': smu, 'slog': slog,
            'gmu': gmu, 'glog': glog,
            'z_spatial': z_spatial,
            'z_geometry': z_geometry
        }


def hungarian_assignment_loss(pred_centers, pred_lengths, pred_angles, gt_centers, gt_lengths, gt_angles, valid_mask):
    B = pred_centers.shape[0]
    total_loss = 0.0

    for b in range(B):
        n_valid = int(valid_mask[b].sum().item())
        if n_valid == 0:
            continue

        pred_c = pred_centers[b]
        pred_l = pred_lengths[b]
        pred_a = pred_angles[b]

        gt_c = gt_centers[b][:n_valid]
        gt_l = gt_lengths[b][:n_valid]
        gt_a = gt_angles[b][:n_valid]

        cost_center = torch.cdist(pred_c, gt_c, p=2)
        cost_length = torch.cdist(pred_l.unsqueeze(1), gt_l.unsqueeze(1), p=1)
        cost_angle = torch.abs(torch.sin(pred_a.unsqueeze(1) - gt_a.unsqueeze(0)))

        cost_matrix = 1.0 * cost_center + 0.5 * cost_length + 1.5 * cost_angle

        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

        center_loss = F.mse_loss(pred_c[row_ind], gt_c[col_ind])
        length_loss = F.mse_loss(pred_l[row_ind], gt_l[col_ind])
        angle_loss = F.mse_loss(pred_a[row_ind], gt_a[col_ind])

        total_loss += center_loss + length_loss + angle_loss

    return total_loss / B


def sample_from_priors(priors, n_samples=1):
    count_stats = priors['count_stats']
    k_samples = []
    for _ in range(n_samples):
        k = max(1, int(np.random.normal(count_stats['mean'], count_stats['std'])))
        k = np.clip(k, count_stats['min'], count_stats['max'])
        k_samples.append(k)
    return k_samples


@torch.no_grad()
def guided_inference_selection(
        centers, lengths, angles, exist_logits, vm_mu, k_target,
        T=None, q=None, alpha=None, K=None, angle_weight=None,
        nms_radius=None, length_mean=None, length_sigma=None, target_spacing=None,
        bin_targets=None
):
    T = T if T is not None else CONFIG["inference"]["temperature"]
    q = q if q is not None else CONFIG["inference"]["q_threshold"]
    alpha = alpha if alpha is not None else CONFIG["inference"]["alpha_nudge"]
    K = K if K is not None else CONFIG["inference"]["shortlist_K"]
    angle_weight = angle_weight if angle_weight is not None else CONFIG["inference"]["angle_weight"]
    nms_radius = nms_radius if nms_radius is not None else CONFIG["inference"]["nms_radius"]

    device = angles.device
    scores = torch.sigmoid((exist_logits.to(device)) / T)

    k_needed = int(max(float(k_target) * 1.5, float(k_target) + 5.0))
    k_needed = min(int(scores.numel()), max(1, k_needed))

    thr_q = torch.quantile(scores, float(q))
    sorted_scores, _ = torch.sort(scores, descending=True)
    thr_needed = sorted_scores[k_needed - 1]
    thr = torch.minimum(thr_q, thr_needed)

    keep = torch.nonzero(scores >= thr, as_tuple=True)[0]
    if keep.numel() == 0:
        keep = torch.topk(scores, k=1).indices

    vm_mu_t = torch.as_tensor(vm_mu, dtype=angles.dtype, device=device)
    ak = angles.clone()
    if vm_mu_t.numel() > 0:
        cos_sim = torch.cos(angles[keep].unsqueeze(1) - vm_mu_t.unsqueeze(0))
        nearest_idx = torch.argmax(cos_sim, dim=1)
        nearest_mu = vm_mu_t[nearest_idx]
        ak[keep] = (1.0 - alpha) * angles[keep] + alpha * nearest_mu

    angles_deg = (torch.rad2deg(ak[keep]) % 180.0)

    if bin_targets is not None:
        bin_labels = torch.zeros(keep.numel(), dtype=torch.long, device=device)
        for idx, (bin_min, bin_max) in enumerate([(60, 80), (80, 100), (100, 120)]):
            mask = (angles_deg >= bin_min) & (angles_deg < bin_max)
            bin_labels[mask] = idx

        bin_means = []
        bin_stds = []
        for idx in range(3):
            bin_mask = (bin_labels == idx)
            if bin_mask.sum() > 0:
                bin_lens = lengths[keep][bin_mask]
                bin_means.append(bin_lens.mean())
                bin_stds.append(bin_lens.std() + 1e-6)
            else:
                bin_means.append(lengths.mean())
                bin_stds.append(torch.tensor(0.1, dtype=lengths.dtype, device=device))

        s_len_norm = torch.zeros_like(scores[keep])
        for idx in range(3):
            bin_mask = (bin_labels == idx)
            if bin_mask.sum() > 0:
                z = (lengths[keep][bin_mask] - bin_means[idx]) / (bin_stds[idx] + 1e-8)
                s_len_norm[bin_mask] = torch.exp(-0.5 * z ** 2)
    else:
        if length_mean is None:
            lmean = lengths.mean()
        else:
            lmean = torch.as_tensor(float(length_mean), dtype=lengths.dtype, device=device)
        if length_sigma is None:
            lsig = torch.tensor(0.1, dtype=lengths.dtype, device=device)
        else:
            lsig = torch.as_tensor(max(1e-6, float(length_sigma)), dtype=lengths.dtype, device=device)

        s_len_norm = torch.exp(-((lengths[keep] - lmean) / (lsig + 1e-8)) ** 2)

    if vm_mu_t.numel() > 0:
        sin_abs = torch.abs(torch.sin(ak[keep].unsqueeze(1) - vm_mu_t.unsqueeze(0)))
        geom_goodness, _ = torch.min(sin_abs, dim=1)
        geom_score = 1.0 - geom_goodness
    else:
        geom_score = torch.zeros_like(scores[keep])

    base_score = (1.0 - angle_weight) * scores[keep] + angle_weight * geom_score

    topk = min(int(K), base_score.shape[0]) if int(K) > 0 else base_score.shape[0]
    order = torch.topk(base_score, k=topk, sorted=True).indices
    cand_idx = keep[order]

    if target_spacing is None:
        eff_radius = torch.as_tensor(float(nms_radius), dtype=centers.dtype, device=device)
    else:
        eff_radius = torch.as_tensor(max(float(nms_radius), 0.5 * float(target_spacing)),
                                     dtype=centers.dtype, device=device)
    eff_radius = torch.clamp(eff_radius, min=1e-6)

    if bin_targets is not None:
        bin_quota = [int(float(k_target) * share) for share in bin_targets]
        bin_counts = [0, 0, 0]
        slack = 2

        chosen = []
        remaining = cand_idx.tolist()

        stage = 0
        while len(chosen) < int(k_target) and len(remaining) > 0:
            if stage == 0:
                need_quota = any(bin_counts[i] < bin_quota[i] - slack for i in range(3))
                if not need_quota:
                    stage = 1

            best_j = None
            best_sc = None

            for j in remaining:
                idx_in_keep = (keep == j).nonzero(as_tuple=True)[0]
                if idx_in_keep.numel() == 0:
                    continue
                idx_val = idx_in_keep[0].item()

                ang_deg = float((torch.rad2deg(ak[j]) % 180.0).item())
                if 60 <= ang_deg < 80:
                    bin_id = 0
                elif 80 <= ang_deg < 100:
                    bin_id = 1
                elif 100 <= ang_deg < 120:
                    bin_id = 2
                else:
                    bin_id = -1

                if stage == 0 and bin_id >= 0:
                    if bin_counts[bin_id] >= bin_quota[bin_id] + slack:
                        continue

                if len(chosen) == 0:
                    min_dist = 1.0
                else:
                    d = torch.norm(centers[j].unsqueeze(0) - centers[torch.as_tensor(chosen, device=device)], dim=1,
                                   p=2)
                    min_dist = float(d.min().item())

                s_space = min(max(min_dist / float(eff_radius.item()), 0.0), 1.0)

                idx_in_base = (cand_idx == j).nonzero(as_tuple=True)[0]
                bs = float(base_score[idx_in_base].item())
                sl = float(s_len_norm[idx_val])

                sc = 0.60 * bs + 0.25 * s_space + 0.15 * sl

                if stage == 0 and bin_id >= 0:
                    deficit = max(0.0, (bin_quota[bin_id] - bin_counts[bin_id]) / max(1.0, float(bin_quota[bin_id])))
                    sc = sc * (1.0 + 0.3 * deficit)

                if (best_sc is None) or (sc > best_sc):
                    best_sc = sc
                    best_j = j

            if best_j is None:
                break

            if len(chosen) == 0:
                chosen.append(best_j)
                ang_deg = float((torch.rad2deg(ak[best_j]) % 180.0).item())
                if 60 <= ang_deg < 80:
                    bin_counts[0] += 1
                elif 80 <= ang_deg < 100:
                    bin_counts[1] += 1
                elif 100 <= ang_deg < 120:
                    bin_counts[2] += 1
            else:
                d = torch.norm(centers[best_j].unsqueeze(0) - centers[torch.as_tensor(chosen, device=device)], dim=1,
                               p=2)
                if float(d.min().item()) >= 0.5 * float(eff_radius.item()):
                    chosen.append(best_j)
                    ang_deg = float((torch.rad2deg(ak[best_j]) % 180.0).item())
                    if 60 <= ang_deg < 80:
                        bin_counts[0] += 1
                    elif 80 <= ang_deg < 100:
                        bin_counts[1] += 1
                    elif 100 <= ang_deg < 120:
                        bin_counts[2] += 1

            remaining = [r for r in remaining if r != best_j]

        if len(remaining) == 0 and len(chosen) < int(k_target):
            fill = [i.item() for i in keep if i.item() not in chosen]
            for j in fill:
                if len(chosen) >= int(k_target):
                    break
                chosen.append(j)
    else:
        chosen = []
        remaining = cand_idx.tolist()
        while len(chosen) < int(k_target) and len(remaining) > 0:
            best_j = None
            best_sc = None
            for j in remaining:
                if len(chosen) == 0:
                    min_dist = 1.0
                else:
                    d = torch.norm(centers[j].unsqueeze(0) - centers[torch.as_tensor(chosen, device=device)], dim=1,
                                   p=2)
                    min_dist = float(d.min().item())

                s_space = min(max(min_dist / float(eff_radius.item()), 0.0), 1.0)

                idx_in_base = (cand_idx == j).nonzero(as_tuple=True)[0]
                bs = float(base_score[idx_in_base].item())

                idx_in_keep = (keep == j).nonzero(as_tuple=True)[0]
                sl = float(s_len_norm[idx_in_keep].item())

                sc = 0.60 * bs + 0.25 * s_space + 0.15 * sl

                if (best_sc is None) or (sc > best_sc):
                    best_sc = sc
                    best_j = j

            if best_j is None:
                break

            if len(chosen) == 0:
                chosen.append(best_j)
            else:
                d = torch.norm(centers[best_j].unsqueeze(0) - centers[torch.as_tensor(chosen, device=device)], dim=1,
                               p=2)
                if float(d.min().item()) >= 0.5 * float(eff_radius.item()):
                    chosen.append(best_j)

            remaining = [r for r in remaining if r != best_j]
            if len(remaining) == 0 and len(chosen) < int(k_target):
                fill = [i.item() for i in keep if i.item() not in chosen]
                for j in fill:
                    if len(chosen) >= int(k_target):
                        break
                    chosen.append(j)

    if len(chosen) == 0:
        chosen = cand_idx[:int(k_target)].tolist()

    chosen = torch.as_tensor(chosen[:int(k_target)], device=device, dtype=torch.long)
    return centers[chosen], lengths[chosen], ak[chosen]

def compute_geoq(X, Y, angle_sim, len_sim):
    cov = 0.5 * (X + Y)
    geom = 0.6 * angle_sim + 0.4 * len_sim
    return 0.5 * cov + 0.5 * geom


def l1_histogram_similarity(hist1, hist2):
    hist1_norm = hist1 / (hist1.sum() + 1e-8)
    hist2_norm = hist2 / (hist2.sum() + 1e-8)
    return 1.0 - 0.5 * torch.sum(torch.abs(hist1_norm - hist2_norm))


def compute_coverage(lines, W=None, H=None):
    W = W if W is not None else CONFIG["data"]["IMG_WIDTH"]
    H = H if H is not None else CONFIG["data"]["IMG_HEIGHT"]

    if len(lines) == 0:
        return 0.0, 0.0

    xs = []
    ys = []
    for line in lines:
        if isinstance(line, dict):
            xs.extend([line['start'][0], line['end'][0]])
            ys.extend([line['start'][1], line['end'][1]])
        else:
            xs.extend([line[0], line[2]])
            ys.extend([line[1], line[3]])

    x_cov = (max(xs) - min(xs)) / W if xs else 0.0
    y_cov = (max(ys) - min(ys)) / H if ys else 0.0
    return x_cov, y_cov


def connectivity_from_centers(centers_xy_norm, sigma=None):
    sigma = sigma if sigma is not None else CONFIG["physics"]["SIGMA_CONN"]

    if centers_xy_norm.shape[0] < 2:
        return 0.0
    D = torch.cdist(centers_xy_norm, centers_xy_norm, p=2)
    K = torch.exp(-D / sigma)
    K = K - torch.diag(torch.diag(K))
    n = centers_xy_norm.shape[0]
    return float(torch.triu(K, diagonal=1).sum() / (n * (n - 1) / 2.0 + 1e-8))


def stochastic_baseline_sampler(priors, W=None, H=None, n_samples=1):
    W = W if W is not None else CONFIG["data"]["IMG_WIDTH"]
    H = H if H is not None else CONFIG["data"]["IMG_HEIGHT"]

    results = []

    angle_hist   = np.array(priors['global_angle_hist'], dtype=np.float64)
    length_hist  = np.array(priors['global_length_hist'], dtype=np.float64)
    length_edges = np.array(priors['length_edges'], dtype=np.float64)
    count_stats  = priors['count_stats']

    for _ in range(n_samples):
        k = int(np.clip(max(1, int(np.random.normal(count_stats['mean'], max(1e-6, count_stats['std'])))),
                        count_stats['min'], count_stats['max']))

        lines = []
        for _ in range(k):
            angle_bin = np.random.choice(len(angle_hist), p=angle_hist / (angle_hist.sum() + 1e-12))
            theta_deg = angle_bin * 180.0 / len(angle_hist) + np.random.uniform(0, 180.0 / len(angle_hist))
            theta_rad = np.radians(theta_deg)

            length_bin = np.random.choice(len(length_hist), p=length_hist / (length_hist.sum() + 1e-12))
            L = length_edges[length_bin] + np.random.uniform(0, max(1e-12, length_edges[length_bin + 1] - length_edges[length_bin]))

            for _try in range(50):
                cx, cy = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
                dx = 0.5 * L * np.cos(theta_rad)
                dy = 0.5 * L * np.sin(theta_rad)

                x1, y1 = cx - dx, cy - dy
                x2, y2 = cx + dx, cy + dy

                if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
                    lines.append({'start': [x1 * W, y1 * H], 'end': [x2 * W, y2 * H]})
                    break

        results.append(lines)

    return results
