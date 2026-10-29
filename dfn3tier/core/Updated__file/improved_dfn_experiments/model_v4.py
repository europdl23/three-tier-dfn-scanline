"""
model_v4.py  –  EVAE v4 retraining on clipped-fragment-to-full-panel pairs.

Architecture: identical DualLatentVAE from Evae_sequential_hybrid_revised/model.py
What's new:
  - Dataset: (input=clipped fragments, target=full panel) instead of (full, full)
  - Hungarian matching cost: 1.0·centre + 0.5·|ΔL| + 0.3·(family mismatch)
  - Per-family length MSE weights: F0=1.0, F1=1.5, F2=2.5, random=1.0
  - KL anneal β 0→0.3 over 30 epochs, capped at 0.3 (anti posterior-collapse)
  - Early stop on val F1 length error (patience=20)
  - Saves to outputs/models/evae_fragment_to_full_v4.pt
"""
from __future__ import annotations
import math, sys, json, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(Path((__import__('os').environ['DFN3_CORE'] + r"\Evae_sequential_hybrid_revised".replace(chr(92), '/')))))

from exp_common import (get_physics_stats, assign_family_ids, N_MAX_EVAE,
                         EVAE_CKPT_LEGACY)
from model import DualLatentVAE, kl_loss

DATA_PATH   = EXP_DIR / "outputs" / "data" / "fragment_to_full_dataset.pt"
MODEL_OUT   = EXP_DIR / "outputs" / "models" / "evae_fragment_to_full_v4.pt"
LOG_OUT     = EXP_DIR / "outputs" / "results" / "task2_evae_v4_training_log.txt"
VAL_OUT     = EXP_DIR / "outputs" / "results" / "task2_evae_v4_validation.txt"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
VAL_OUT.parent.mkdir(parents=True, exist_ok=True)

FAM_LEN_WEIGHTS = torch.tensor([1.0, 1.5, 2.5, 1.0], dtype=torch.float32)


# ── Fragment → full batch collator ───────────────────────────────────────────
def pad_to(arr: np.ndarray, N: int, dim: int) -> np.ndarray:
    out = np.zeros((N, dim), dtype=np.float32)
    n = min(len(arr), N)
    out[:n] = arr[:n]
    return out, n


class FragmentDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[Dict], N_IN: int = 120, N_OUT: int = 120):
        self.pairs = pairs
        self.N_IN  = N_IN
        self.N_OUT = N_OUT

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        frag = p["frag"]  # (n_frag, 4) [cx,cy,L,A]
        full = p["full"]  # (n_full, 4)
        fam_full = p["fam_full"]  # (n_full,)

        # Build input tensors for encoder (pad to N_IN)
        C_in   = np.zeros((self.N_IN, 2), dtype=np.float32)
        L_in   = np.zeros((self.N_IN, 1), dtype=np.float32)
        A_in   = np.zeros((self.N_IN, 1), dtype=np.float32)
        mask   = np.zeros((self.N_IN,),    dtype=bool)
        n_in   = min(len(frag), self.N_IN)
        C_in[:n_in] = frag[:n_in, 0:2]
        L_in[:n_in, 0] = frag[:n_in, 2]
        A_in[:n_in, 0] = frag[:n_in, 3]
        mask[:n_in] = True

        # Target (padded)
        C_tgt   = np.zeros((self.N_OUT, 2), dtype=np.float32)
        L_tgt   = np.zeros((self.N_OUT, 1), dtype=np.float32)
        A_tgt   = np.zeros((self.N_OUT, 1), dtype=np.float32)
        F_tgt   = np.full( (self.N_OUT,),   -1, dtype=np.int64)
        E_tgt   = np.zeros((self.N_OUT,),    dtype=np.float32)
        n_out = min(len(full), self.N_OUT)
        C_tgt[:n_out] = full[:n_out, 0:2]
        L_tgt[:n_out, 0] = full[:n_out, 2]
        A_tgt[:n_out, 0] = full[:n_out, 3]
        F_tgt[:n_out] = fam_full[:n_out]
        E_tgt[:n_out] = 1.0

        return {
            "C_in": torch.from_numpy(C_in),
            "L_in": torch.from_numpy(L_in),
            "A_in": torch.from_numpy(A_in),
            "mask": torch.from_numpy(mask),
            "C_tgt": torch.from_numpy(C_tgt),
            "L_tgt": torch.from_numpy(L_tgt),
            "A_tgt": torch.from_numpy(A_tgt),
            "F_tgt": torch.from_numpy(F_tgt),
            "E_tgt": torch.from_numpy(E_tgt),
            "n_in":  n_in,
            "n_out": n_out,
        }


# ── Hungarian matcher (centre + length + family) ─────────────────────────────
def hungarian_match_batch(pc, pl, pf_logits, C_tgt, L_tgt, F_tgt, E_tgt,
                           mask_valid):
    """
    pc: (B, N, 2)  predicted centres
    pl: (B, N, 1)  predicted sigmoid-scaled length (in [0,1])
    pf_logits: (B, N, 4)  family logits (F0..F2 + random-as-4th)
    C_tgt, L_tgt, F_tgt: ground truth padded
    E_tgt: (B, N) mask of which targets are real (1.0) vs padding (0.0)
    Returns list of (pred_idx, tgt_idx) per batch item.
    """
    B, N, _ = pc.shape
    assignments = []
    for b in range(B):
        n_tgt = int(E_tgt[b].sum().item())
        if n_tgt == 0:
            assignments.append((np.array([], dtype=int), np.array([], dtype=int)))
            continue
        c_pred = pc[b].detach()            # (N, 2)
        l_pred = pl[b, :, 0].detach()      # (N,)   in [0,1]
        f_pred = F.softmax(pf_logits[b, :, :3], dim=-1).detach()  # (N, 3) — systematic only
        c_tgt  = C_tgt[b, :n_tgt]          # (n_tgt, 2)
        l_tgt  = L_tgt[b, :n_tgt, 0]       # (n_tgt,)
        f_tgt  = F_tgt[b, :n_tgt]          # (n_tgt,)
        # Cost matrix
        # centre L2
        dist = torch.cdist(c_pred.unsqueeze(0), c_tgt.unsqueeze(0))[0]  # (N, n_tgt)
        # length
        lcost = (l_pred.unsqueeze(1) - l_tgt.unsqueeze(0)).abs()        # (N, n_tgt)
        # family mismatch  (treat GT "random"=3 as any, cost = 0 for all preds)
        f_tgt_clip = f_tgt.clamp(0, 2)
        # correct-class prob per prediction/target pair
        f_probs = f_pred.gather(1, f_tgt_clip.unsqueeze(0).expand(N, -1))  # (N, n_tgt)
        f_cost  = 1.0 - f_probs
        # Where target is random family (3), don't penalise family mismatch
        is_random = (f_tgt == 3).float().unsqueeze(0).expand(N, -1)
        f_cost   = f_cost * (1.0 - is_random)

        cost = (1.0 * dist + 0.5 * lcost + 0.3 * f_cost).cpu().numpy()
        pi, gi = linear_sum_assignment(cost)
        assignments.append((pi, gi))
    return assignments


# ── Full loss ────────────────────────────────────────────────────────────────
def compute_loss(pred_out, batch, beta_kl: float,
                  fam_len_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    pc = pred_out["pc"]
    pl = pred_out["pl"]
    ml = pred_out["ml"]        # (B, N, 4) family logits (3 fam + random slot)
    kp = pred_out["kp"]        # (B, N, 4) von Mises concentrations
    el = pred_out["el"]        # (B, N) existence logits
    sm, gm = pred_out["sm"], pred_out["gm"]
    gl = pred_out["gl"]

    C_tgt = batch["C_tgt"].to(pc.device)
    L_tgt = batch["L_tgt"].to(pc.device)
    A_tgt = batch["A_tgt"].to(pc.device)
    F_tgt = batch["F_tgt"].to(pc.device)
    E_tgt = batch["E_tgt"].to(pc.device)

    # Hungarian match
    assigns = hungarian_match_batch(pc, pl, ml, C_tgt, L_tgt, F_tgt, E_tgt, E_tgt)

    B, N, _ = pc.shape
    loss_centre = 0.0
    loss_length = 0.0
    loss_family = 0.0
    loss_angle  = 0.0
    n_matches   = 0

    for b, (pi, gi) in enumerate(assigns):
        if len(pi) == 0:
            continue
        pi_t = torch.from_numpy(pi).to(pc.device)
        gi_t = torch.from_numpy(gi).to(pc.device)
        # centre
        loss_centre += F.mse_loss(pc[b, pi_t], C_tgt[b, gi_t], reduction="sum")
        # length (weighted per family)
        gt_L = L_tgt[b, gi_t, 0]
        pr_L = pl[b, pi_t, 0]
        gt_f = F_tgt[b, gi_t].clamp(0, 3)
        w    = fam_len_weights.to(pc.device)[gt_f]
        loss_length += (w * (pr_L - gt_L).pow(2)).sum()
        # family CE (3-way on systematic; random gets -log(1-max(p3)) )
        fam_logits_sys = ml[b, pi_t, :3]
        gt_f_sys = gt_f.clamp(0, 2)
        is_random = (gt_f == 3)
        if (~is_random).any():
            loss_family += F.cross_entropy(
                fam_logits_sys[~is_random],
                gt_f_sys[~is_random],
                reduction="sum",
            )
        # angle NLL for matched
        gt_A = A_tgt[b, gi_t, 0]
        p3   = F.softmax(fam_logits_sys, dim=-1)
        kpr  = kp[b, pi_t, :3].clamp(0.1, 30.0)
        fct  = torch.tensor([0.623, 1.165, 2.221], device=pc.device)
        nll_k = []
        for k in range(3):
            diff = gt_A - fct[k]
            # Work in axial sense: align diff to [-pi/2, pi/2]
            diff = torch.remainder(diff + math.pi/2, math.pi) - math.pi/2
            nll_i = -kpr[:, k] * torch.cos(2*diff)  # doubled angle for axial
            nll_k.append(nll_i)
        nll_stack = torch.stack(nll_k, dim=-1)     # (n_match, 3)
        loss_angle += (p3 * nll_stack).sum()
        n_matches += len(pi)

    norm = max(n_matches, 1)
    loss_centre /= norm
    loss_length /= norm
    loss_family /= norm
    loss_angle  /= norm

    # Existence BCE
    el_pred = el  # (B, N) logits
    # Build ground-truth existence mask from Hungarian matches
    E_match = torch.zeros_like(el_pred)
    for b, (pi, _gi) in enumerate(assigns):
        if len(pi) > 0:
            E_match[b, torch.from_numpy(pi).to(el_pred.device)] = 1.0
    loss_exist = F.binary_cross_entropy_with_logits(el_pred, E_match)

    # KL (β-weighted)
    kl_s = kl_loss(sm, torch.zeros_like(sm))
    kl_g = kl_loss(gm, gl)
    loss_kl = beta_kl * (kl_s + kl_g)

    total = (loss_centre + loss_length + 0.3*loss_family +
             0.1*loss_angle + 0.5*loss_exist + loss_kl)

    return total, {
        "total": float(total.detach()),
        "centre": float(loss_centre.detach()) if isinstance(loss_centre, torch.Tensor) else float(loss_centre),
        "length": float(loss_length.detach()) if isinstance(loss_length, torch.Tensor) else float(loss_length),
        "family": float(loss_family.detach()) if isinstance(loss_family, torch.Tensor) else float(loss_family),
        "angle":  float(loss_angle.detach())  if isinstance(loss_angle,  torch.Tensor) else float(loss_angle),
        "exist":  float(loss_exist.detach()),
        "kl":     float(loss_kl.detach()),
    }


# ── Validation: per-family length error ──────────────────────────────────────
def evaluate(model, val_loader, ps, device):
    model.eval()
    fam_centers = torch.tensor(ps["fam_centers_rad"], device=device)
    per_fam_lens = {0: [], 1: [], 2: []}
    per_fam_lens_gt = {0: [], 1: [], 2: []}
    per_fam_angs = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for batch in val_loader:
            C_in = batch["C_in"].to(device); L_in = batch["L_in"].to(device)
            A_in = batch["A_in"].to(device); mask = batch["mask"].to(device)
            out = model(C_in, L_in, A_in, mask, mode="B")
            pc = out["pc"]; pl = out["pl"]; el = out["el"]; ml = out["ml"]
            B = pc.size(0)
            for b in range(B):
                # Model v4 directly outputs normalised length via sigmoid
                # (trained with raw gt_L in [0, ~0.3]), so no length_max scaling.
                el_b = el[b]
                pl_b = pl[b, :, 0]
                pc_b = pc[b]
                # family argmax
                ml_probs = F.softmax(ml[b, :, :3], dim=-1)
                fam_pred = ml_probs.argmax(-1)
                # Take top-N real existence lines where N = batch n_out
                n_out = batch["n_out"][b].item()
                top_idx = torch.topk(torch.sigmoid(el_b), k=n_out).indices
                sel_lens = pl_b[top_idx]
                sel_fams = fam_pred[top_idx]
                for f in range(3):
                    fl = sel_lens[sel_fams == f]
                    if len(fl):
                        per_fam_lens[f].append(float(fl.mean().item()))
                # GT fam lengths
                F_tgt = batch["F_tgt"][b]
                L_tgt = batch["L_tgt"][b, :, 0]
                E_tgt = batch["E_tgt"][b]
                for f in range(3):
                    mask_f = (F_tgt == f) & (E_tgt > 0.5)
                    if mask_f.any():
                        per_fam_lens_gt[f].append(float(L_tgt[mask_f].mean().item()))
    res = {}
    for f in range(3):
        gen_mean = float(np.mean(per_fam_lens[f])) if per_fam_lens[f] else 0.0
        gt_mean  = float(np.mean(per_fam_lens_gt[f])) if per_fam_lens_gt[f] else 0.0
        res[f"F{f}_gen_len"] = gen_mean
        res[f"F{f}_gt_len"]  = gt_mean
        res[f"F{f}_err"]     = abs(gen_mean - gt_mean)
    res["primary"] = res["F1_err"]  # early stopping key
    return res


# ── Training loop ────────────────────────────────────────────────────────────
def train_evae_v4(
    max_epochs: int = 150,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    beta_max: float = 0.3,
    beta_warmup: int = 30,
    patience: int = 20,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = torch.load(DATA_PATH, weights_only=False)
    ps   = get_physics_stats()
    train_ds = FragmentDataset(data["train"], N_IN=N_MAX_EVAE, N_OUT=N_MAX_EVAE)
    val_ds   = FragmentDataset(data["val"],   N_IN=N_MAX_EVAE, N_OUT=N_MAX_EVAE)
    print(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader   = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model (fresh or warm-start from legacy)
    model = DualLatentVAE(sd=12, gd=20, N=N_MAX_EVAE, yscale=ps["yscale"]).to(device)
    if EVAE_CKPT_LEGACY.exists():
        print(f"Warm-starting from {EVAE_CKPT_LEGACY}")
        try:
            model.load_state_dict(torch.load(EVAE_CKPT_LEGACY, map_location=device),
                                   strict=False)
        except Exception as e:
            print(f"  warm start failed, training from scratch: {e}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience_ctr = 0
    log_lines = ["epoch,beta,train_total,train_centre,train_len,val_primary,F0_gen,F1_gen,F2_gen"]
    t0 = time.time()

    for epoch in range(1, max_epochs+1):
        beta = beta_max * min(1.0, epoch / beta_warmup)
        model.train()
        ep_stats = {"total":0.0, "centre":0.0, "length":0.0, "family":0.0}
        n_b = 0
        for batch in train_loader:
            C_in = batch["C_in"].to(device); L_in = batch["L_in"].to(device)
            A_in = batch["A_in"].to(device); mask = batch["mask"].to(device)
            out = model(C_in, L_in, A_in, mask, mode="B")
            loss, stats = compute_loss(out, batch, beta, FAM_LEN_WEIGHTS)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            for k in ep_stats: ep_stats[k] += stats[k]
            n_b += 1
        for k in ep_stats: ep_stats[k] /= max(n_b, 1)

        val_res = evaluate(model, val_loader, ps, device)
        primary = val_res["primary"]

        log_line = (f"{epoch},{beta:.3f},{ep_stats['total']:.4f},"
                    f"{ep_stats['centre']:.4f},{ep_stats['length']:.4f},"
                    f"{primary:.6f},{val_res['F0_gen_len']:.4f},"
                    f"{val_res['F1_gen_len']:.4f},{val_res['F2_gen_len']:.4f}")
        log_lines.append(log_line)
        if epoch % 5 == 0 or epoch == 1:
            print(f"ep{epoch:03d} beta={beta:.2f}  train={ep_stats['total']:.3f} "
                  f"val_F1err={primary:.4f}  "
                  f"F0_gen={val_res['F0_gen_len']:.3f} "
                  f"F1_gen={val_res['F1_gen_len']:.3f} "
                  f"F2_gen={val_res['F2_gen_len']:.3f}")

        if primary < best_val:
            best_val = primary
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch} (best was ep{best_epoch})")
                break

    # Save best
    torch.save({
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "val_F1_err": best_val,
        "physics_stats": ps,
        "config": {
            "sd": 12, "gd": 20, "N": N_MAX_EVAE,
            "beta_max": beta_max, "beta_warmup": beta_warmup,
            "lr": lr, "batch_size": batch_size,
        },
    }, MODEL_OUT)
    dt = time.time() - t0
    print(f"\nBest epoch: {best_epoch}  val F1 err: {best_val:.4f}")
    print(f"Saved: {MODEL_OUT}")
    print(f"Training time: {dt:.1f}s")

    LOG_OUT.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Training log: {LOG_OUT}")
    return best_val, best_epoch


if __name__ == "__main__":
    train_evae_v4()
