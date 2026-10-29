"""
transformer_v4.py  –  Task 3: retrain sequential transformer with MAX_CTX=80.

Dataset: reuse fragment_to_full_dataset.pt from Task 2.
  INPUT  = fragment tokens (sorted by cx)
  TARGET = lines outside the strip region  (sorted by cx)
Token format: (cx, cy, logL, sin θ, cos θ)  — same 5-dim as before.

Architecture: SeqFracTransformerV4
  D_MODEL=128, N_HEAD=4, N_LAYERS=3, DROPOUT=0.1  (same as v3)
  MAX_CTX=80  (was 40)
  MAX_GEN=80
  sd=12, gd=20 (matches v4 EVAE latent dims)

Training:
  Adam lr=3e-4, batch_size=8, max_epochs=50, early stop patience=10 on val position loss.
  Save to outputs/models/transformer_v4.pt
"""
from __future__ import annotations
import math, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(Path((__import__('os').environ['DFN3_CORE'] + r"\Evae_sequential_hybrid_revised".replace(chr(92), '/')))))

from exp_common import (get_physics_stats, assign_family_ids, N_MAX_EVAE,
                         split_panel, strips_for_interval, load_panels_raw)
from model import DualLatentVAE

DATA_PATH    = EXP_DIR / "outputs" / "data" / "fragment_to_full_dataset.pt"
MODEL_OUT    = EXP_DIR / "outputs" / "models" / "transformer_v4.pt"
LOG_OUT      = EXP_DIR / "outputs" / "results" / "task3_transformer_v4_training_log.txt"
EVAE_V4_CKPT = EXP_DIR / "outputs" / "models" / "evae_fragment_to_full_v4.pt"

MAX_CTX = 80
MAX_GEN = 80
D_MODEL = 128
N_HEAD  = 4
N_LAYERS = 3
DROPOUT = 0.1
N_FAM   = 4


# ── Tokenisation ─────────────────────────────────────────────────────────────
def line_to_token(line):
    """(cx, cy, L, A) → (cx, cy, logL, sin A, cos A)"""
    cx, cy, L, A = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    return [cx, cy, math.log(max(L, 1e-8)), math.sin(A), math.cos(A)]


def sort_by_cx(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0: return arr
    order = np.argsort(arr[:, 0])
    return arr[order]


# ── Dataset ──────────────────────────────────────────────────────────────────
class SeqV4Dataset(torch.utils.data.Dataset):
    """
    For each (panel, interval, aug) pair:
      ctx  = fragments (sorted cx)        → tokens + family
      tgt  = LINES OUTSIDE strips (from the full panel minus fragments)
             → Re-compute by taking full panel and removing fragments via
             center proximity, OR by re-clipping the full panel with its strips.
    We re-clip the full panel using strips_for_interval(iv) and take the
    target = outside-strip lines.
    """
    def __init__(self, pairs: List[Dict], fam_centers):
        self.fam_centers = np.asarray(fam_centers, dtype=float)
        self.seqs = []
        for p in pairs:
            # p['frag'] is inside-strip fragments
            # p['full'] is whole panel
            # Reconstruct the outside-strip target by clipping the full panel
            # with its (recomputed) strip layout.
            iv = float(p["interval"])
            strips = strips_for_interval(iv)
            measured, outside = split_panel(p["full"], strips)
            # Use the fragments directly as ctx (= measured from this data spec)
            ctx = sort_by_cx(p["frag"].astype(np.float32))
            tgt = sort_by_cx(outside.astype(np.float32))
            if len(tgt) == 0: continue
            ctx_t = np.array([line_to_token(l) for l in ctx], dtype=np.float32)
            tgt_t = np.array([line_to_token(l) for l in tgt], dtype=np.float32)
            ctx_f = assign_family_ids(ctx, self.fam_centers).astype(np.int64)
            tgt_f = assign_family_ids(tgt, self.fam_centers).astype(np.int64)
            # Truncate/pad sizes
            ctx_t = ctx_t[:MAX_CTX]
            ctx_f = ctx_f[:MAX_CTX]
            tgt_t = tgt_t[:MAX_GEN]
            tgt_f = tgt_f[:MAX_GEN]
            self.seqs.append({
                "ctx": ctx_t, "ctx_f": ctx_f,
                "tgt": tgt_t, "tgt_f": tgt_f,
            })

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def collate(batch):
    B = len(batch)
    cp = min(max(len(b["ctx"]) for b in batch), MAX_CTX)
    tp = min(max(len(b["tgt"]) for b in batch), MAX_GEN)
    ctx = torch.zeros(B, cp, 5, dtype=torch.float32)
    ctx_m = torch.ones(B, cp, dtype=torch.bool)    # True = pad
    tgt = torch.zeros(B, tp, 5, dtype=torch.float32)
    tgt_m = torch.ones(B, tp, dtype=torch.bool)
    tgt_fids = torch.zeros(B, tp, dtype=torch.long)
    for bi, b in enumerate(batch):
        nc = len(b["ctx"])
        nt = len(b["tgt"])
        ctx[bi, :nc] = torch.from_numpy(b["ctx"])
        ctx_m[bi, :nc] = False
        tgt[bi, :nt] = torch.from_numpy(b["tgt"])
        tgt_m[bi, :nt] = False
        tgt_fids[bi, :nt] = torch.from_numpy(b["tgt_f"])
    return ctx, ctx_m, tgt, tgt_m, tgt_fids


# ── Model (v4) ───────────────────────────────────────────────────────────────
class SeqFracTransformerV4(nn.Module):
    def __init__(self, sd=12, gd=20):
        super().__init__()
        self.sd = sd; self.gd = gd
        self.ctx_proj = nn.Sequential(nn.Linear(5, D_MODEL), nn.ReLU(),
                                        nn.Linear(D_MODEL, D_MODEL))
        self.tgt_proj = nn.Sequential(nn.Linear(5, D_MODEL), nn.ReLU(),
                                        nn.Linear(D_MODEL, D_MODEL))
        self.ctx_pos = nn.Parameter(torch.randn(1, MAX_CTX + 1, D_MODEL) * 0.02)
        self.tgt_pos = nn.Parameter(torch.randn(1, MAX_GEN, D_MODEL) * 0.02)
        enc_l = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL * 4, DROPOUT,
                                             batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_l, N_LAYERS)
        dec_l = nn.TransformerDecoderLayer(D_MODEL, N_HEAD, D_MODEL * 4, DROPOUT,
                                             batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_l, N_LAYERS)
        self.latent_proj = nn.Linear(sd + gd, D_MODEL)
        self.pos_head    = nn.Sequential(nn.Linear(D_MODEL, 64), nn.ReLU(),
                                           nn.Linear(64, 2))
        self.loglen_head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.ReLU(),
                                           nn.Linear(64, 1))
        self.ang_head    = nn.Sequential(nn.Linear(D_MODEL, 64), nn.ReLU(),
                                           nn.Linear(64, 2))
        self.fam_head    = nn.Sequential(nn.Linear(D_MODEL, 64), nn.ReLU(),
                                           nn.Linear(64, N_FAM))

    def encode(self, ctx, ctx_mask, sz=None, gz=None):
        B = ctx.size(0); n = ctx.size(1)
        x = self.ctx_proj(ctx)
        if sz is not None and gz is not None:
            latent = torch.cat([sz, gz], dim=-1)
            lt = self.latent_proj(latent).unsqueeze(1)
            x = torch.cat([lt, x], dim=1)
            pos = self.ctx_pos[:, : n + 1, :]
            ctx_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=ctx_mask.device),
                ctx_mask], dim=1)
        else:
            pos = self.ctx_pos[:, 1 : n + 1, :]
        x = x + pos
        memory = self.encoder(x, src_key_padding_mask=ctx_mask)
        return memory, ctx_mask

    def forward(self, ctx, ctx_mask, tgt, tgt_mask, sz=None, gz=None):
        memory, ctx_mask_out = self.encode(ctx, ctx_mask, sz, gz)
        n = tgt.size(1)
        x = self.tgt_proj(tgt) + self.tgt_pos[:, :n, :]
        causal = nn.Transformer.generate_square_subsequent_mask(n, device=tgt.device)
        out = self.decoder(x, memory, tgt_mask=causal,
                            tgt_key_padding_mask=tgt_mask,
                            memory_key_padding_mask=ctx_mask_out)
        pos = torch.sigmoid(self.pos_head(out))
        logL = self.loglen_head(out)
        angle = self.ang_head(out)
        family = self.fam_head(out)
        return pos, logL, angle, family


def compute_loss(pos_p, logL_p, ang_p, fam_p, tgt, tgt_fids, tgt_mask, ps):
    valid = ~tgt_mask
    pos_loss = F.mse_loss(pos_p[valid], tgt[:, :, :2][valid])
    len_loss = F.mse_loss(logL_p[valid], tgt[:, :, 2:3][valid])
    tgt_s = tgt[:, :, 3]; tgt_c = tgt[:, :, 4]
    ang_true = torch.atan2(tgt_s[valid], tgt_c[valid])
    ang_vec  = F.normalize(ang_p[valid], dim=-1, eps=1e-6)
    ang_pred = torch.atan2(ang_vec[:, 0], ang_vec[:, 1])
    ang_loss = (1.0 - torch.cos(ang_true - ang_pred)).mean()
    fam_loss = F.cross_entropy(fam_p[valid], tgt_fids[valid])
    total = pos_loss + 0.5 * len_loss + 0.3 * ang_loss + 0.3 * fam_loss
    return total, {"total": float(total.detach()),
                   "pos": float(pos_loss.detach()),
                   "len": float(len_loss.detach()),
                   "ang": float(ang_loss.detach()),
                   "fam": float(fam_loss.detach())}


# ── Training ─────────────────────────────────────────────────────────────────
def train_transformer_v4(max_epochs=50, batch_size=8, lr=3e-4,
                          grad_clip=1.0, patience=10, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  MAX_CTX={MAX_CTX}")

    data = torch.load(DATA_PATH, weights_only=False)
    ps   = get_physics_stats()
    fam_centers = ps["fam_centers_rad"]
    train_ds = SeqV4Dataset(data["train"], fam_centers)
    val_ds   = SeqV4Dataset(data["val"],   fam_centers)
    print(f"Train seqs: {len(train_ds)}  Val seqs: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size,
                                                 shuffle=False, collate_fn=collate)

    # Build EVAE v4 to extract sz/gz latents for conditioning
    evae_v4 = DualLatentVAE(sd=12, gd=20, N=N_MAX_EVAE, yscale=ps["yscale"]).to(device)
    ck_ev = torch.load(EVAE_V4_CKPT, map_location=device, weights_only=False)
    evae_v4.load_state_dict(ck_ev["model_state_dict"])
    evae_v4.eval()

    model = SeqFracTransformerV4().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def get_latents(ctx_tokens, ctx_mask):
        """Derive sz,gz from fragment context via v4 EVAE encoder (detokenised)."""
        B = ctx_tokens.size(0)
        # tokens are (cx, cy, logL, sin, cos) → reconstruct (cx,cy,L,A)
        cx = ctx_tokens[:, :, 0]
        cy = ctx_tokens[:, :, 1]
        L  = torch.exp(ctx_tokens[:, :, 2])
        A  = torch.atan2(ctx_tokens[:, :, 3], ctx_tokens[:, :, 4]) % math.pi
        C = torch.stack([cx, cy], dim=-1)
        Lv = L.unsqueeze(-1); Av = A.unsqueeze(-1)
        mk = ~ctx_mask
        # Pad/truncate to N_MAX_EVAE
        n_in = min(C.size(1), N_MAX_EVAE)
        C  = C[:, :n_in]; Lv = Lv[:, :n_in]; Av = Av[:, :n_in]; mk = mk[:, :n_in]
        if C.size(1) < N_MAX_EVAE:
            pad_n = N_MAX_EVAE - C.size(1)
            C  = F.pad(C,  (0,0,0,pad_n))
            Lv = F.pad(Lv, (0,0,0,pad_n))
            Av = F.pad(Av, (0,0,0,pad_n))
            mk = F.pad(mk, (0,pad_n), value=False)
        with torch.no_grad():
            sm, _, gm, _ = evae_v4.encode(C, Lv, Av, mk)
        return sm, gm

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience_ctr = 0
    log_lines = ["epoch,train_total,val_pos,val_total"]
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_stats = {"total":0.0, "pos":0.0, "len":0.0, "ang":0.0, "fam":0.0}
        n_b = 0
        for ctx, ctx_m, tgt, tgt_m, tgt_fids in train_loader:
            ctx = ctx.to(device); ctx_m = ctx_m.to(device)
            tgt = tgt.to(device); tgt_m = tgt_m.to(device); tgt_fids = tgt_fids.to(device)
            sz, gz = get_latents(ctx, ctx_m)
            # Teacher forcing with proper shift: decoder input = <bos> + tgt[:-1]
            # Labels = tgt. This prevents the degenerate identity mapping.
            dec_in = torch.cat([
                torch.zeros(tgt.size(0), 1, tgt.size(2), device=tgt.device),
                tgt[:, :-1, :]
            ], dim=1)
            pos_p, logL_p, ang_p, fam_p = model(ctx, ctx_m, dec_in, tgt_m, sz=sz, gz=gz)
            loss, stats = compute_loss(pos_p, logL_p, ang_p, fam_p,
                                        tgt, tgt_fids, tgt_m, ps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            for k in ep_stats: ep_stats[k] += stats[k]
            n_b += 1
        for k in ep_stats: ep_stats[k] /= max(n_b, 1)

        # Validation
        model.eval()
        val_pos_loss = 0.0; val_total = 0.0; n_vb = 0
        with torch.no_grad():
            for ctx, ctx_m, tgt, tgt_m, tgt_fids in val_loader:
                ctx = ctx.to(device); ctx_m = ctx_m.to(device)
                tgt = tgt.to(device); tgt_m = tgt_m.to(device); tgt_fids = tgt_fids.to(device)
                sz, gz = get_latents(ctx, ctx_m)
                dec_in = torch.cat([
                    torch.zeros(tgt.size(0), 1, tgt.size(2), device=tgt.device),
                    tgt[:, :-1, :]
                ], dim=1)
                pos_p, logL_p, ang_p, fam_p = model(ctx, ctx_m, dec_in, tgt_m, sz=sz, gz=gz)
                loss, stats = compute_loss(pos_p, logL_p, ang_p, fam_p,
                                            tgt, tgt_fids, tgt_m, ps)
                val_pos_loss += stats["pos"]; val_total += stats["total"]; n_vb += 1
        val_pos_loss /= max(n_vb, 1); val_total /= max(n_vb, 1)

        log_lines.append(f"{epoch},{ep_stats['total']:.4f},{val_pos_loss:.4f},{val_total:.4f}")
        if epoch % 5 == 0 or epoch == 1:
            print(f"ep{epoch:03d}  train_total={ep_stats['total']:.3f}  "
                  f"val_pos={val_pos_loss:.4f}  val_total={val_total:.4f}")

        if val_pos_loss < best_val:
            best_val = val_pos_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stop at epoch {epoch} (best={best_epoch})")
                break

    torch.save({
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "val_pos_loss": best_val,
        "physics_stats": ps,
        "config": {"MAX_CTX": MAX_CTX, "MAX_GEN": MAX_GEN,
                    "D_MODEL": D_MODEL, "N_HEAD": N_HEAD, "N_LAYERS": N_LAYERS},
    }, MODEL_OUT)
    dt = time.time() - t0
    print(f"Best epoch={best_epoch}  val_pos={best_val:.4f}")
    print(f"Saved: {MODEL_OUT}")
    print(f"Training time: {dt:.1f}s")
    LOG_OUT.write_text("\n".join(log_lines), encoding="utf-8")
    return best_val, best_epoch


# ── Inference wrapper ────────────────────────────────────────────────────────
def generate_v4(model, evae_v4, fragments, n_gen, ps, device, strips=None,
                temperature: float = 1.0, seed: int = 0):
    """
    Autoregressive generation using v4 transformer + v4 EVAE latent.

    temperature: noise multiplier applied ONLY to spatial position (cx, cy)
        and length (logL) to break the mode-collapse pattern where
        adjacent cx values cluster too tightly (observed in Task 3).
        Angle is NEVER perturbed — it's already within 1-3° of GT and
        must stay tight to the family centre.
        T=0   : deterministic (original behaviour)
        T=1.0 : noise scale 0.05 norm units ≈ 1.8 m on cx/cy, ~5% on L
        T=2.0 : 2x that
    """
    if len(fragments) == 0 or n_gen < 1:
        return np.zeros((0, 4), dtype=np.float32)
    fam_centers = np.asarray(ps["fam_centers_rad"], dtype=float)
    # Sort ctx by cx, build tokens
    arr = np.asarray(fragments, dtype=np.float32)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    tokens = np.array([line_to_token(l) for l in arr[:MAX_CTX]], dtype=np.float32)
    n_in = len(tokens)
    ctx = torch.from_numpy(tokens).unsqueeze(0).to(device)
    ctx_m = torch.zeros(1, n_in, dtype=torch.bool, device=device)

    # EVAE latent from fragments
    C = torch.from_numpy(arr[:n_in, 0:2].reshape(1, n_in, 2)).to(device)
    Lv = torch.from_numpy(arr[:n_in, 2:3].reshape(1, n_in, 1)).to(device)
    Av = torch.from_numpy(arr[:n_in, 3:4].reshape(1, n_in, 1)).to(device)
    mk_e = torch.ones(1, n_in, dtype=torch.bool, device=device)
    # pad to N_MAX_EVAE=120
    if n_in < N_MAX_EVAE:
        C  = F.pad(C, (0,0,0,N_MAX_EVAE-n_in))
        Lv = F.pad(Lv,(0,0,0,N_MAX_EVAE-n_in))
        Av = F.pad(Av,(0,0,0,N_MAX_EVAE-n_in))
        mk_e = F.pad(mk_e,(0,N_MAX_EVAE-n_in),value=False)
    with torch.no_grad():
        sm, _, gm, _ = evae_v4.encode(C, Lv, Av, mk_e)

    model.eval()
    y_max = float(ps.get("yscale", 0.315))
    generated_lines = []
    np.random.seed(seed)  # for compatibility with old sampling functions

    # Data-derived sampling parameters
    fam_centers_rad = np.asarray(ps.get("fam_centers_rad", [0.62, 1.17, 2.22]), dtype=float)
    # FIX 5: Use actual-angle-space kappa (not doubled-angle from ps["fam_kappa"]).
    # These match the statistical branch in hybrid_v6.py and give ~18-20° spread.
    fam_kappa = np.array([8.69, 9.51, 10.61], dtype=float)
    fam_logmu = np.asarray(ps.get("fam_logmu", [-2.3, -2.3, -2.0]), dtype=float)
    fam_logsig = np.asarray(ps.get("fam_logsig", [0.4, 0.4, 0.4]), dtype=float)
    cx_std = max(float(ps.get("cx_std", 0.2)) * 0.2, 0.01)
    cy_std = max(float(ps.get("cy_std", 0.06)) * 0.2, 0.003)
    N_CANDIDATES = 24
    strips_eff = strips if strips is not None else []

    def _angle_diff(a, b):
        d = abs(float(a) - float(b))
        return min(d, math.pi - d)

    def _sample_angle_posterior(mu_pred, fam):
        """Von Mises posterior: combine model prediction with family prior."""
        fam_idx = int(np.clip(fam, 0, 2))
        mu_prior = float(fam_centers_rad[fam_idx])
        k_prior = float(np.clip(fam_kappa[fam_idx], 0.1, 100.0))
        k_model = max(1.0, 0.5 * k_prior)
        vx = k_model * np.cos(mu_pred) + k_prior * np.cos(mu_prior)
        vy = k_model * np.sin(mu_pred) + k_prior * np.sin(mu_prior)
        k_post = max(float(np.hypot(vx, vy)), 1e-3)
        mu_post = float(np.arctan2(vy, vx))
        return float(np.random.vonmises(mu_post, k_post) % np.pi)

    def _sample_loglen_posterior(mu_pred, fam):
        """Gaussian posterior: combine model prediction with family prior."""
        if fam <= 2:
            fam_idx = int(np.clip(fam, 0, 2))
            mu_prior = float(fam_logmu[fam_idx])
            sig_prior = float(fam_logsig[fam_idx])
        else:
            mu_prior = float(ps.get("rand_logmu", -2.0))
            sig_prior = float(ps.get("rand_logsig", 0.4))
        sig_prior = max(sig_prior, 0.08)
        sig_model = max(sig_prior * 0.7, 0.06)
        p_model = 1.0 / (sig_model ** 2)
        p_prior = 1.0 / (sig_prior ** 2)
        mu_post = (mu_pred * p_model + mu_prior * p_prior) / (p_model + p_prior)
        sig_post = np.sqrt(1.0 / (p_model + p_prior))
        return float(np.random.normal(mu_post, sig_post))

    def _loglen_posterior_stats(mu_pred, fam):
        if fam <= 2:
            fam_idx = int(np.clip(fam, 0, 2))
            mu_prior = float(fam_logmu[fam_idx])
            sig_prior = float(fam_logsig[fam_idx])
        else:
            mu_prior = float(ps.get("rand_logmu", -2.0))
            sig_prior = float(ps.get("rand_logsig", 0.4))
        sig_prior = max(sig_prior, 0.08)
        sig_model = max(sig_prior * 0.7, 0.06)
        p_model = 1.0 / (sig_model ** 2)
        p_prior = 1.0 / (sig_prior ** 2)
        mu_post = (mu_pred * p_model + mu_prior * p_prior) / (p_model + p_prior)
        sig_post = np.sqrt(1.0 / (p_model + p_prior))
        return float(mu_post), float(sig_post)

    # Multi-candidate autoregressive generation
    # Ported from sequential_transformer.py:560-644
    tgt = torch.zeros(1, 1, 5, device=device)
    for step in range(min(n_gen, MAX_GEN)):
        tgt_m = torch.zeros(1, tgt.size(1), dtype=torch.bool, device=device)
        with torch.no_grad():
            pos_p, logL_p, ang_p, fam_p = model(ctx, ctx_m, tgt, tgt_m, sz=sm, gz=gm)

        # Model's predicted distributions for this step
        last_pos = pos_p[0, -1, :].detach().cpu().numpy()
        last_logL = float(logL_p[0, -1, 0].item())
        last_ang = ang_p[0, -1, :].detach().cpu().numpy()
        last_fam = fam_p[0, -1, :].detach().cpu().numpy()

        # Model family probabilities (from learned family head)
        fam_probs = np.exp(last_fam - np.max(last_fam))
        fam_probs = fam_probs / np.maximum(fam_probs.sum(), 1e-12)

        # Model predicted angle (centre of posterior)
        ang_vec = last_ang / max(float(np.linalg.norm(last_ang)), 1e-6)
        mu_pred = float(np.arctan2(ang_vec[0], ang_vec[1]) % np.pi)

        # Sample N_CANDIDATES candidates from learned posteriors
        best = None
        best_score = np.inf
        for _k in range(N_CANDIDATES):
            # Family: sampled from model's learned family head
            fam = int(np.random.choice(N_FAM, p=fam_probs))
            # Angle: von Mises posterior (model prediction + family prior)
            A_rad = _sample_angle_posterior(mu_pred, fam)
            # Length: Gaussian posterior (model prediction + family prior)
            logL_cand = _sample_loglen_posterior(last_logL, fam)
            L = float(np.exp(logL_cand))
            # Position: Normal around model's predicted position
            cx = float(np.clip(np.random.normal(float(last_pos[0]),
                        cx_std * float(temperature)), 0.0, 1.0))
            cy = float(np.clip(np.random.normal(float(last_pos[1]),
                        cy_std * float(temperature)), 0.0, float(y_max)))
            # Strip exclusion
            if strips_eff and any(float(lo) <= cx <= float(hi) for lo, hi in strips_eff):
                continue
            # Bounds check
            x1 = cx - 0.5 * L * math.cos(A_rad)
            y1 = cy - 0.5 * L * math.sin(A_rad)
            x2 = cx + 0.5 * L * math.cos(A_rad)
            y2 = cy + 0.5 * L * math.sin(A_rad)
            if x1 < -1e-3 or x2 < -1e-3 or x1 > 1.0 + 1e-3 or x2 > 1.0 + 1e-3:
                continue
            if y1 < -1e-3 or y2 < -1e-3 or y1 > y_max + 1e-3 or y2 > y_max + 1e-3:
                continue
            # Score: sum of squared z-scores (lower = more probable)
            d_ang = _angle_diff(A_rad, mu_pred)
            fam_ll = -np.log(max(float(fam_probs[fam]), 1e-8))
            mu_lp, sig_lp = _loglen_posterior_stats(last_logL, fam)
            score = (
                ((cx - float(last_pos[0])) / cx_std) ** 2
                + ((cy - float(last_pos[1])) / cy_std) ** 2
                + ((logL_cand - mu_lp) / max(sig_lp, 0.08)) ** 2
                + (d_ang / np.deg2rad(12.0)) ** 2
                + fam_ll
            )
            if score < best_score:
                best_score = score
                best = (cx, cy, L, A_rad, fam)

        # Fallback if all candidates rejected
        if best is None:
            fam = int(np.argmax(fam_probs))
            A_rad = float(mu_pred)
            L = float(np.exp(last_logL))
            cx = float(np.clip(float(last_pos[0]), 0.0, 1.0))
            cy = float(np.clip(float(last_pos[1]), 0.0, float(y_max)))
            best = (cx, cy, L, A_rad, fam)

        cx, cy, L, A_rad, fam = best

        # ALWAYS advance decoder state (even if line filtered out later)
        new_tok = torch.tensor([[[cx, cy, math.log(max(L, 1e-8)),
                                   math.sin(A_rad), math.cos(A_rad)]]],
                                device=device, dtype=torch.float32)
        tgt = torch.cat([tgt, new_tok], dim=1)

        # Output filters
        if L < 1e-4:
            continue
        x1 = cx - 0.5 * L * math.cos(A_rad)
        x2 = cx + 0.5 * L * math.cos(A_rad)
        y1 = cy - 0.5 * L * math.sin(A_rad)
        y2 = cy + 0.5 * L * math.sin(A_rad)
        if min(x1, x2) < -0.02 or max(x1, x2) > 1.02:
            continue
        if min(y1, y2) < -0.02 or max(y1, y2) > y_max + 0.02:
            continue
        if strips_eff and any(float(lo) <= cx <= float(hi) for lo, hi in strips_eff):
            continue
        generated_lines.append([cx, cy, L, A_rad])
    return (np.array(generated_lines, dtype=np.float32) if generated_lines
            else np.zeros((0, 4), dtype=np.float32))


if __name__ == "__main__":
    train_transformer_v4()
