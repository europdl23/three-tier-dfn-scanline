# -*- coding: utf-8 -*-
"""GAST - geometry-aware set transformer for tier 3 (inferred) fractures.
Sequence = [set tokens (one per trace-direction cluster) | BOS | evidence traces (measured + indicated) | generated traces].
Every position attends to the set tokens and the evidence; generated positions attend causally to earlier generated
positions (prefix model). Attention scores = QK^T / sqrt(d) + b(edge), where b is a learned per-head bias computed from
explicit pair geometry (segment distance, angle difference, crossing, overlap along strike, parallel gap, relative offset).
Token embedding = trace features + Fourier encoding of midpoint and both endpoints + learned set embedding + learned tier
embedding. Set tokens carry the cluster's distribution from the training benches (mean direction, spread, share, length
quantiles, median spacing). Heads: next trace as mixtures (2D position, log length, axial von Mises on 2 theta), set, stop.
Loss: mixture NLL + set CE + stop BCE + Paper 1 geology guidance (direction-distribution W1, set-share L1) + interaction
losses on the expected next trace against the context (soft crossings vs the real next trace's crossings, near-duplicate
penalty, nearest same-set gap vs the real one). Trained per whole-bench fold (S2) and Paper 1 run seed on the other six
benches with +-40 deg rotations and random scanline layouts."""
import sys, math, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
import tr_common as TC
import tr_gen as TG

MODELS = HERE / 'tr_models'
K1MAX, NF, NE, D, NH, NL = 5, 4, 8, 96, 4, 3
NMAX = 40
FREQ = [1, 2, 4, 8, 16]
FEAT = 5 + 4 * 2 * 2 * len(FREQ) + 1   # x, y, logL, cos2t, sin2t + Fourier(mid, 2 ends, cross-corner) + dist-to-line
SFEAT = 2 + 1 + 1 + 4 + 1                # set token: cos2mu, sin2mu, spread, share, 4 log-length quantiles, spacing
KP, KL, KA = 4, 3, 3
WINX = False          # window-level crossing loss (GAST2)
TAG = 'gast'


# ------------------------------------------------------------------ features
def fourier(xy):
    out = []
    for f in FREQ:
        out += [np.sin(math.pi * f * xy / 20), np.cos(math.pi * f * xy / 20)]
    return np.concatenate(out, -1)


def trace_feats(L, lines):
    L = np.asarray(L, float).reshape(-1, 4)
    if not len(L): return np.zeros((0, FEAT), np.float32)
    th = P.segdir(L); mx, my = (L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2
    d = np.array([min([abs(x - l['c']) for l in lines] or [20.0]) for x in mx])
    base = np.column_stack([mx / 20, my / 20, np.log(np.maximum(P.seglen(L), 0.05)) / 3, np.cos(2 * th), np.sin(2 * th)])
    four = np.concatenate([fourier(np.column_stack([mx, my])), fourier(L[:, :2]), fourier(L[:, 2:]), fourier(np.column_stack([L[:, 0], L[:, 3]]))], -1)
    return np.concatenate([base, four, (d / 20)[:, None]], -1).astype(np.float32)


def set_tokens(C):
    model = C['model']; K1 = model['K'] + 1; th = C['th2']; ln = C['tr_len2d']; cls = C['cls2']
    out = np.zeros((K1MAX, SFEAT), np.float32)
    for k in range(K1):
        t, l = th[cls == k], ln[cls == k]
        if len(t) < 2: continue
        z = np.mean(np.exp(2j * t))
        out[k, :2] = [z.real, z.imag]; out[k, 2] = math.sqrt(max(-2 * math.log(max(abs(z), 1e-6)), 0)) / 2
        out[k, 3] = len(t) / len(th); out[k, 4:8] = np.log(np.percentile(l, [25, 50, 75, 90]) + 0.05) / 3
        out[k, 8] = C['set_gap'].get(k, 3.0) / 20
    return out


def edge_feats(A, B):
    """pair geometry between segment sets A (n,4) and B (m,4) -> (n, m, NE)"""
    n, m = len(A), len(B)
    if n == 0 or m == 0: return np.zeros((n, m, NE), np.float32)
    ta, tb = P.segdir(A), P.segdir(B)
    dth = ta[:, None] - tb[None]
    ma, mb = (A[:, :2] + A[:, 2:]) / 2, (B[:, :2] + B[:, 2:]) / 2
    rel = (mb[None] - ma[:, None]) / 20
    E = np.vstack([A[:, :2], A[:, 2:]])
    dmin = np.minimum(P._pt_seg(A[:, :2], B[:, :2], B[:, 2:]), P._pt_seg(A[:, 2:], B[:, :2], B[:, 2:]))
    cross = np.zeros((n, m), np.float32)
    for i in range(n):
        cross[i] = P.common.seg_cross_one(A[i], B)
    ua = np.column_stack([np.cos(ta), np.sin(ta)])
    perp = np.abs(-ua[:, 1:2] * (mb[None, :, 0] - ma[:, None, 0]) + ua[:, 0:1] * (mb[None, :, 1] - ma[:, None, 1]))
    along = np.abs(ua[:, 0:1] * (mb[None, :, 0] - ma[:, None, 0]) + ua[:, 1:2] * (mb[None, :, 1] - ma[:, None, 1]))
    la, lb = P.seglen(A), P.seglen(B)
    ov = np.clip((la[:, None] + lb[None]) / 2 - along, 0, None) / np.maximum(np.minimum(la[:, None], lb[None]), 0.1)
    F_ = np.stack([np.minimum(dmin, 20) / 20, np.abs(np.sin(dth)), np.cos(2 * dth), cross, np.minimum(ov, 1),
                   np.minimum(perp, 20) / 20, rel[..., 0], rel[..., 1]], -1)
    return F_.astype(np.float32)


# ------------------------------------------------------------------ model
class GeoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D); self.o = nn.Linear(D, D)
        self.eb = nn.Sequential(nn.Linear(NE + 2, 32), nn.ReLU(), nn.Linear(32, NH))

    def forward(self, x, edge, emask, amask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, NH, D // NH).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(D // NH)
        s = s + self.eb(torch.cat([edge, emask], -1)).permute(0, 3, 1, 2)
        s = s.masked_fill(~amask.unsqueeze(1), -1e9)
        return self.o((F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = GeoAttention(); self.n1 = nn.LayerNorm(D); self.n2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D)); self.dr = nn.Dropout(0.1)

    def forward(self, x, edge, emask, amask):
        x = x + self.dr(self.att(self.n1(x), edge, emask, amask))
        return x + self.dr(self.ff(self.n2(x)))


class GAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.tin = nn.Sequential(nn.Linear(FEAT, D), nn.ReLU(), nn.Linear(D, D))
        self.sin = nn.Sequential(nn.Linear(SFEAT, D), nn.ReLU(), nn.Linear(D, D))
        self.set_emb = nn.Embedding(K1MAX + 1, D)      # +1 = unknown (BOS / set tokens)
        self.tier_emb = nn.Embedding(NF + 1, D)         # 0 measured, 1 indicated, 2 generated, 3 BOS, 4 set token
        self.blocks = nn.ModuleList([Block() for _ in range(NL)]); self.nf = nn.LayerNorm(D)
        self.h_pos = nn.Linear(D, KP * 5); self.h_len = nn.Linear(D, KL * 3); self.h_ang = nn.Linear(D, KA * 3)
        self.h_set = nn.Linear(D, K1MAX); self.h_stop = nn.Linear(D, 1)

    def forward(self, tf, sf, tset, ttier, edge, emask, amask):
        """tf (B,Tt,FEAT) trace tokens incl. BOS row; sf (B,K1MAX,SFEAT); full sequence = set tokens + trace tokens"""
        B = tf.shape[0]
        xs = self.sin(sf) + self.set_emb.weight[:K1MAX][None] + self.tier_emb.weight[4][None, None]
        xt = self.tin(tf) + self.set_emb(tset) + self.tier_emb(ttier)
        x = torch.cat([xs, xt], 1)
        for b in self.blocks:
            x = b(x, edge, emask, amask)
        return self.nf(x)[:, K1MAX:]

    def heads(self, z, Hn):
        B, T, _ = z.shape
        p = self.h_pos(z).view(B, T, KP, 5)
        pw = F.log_softmax(p[..., 0], -1); pm = torch.stack([torch.sigmoid(p[..., 1]), torch.sigmoid(p[..., 2]) * Hn.view(B, 1, 1)], -1)
        ps = F.softplus(p[..., 3:5]) * 0.2 + 0.01
        l = self.h_len(z).view(B, T, KL, 3); lw = F.log_softmax(l[..., 0], -1); lm = l[..., 1]; ls = F.softplus(l[..., 2]) * 0.5 + 0.05
        a = self.h_ang(z).view(B, T, KA, 3); aw = F.log_softmax(a[..., 0], -1); am = math.pi * torch.tanh(a[..., 1]); ak = F.softplus(a[..., 2]) + 0.5
        return dict(pw=pw, pm=pm, ps=ps, lw=lw, lm=lm, ls=ls, aw=aw, am=am, ak=ak, set=self.h_set(z), stop=self.h_stop(z).squeeze(-1))


# ------------------------------------------------------------------ sequence building
def build_seq(C, ctx_traces, ctx_tiers, gen_traces, lines):
    """returns numpy arrays for one sequence: trace tokens = [BOS] + ctx + gen"""
    model = C['model']
    allL = np.vstack([np.asarray(ctx_traces).reshape(-1, 4), np.asarray(gen_traces).reshape(-1, 4)])
    n_ctx, n_gen = len(ctx_traces), len(gen_traces)
    tf = np.vstack([np.zeros((1, FEAT), np.float32), trace_feats(allL, lines)])
    cls = V.setbg.classify(P.segdir(allL), model) if len(allL) else np.zeros(0, int)
    tset = np.r_[K1MAX, cls].astype(np.int64)
    ttier = np.r_[3, ctx_tiers, np.full(n_gen, 2)].astype(np.int64)
    return tf, tset, ttier, allL, n_ctx, n_gen


def batch(C, seqs):
    """seqs: list of (tf, tset, ttier, allL, n_ctx, n_gen). full length = K1MAX + 1 + n_ctx + n_gen"""
    B = len(seqs); T = max(K1MAX + len(s[0]) for s in seqs); Tt = T - K1MAX
    tf = np.zeros((B, Tt, FEAT), np.float32); tset = np.full((B, Tt), K1MAX, np.int64); ttier = np.full((B, Tt), 3, np.int64)
    edge = np.zeros((B, T, T, NE), np.float32); emask = np.zeros((B, T, T, 2), np.float32); amask = np.zeros((B, T, T), bool)
    sf = np.repeat(C['set_tokens'][None], B, 0)
    for b, (f, s, t, L, nc, ng) in enumerate(seqs):
        n = len(f); tf[b, :n] = f; tset[b, :n] = s; ttier[b, :n] = t
        i0 = K1MAX + 1                                     # first trace position in the full sequence
        if len(L):
            edge[b, i0:i0 + len(L), i0:i0 + len(L)] = edge_feats(L, L); emask[b, i0:i0 + len(L), i0:i0 + len(L), 0] = 1
        emask[b, :K1MAX, :, 1] = 1; emask[b, :, :K1MAX, 1] = 1
        full = K1MAX + n
        amask[b, :full, :K1MAX + 1 + nc] = True             # everyone sees set tokens, BOS and evidence
        g0 = K1MAX + 1 + nc
        for i in range(g0, full):
            amask[b, i, g0:i + 1] = True                    # generated: causal over generated
        amask[b, full:, :] = False; amask[b, np.arange(T), np.arange(T)] = True
    return [torch.as_tensor(a) for a in (tf, sf, tset, ttier, edge, emask, amask)]


# ------------------------------------------------------------------ training examples
def make_example(C, ev, rng):
    k = rng.integers(len(C['TW'])); H = C['TH'][k]
    L = C['TW'][k]
    rot = rng.uniform(-40, 40); L = V.rotate(L, H, rot) if len(L) else L
    if len(L) < 3: return None
    lines = TG.random_layout(H, rng); obs, t1, t2, hidden = TG.evidence(C, L, H, lines, rng)
    if not len(hidden): return None
    hidden = hidden[TG.order_from_evidence(hidden, lines)][:NMAX - 1]
    ctx = np.vstack([t1, t2]) if len(t2) else t1
    tiers = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    return build_seq(C, ctx, tiers, hidden, lines), H, hidden


def soft_cross(a, B, k=30.0):
    """differentiable crossing indicator of segment a (4,) with segments B (m,4)"""
    p, r = a[:2], a[2:] - a[:2]; q, s = B[:, :2], B[:, 2:] - B[:, :2]
    den = r[0] * s[:, 1] - r[1] * s[:, 0]; den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
    d = q - p; t = (d[:, 0] * s[:, 1] - d[:, 1] * s[:, 0]) / den; u = (d[:, 0] * r[1] - d[:, 1] * r[0]) / den
    return torch.sigmoid(k * t) * torch.sigmoid(k * (1 - t)) * torch.sigmoid(k * u) * torch.sigmoid(k * (1 - u))


def loss_fn(net, C, items):
    seqs = [it[0] for it in items]; Hs = torch.tensor([it[1] for it in items], dtype=torch.float32)
    tf, sf, tset, ttier, edge, emask, amask = batch(C, seqs)
    z = net(tf, sf, tset, ttier, edge, emask, amask); h = net.heads(z, Hs / 20)
    tot = 0.0; nll, cnt = 0.0, 0; geo_dir, geo_set, inter = [], [], []
    grid = (torch.arange(36) + 0.5) * (2 * math.pi / 36)
    for b, (seq, H, hidden) in enumerate(items):
        f, s, t, L, nc, ng = seq
        pred = torch.arange(nc, nc + ng)                      # trace-token positions predicting each hidden trace (BOS at 0)
        tgt = torch.as_tensor(hidden, dtype=torch.float32)
        mx = (tgt[:, 0] + tgt[:, 2]) / 40; my = (tgt[:, 1] + tgt[:, 3]) / 40
        ln = torch.log(torch.clamp(torch.hypot(tgt[:, 2] - tgt[:, 0], tgt[:, 3] - tgt[:, 1]), min=0.05))
        th = torch.remainder(torch.atan2(tgt[:, 3] - tgt[:, 1], tgt[:, 2] - tgt[:, 0]), math.pi)
        xy = torch.stack([mx, my], -1).unsqueeze(1)
        lp = (h['pw'][b, pred] + (-0.5 * ((xy - h['pm'][b, pred]) / h['ps'][b, pred]) ** 2 - torch.log(h['ps'][b, pred]) - 0.919).sum(-1)).logsumexp(-1)
        ll = (h['lw'][b, pred] + (-0.5 * ((ln.unsqueeze(1) - h['lm'][b, pred]) / h['ls'][b, pred]) ** 2 - torch.log(h['ls'][b, pred]) - 0.919)).logsumexp(-1)
        k_ = h['ak'][b, pred]
        la = (h['aw'][b, pred] + k_ * torch.cos(2 * th.unsqueeze(1) - h['am'][b, pred]) - math.log(2 * math.pi) - (torch.log(torch.special.i0e(k_)) + k_)).logsumexp(-1)
        nll = nll - (lp + ll + la).sum(); cnt += ng
        cls = torch.as_tensor(s[1 + nc:1 + nc + ng]); tot = tot + F.cross_entropy(h['set'][b, pred], cls, reduction='sum')
        stop_t = torch.zeros(ng + 1); stop_t[-1] = 1
        tot = tot + F.binary_cross_entropy_with_logits(h['stop'][b, nc:nc + ng + 1], stop_t, reduction='sum')
        # geology guidance (Paper 1): generated direction distribution and set shares vs real hidden traces
        dens = (h['aw'][b, pred].exp().unsqueeze(-1) * torch.exp(h['ak'][b, pred].unsqueeze(-1) * (torch.cos(grid - h['am'][b, pred].unsqueeze(-1)) - 1))).sum(-2)
        G = (dens / dens.sum(-1, keepdim=True)).mean(0)
        Rr = torch.exp(20 * (torch.cos(grid.view(1, -1) - 2 * th.view(-1, 1)) - 1)); Rr = (Rr / Rr.sum(-1, keepdim=True)).mean(0)
        Dc = torch.cumsum(G - Rr, -1); geo_dir.append((Dc - Dc.median()).abs().mean())
        ps = F.softmax(h['set'][b, pred], -1).mean(0); rs = torch.bincount(cls, minlength=K1MAX).float() / ng
        geo_set.append((ps - rs).abs().sum())
        # interaction losses on the expected next trace
        w = h['pw'][b, pred].exp().unsqueeze(-1); emxy = (w * h['pm'][b, pred]).sum(1) * 20
        eL = torch.exp((h['lw'][b, pred].exp() * h['lm'][b, pred]).sum(1))
        aw_ = h['aw'][b, pred].exp(); et = torch.atan2((aw_ * torch.sin(h['am'][b, pred])).sum(1), (aw_ * torch.cos(h['am'][b, pred])).sum(1)) / 2
        eseg = torch.stack([emxy[:, 0] - torch.cos(et) * eL / 2, emxy[:, 1] - torch.sin(et) * eL / 2,
                            emxy[:, 0] + torch.cos(et) * eL / 2, emxy[:, 1] + torch.sin(et) * eL / 2], -1)
        ctxL = torch.as_tensor(L[:nc], dtype=torch.float32)
        if WINX and ng > 0:
            # window-level: expected crossings of the generated window (with context and among themselves) vs real
            ex_tot = sum(soft_cross(eseg[j], torch.cat([ctxL, eseg[:j]], 0)).sum() for j in range(ng) if nc + j > 0)
            real_tot = sum(float(P.common.seg_cross_one(hidden[j], np.vstack([L[:nc], hidden[:j]]).reshape(-1, 4)).sum()) for j in range(ng) if nc + j > 0)
            inter.append(3.0 * (ex_tot - real_tot) ** 2 / max(ng, 1))
        for j in range(ng):
            Bj = torch.cat([ctxL, tgt[:j]], 0)
            if not len(Bj): continue
            real_x = float(P.common.seg_cross_one(hidden[j], Bj.numpy()).sum())
            inter.append((soft_cross(eseg[j], Bj).sum() - real_x) ** 2 / 4)
            # near-duplicate: expected trace parallel, overlapping and closer than the real 5th-percentile gap
            tb = torch.remainder(torch.atan2(Bj[:, 3] - Bj[:, 1], Bj[:, 2] - Bj[:, 0]), math.pi)
            par = torch.cos(2 * (et[j] - tb)).clamp(min=0) ** 8
            mb = (Bj[:, :2] + Bj[:, 2:]) / 2; u = torch.stack([torch.cos(et[j]), torch.sin(et[j])])
            gap = ((mb - emxy[j]) @ torch.stack([-u[1], u[0]])).abs()
            along = ((mb - emxy[j]) @ u).abs()
            ov = torch.sigmoid(2 * ((eL[j] + torch.hypot(Bj[:, 2] - Bj[:, 0], Bj[:, 3] - Bj[:, 1])) / 2 - along))
            inter.append((par * ov * F.relu(C['dmin'] - gap)).sum())
    loss = (nll + tot) / max(cnt, 1)
    if geo_dir: loss = loss + torch.stack(geo_dir).mean() + torch.stack(geo_set).mean()
    if inter: loss = loss + torch.stack(inter).mean()
    return loss


def prep_ctx(scheme, fi):
    C = TC.fold_ctx(scheme, fi)
    C['cls2'] = V.setbg.classify(C['th2'], C['model'])
    gaps = {}
    for k in range(C['model']['K'] + 1):
        g = []
        for L in C['TW']:
            if len(L) < 2: continue
            c = V.setbg.classify(P.segdir(L), C['model']); S = L[c == k]
            if len(S) > 1: g += list(P.common.nn_distances((S[:, :2] + S[:, 2:]) / 2))
        gaps[k] = float(np.median(g)) if g else 3.0
    C['set_gap'] = gaps; C['set_tokens'] = set_tokens(C)
    return C


def train(fi, seed, scheme='S2', epochs=40, per=400):
    torch.manual_seed(seed); torch.set_num_threads(2); rng = np.random.default_rng(seed)
    C = prep_ctx(scheme, fi); B_ = C['boxes']; sp = C['sp']; held = set().union(*[set(B_[t]['ids']) for t in sp['test']])
    def wins(keys):
        W_, H_ = [], []
        for k in keys:
            b = B_[k]; keep = np.array([i not in held for i in b['ids']], bool); W_.append(P.clip(b['L'][keep], b['H'])); H_.append(b['H'])
        return W_, H_
    FW, FH = wins(sp['fitting']); VW, VH = wins(sp['validation'])
    Ct = dict(C, TW=FW, TH=FH); Cv = dict(C, TW=VW, TH=VH)
    vr = np.random.default_rng(11); vit = [x for x in (make_example(Cv, None, vr) for _ in range(120)) if x is not None]
    net = GAST(); opt = torch.optim.Adam(net.parameters(), 5e-4, weight_decay=1e-5); best, bad, state = 1e9, 0, None
    for ep in range(epochs):
        it = [x for x in (make_example(Ct, None, rng) for _ in range(per)) if x is not None]; net.train()
        for i in range(0, len(it), 16):
            l = loss_fn(net, Ct, it[i:i + 16]); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(net, Cv, vit[i:i + 30]).item() for i in range(0, len(vit), 30)]))
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    torch.save(state, MODELS / f'{TAG}_{scheme}_f{fi}_s{seed}.pt'); json.dump(dict(best=best, epochs=ep + 1), open(MODELS / f'{TAG}_{scheme}_f{fi}_s{seed}.json', 'w'))


def load(fi, seed, scheme='S2', tag='gast'):
    n = GAST(); n.load_state_dict(torch.load(MODELS / f'{tag}_{scheme}_f{fi}_s{seed}.pt')); n.eval(); return n


@torch.no_grad()
def generate(net, C, t1, t2, lines, H, rng, tries=5, len_fn=None):
    ctx = np.vstack([t1, t2]) if len(t2) else np.asarray(t1).reshape(-1, 4)
    tiers = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    gen = np.zeros((0, 4)); torch.manual_seed(int(rng.integers(1 << 30)))
    for step in range(NMAX):
        seq = build_seq(C, ctx, tiers, gen, lines)
        z = net(*batch(C, [seq])); h = net.heads(z[:, -1:], torch.tensor([H / 20]))
        if rng.random() < torch.sigmoid(h['stop'][0, 0]).item(): break
        seg = None
        for _ in range(tries):
            kp = rng.choice(KP, p=h['pw'][0, 0].exp().numpy()); mx, my = (h['pm'][0, 0, kp] + h['ps'][0, 0, kp] * torch.randn(2)).numpy() * 20
            kl = rng.choice(KL, p=h['lw'][0, 0].exp().numpy()); Lg = math.exp(float(h['lm'][0, 0, kl] + h['ls'][0, 0, kl] * torch.randn(())))
            ka = rng.choice(KA, p=h['aw'][0, 0].exp().numpy()); th = (rng.vonmises(float(h['am'][0, 0, ka]), float(h['ak'][0, 0, ka])) / 2) % math.pi
            if len_fn is not None:
                Lg = float(len_fn(int(V.setbg.classify(np.array([th]), C['model'])[0]), rng))
            s = P.clip(np.array([[mx - math.cos(th) * Lg / 2, my - math.sin(th) * Lg / 2, mx + math.cos(th) * Lg / 2, my + math.sin(th) * Lg / 2]]), H)
            if len(s) and not any(V.crosses(s, l)[0] for l in lines):
                seg = s[0]; break
        if seg is None: continue
        gen = np.vstack([gen, seg[None]])
    return gen


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        C = prep_ctx('S2', 0); rng = np.random.default_rng(0)
        it = [x for x in (make_example(C, None, rng) for _ in range(8)) if x is not None]
        net = GAST(); t = time.time(); l = loss_fn(net, C, it); l.backward(); print('loss', l.item(), f'{time.time()-t:.1f}s', sum(p.numel() for p in net.parameters()), 'params')
        L = C['TW'][0]; H = C['TH'][0]; lines = V.lines_layout('v', 5.0, 0.1, H); obs, t1, t2, hid = TG.evidence(C, L, H, lines, rng)
        t = time.time(); print('gen', generate(net, C, t1, t2, lines, H, rng).shape, f'{time.time()-t:.1f}s')
    else:
        if len(sys.argv) > 3 and sys.argv[3] == 'winx':
            WINX = True; TAG = 'gast2'
        fi, seed = int(sys.argv[1]), int(sys.argv[2]); t = time.time(); train(fi, seed); print('done', TAG, fi, seed, f'{time.time()-t:.0f}s')
