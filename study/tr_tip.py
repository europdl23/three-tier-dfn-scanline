# -*- coding: utf-8 -*-
"""Transformer as a PORTABLE INTERACTION PRIOR (TIP) - last chance test for the sequential transformer.
It does not generate fractures. Given the scanline evidence, the traces already placed and a proposed trace (from
statistics or the EVAE), it predicts
  accept    is the proposal real-like in this context (accept / reject),
  priority  placement order (traces that others end on come first),
  stop      for every existing trace the proposal meets: does it stop there (Y) or cross (X).
Directions are expressed relative to the local reference direction (axial mean of the scanline hits), so that rotated
geology at a new place looks the same. Trained per whole-bench fold (S2) and Paper 1 run seed on the other six benches,
with random rotations of the training windows (+-40 deg)."""
import os, sys, math, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
import tr_common as TC
import tr_gen as TG
from run_v3_term import protect_spans

MODELS = HERE / 'tr_models'
FD = 10


def theta_ref(obs, fallback):
    if len(obs) >= 2:
        z = np.mean([np.exp(2j * o['th']) for o in obs]); return (np.angle(z) / 2) % math.pi
    return fallback


def tok(L, tier, lines, tref):
    L = np.asarray(L, float).reshape(-1, 4)
    if not len(L): return np.zeros((0, FD), np.float32)
    th = P.segdir(L) - tref; mx, my = (L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2
    d = np.array([min([abs(x - l['c']) for l in lines] or [20.0]) for x in mx])
    T = np.zeros((len(L), FD), np.float32)
    T[:, 0] = mx / 20; T[:, 1] = my / 20; T[:, 2] = np.log(np.maximum(P.seglen(L), 0.05)) / 3
    T[:, 3] = np.cos(2 * th); T[:, 4] = np.sin(2 * th); T[np.arange(len(L)), 5 + tier] = 1; T[:, 9] = d / 20
    return T


class TIP(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FD, d), nn.ReLU(), nn.Linear(d, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True), 3)
        self.acc = nn.Linear(d, 1); self.pri = nn.Linear(d, 1)
        self.stop = nn.Sequential(nn.Linear(2 * d + 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, tok, pad):
        h = self.enc(self.inp(tok), src_key_padding_mask=pad)
        return h, self.acc(h).squeeze(-1), self.pri(h).squeeze(-1)

    def stop_logit(self, hc, hj, geo):
        return self.stop(torch.cat([hc, hj, geo], -1)).squeeze(-1)


def pair_geo(a, b, t):
    ta, tb = P.segdir(a[None])[0], P.segdir(b[None])[0]
    return [abs(math.sin(ta - tb)), min(t, 1 - t), math.log(max(P.seglen(b[None])[0], 0.05)) / 3]


def meets(a, b, tol=0.3):
    """does trace a cross b (return ('X', t)) or end on b (('Y', t)), else None; t = position along a"""
    p, r = a[:2], a[2:] - a[:2]; q, s = b[:2], b[2:] - b[:2]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) > 1e-12:
        d = q - p; t = (d[0] * s[1] - d[1] * s[0]) / den; u = (d[0] * r[1] - d[1] * r[0]) / den
        if 0 < t < 1 and 0 < u < 1:
            return ('X', t)
    for e, t in ((a[:2], 0.0), (a[2:], 1.0)):
        if P._pt_seg(e[None], b[None, :2], b[None, 2:])[0, 0] <= tol:
            return ('Y', t)
    return None


def rot_window(L, H, deg):
    return V.rotate(L, H, deg) if deg else L


def sample(C, ev, rng):
    k = rng.integers(len(C['TW'])); H = C['TH'][k]
    L = rot_window(C['TW'][k], H, rng.uniform(-40, 40))
    if len(L) < 3: return None
    lines = TG.random_layout(H, rng); obs, t1, t2, hidden = TG.evidence(C, L, H, lines, rng)
    if len(hidden) < 2: return None
    tref = theta_ref(obs, float(np.angle(np.mean(np.exp(2j * C['th2']))) / 2))
    perm = rng.permutation(len(hidden)); na = rng.integers(1, len(hidden)); acc, rest = hidden[perm[:na]], hidden[perm[na:]]
    if rng.random() < 0.5:
        cand, lab = rest[rng.integers(len(rest))], 1.0
    else:
        src = P.clip(ev(H, rng.integers(1 << 30)), H) if rng.random() < 0.5 else P.clip(P.unif_proposals(20, C['tr_th'], C['tr_len3d'], H, rng), H)
        src = src[[not any(V.crosses(t[None], l)[0] for l in lines) for t in src]] if len(src) else src
        if not len(src): return None
        cand, lab = src[rng.integers(len(src))], 0.0
    ctx = np.vstack([tok(t1, 0, lines, tref), tok(t2, 1, lines, tref), tok(acc, 2, lines, tref)])
    seq = np.vstack([ctx, tok(cand[None], 3, lines, tref)])
    n_ctx_real0 = len(t1) + len(t2)
    stops, pri = [], []
    # every real encounter among the real traces in the sequence (accepted real traces, plus the candidate if real)
    real = [(n_ctx_real0 + j, b) for j, b in enumerate(acc)] + ([(len(seq) - 1, cand)] if lab == 1.0 else [])
    for ia, a in real:
        for ib, b in real:
            if ia == ib: continue
            m = meets(a, b)
            if not m: continue
            stops.append((ia, ib, 1.0 if m[0] == 'Y' else 0.0, pair_geo(a, b, m[1])))
            if m[0] == 'Y': pri.append((ib, ia))        # b placed before a
    return seq[-120:], lab, stops, pri, len(seq) - 1 - max(0, len(seq) - 120)


def batch_loss(net, items):
    T = max(len(i[0]) for i in items); B = len(items)
    tk = np.zeros((B, T, FD), np.float32); pad = np.ones((B, T), bool); ci = np.zeros(B, int)
    for b, it in enumerate(items):
        tk[b, :len(it[0])] = it[0]; pad[b, :len(it[0])] = False; ci[b] = len(it[0]) - 1
    h, acc, pri = net(torch.as_tensor(tk), torch.as_tensor(pad))
    lab = torch.tensor([i[1] for i in items])
    loss = F.binary_cross_entropy_with_logits(acc[torch.arange(B), ci], lab)
    sl, sy, pl = [], [], []
    shift = [len(i[0]) - 1 - i[4] for i in items]
    for b, it in enumerate(items):
        for ia, ib, y, g in it[2]:
            a2, b2 = ia - shift[b], ib - shift[b]
            if 0 <= a2 < len(it[0]) and 0 <= b2 < len(it[0]):
                sl.append(net.stop_logit(h[b, a2], h[b, b2], torch.tensor(g, dtype=torch.float32))); sy.append(y)
        for first, later in it[3]:
            f2, l2 = first - shift[b], later - shift[b]
            if 0 <= f2 < len(it[0]) and 0 <= l2 < len(it[0]): pl.append(F.softplus(pri[b, l2] - pri[b, f2]))
    if sl: loss = loss + F.binary_cross_entropy_with_logits(torch.stack(sl), torch.tensor(sy))
    if pl: loss = loss + torch.stack(pl).mean()
    return loss


def train(fi, seed, epochs=30, per=1500, scheme='S2'):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); torch.set_num_threads(2)
    C = TC.fold_ctx(scheme, fi); B = C['boxes']; sp = C['sp']; held = set().union(*[set(B[t]['ids']) for t in sp['test']])
    def wins(keys):
        W_, H_ = [], []
        for k in keys:
            b = B[k]; keep = np.array([i not in held for i in b['ids']], bool); W_.append(P.clip(b['L'][keep], b['H'])); H_.append(b['H'])
        return W_, H_
    FW, FH = wins(sp['fitting']); VW, VH = wins(sp['validation'])
    Ct = dict(C, TW=FW, TH=FH); Cv = dict(C, TW=VW, TH=VH)
    ev = TG.evae_sampler(scheme, fi, seed)
    vr = np.random.default_rng(3); vit = [x for x in (sample(Cv, ev, vr) for _ in range(300)) if x is not None]
    net = TIP(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5); best, bad, state = 1e9, 0, None
    for ep in range(epochs):
        it = [x for x in (sample(Ct, ev, rng) for _ in range(per)) if x is not None]; net.train()
        for i in range(0, len(it), 32):
            l = batch_loss(net, it[i:i + 32]); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([batch_loss(net, vit[i:i + 50]).item() for i in range(0, len(vit), 50)]))
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    torch.save(state, MODELS / f'tip_{scheme}_f{fi}_s{seed}.pt'); json.dump(dict(best=best, epochs=ep + 1), open(MODELS / f'tip_{scheme}_f{fi}_s{seed}.json', 'w'))


def load(fi, seed, scheme='S2'):
    n = TIP(); n.load_state_dict(torch.load(MODELS / f'tip_{scheme}_f{fi}_s{seed}.pt')); n.eval(); return n


@torch.no_grad()
def place(net, C, obs, lines, H, rng, pool, bud, tref):
    """TIP-driven sequential placement: score all gate-passing proposals, place them in learned priority order, accept
    with the learned probability, cut at encounters with the learned stop probability, until the length budget."""
    groups = V.merge(obs); t1 = V.tier1(obs)
    t2 = V.tier2(groups, C['th2'], C['tr_len2d'], H, rng)
    if len(t2) > 1:
        t2 = V.terminate(t2, 0, rng, C['p_pass'], protect=protect_spans(t2, obs))
    rec = t2; ntier = [1] * len(t2)
    cand = P.clip(pool, H)
    cand = cand[[not any(V.crosses(t[None], l)[0] for l in lines) for t in cand]] if len(cand) else cand
    tries = 0
    while len(cand) and P.seglen(rec).sum() < bud and tries < 6:
        tries += 1
        ctx = np.vstack([tok(t1, 0, lines, tref), tok(rec[:len(t2)], 1, lines, tref), tok(rec[len(t2):], 2, lines, tref)])
        seqs = [np.vstack([ctx, tok(c[None], 3, lines, tref)])[-120:] for c in cand]
        T = max(len(s) for s in seqs); tk = np.zeros((len(seqs), T, FD), np.float32); pad = np.ones((len(seqs), T), bool)
        for i, s in enumerate(seqs): tk[i, :len(s)] = s; pad[i, :len(s)] = False
        h, acc, pri = net(torch.as_tensor(tk), torch.as_tensor(pad))
        ci = torch.as_tensor([len(s) - 1 for s in seqs])
        pa = torch.sigmoid(acc[torch.arange(len(seqs)), ci]).numpy(); pp = pri[torch.arange(len(seqs)), ci].numpy()
        used = np.zeros(len(cand), bool)
        for i in np.argsort(-pp):
            if P.seglen(rec).sum() >= bud: break
            if rng.random() >= pa[i]: continue
            c = cand[i].copy(); L = P.seglen(c[None])[0]; lo, hi = 0.0, L
            for j, b in enumerate(rec):
                m = meets(c, b)
                if not m or m[0] != 'X': continue
                g = torch.tensor(pair_geo(c, b, m[1]), dtype=torch.float32)
                hj = h[i, min(len(t1) + j, len(seqs[i]) - 2)] if len(seqs[i]) > 1 else h[i, ci[i]]
                if rng.random() < torch.sigmoid(net.stop_logit(h[i, ci[i]], hj, g)).item():
                    a = m[1] * L
                    if a < L / 2: lo = max(lo, a)
                    else: hi = min(hi, a)
            ux, uy = (c[2] - c[0]) / max(L, 1e-9), (c[3] - c[1]) / max(L, 1e-9)
            c = np.array([c[0] + lo * ux, c[1] + lo * uy, c[0] + hi * ux, c[1] + hi * uy])
            if P.seglen(c[None])[0] > 0.3:
                rec = np.vstack([rec, c[None]]) if len(rec) else c[None]
            used[i] = True
        cand = cand[~used]
    return rec, len(t2)


if __name__ == '__main__':
    sch = sys.argv[3] if len(sys.argv) > 3 else 'S2'
    fi, seed = int(sys.argv[1]), int(sys.argv[2]); t = time.time(); train(fi, seed, scheme=sch); print('done', sch, fi, seed, f'{time.time() - t:.0f}s')
