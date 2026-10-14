# -*- coding: utf-8 -*-
"""Fully learned components for the three-tier rebuild (round 9). Trained per whole-bench fold (S2) and Paper 1 run
seed on the other six benches, real geology only (no rotation).
  EVN  evidence network (C1 + C2): transformer over the observed hit groups (+ set tokens + a global token).
       Per group: full in-window length of the trace (3-component log-normal mixture) and the position of the hits along
       it (logit-normal fraction). Global: total length of the hidden (unseen) traces (log-normal).
  CRT  critic (C4): GAST backbone (geometry-aware attention) over [set tokens | evidence | accepted traces | candidate]
       -> accept probability. Positives: real hidden traces; negatives: Paper 1 EVAE samples of the fold, near-duplicates
       of existing traces, real-length traces at random positions, and real hidden traces displaced along their normal.
  C3   generator = GAST2 (sequential transformer, trained earlier) with EVAE lengths; C5 termination = TermNet (tr_gen).
"""
import sys, math, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
import tr_gen as TG
import gast as G

MODELS = HERE / 'tr_models'
D = 96
GF = 6 + 4 * len(G.FREQ)          # group features: x, y, cos2t, sin2t, n hits, span + Fourier(x, y)


# ------------------------------------------------------------------ data helpers
def windows(C, keys):
    B = C['boxes']; held = set().union(*[set(B[t]['ids']) for t in C['sp']['test']])
    W_, H_ = [], []
    for k in keys:
        b = B[k]; keep = np.array([i not in held for i in b['ids']], bool); W_.append(P.clip(b['L'][keep], b['H'])); H_.append(b['H'])
    return W_, H_


def group_feats(groups, H):
    out = []
    for g in groups:
        th = math.atan2(sum(math.sin(2 * o['th']) for o in g), sum(math.cos(2 * o['th']) for o in g)) / 2 % math.pi
        mx, my = np.mean([o['x'] for o in g]), np.mean([o['y'] for o in g])
        s = np.array([o['x'] * math.cos(th) + o['y'] * math.sin(th) for o in g]); span = s.max() - s.min()
        f = [mx / 20, my / 20, math.cos(2 * th), math.sin(2 * th), len(g) / 5, span / 20]
        for fr in G.FREQ:
            f += [math.sin(math.pi * fr * mx / 20), math.cos(math.pi * fr * mx / 20), math.sin(math.pi * fr * my / 20), math.cos(math.pi * fr * my / 20)]
        out.append((f, th, span, s.min(), g))
    return out


# ------------------------------------------------------------------ EVN
class EVN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gin = nn.Sequential(nn.Linear(GF, D), nn.ReLU(), nn.Linear(D, D))
        self.sin = nn.Sequential(nn.Linear(G.SFEAT, D), nn.ReLU(), nn.Linear(D, D))
        self.glob = nn.Sequential(nn.Linear(3, D), nn.ReLU(), nn.Linear(D, D))
        self.set_emb = nn.Embedding(G.K1MAX, D)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.1, batch_first=True), 3)
        self.h_len = nn.Linear(D, 9); self.h_off = nn.Linear(D, 2); self.h_hid = nn.Linear(D, 2)

    def forward(self, gf, gset, gpad, sf, glob):
        B = gf.shape[0]
        xs = self.sin(sf) + self.set_emb.weight[None]
        xg = self.glob(glob).unsqueeze(1)
        x = torch.cat([xg, xs.expand(B, -1, -1), self.gin(gf) + self.set_emb(gset)], 1)
        pad = torch.cat([torch.zeros(B, 1 + G.K1MAX, dtype=torch.bool), gpad], 1)
        z = self.enc(x, src_key_padding_mask=pad)
        zg, zt = z[:, 0], z[:, 1 + G.K1MAX:]
        l = self.h_len(zt).view(B, -1, 3, 3)
        return dict(lw=F.log_softmax(l[..., 0], -1), lm=l[..., 1] + 1.5, ls=F.softplus(l[..., 2]) * 0.5 + 0.05,
                    om=self.h_off(zt)[..., 0], os=F.softplus(self.h_off(zt)[..., 1]) + 0.1,
                    hm=self.h_hid(zg)[:, 0] + 3.5, hs=F.softplus(self.h_hid(zg)[:, 1]) * 0.5 + 0.05)


def evn_inputs(C, groups, lines, H):
    gfs = group_feats(groups, H)
    f = np.array([g[0] for g in gfs], np.float32).reshape(-1, GF)
    th = np.array([g[1] for g in gfs])
    cls = V.setbg.classify(th, C['model']) if len(th) else np.zeros(0, int)
    glob = np.array([H / 20, len(lines) / 10, sum((H if l['o'] == 'v' else 20) for l in lines) / 100], np.float32)
    return f, cls, glob, gfs


def evn_example(C, L, H, rng):
    lines = TG.random_layout(H, rng)
    obs = V.observe(L, lines, H, rng)
    groups = V.merge(obs)
    f, cls, glob, gfs = evn_inputs(C, groups, lines, H)
    seen = {o['tid'] for o in obs}
    hid_len = float(P.seglen(np.array([L[i] for i in range(len(L)) if i not in seen]).reshape(-1, 4)).sum())
    lt, ot = [], []
    for (feat, th, span, smin, g) in gfs:
        tid = np.bincount([o['tid'] for o in g]).argmax()
        tr = L[tid]; Lt = P.seglen(tr[None])[0]
        ux, uy = math.cos(th), math.sin(th)
        p0 = np.array([tr[0] * ux + tr[1] * uy, tr[2] * ux + tr[3] * uy])   # endpoint positions along the direction
        start = p0.min()
        frac = (smin - start) / max(Lt - span, 1e-3)
        lt.append(max(Lt, span + 0.05)); ot.append(float(np.clip(frac, 0.01, 0.99)))
    return f, cls, glob, np.array(lt, np.float32), np.array(ot, np.float32), max(hid_len, 0.5)


def evn_batch(C, items):
    B = len(items); T = max(1, max(len(i[0]) for i in items))
    gf = np.zeros((B, T, GF), np.float32); gs = np.zeros((B, T), np.int64); pad = np.ones((B, T), bool)
    lt = np.ones((B, T), np.float32); ot = np.full((B, T), 0.5, np.float32)
    for b, it in enumerate(items):
        n = len(it[0]); gf[b, :n] = it[0]; gs[b, :n] = it[1]; pad[b, :n] = False; lt[b, :n] = it[3]; ot[b, :n] = it[4]
        if n == 0: pad[b, 0] = False; gf[b, 0] = 0
    sf = torch.as_tensor(C['set_tokens'])[None]
    glob = torch.as_tensor(np.stack([it[2] for it in items])); hid = torch.as_tensor(np.array([it[5] for it in items], np.float32))
    valid = torch.as_tensor(np.array([[j < len(it[0]) for j in range(T)] for it in items]))
    return [torch.as_tensor(x) for x in (gf, gs, pad)] + [sf, glob], torch.as_tensor(lt), torch.as_tensor(ot), hid, valid


def evn_loss(net, C, items):
    inp, lt, ot, hid, valid = evn_batch(C, items)
    h = net(*inp)
    ll = torch.log(lt).unsqueeze(-1)
    nll_l = -(h['lw'] + (-0.5 * ((ll - h['lm']) / h['ls']) ** 2 - torch.log(h['ls']) - 0.919)).logsumexp(-1)
    lo = torch.log(ot / (1 - ot))
    nll_o = 0.5 * ((lo - h['om']) / h['os']) ** 2 + torch.log(h['os'])
    nll_h = 0.5 * ((torch.log(hid) - h['hm']) / h['hs']) ** 2 + torch.log(h['hs'])
    v = valid.float()
    return ((nll_l + nll_o) * v).sum() / v.sum().clamp_min(1) + nll_h.mean()


# ------------------------------------------------------------------ CRT
class CRT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = G.GAST(); self.acc = nn.Linear(G.D, 1)

    def forward(self, *inp):
        z = self.bb(*inp)
        return z


def crt_negative(C, rest, ctx, lines, H, rng, ev, kind=None):
    kind = rng.choice(4) if kind is None else kind
    if kind == 0:                                                    # EVAE sample of the fold
        s = P.clip(ev(H, rng.integers(1 << 30)), H)
        if not len(s): return None
        cand = s[rng.integers(len(s))]
    elif kind == 1 and len(ctx):                                     # near-duplicate of an existing trace
        b = ctx[rng.integers(len(ctx))]; th = math.atan2(b[3] - b[1], b[2] - b[0]); off = rng.choice([-1, 1]) * rng.uniform(0.1, max(C['dmin'], 0.3))
        cand = b + np.array([-math.sin(th) * off, math.cos(th) * off] * 2)
    elif kind == 2 or kind == 1:                                     # real trace at a random position
        r = rest[rng.integers(len(rest))]; c = (r[:2] + r[2:]) / 2; nc = np.array([rng.uniform(0, 20), rng.uniform(0, H)])
        cand = r + np.r_[nc - c, nc - c]
    else:                                                            # real hidden trace displaced along its normal
        r = rest[rng.integers(len(rest))]; th = math.atan2(r[3] - r[1], r[2] - r[0]); off = rng.choice([-1, 1]) * rng.uniform(1.5, 5)
        cand = r + np.array([-math.sin(th) * off, math.cos(th) * off] * 2)
    cand = P.clip(cand[None], H)
    if not len(cand) or any(V.crosses(cand, l)[0] for l in lines): return None
    return cand[0]


def crt_example(C, L, H, rng, ev):
    lines = TG.random_layout(H, rng); obs, t1, t2, hidden = TG.evidence(C, L, H, lines, rng)
    if len(hidden) < 2: return None
    perm = rng.permutation(len(hidden)); na = rng.integers(0, len(hidden)); acc, rest = hidden[perm[:na]], hidden[perm[na:]]
    ctx = np.vstack([t1, t2, acc]) if len(t2) else np.vstack([t1, acc]) if len(t1) else acc
    tiers = np.r_[np.zeros(len(t1)), np.ones(len(t2)), np.ones(len(acc))]
    if rng.random() < 0.5:
        cand, lab = rest[rng.integers(len(rest))], 1.0
    else:
        lab, cand = 0.0, None
        for _ in range(8):                                           # retry so negatives keep their 50% share
            c = crt_negative(C, rest, ctx, lines, H, rng, ev)
            if c is not None: cand = c; break
        if cand is None: return None
    seq = G.build_seq(C, ctx, tiers, cand[None], lines)
    return seq, lab


def crt_logits(net, C, seqs):
    inp = G.batch(C, seqs); z = net(*inp)
    idx = torch.as_tensor([len(s[0]) - 1 for s in seqs])
    return net.acc(z[torch.arange(len(seqs)), idx]).squeeze(-1)


# ------------------------------------------------------------------ training
def train(fi, seed, scheme='S2', epochs=60, per=800):
    torch.manual_seed(seed); torch.set_num_threads(2); rng = np.random.default_rng(seed)
    C = G.prep_ctx(scheme, fi)
    FW, FH = windows(C, C['sp']['fitting']); VW, VH = windows(C, C['sp']['validation'])
    ev = TG.evae_sampler(scheme, fi, seed)
    # --- EVN
    vr = np.random.default_rng(5); vex = [evn_example(C, VW[i % len(VW)], VH[i % len(VW)], vr) for i in range(400)]
    net = EVN(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5); best, bad, state = 1e9, 0, None
    for ep in range(epochs):
        ex = [evn_example(C, FW[k], FH[k], rng) for k in rng.integers(0, len(FW), per)]; net.train()
        for i in range(0, len(ex), 32):
            l = evn_loss(net, C, ex[i:i + 32]); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad(): vl = float(np.mean([evn_loss(net, C, vex[i:i + 50]).item() for i in range(0, len(vex), 50)]))
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12: break
    evn_ep = ep + 1
    torch.save(state, MODELS / f'evn_{scheme}_f{fi}_s{seed}.pt'); evn_best = best
    # --- CRT
    vex = [x for x in (crt_example(C, VW[i % len(VW)], VH[i % len(VW)], vr, ev) for i in range(600)) if x is not None]
    net = CRT(); opt = torch.optim.Adam(net.parameters(), 5e-4, weight_decay=1e-5); best, bad, state = 1e9, 0, None
    for ep in range(epochs):
        ex = [x for x in (crt_example(C, FW[k], FH[k], rng, ev) for k in rng.integers(0, len(FW), per)) if x is not None]; net.train()
        for i in range(0, len(ex), 32):
            b = ex[i:i + 32]; lg = crt_logits(net, C, [e[0] for e in b])
            l = F.binary_cross_entropy_with_logits(lg, torch.tensor([e[1] for e in b], dtype=torch.float32))
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            lg = torch.cat([crt_logits(net, C, [e[0] for e in vex[i:i + 50]]) for i in range(0, len(vex), 50)])
            y = torch.tensor([e[1] for e in vex], dtype=torch.float32)
            vl = F.binary_cross_entropy_with_logits(lg, y).item(); vacc = float(((lg > 0).float() == y).float().mean())
        if vl < best - 1e-4: best, bad, state, bacc = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}, vacc
        else:
            bad += 1
            if bad >= 12: break
    torch.save(state, MODELS / f'crt_{scheme}_f{fi}_s{seed}.pt')
    json.dump(dict(evn_val=evn_best, evn_epochs=evn_ep, crt_val=best, crt_val_acc=bacc, crt_epochs=ep + 1), open(MODELS / f'learned_{scheme}_f{fi}_s{seed}.json', 'w'))


def load(fi, seed, scheme='S2'):
    e = EVN(); e.load_state_dict(torch.load(MODELS / f'evn_{scheme}_f{fi}_s{seed}.pt')); e.eval()
    c = CRT(); c.load_state_dict(torch.load(MODELS / f'crt_{scheme}_f{fi}_s{seed}.pt')); c.eval()
    return e, c


# ------------------------------------------------------------------ learned components at inference
@torch.no_grad()
def tier2_learned(net, C, groups, lines, H, rng):
    if not groups: return np.zeros((0, 4)), None
    f, cls, glob, gfs = evn_inputs(C, groups, lines, H)
    inp, _, _, _, _ = evn_batch(C, [(f, cls, glob, np.ones(len(f), np.float32), np.full(len(f), .5, np.float32), 1.0)])
    h = net(*inp)
    out = []
    for j, (feat, th, span, smin, g) in enumerate(gfs):
        k = rng.choice(3, p=h['lw'][0, j].exp().numpy())
        Lf = max(math.exp(float(h['lm'][0, j, k] + h['ls'][0, j, k] * rng.standard_normal())), span + 0.05)
        u = 1 / (1 + math.exp(-float(h['om'][0, j] + h['os'][0, j] * rng.standard_normal())))
        ux, uy = math.cos(th), math.sin(th)
        a = u * (Lf - span); o0 = [o for o in g if abs(o['x'] * ux + o['y'] * uy - smin) < 1e-6][0]
        p1 = (o0['x'] - a * ux, o0['y'] - a * uy); out.append([p1[0], p1[1], p1[0] + Lf * ux, p1[1] + Lf * uy])
    hid = math.exp(float(h['hm'][0] + h['hs'][0] * rng.standard_normal()))
    return P.clip(np.array(out).reshape(-1, 4), H), hid


@torch.no_grad()
def hidden_length(net, C, groups, lines, H, rng):
    f, cls, glob, _ = evn_inputs(C, groups, lines, H)
    inp, _, _, _, _ = evn_batch(C, [(f, cls, glob, np.ones(len(f), np.float32), np.full(len(f), .5, np.float32), 1.0)])
    h = net(*inp)
    return math.exp(float(h['hm'][0] + h['hs'][0] * rng.standard_normal()))


@torch.no_grad()
def critic_accept(net, C, ctx, tiers, cands, lines, rng):
    if not len(cands): return np.zeros(len(cands), bool)
    seqs = [G.build_seq(C, ctx, tiers, c[None], lines) for c in cands]
    p = torch.sigmoid(torch.cat([crt_logits(net, C, seqs[i:i + 64]) for i in range(0, len(seqs), 64)])).numpy()
    return rng.random(len(cands)) < p


@torch.no_grad()
def critic_prob(net, C, ctx, tiers, cands, lines):
    seqs = [G.build_seq(C, ctx, tiers, c[None], lines) for c in cands]
    return torch.sigmoid(torch.cat([crt_logits(net, C, seqs[i:i + 64]) for i in range(0, len(seqs), 64)])).numpy().astype(float)


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        C = G.prep_ctx('S2', 0); rng = np.random.default_rng(0)
        FW, FH = windows(C, C['sp']['fitting']); ev = TG.evae_sampler('S2', 0, 1337)
        ex = [evn_example(C, FW[i], FH[i], rng) for i in range(8)]; n = EVN(); l = evn_loss(n, C, ex); l.backward(); print('evn loss', l.item())
        cx = [x for x in (crt_example(C, FW[i], FH[i], rng, ev) for i in range(12)) if x is not None]
        c = CRT(); lg = crt_logits(c, C, [e[0] for e in cx]); print('crt logits', lg.shape, 'labels', [e[1] for e in cx])
        L, H = FW[0], FH[0]; lines = V.lines_layout('v', 5.0, 0.1, H); obs = V.observe(L, lines, H, rng)
        t2, hid = tier2_learned(n, C, V.merge(obs), lines, H, rng); print('learned tier2', t2.shape, 'hidden length', round(hid, 1))
    else:
        fi, seed = int(sys.argv[1]), int(sys.argv[2]); t = time.time(); train(fi, seed); print('done', fi, seed, f'{time.time()-t:.0f}s')
