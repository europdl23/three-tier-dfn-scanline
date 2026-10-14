# -*- coding: utf-8 -*-
"""Transformer directions 2-4 (trained per fold and Paper 1 run seed; training windows of the fold only).
  TERM   learned termination: probability that an encounter between two traces is a crossing (X) rather than one trace
         ending on the other (Y), from the two traces' sets, angle, lengths and where the encounter lies along them;
         labels = real encounters in the fold's training windows. Used by EDIT and GEN instead of one global probability.
  EDIT   (direction 2) transformer editor: given the window's evidence (measured + indicated) and the traces already
         accepted, decides whether a proposed trace is real-like; trained on real hidden traces (positive) vs traces from
         the fold's own Paper 1 EVAE models sampled on the training windows (negative).
  GEN    (direction 3) geology-aware sequential generator: context = evidence tokens; generates hidden traces from the
         scanlines outward (ordered by distance to the nearest scanline), with mixture outputs (2D position mixture,
         log-length mixture, axial von Mises mixture on 2 theta, set head, stop head) and the Paper 1 geology guidance
         (direction-distribution W1 and set-share L1 of the generated window against the real hidden traces).
  GENPT  (direction 4) GEN pretrained on windows sampled from the fold's Paper 1 EVAE models, then fine-tuned on real.
Units: metres (window 20 m x H); tokens scaled by 20 m."""
import os, sys, math, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p2lib as P
import p2v2 as V
import tr_common as TC
from run_v3_term import protect_spans

MODELS = HERE / 'tr_models'; MODELS.mkdir(exist_ok=True)
K1MAX, NT = 5, 4                      # max set classes (4 sets + background), tier flags
FDIM = 5 + K1MAX + NT + 1
NMAXSEQ = 40
SEEDS = (1337, 20260903, 7)


# ------------------------------------------------------------------ tokens
def tokens(L, tier, H, lines, model):
    L = np.asarray(L, float).reshape(-1, 4)
    if not len(L):
        return np.zeros((0, FDIM), np.float32)
    th = P.segdir(L); c = V.setbg.classify(th, model)
    mx, my = (L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2
    d = np.array([min([abs(x - l['c']) for l in lines] or [20.0]) for x in mx])
    T = np.zeros((len(L), FDIM), np.float32)
    T[:, 0] = mx / 20; T[:, 1] = my / 20; T[:, 2] = np.log(np.maximum(P.seglen(L), 0.05)) / 3; T[:, 3] = np.cos(2 * th); T[:, 4] = np.sin(2 * th)
    T[np.arange(len(L)), 5 + c] = 1; T[:, 5 + K1MAX + tier] = 1; T[:, -1] = d / 20
    return T


def order_from_evidence(L, lines):
    mx = (L[:, 0] + L[:, 2]) / 2; my = (L[:, 1] + L[:, 3]) / 2
    d = np.array([min([abs(x - l['c']) for l in lines] or [20.0]) for x in mx])
    return np.lexsort((my, d))


# ------------------------------------------------------------------ learned termination (TERM)
def enc_features(a, b, t_a, model):
    """features of an encounter where trace a meets trace b at arc fraction t_a along a"""
    tha, thb = math.atan2(a[3] - a[1], a[2] - a[0]) % math.pi, math.atan2(b[3] - b[1], b[2] - b[0]) % math.pi
    ca, cb = V.setbg.classify(np.array([tha, thb]), model)
    f = np.zeros(4 + 2 * K1MAX, np.float32)
    f[0] = abs(math.sin(tha - thb)); f[1] = math.log(max(math.hypot(a[2] - a[0], a[3] - a[1]), 0.05)) / 3
    f[2] = math.log(max(math.hypot(b[2] - b[0], b[3] - b[1]), 0.05)) / 3; f[3] = min(t_a, 1 - t_a)
    f[4 + ca] = 1; f[4 + K1MAX + cb] = 1
    return f


def encounters(L, tol=0.3):
    """real encounters: (i, j, t_i, label) label 1 = crossing (X), 0 = i ends on j (Y)."""
    out = []
    for i, j in P.crossing_pairs(L):
        p, r = L[i, :2], L[i, 2:] - L[i, :2]; q, s = L[j, :2], L[j, 2:] - L[j, :2]
        den = r[0] * s[1] - r[1] * s[0]
        if abs(den) < 1e-12: continue
        d = q - p; t = (d[0] * s[1] - d[1] * s[0]) / den; u = (d[0] * r[1] - d[1] * r[0]) / den
        out.append((i, j, t, 1) if (i + j) % 2 else (j, i, u, 1))   # one decision per crossing (as in terminate)
    n = len(L)
    if n > 1:
        E = np.vstack([L[:, :2], L[:, 2:]]); own = np.r_[np.arange(n), np.arange(n)]
        D = P._pt_seg(E, L[:, :2], L[:, 2:]); D[np.arange(2 * n), own] = np.inf
        for e in np.where(D.min(1) <= tol)[0]:
            i = own[e]; out.append((i, int(D[e].argmin()), 0.0 if e < n else 1.0, 0))
    return out


class TermNet(nn.Module):
    def __init__(self):
        super().__init__(); self.f = nn.Sequential(nn.Linear(4 + 2 * K1MAX, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.f(x).squeeze(-1)


def train_term(C, seed):
    torch.manual_seed(seed)
    X, Y = [], []
    for L in C['TW']:
        for i, j, t, lab in encounters(L):
            X.append(enc_features(L[i], L[j], t, C['model'])); Y.append(lab)
    X = torch.as_tensor(np.array(X)); Y = torch.as_tensor(np.array(Y, np.float32))
    net = TermNet(); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    for ep in range(300):
        l = F.binary_cross_entropy_with_logits(net(X), Y); opt.zero_grad(); l.backward(); opt.step()
    net.eval(); return net


def terminate_learned(rec, keep_n, rng, term, model, protect=None):
    """as p2v2.terminate, but the pass probability of each encounter comes from the learned TERM network"""
    rec = np.array(rec, float).reshape(-1, 4); keep_n = min(keep_n, len(rec))
    order = list(range(keep_n)) + list(keep_n + rng.permutation(len(rec) - keep_n)); done = []
    for i in order:
        x1, y1, x2, y2 = rec[i]; L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-6 or not done:
            done.append(i); continue
        ps = protect[i] if protect is not None and i < len(protect) and protect[i] is not None else (L / 2, L / 2)
        lo, hi = 0.0, L
        for j in done:
            q = rec[j]; p, r = np.array([x1, y1]), np.array([x2 - x1, y2 - y1]); qq, s = q[:2], q[2:] - q[:2]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12: continue
            d = qq - p; t = (d[0] * s[1] - d[1] * s[0]) / den; u = (d[0] * r[1] - d[1] * r[0]) / den
            if not (0 < t < 1 and 0 < u < 1): continue
            a = t * L
            if ps[0] <= a <= ps[1]: continue
            with torch.no_grad():
                pp = torch.sigmoid(term(torch.as_tensor(enc_features(rec[i], q, t, model)[None]))).item()
            if rng.random() < pp: continue
            if a < ps[0]: lo = max(lo, a)
            else: hi = min(hi, a)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        rec[i] = [x1 + lo * ux, y1 + lo * uy, x1 + hi * ux, y1 + hi * uy]; done.append(i)
    return rec


# ------------------------------------------------------------------ EVAE samples on training windows (no test leakage)
_EV = {}
def evae_sampler(scheme, fi, seed):
    key = (scheme, fi, seed)
    if key not in _EV:
        sys.path.insert(0, str(P.REL / 'src' / 'evaluation'))
        import generators as G
        G.BRANCH = P.REL / 'src' / 'model'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        m, dev = G.load_evae(P.REL / 'models' / 'evae' / f'{scheme}_f{fi}_s{seed}' / 'checkpoint.pt')
        _EV[key] = (G, m, dev)
    G, m, dev = _EV[key]
    return lambda H, s: G.generate_evae(m, dev, H / 20.0, int(s)).reshape(-1, 4) * 20.0


# ------------------------------------------------------------------ examples
def random_layout(H, rng):
    s = rng.choice([None, 10.0, 7.0, 5.0, 3.0], p=[0.25, 0.2, 0.15, 0.25, 0.15])
    return V.lines_layout('v', s, 0.10, H, rng=rng, shift=True)


def evidence(C, L, H, lines, rng):
    obs = V.observe(L, lines, H, rng); groups = V.merge(obs)
    t1 = V.tier1(obs)
    t2 = V.tier2(groups, C['th2'], C['tr_len2d'], H, rng)
    if len(t2) > 1:
        t2 = V.terminate(t2, 0, rng, C['p_pass'], protect=protect_spans(t2, obs))
    seen = {o['tid'] for o in obs}
    hidden = np.array([L[i] for i in range(len(L)) if i not in seen]).reshape(-1, 4)
    return obs, t1, t2, hidden


# ------------------------------------------------------------------ EDIT
class Editor(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FDIM, d), nn.ReLU(), nn.Linear(d, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True), 3)
        self.out = nn.Linear(d, 1)
    def forward(self, tok, pad):          # last valid token = candidate
        z = self.enc(self.inp(tok), src_key_padding_mask=pad)
        idx = (~pad).sum(1) - 1
        return self.out(z[torch.arange(len(z)), idx]).squeeze(-1)


def pad_batch(seqs):
    T = max(len(s) for s in seqs); B = len(seqs)
    tok = np.zeros((B, T, FDIM), np.float32); pad = np.ones((B, T), bool)
    for i, s in enumerate(seqs):
        tok[i, :len(s)] = s; pad[i, :len(s)] = False
    return torch.as_tensor(tok), torch.as_tensor(pad)


def edit_examples(C, ev, rng, n_win=300):
    model = C['model']; out = []
    for _ in range(n_win):
        k = rng.integers(len(C['TW'])); L, H = C['TW'][k], C['TH'][k]
        lines = random_layout(H, rng); obs, t1, t2, hidden = evidence(C, L, H, lines, rng)
        if not len(hidden): continue
        acc_n = rng.integers(0, len(hidden)); perm = rng.permutation(len(hidden))
        acc, rest = hidden[perm[:acc_n]], hidden[perm[acc_n:]]
        ctx = np.vstack([tokens(t1, 0, H, lines, model), tokens(t2, 1, H, lines, model), tokens(acc, 2, H, lines, model)])
        neg = P.clip(ev(H, rng.integers(1 << 30)), H)
        neg = neg[[not any(V.crosses(t[None], l)[0] for l in lines) for t in neg]] if len(neg) else neg
        for t in rest[:3]:
            out.append((np.vstack([ctx, tokens(t[None], 3, H, lines, model)])[-NMAXSEQ * 2:], 1.0))
        for t in (neg[rng.permutation(len(neg))[:3]] if len(neg) else []):
            out.append((np.vstack([ctx, tokens(t[None], 3, H, lines, model)])[-NMAXSEQ * 2:], 0.0))
    return out


def train_edit(scheme, fi, seed, C, epochs=30):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); ev = evae_sampler(scheme, fi, seed)
    net = Editor(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    Cv = dict(C, TW=C['VW'], TH=C['VH']); vex = edit_examples(Cv, ev, np.random.default_rng(9), 120)
    best, bad, state = 1e9, 0, None
    for ep in range(epochs):
        ex = edit_examples(C, ev, rng); net.train()
        for i in range(0, len(ex), 64):
            tok, pad = pad_batch([e[0] for e in ex[i:i + 64]]); y = torch.as_tensor([e[1] for e in ex[i:i + 64]], dtype=torch.float32)
            l = F.binary_cross_entropy_with_logits(net(tok, pad), y); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tok, pad = pad_batch([e[0] for e in vex]); y = torch.as_tensor([e[1] for e in vex], dtype=torch.float32)
            vl = F.binary_cross_entropy_with_logits(net(tok, pad), y).item()
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    net.load_state_dict(state); net.eval(); return net, best


def make_editor(net, model, lines, H, rng, n_t1):
    """returns editor(rec, pool, n2) -> accepted candidates (sampled with the predicted probability)"""
    def ed(rec, pool, n2):
        pool = P.clip(pool, H)
        if not len(pool): return pool
        tiers = np.r_[np.full(n2, 1), np.full(len(rec) - n2, 2)] if len(rec) else np.zeros(0)
        ctx = np.vstack([tokens(rec[tiers == 1], 1, H, lines, model), tokens(rec[tiers == 2], 2, H, lines, model)]) if len(rec) else np.zeros((0, FDIM), np.float32)
        seqs = [np.vstack([ctx, tokens(t[None], 3, H, lines, model)])[-NMAXSEQ * 2:] for t in pool]
        with torch.no_grad():
            p = torch.sigmoid(net(*pad_batch(seqs))).numpy()
        return pool[rng.random(len(pool)) < p]
    return ed


# ------------------------------------------------------------------ GEN (directions 3 and 4)
KP, KL, KA = 4, 3, 3


class Gen(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.ci = nn.Sequential(nn.Linear(FDIM, d), nn.ReLU(), nn.Linear(d, d))
        self.ti = nn.Sequential(nn.Linear(FDIM, d), nn.ReLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, NMAXSEQ + 1, d) * 0.02)
        self.g = nn.Parameter(torch.randn(1, 1, d) * 0.02)          # global token (always present, also with no evidence)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True), 3)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 4 * d, 0.1, batch_first=True), 3)
        self.h_pos = nn.Linear(d, KP * 5); self.h_len = nn.Linear(d, KL * 3); self.h_ang = nn.Linear(d, KA * 3)
        self.h_set = nn.Linear(d, K1MAX); self.h_stop = nn.Linear(d, 1)

    def forward(self, ctx, cpad, tgt, tpad):
        B = ctx.shape[0]
        mem = self.enc(torch.cat([self.g.expand(B, -1, -1), self.ci(ctx)], 1),
                       src_key_padding_mask=torch.cat([torch.zeros(B, 1, dtype=torch.bool), cpad], 1))
        n = tgt.shape[1]
        x = self.ti(tgt) + self.pos[:, :n]
        cm = nn.Transformer.generate_square_subsequent_mask(n)
        z = self.dec(x, mem, tgt_mask=cm, tgt_key_padding_mask=tpad,
                     memory_key_padding_mask=torch.cat([torch.zeros(B, 1, dtype=torch.bool), cpad], 1))
        return z

    def heads(self, z, Hn):
        B, T, _ = z.shape
        p = self.h_pos(z).view(B, T, KP, 5)
        pw = F.log_softmax(p[..., 0], -1); pm = torch.stack([torch.sigmoid(p[..., 1]), torch.sigmoid(p[..., 2]) * Hn.view(B, 1, 1)], -1)
        ps = F.softplus(p[..., 3:5]) * 0.2 + 0.01
        l = self.h_len(z).view(B, T, KL, 3); lw = F.log_softmax(l[..., 0], -1); lm = l[..., 1]; ls = F.softplus(l[..., 2]) * 0.5 + 0.05
        a = self.h_ang(z).view(B, T, KA, 3); aw = F.log_softmax(a[..., 0], -1); am = math.pi * torch.tanh(a[..., 1]); ak = F.softplus(a[..., 2]) + 0.5
        return dict(pw=pw, pm=pm, ps=ps, lw=lw, lm=lm, ls=ls, aw=aw, am=am, ak=ak, set=self.h_set(z), stop=self.h_stop(z).squeeze(-1))


def _vm_logpdf(x, mu, k):
    return k * torch.cos(x - mu) - math.log(2 * math.pi) - (torch.log(torch.special.i0e(k)) + k)


def gen_batch(items):
    B = len(items); Tc = max(1, max(len(i[0]) for i in items)); Tt = max(len(i[1]) for i in items) + 1
    ctx = np.zeros((B, Tc, FDIM), np.float32); cpad = np.ones((B, Tc), bool)
    tin = np.zeros((B, Tt, FDIM), np.float32); tpad = np.ones((B, Tt), bool)
    tgt = np.zeros((B, Tt, 6), np.float32); stop = np.zeros((B, Tt), np.float32); setl = np.full((B, Tt), -1, np.int64)
    Hn = np.zeros(B, np.float32); realdir = []; realset = np.zeros((B, K1MAX), np.float32)
    for b, (c, t, raw, cls, H) in enumerate(items):
        ctx[b, :len(c)] = c; cpad[b, :len(c)] = False
        if not len(c): cpad[b, 0] = False
        n = len(t); tin[b, 1:n + 1] = t; tpad[b, :n + 1] = False
        tgt[b, :n] = raw; stop[b, n] = 1; setl[b, :n] = cls; Hn[b] = H / 20
        realdir.append(raw[:, 3] if n else np.zeros(0)); realset[b] = np.bincount(cls, minlength=K1MAX) if n else 0
    return [torch.as_tensor(a) for a in (ctx, cpad, tin, tpad, tgt, stop, setl, Hn, realset)], realdir


def gen_items(C, rng, n, source='real', ev=None):
    """(context tokens, target tokens, raw target [x,y,logL,2theta,..], set classes, H)"""
    model = C['model']; out = []
    while len(out) < n:
        k = rng.integers(len(C['TW'])); H = C['TH'][k]
        L = C['TW'][k] if source == 'real' else P.clip(ev(H, rng.integers(1 << 30)), H)
        lines = random_layout(H, rng); obs, t1, t2, hidden = evidence(C, L, H, lines, rng)
        ctx = np.vstack([tokens(t1, 0, H, lines, model), tokens(t2, 1, H, lines, model)])
        if len(hidden):
            hidden = hidden[order_from_evidence(hidden, lines)][:NMAXSEQ - 1]
        tt = tokens(hidden, 2, H, lines, model)
        th = P.segdir(hidden) if len(hidden) else np.zeros(0)
        raw = np.column_stack([(hidden[:, 0] + hidden[:, 2]) / 2 / 20, (hidden[:, 1] + hidden[:, 3]) / 2 / 20,
                               np.log(np.maximum(P.seglen(hidden), 0.05)), 2 * th, np.zeros(len(th)), np.zeros(len(th))]) if len(hidden) else np.zeros((0, 6))
        cls = V.setbg.classify(th, model) if len(th) else np.zeros(0, int)
        out.append((ctx[-NMAXSEQ * 2:], tt, raw.astype(np.float32), cls, H))
    return out


def gen_loss(net, batch, realdir, geo=True):
    ctx, cpad, tin, tpad, tgt, stop, setl, Hn, realset = batch
    z = net(ctx, cpad, tin, tpad); h = net.heads(z, Hn)
    valid = setl >= 0
    x = tgt[..., :2].unsqueeze(-2)
    lp = (h['pw'] + (-0.5 * ((x - h['pm']) / h['ps']) ** 2 - torch.log(h['ps']) - 0.919).sum(-1)).logsumexp(-1)
    ll = (h['lw'] + (-0.5 * ((tgt[..., 2:3] - h['lm']) / h['ls']) ** 2 - torch.log(h['ls']) - 0.919)).logsumexp(-1)
    la = (h['aw'] + _vm_logpdf(tgt[..., 3:4], h['am'], h['ak'])).logsumexp(-1)
    nll = -(lp + ll + la)[valid].mean() if valid.any() else torch.zeros(())
    lst = F.binary_cross_entropy_with_logits(h['stop'][~tpad], stop[~tpad])
    lset = F.cross_entropy(h['set'][valid], setl[valid]) if valid.any() else torch.zeros(())
    loss = nll + lst + lset
    if geo and valid.any():
        # Paper 1-style geology guidance: generated direction distribution (step mixtures averaged over the window,
        # weighted by the probability of not having stopped) vs the real hidden directions; set-share L1
        nb = 36; grid = (torch.arange(nb) + 0.5) * (2 * math.pi / nb)
        dens = (h['aw'].exp().unsqueeze(-1) * torch.exp(h['ak'].unsqueeze(-1) * (torch.cos(grid - h['am'].unsqueeze(-1)) - 1))).sum(-2)
        dens = dens / dens.sum(-1, keepdim=True).clamp_min(1e-9)
        G = (dens * valid.unsqueeze(-1)).sum(1); G = G / G.sum(-1, keepdim=True).clamp_min(1e-9)
        Rr = torch.zeros_like(G)
        for b, d in enumerate(realdir):
            if len(d):
                r = torch.exp(20 * (torch.cos(grid.view(1, -1) - torch.as_tensor(d).view(-1, 1)) - 1)); r = r / r.sum(-1, keepdim=True)
                Rr[b] = r.mean(0)
        has = Rr.sum(-1) > 0
        D = torch.cumsum(G - Rr, -1); w1 = (D - D.median(-1, keepdim=True).values).abs().mean(-1)
        ps = (F.softmax(h['set'], -1) * valid.unsqueeze(-1)).sum(1); ps = ps / ps.sum(-1, keepdim=True).clamp_min(1e-9)
        rs = realset / realset.sum(-1, keepdim=True).clamp_min(1e-9)
        loss = loss + w1[has].mean() + (ps - rs).abs().sum(-1)[has].mean()
    return loss


def train_gen(C, seed, epochs=60, pretrain=None, n_per_epoch=400, tag=''):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Gen(); opt = torch.optim.Adam(net.parameters(), 5e-4, weight_decay=1e-5)
    Cv = dict(C, TW=C['VW'], TH=C['VH']); vit = gen_items(Cv, np.random.default_rng(7), 150)
    def run(src, ev, n_ep, val=True):
        best, bad, state = 1e9, 0, None
        for ep in range(n_ep):
            it = gen_items(C, rng, n_per_epoch, src, ev); net.train()
            for i in range(0, len(it), 32):
                b, rd = gen_batch(it[i:i + 32]); l = gen_loss(net, b, rd)
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if not val: continue
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([gen_loss(net, *gen_batch(vit[i:i + 50])).item() for i in range(0, len(vit), 50)]))
            if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 8: break
        if state: net.load_state_dict(state)
        return best
    if pretrain is not None:
        run('evae', pretrain, 25, val=False)
    best = run('real', None, epochs)
    net.eval(); return net, best


@torch.no_grad()
def generate(net, C, t1, t2, lines, H, rng, max_n=NMAXSEQ, tries=5):
    """sample hidden traces sequentially; a trace crossing a scanline is resampled (up to `tries`), then dropped"""
    model = C['model']
    ctx = np.vstack([tokens(t1, 0, H, lines, model), tokens(t2, 1, H, lines, model)])
    c = torch.as_tensor(ctx[None]) if len(ctx) else torch.zeros(1, 1, FDIM)
    cpad = torch.zeros(1, c.shape[1], dtype=torch.bool)
    out = []; tin = torch.zeros(1, 1, FDIM); Hn = torch.tensor([H / 20])
    torch.manual_seed(int(rng.integers(1 << 30)))
    for step in range(max_n):
        z = net(c, cpad, tin, torch.zeros(1, tin.shape[1], dtype=torch.bool)); h = net.heads(z[:, -1:], Hn)
        if rng.random() < torch.sigmoid(h['stop'][0, 0]).item():
            break
        seg = None
        for _ in range(tries):
            kp = rng.choice(KP, p=h['pw'][0, 0].exp().numpy()); mx, my = (h['pm'][0, 0, kp] + h['ps'][0, 0, kp] * torch.randn(2)).numpy() * 20
            kl = rng.choice(KL, p=h['lw'][0, 0].exp().numpy()); Lg = math.exp(float(h['lm'][0, 0, kl] + h['ls'][0, 0, kl] * torch.randn(())))
            ka = rng.choice(KA, p=h['aw'][0, 0].exp().numpy()); th = (rng.vonmises(float(h['am'][0, 0, ka]), float(h['ak'][0, 0, ka])) / 2) % math.pi
            s = np.array([[mx - math.cos(th) * Lg / 2, my - math.sin(th) * Lg / 2, mx + math.cos(th) * Lg / 2, my + math.sin(th) * Lg / 2]])
            s = P.clip(s, H)
            if len(s) and not any(V.crosses(s, l)[0] for l in lines):
                seg = s[0]; break
        if seg is None:
            continue
        out.append(seg)
        tin = torch.cat([tin, torch.as_tensor(tokens(seg[None], 2, H, lines, model)[None])], 1)
    return np.array(out).reshape(-1, 4)


# ------------------------------------------------------------------ training driver
def train_all(scheme, fi, seed):
    C = TC.fold_ctx(scheme, fi)
    B = C['boxes']; sp = C['sp']; held = set().union(*[set(B[t]['ids']) for t in sp['test']])
    VW, VH = [], []
    for k in sp['validation']:
        b = B[k]; keep = np.array([i not in held for i in b['ids']], bool); VW.append(P.clip(b['L'][keep], b['H'])); VH.append(b['H'])
    # training windows only = fitting (validation kept apart for early stopping)
    FW, FH = [], []
    for k in sp['fitting']:
        b = B[k]; keep = np.array([i not in held for i in b['ids']], bool); FW.append(P.clip(b['L'][keep], b['H'])); FH.append(b['H'])
    Ct = dict(C, TW=FW, TH=FH, VW=VW, VH=VH)
    t0 = time.time(); tag = f'{scheme}_f{fi}_s{seed}'
    term = train_term(Ct, seed); torch.save(term.state_dict(), MODELS / f'term_{tag}.pt')
    ed, bed = train_edit(scheme, fi, seed, Ct); torch.save(ed.state_dict(), MODELS / f'edit_{tag}.pt')
    g, bg = train_gen(Ct, seed); torch.save(g.state_dict(), MODELS / f'gen_{tag}.pt')
    gp, bgp = train_gen(Ct, seed, pretrain=evae_sampler(scheme, fi, seed)); torch.save(gp.state_dict(), MODELS / f'genpt_{tag}.pt')
    json.dump(dict(edit_val=bed, gen_val=bg, genpt_val=bgp, minutes=(time.time() - t0) / 60), open(MODELS / f'gen_{tag}.json', 'w'))
    print('done', tag, round((time.time() - t0) / 60, 1), 'min', flush=True)


def load_models(scheme, fi, seed):
    tag = f'{scheme}_f{fi}_s{seed}'
    t = TermNet(); t.load_state_dict(torch.load(MODELS / f'term_{tag}.pt')); t.eval()
    e = Editor(); e.load_state_dict(torch.load(MODELS / f'edit_{tag}.pt')); e.eval()
    g = Gen(); g.load_state_dict(torch.load(MODELS / f'gen_{tag}.pt')); g.eval()
    gp = Gen(); gp.load_state_dict(torch.load(MODELS / f'genpt_{tag}.pt')); gp.eval()
    return t, e, g, gp


if __name__ == '__main__':
    torch.set_num_threads(2)
    if sys.argv[1:2] == ['smoke']:
        C = TC.fold_ctx('S1', 0); rng = np.random.default_rng(0)
        B = C['boxes']; sp = C['sp']
        Ct = dict(C, VW=C['TW'][:3], VH=C['TH'][:3])
        t = time.time(); term = train_term(Ct, 1); print('term', time.time() - t)
        it = gen_items(Ct, rng, 8); b, rd = gen_batch(it); g = Gen(); print('gen loss', gen_loss(g, b, rd).item())
        ev = evae_sampler('S1', 0, 1337); s = ev(20.0, 3); print('evae sample', s.shape)
        ex = edit_examples(Ct, ev, rng, 5); print('edit ex', len(ex))
        L = Ct['TW'][0]; H = Ct['TH'][0]; lines = V.lines_layout('v', 5.0, 0.1, H); obs, t1, t2, hid = evidence(Ct, L, H, lines, rng)
        print('gen sample', generate(g, Ct, t1, t2, lines, H, rng).shape)
    else:
        train_all(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
