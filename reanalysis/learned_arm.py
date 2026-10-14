# -*- coding: utf-8 -*-
"""RA-32: frozen learned arm candidate (MIX_ETP-type) with a switchable termination module.

One reconstruction of the learned configuration of AD-05, exactly as the MIX_ETP arm of the study's east_test.py
(run_job, lines 95-111), except that the termination module can be chosen:
  learned indicated tier   learned.tier2_learned (evidence network of the fold and run seed)
  intensity                rule R2, p2v2.budget from the scanline hits (prior weight A_PRIOR = 2.0)
  candidate pool           sequential transformer (gast.generate, variant-2 models) proposes centres; each proposed trace
                           is replaced by a randomly chosen trace of EVAE realisation (offset + k) mod 24, translated to the
                           proposed centre (direction and length come from the EVAE trace)
  placement                rules R1, R4, R3 to the budget, then the R5 clean-up (eval_learned.rule_place)
  termination              observed spans of the indicated traces protected (eval_learned.protect_spans); then pieces of
                           0.05 m or less removed. Modes:
                             'old'     study TermNet (tr_gen.terminate_learned; feature f3 leaks the label, REV-05)
                             'rebuilt' RA-32 network TermNet2 with features known at inference for both classes
                             'rule'    p2v2.terminate with the fold's training pass probability C['p_pass']

The rebuilt features (enc_features_v2) are shared by training (ra32_train_termnet.py) and inference (terminate), so the
same code computes them in both places. Units: metres; window [0, 20] x [0, H].
Study code and release code are imported read-only; nothing is written by this module."""
import os, sys, math, time, json
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np

STUDY = Path(__file__).resolve().parents[1] / 'study'
REL = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[1]
RA32_MODELS = PROJECT / 'models' / 'termination'

# import order as in east_test.py: study folder first, then the release scripts and src/study in front of it
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))
import p2lib as P            # noqa: E402
import p2v2 as V             # noqa: E402
import final_eval as FE      # noqa: E402
from run_v2 import sd        # noqa: E402
for _p in (str(P.REL / 'scripts'), str(P.REL / 'src' / 'study')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                 # noqa: E402
import torch.nn as nn        # noqa: E402

VERSION = 'RA32 learned_arm 1.0 (2026-10-09)'
K1MAX = 5                    # 4 trace-direction clusters + unclustered (as tr_gen.K1MAX)
NF2 = 1 + 4 + 2 * K1MAX + 3  # 18 rebuilt features
R_LOCAL = 2.0                # radius of the local-intensity disc (m)
P21_SCALE = 0.25             # scale of the local intensity (m/m2)
NGRID = 81                   # grid for the disc area inside the window
TOL, EDGE = 0.3, 0.1         # Paper 1 node tolerance and window-edge margin (p2lib.TOL, p2lib.EDGE)
MIN_ANGLE_DEG = 10.0         # encounters below this angle are excluded from training and evaluation (both classes)
DEFAULT_ARM = 'LEARNED_ETP'  # name and seed key proposed for the frozen arm (a new key; MIX_ETP is the old arm)
MODES = ('old', 'rebuilt', 'rule')

_G = (np.arange(NGRID) + 0.5) / NGRID * 2.0 - 1.0
_GX, _GY = np.meshgrid(_G, _G)
_IN_DISC = (_GX ** 2 + _GY ** 2) <= 1.0


# ------------------------------------------------------------------ rebuilt network
class TermNet2(nn.Module):
    """same widths as tr_gen.TermNet; input = the 18 rebuilt features; output = logit of P(the trace passes = crossing)"""
    def __init__(self, nin=NF2):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


# ------------------------------------------------------------------ rebuilt features (training and inference)
def seg_len_in_disc(S, c, R):
    """length (m) of each segment of S (n, 4) inside the disc of radius R centred at c"""
    S = np.asarray(S, float).reshape(-1, 4)
    if not len(S):
        return np.zeros(0)
    p0 = S[:, :2]; d = S[:, 2:] - S[:, :2]
    a = (d ** 2).sum(1); f = p0 - np.asarray(c, float)[None]
    b = 2.0 * (f * d).sum(1); cc = (f ** 2).sum(1) - R * R
    disc = b * b - 4.0 * a * cc
    ok = (a > 1e-18) & (disc > 0)
    aa = np.where(ok, a, 1.0); sq = np.sqrt(np.where(ok, disc, 0.0))
    s1 = np.clip((-b - sq) / (2.0 * aa), 0.0, 1.0); s2 = np.clip((-b + sq) / (2.0 * aa), 0.0, 1.0)
    return np.where(ok, np.clip(s2 - s1, 0.0, None) * np.sqrt(a), 0.0)


def disc_area_in_window(c, R, H, W=20.0):
    """area (m2) of the disc of radius R at c that lies inside [0, W] x [0, H] (fixed NGRID x NGRID grid)"""
    X = c[0] + R * _GX; Y = c[1] + R * _GY
    inside = _IN_DISC & (X >= 0) & (X <= W) & (Y >= 0) & (Y <= H)
    return float(inside.mean()) * (2.0 * R) ** 2


def enc_features_v2(a, b, u, others, H, model):
    """features of an encounter of trace a (the deciding trace) with trace b at fraction u along b.
    Depends only on the directions of a and b, the final geometry of b, the encounter point on b and the other traces
    (a and b excluded): never on the length of a or on where the encounter lies along a."""
    tha = math.atan2(a[3] - a[1], a[2] - a[0]) % math.pi
    thb = math.atan2(b[3] - b[1], b[2] - b[0]) % math.pi
    ca, cb = V.setbg.classify(np.array([tha, thb]), model)
    f = np.zeros(NF2, np.float32)
    f[0] = abs(math.sin(tha - thb))
    f[1] = math.cos(2 * tha); f[2] = math.sin(2 * tha); f[3] = math.cos(2 * thb); f[4] = math.sin(2 * thb)
    f[5 + ca] = 1; f[5 + K1MAX + cb] = 1
    Lb = math.hypot(b[2] - b[0], b[3] - b[1])
    f[15] = math.log(max(Lb, 0.05)) / 3
    f[16] = min(u, 1 - u)
    Pp = np.array([b[0] + u * (b[2] - b[0]), b[1] + u * (b[3] - b[1])])
    O = np.asarray(others, float).reshape(-1, 4)
    ln = float(seg_len_in_disc(O, Pp, R_LOCAL).sum()) if len(O) else 0.0
    area = disc_area_in_window(Pp, R_LOCAL, H)
    f[17] = math.log1p((ln / area if area > 0 else 0.0) / P21_SCALE)
    return f


def line_params(a, b):
    """t along a and u along b of the intersection of the lines of a and b (formula of tr_gen / p2v2.terminate)"""
    p, r = a[:2], a[2:] - a[:2]; q, s = b[:2], b[2:] - b[:2]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    d = q - p
    return (d[0] * s[1] - d[1] * s[0]) / den, (d[0] * r[1] - d[1] * r[0]) / den


def encounters_v2(L, H, model, return_old=True):
    """real encounters of one window, RA-32 definition (RA32_PRESPECIFICATION.md Sec. 3).
    Returns (examples, counts). Each example: dict(a, b, label (1 crossing, 0 termination), weight, u, t_old, x (rebuilt
    features), x_old (old tr_gen features, as the old network was trained: t_old = 0/1 for a termination)."""
    import tr_gen as TG
    L = np.asarray(L, float).reshape(-1, 4); n = len(L)
    cnt = dict(Y_cand=0, Y_edge=0, Y_outside_b=0, Y_angle=0, Y=0, X_pairs=0, X_overshoot=0, X_angle=0, X=0, Y_gmax=0.0)
    ex = []
    if n < 2:
        return ex, cnt
    smin = math.sin(math.radians(MIN_ANGLE_DEG))
    th = P.segdir(L)
    # terminations: an end inside the window lying within TOL of its nearest other trace
    E = np.vstack([L[:, :2], L[:, 2:]]); own = np.r_[np.arange(n), np.arange(n)]
    D = P._pt_seg(E, L[:, :2], L[:, 2:]); D[np.arange(2 * n), own] = np.inf
    ypairs = set()
    for e in np.where(D.min(1) <= TOL)[0]:
        cnt['Y_cand'] += 1
        x, y = E[e]
        if not (EDGE < x < P.M - EDGE and EDGE < y < H - EDGE):
            cnt['Y_edge'] += 1; continue
        i = int(own[e]); j = int(D[e].argmin())
        if abs(math.sin(th[i] - th[j])) < smin:
            cnt['Y_angle'] += 1; continue
        tu = line_params(L[i], L[j])
        if tu is None or not (0 < tu[1] < 1):
            cnt['Y_outside_b'] += 1; continue
        t_e = 0.0 if e < n else 1.0
        g = abs(tu[0] - t_e) * math.hypot(L[i, 2] - L[i, 0], L[i, 3] - L[i, 1])
        cnt['Y_gmax'] = max(cnt['Y_gmax'], g)
        others = L[[k for k in range(n) if k != i and k != j]]
        d_ = dict(a=i, b=j, label=0, weight=1.0, u=float(tu[1]), t_old=t_e, end=int(e >= n),
                  x=enc_features_v2(L[i], L[j], float(tu[1]), others, H, model))
        if return_old:
            d_['x_old'] = TG.enc_features(L[i], L[j], t_e, model)
        ex.append(d_); ypairs.add((i, j)); cnt['Y'] += 1
    # crossings: strict intersections, not an overshooting termination of one trace on the other
    for i, j in P.crossing_pairs(L):
        cnt['X_pairs'] += 1
        if (i, j) in ypairs or (j, i) in ypairs:
            cnt['X_overshoot'] += 1; continue
        if abs(math.sin(th[i] - th[j])) < smin:
            cnt['X_angle'] += 1; continue
        tu = line_params(L[i], L[j])
        if tu is None:
            cnt['X_angle'] += 1; continue
        t, u = float(tu[0]), float(tu[1])
        others = L[[k for k in range(n) if k != i and k != j]]
        for a_, b_, ta, ub in ((i, j, t, u), (j, i, u, t)):
            d_ = dict(a=a_, b=b_, label=1, weight=0.5, u=ub, t_old=ta, end=-1,
                      x=enc_features_v2(L[a_], L[b_], ub, others, H, model))
            if return_old:
                d_['x_old'] = TG.enc_features(L[a_], L[b_], ta, model)
            ex.append(d_)
        cnt['X'] += 1
    return ex, cnt


# ------------------------------------------------------------------ termination (inference)
def terminate(rec, keep_n, rng, model, H, mode, net=None, p_pass=None, protect=None):
    """p2v2.terminate / tr_gen.terminate_learned with a switchable pass probability. Traces keep_n.. are visited in random
    order (indicated traces first); where a trace crosses an earlier (already decided) trace outside its protected span,
    it passes with probability p, otherwise it is cut there. p = p_pass ('rule'), sigmoid(old TermNet) ('old') or
    sigmoid(TermNet2) ('rebuilt'). Random draws: one rng.random() per unprotected crossing in every mode."""
    if mode not in MODES:
        raise ValueError(mode)
    rec = np.array(rec, float).reshape(-1, 4); keep_n = min(keep_n, len(rec))
    order = list(range(keep_n)) + list(keep_n + rng.permutation(len(rec) - keep_n)); done = []
    if mode == 'old':
        import tr_gen as TG
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
            if mode == 'rule':
                if rng.random() < p_pass: continue
            else:
                if mode == 'old':
                    x = TG.enc_features(rec[i], q, t, model)
                else:
                    others = rec[[k for k in range(len(rec)) if k != i and k != j]]
                    x = enc_features_v2(rec[i], q, u, others, H, model)
                with torch.no_grad():
                    pp = torch.sigmoid(net(torch.as_tensor(x[None]))).item()
                if rng.random() < pp: continue
            if a < ps[0]: lo = max(lo, a)
            else: hi = min(hi, a)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        rec[i] = [x1 + lo * ux, y1 + lo * uy, x1 + hi * ux, y1 + hi * uy]; done.append(i)
    return rec


# ------------------------------------------------------------------ models and EVAE realisations
def term_path(fi, seed, mode, scheme='S2', term_dir=None):
    if mode == 'old':
        return STUDY / 'tr_models' / f'term_{scheme}_f{fi}_s{seed}.pt'
    if mode == 'rebuilt':
        return Path(term_dir or RA32_MODELS) / f'term2_{scheme}_f{fi}_s{seed}.pt'
    return None


def load_models(fi, seed, mode, scheme='S2', term_dir=None):
    """models of one S2 fold and run seed: sequential transformer variant 2 (gast2), evidence network, termination"""
    import gast as G
    import learned as LN
    import tr_gen as TG
    gnet = G.load(fi, seed, scheme, 'gast2')
    evn = LN.EVN(); evn.load_state_dict(torch.load(LN.MODELS / f'evn_{scheme}_f{fi}_s{seed}.pt')); evn.eval()
    term = None
    if mode == 'old':
        term = TG.TermNet(); term.load_state_dict(torch.load(term_path(fi, seed, mode, scheme))); term.eval()
    elif mode == 'rebuilt':
        term = TermNet2(); term.load_state_dict(torch.load(term_path(fi, seed, mode, scheme, term_dir))); term.eval()
    elif mode != 'rule':
        raise ValueError(mode)
    return dict(gnet=gnet, evn=evn, term=term, mode=mode, fold=fi, seed=seed, scheme=scheme)


def evae_prior_nets(fi, H, x0, piece=''):
    """the 24 EVAE prior realisations of east_test.nets_for (EVAE part only; ADFNE and KDE are not generated):
    run seeds x 8 draws, fold models of S2, seed study.gen_seed('EVAE', 'east' + piece, fold, int(x0), seed, draw)"""
    import study as S
    import tr_gen as TG
    out = []
    for s in P.SEEDS:
        ev = TG.evae_sampler('S2', fi, s)
        for d in range(P.NDRAW):
            out.append(P.clip(ev(H, S.gen_seed('EVAE', 'east' + piece, fi, int(x0), s, d)), H))
    return out


def evae_stored_nets(scheme, fi, box, H):
    """the 24 stored Paper 1 EVAE realisations of a trained-wall window (eval_learned.run_window convention)"""
    return [P.clip(P.stored_net(scheme, 'EVAE', fi, box, s_, d_), H) for s_ in P.SEEDS for d_ in range(P.NDRAW)]


# ------------------------------------------------------------------ the arm
def reconstruct(C, obs, lines, H, rng, models, evae_nets, net_offset, dmc):
    """one reconstruction of the frozen learned arm. C = gast.prep_ctx(scheme, fold); obs = p2v2.observe output;
    lines = p2v2.lines_layout output; rng = the arm's generator (one per realisation); models = load_models(...);
    evae_nets = 24 EVAE realisations of the window; net_offset = index offset into evae_nets (east_test: 8 x seed index;
    eval_learned: realisation r); dmc = dict shared by the calls of one window (dmin cache, as east_test)."""
    import gast as G
    import learned as LN
    import eval_learned as EL
    t1 = V.tier1(obs); groups = V.merge(obs, FE.BASE['JOIN_DTH'], FE.BASE['JOIN_DPERP'])
    t2, _ = LN.tier2_learned(models['evn'], C, groups, lines, H, rng)
    bud, p21h = V.budget(obs, lines, H, C['tr_th'], C['tr_len3d'], C['prior'], FE.BASE['A_PRIOR'])
    gnet = models['gnet']

    def pf(k, rec, src=evae_nets):
        g = G.generate(gnet, C, t1, rec, lines, H, rng); s_ = src[(net_offset + k) % 24]
        if not len(g) or not len(s_): return np.zeros((0, 4))
        e = s_[rng.integers(len(s_), size=len(g))]; c = (g[:, :2] + g[:, 2:]) / 2; ce = (e[:, :2] + e[:, 2:]) / 2
        return P.clip(e + np.hstack([c - ce, c - ce]), H)
    rec = EL.rule_place(C, t2, bud, p21h, lines, H, rng, pf, dmc, len(t2))
    if len(rec) > 1:
        pr = EL.protect_spans(t2, obs) + [None] * (len(rec) - len(t2))
        rec = terminate(rec, len(t2), rng, C['model'], H, models['mode'], net=models['term'], p_pass=C['p_pass'], protect=pr)
        rec = rec[P.seglen(rec) > 0.05]
    return rec


# ------------------------------------------------------------------ unseen-piece driver (east_test.run_job conventions)
def piece_traces(labels_path):
    """traces (m) and face height (m) of a mapped piece (east_test.east_traces: 2 cm per pixel)"""
    d = json.load(open(labels_path))
    L = np.array([[s['points'][0][0], s['points'][0][1], s['points'][1][0], s['points'][1][1]] for s in d['shapes'] if s['label'] == 'fracture'], float) * 0.02
    return L, d['imageHeight'] * 0.02


def window(L, x0, H):
    """east_test.window: traces shifted by x0 and clipped to [0, 20] x [0, H]"""
    W = L - np.array([x0, 0, x0, 0]); return P.clip(W, H)


def score_row(rec, gt, H, smodel, scheme_tag, fi, box, cname, arm, r, sec):
    """row of east_test.run_job (Paper 1 scores with one scoring model + P21 map RMSE)"""
    import statistical_tests as ST
    ym = H / 20
    s = ST.score(np.asarray(rec).reshape(-1, 4) / 20, gt / 20, ym, smodel)
    s = {k: (np.nan if (v is not None and np.isinf(v)) else v) for k, v in s.items()}
    return dict(scheme=scheme_tag, fold=fi, box=box, cfg=cname, arm=arm, r=r, n=len(rec), gt_n=len(gt),
                map_rmse=FE.p21map_err(rec, gt, H), sec=sec, **s)


def run_unseen_job(fi, x0, labels_path, piece, term_mode, arm=DEFAULT_ARM, term_dir=None, scheme_tag='EAST'):
    """all realisations of one S2 fold on one 20 m window of a mapped piece, as east_test.run_job does for MIX_ETP:
    3 run seeds x 2 layouts (5 m, centre line); realisation r = 3 fold + seed index; observation seed
    sd(scheme_tag + piece, int(x0), r, cname, 'obs'); arm seed sd(scheme_tag + piece, int(x0), r, cname, arm);
    EVAE prior realisations as east_test.nets_for; scoring model S2 fold 0. With term_mode='old' and arm='MIX_ETP' this
    is the stored MIX_ETP arm (regression); the frozen arm uses its own arm name (new seed key)."""
    torch.set_num_threads(1)
    import gast as G
    Lall, H = piece_traces(labels_path); gt = window(Lall, x0, H)
    C = G.prep_ctx('S2', fi)
    smodel = V.setbg.load(P.REL / 'data' / 'orientation_clusters' / 'S2_f0.json')
    nets = evae_prior_nets(fi, H, x0, piece)
    dmc = {}; rows = []
    for si, seed in enumerate(P.SEEDS):
        models = load_models(fi, seed, term_mode, term_dir=term_dir)
        r = fi * 3 + si
        for spacing in (5.0, None):
            cname = '5m' if spacing else 'centre'
            lines = V.lines_layout('v', spacing, FE.BASE['APERTURE'], H)
            obs = V.observe(gt, lines, H, np.random.default_rng(sd(scheme_tag + piece, int(x0), r, cname, 'obs')))
            rng = np.random.default_rng(sd(scheme_tag + piece, int(x0), r, cname, arm)); t0 = time.time()
            rec = reconstruct(C, obs, lines, H, rng, models, nets, si * 8, dmc)
            rows.append(score_row(rec, gt, H, smodel, scheme_tag + piece, fi, f'x{int(x0):02d}', cname, arm, r, time.time() - t0))
    return rows
