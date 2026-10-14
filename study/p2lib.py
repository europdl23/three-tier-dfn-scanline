# -*- coding: utf-8 -*-
"""Paper 2 revision tests on the Section 1 (Paper 1) data.

Task (Paper 2): rebuild the full trace map of a held-out sampling window from sparse VERTICAL SCANLINES only
(10 cm apertures at a fixed spacing), using the three-tier evidence hierarchy:
  Tier 1 measured  : the piece of each trace inside an aperture (a point + a trace direction), kept exactly;
  Tier 2 indicated : each observed trace extended to a full trace (length drawn from the training traces of a
                     similar direction, length-biased because a scanline hits long traces more often);
  Tier 3 inferred  : traces placed in the unobserved space, from a PROPOSER, accepted only if
                     (a) they do not cross any aperture (they would have been seen there) and
                     (b) until the trace length implied by the scanline crossing rate is reached.
Proposers compared on identical evidence, gates and budget: Paper 1 EVAE, ADFNE, KDE (their stored, fold-matched
realisations, unchanged) and UNIF (uniform centres, direction and length resampled from training traces).

Everything is in metres. Window = 20 m wide x box height. Data and stored realisations are read from the Paper 1
release (GITHUB_UPLOAD/evae-dfn-bench-faces), never modified."""
import json, csv, math, os, sys
from pathlib import Path
import numpy as np

REL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REL / 'src' / 'evaluation'))
import common  # noqa: E402  (Paper 1 release helpers: w1_circular_doubled, clip)

M = 20.0
SEEDS = (1337, 20260903, 7)
NDRAW = 8
ORDER = ['W00', 'W01', 'W02', 'W03', 'W04', 'W05', 'W06']


# ------------------------------------------------------------------ data
def load_library():
    boxes = {}
    for r in csv.DictReader(open(REL / 'data/box_library/box_manifest.csv')):
        if r['usable'] != 'True':
            continue
        d = json.load(open(REL / 'data/box_library/boxes' / (r['box'] + '.json')))
        s = float(max(d['imageWidth'], d['imageHeight']))
        sh = d['shapes']
        L = np.array([[x['points'][0][0], x['points'][0][1], x['points'][1][0], x['points'][1][1]] for x in sh], float).reshape(-1, 4) / s * M
        full = np.array([x.get('full_length_m', np.nan) for x in sh], float)
        boxes[r['box']] = dict(name=r['box'], row=r['row'], ri=int(r['row_index']), col=int(r['col']), L=L,
                               ids=np.array([x['description'] for x in sh], dtype=object), full=full,
                               H=d['imageHeight'] / s * M, cu=float(r['centre_u']), cs=float(r['centre_stack']))
    return boxes


def load_split(scheme, fi):
    return json.load(open(REL / 'data/splits' / f'{scheme}_f{fi}.json'))


def train_traces(boxes, sp):
    """direction (rad) and FULL length (m) of every training trace of the fold (fitting + validation boxes,
    with every fracture of a test box removed), each fracture counted once."""
    held = set()
    for k in sp['test']:
        held |= set(boxes[k]['ids'])
    seen, th, ln, clipped_len = set(), [], [], []
    for k in sp['fitting'] + sp['validation']:
        b = boxes[k]
        for i, tid in enumerate(b['ids']):
            if tid in held or tid in seen:
                continue
            seen.add(tid)
            x1, y1, x2, y2 = b['L'][i]
            th.append(math.atan2(y2 - y1, x2 - x1) % math.pi)
            f = b['full'][i]
            ln.append(f if np.isfinite(f) else math.hypot(x2 - x1, y2 - y1))
    return np.array(th), np.array(ln)


def train_p21(boxes, sp):
    tot_len, tot_area = 0.0, 0.0
    for k in sp['fitting'] + sp['validation']:
        b = boxes[k]
        held = set()
        for t in sp['test']:
            held |= set(boxes[t]['ids'])
        keep = np.array([i not in held for i in b['ids']], bool)
        tot_len += seglen(b['L'][keep]).sum(); tot_area += M * b['H']
    return tot_len / tot_area


def stored_net(scheme, method, fi, box, seed, d):
    p = REL / 'networks' / scheme / method / f'f{fi}_{box}_s{seed}_d{d:02d}.csv'
    a = np.loadtxt(p, delimiter=',', ndmin=2) if p.stat().st_size else np.zeros((0, 4))
    return a.reshape(-1, 4) * M


# ------------------------------------------------------------------ geometry
def seglen(L):
    L = np.asarray(L, float).reshape(-1, 4)
    return np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])


def segdir(L):
    L = np.asarray(L, float).reshape(-1, 4)
    return np.mod(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0]), np.pi)


def clip(L, H):
    L = np.asarray(L, float).reshape(-1, 4)
    if not len(L):
        return np.zeros((0, 4))
    out = common.clip_lines(L / M, H / M)
    return out.reshape(-1, 4) * M


def apertures(spacing, width, W=M, mode='regular', rng=None, jitter=0.0):
    """x-intervals of vertical scanline apertures. spacing=None -> one scanline at the window centre."""
    if spacing is None:
        cs = [W / 2]
    else:
        n = int(math.floor(W / spacing))
        off = (W - (n - 1) * spacing) / 2 if n > 1 else W / 2
        cs = [off + i * spacing for i in range(n)] if n >= 1 else [W / 2]
        if mode == 'irregular' and rng is not None:
            cs = [min(max(c + rng.uniform(-jitter, jitter), width), W - width) for c in cs]
    return [(c - width / 2, c + width / 2) for c in cs]


def crosses_band(L, lo, hi):
    """True where segment has any part with lo <= x <= hi."""
    x1, x2 = L[:, 0], L[:, 2]
    return (np.maximum(x1, x2) >= lo) & (np.minimum(x1, x2) <= hi)


def observe(gt, ap, H, rng, miss=0.0, ang_noise_deg=0.0):
    """Scanline observations of the ground truth: one record per (trace, aperture) hit: point (x,y) at the
    aperture centre line, measured direction, true trace index. A hit is missed with probability `miss`."""
    obs = []
    for j, (lo, hi) in enumerate(ap):
        xc = (lo + hi) / 2
        hit = np.where(crosses_band(gt, lo, hi))[0]
        for i in hit:
            if rng.random() < miss:
                continue
            x1, y1, x2, y2 = gt[i]
            if abs(x2 - x1) < 1e-9:
                y = (y1 + y2) / 2
            else:
                t = np.clip((xc - x1) / (x2 - x1), 0, 1)
                y = y1 + t * (y2 - y1)
            th = math.atan2(y2 - y1, x2 - x1) % math.pi
            th = (th + math.radians(rng.normal(0, ang_noise_deg))) % math.pi if ang_noise_deg else th
            obs.append(dict(ap=j, x=xc, y=y, th=th, tid=int(i), lo=lo, hi=hi))
    return obs


def tier1_pieces(obs):
    """the measured piece of each hit: the part of the trace inside its aperture (a short segment)."""
    out = []
    for o in obs:
        w = (o['hi'] - o['lo']) / 2
        c, s = math.cos(o['th']), math.sin(o['th'])
        if abs(c) < 1e-6:
            continue
        t = w / abs(c)
        out.append([o['x'] - t * c, o['y'] - t * s, o['x'] + t * c, o['y'] + t * s])
    return np.array(out, float).reshape(-1, 4)


def merge_obs(obs, dth_deg=6.0, dperp=0.35):
    """group hits that lie on one straight trace across several scanlines (same direction, collinear)."""
    groups = []
    for o in sorted(obs, key=lambda q: q['x']):
        placed = False
        for g in groups:
            r = g[-1]
            if r['ap'] == o['ap']:
                continue
            dth = abs(((o['th'] - r['th']) + math.pi / 2) % math.pi - math.pi / 2)
            if dth > math.radians(dth_deg):
                continue
            ux, uy = math.cos(r['th']), math.sin(r['th'])
            px, py = o['x'] - r['x'], o['y'] - r['y']
            if abs(-uy * px + ux * py) <= dperp:
                g.append(o); placed = True; break
        if not placed:
            groups.append([o])
    return groups


def tier2_extend(groups, tr_th, tr_len, H, rng, use_bias=True, dth_win=math.radians(15)):
    """full traces from observed groups: length from training traces of similar direction, weighted by length
    (scanline length bias), at least the observed span; the observed point(s) placed at a random position."""
    out = []
    for g in groups:
        th = math.atan2(sum(math.sin(2 * o['th']) for o in g), sum(math.cos(2 * o['th']) for o in g)) / 2 % math.pi
        ux, uy = math.cos(th), math.sin(th)
        s = np.array([o['x'] * ux + o['y'] * uy for o in g])
        span = s.max() - s.min()
        d = np.abs(((tr_th - th) + math.pi / 2) % math.pi - math.pi / 2)
        pool = tr_len[d <= dth_win]
        if len(pool) < 5:
            pool = tr_len
        pool = pool[pool >= span] if (pool >= span).any() else np.array([span * 1.1 + 0.2])
        w = pool / pool.sum() if use_bias else np.full(len(pool), 1 / len(pool))
        Lf = float(rng.choice(pool, p=w))
        extra = Lf - span
        a = rng.uniform(0, extra)
        x0, y0 = g[int(np.argmin(s))]['x'], g[int(np.argmin(s))]['y']
        p1 = (x0 - a * ux, y0 - a * uy)
        p2 = (p1[0] + Lf * ux, p1[1] + Lf * uy)
        out.append([p1[0], p1[1], p2[0], p2[1]])
    return clip(np.array(out, float).reshape(-1, 4), H)


def p21_budget(obs, ap, H, tr_th, tr_len, prior_p21, a_prior=2.0, use_evidence=True):
    """total trace length expected in the window. Crossing rate of a vertical line = P21 * E|cos(theta)|
    (length-weighted over training directions). Gamma-Poisson shrinkage toward the training P21 with a prior
    weight of a_prior crossings (handles scanlines with no hit)."""
    ec = float(np.sum(tr_len * np.abs(np.cos(tr_th))) / np.sum(tr_len))
    if not use_evidence:
        return prior_p21 * M * H, prior_p21
    Ls = len(ap) * H
    lam_prior = prior_p21 * ec
    lam = (len(obs) + a_prior) / (Ls + (a_prior / lam_prior if a_prior > 0 else 0.0))
    p21 = lam / ec
    return p21 * M * H, p21


def tier3_fill(proposals, ap, existing, budget, H, rng, gate=True):
    """accept proposed traces in random order: reject any trace crossing an aperture (it would have been
    observed); stop once total length reaches the budget."""
    have = seglen(existing).sum()
    acc = []
    P = clip(proposals, H)
    if not len(P):
        return np.zeros((0, 4))
    order = rng.permutation(len(P))
    for i in order:
        if have >= budget:
            break
        t = P[i:i + 1]
        if gate and any(crosses_band(t, lo, hi)[0] for lo, hi in ap):
            continue
        acc.append(t[0]); have += seglen(t)[0]
    return np.array(acc, float).reshape(-1, 4)


def unif_proposals(n, tr_th, tr_len, H, rng):
    c = np.column_stack([rng.uniform(0, M, n), rng.uniform(0, H, n)])
    idx = rng.integers(0, len(tr_th), n)
    th, ln = tr_th[idx], tr_len[idx]
    dx, dy = np.cos(th) * ln / 2, np.sin(th) * ln / 2
    return np.column_stack([c[:, 0] - dx, c[:, 1] - dy, c[:, 0] + dx, c[:, 1] + dy])


# ------------------------------------------------------------------ metrics (metres)
TOL, EDGE = 0.3, 0.1


def _pt_seg(P, A, B):
    d = B - A; l2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(2) / l2[None], 0, 1)
    return np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * d[None]), axis=2)


def crossing_pairs(L):
    n = len(L); pr = []
    for i in range(n):
        m = common.seg_cross_one(L[i], L[i + 1:])
        pr += [(i, i + 1 + j) for j in np.where(m)[0]]
    return pr


def topo(L, H):
    n = len(L)
    if n == 0:
        return dict(nX=0, pI=np.nan, pY=np.nan, pX=np.nan, CL=np.nan, lcc=np.nan)
    E = np.vstack([L[:, :2], L[:, 2:]]); own = np.r_[np.arange(n), np.arange(n)]
    inside = (E[:, 0] > EDGE) & (E[:, 0] < M - EDGE) & (E[:, 1] > EDGE) & (E[:, 1] < H - EDGE)
    if n > 1:
        D = _pt_seg(E, L[:, :2], L[:, 2:]); D[np.arange(2 * n), own] = np.inf
        onto = D.min(1) <= TOL
        # Y-node partner for connectivity
        partner = D.argmin(1)
    else:
        onto = np.zeros(2 * n, bool); partner = own
    nY = int((inside & onto).sum()); nI = int((inside & ~onto).sum())
    pr = crossing_pairs(L) if n > 1 else []
    nX = len(pr); tot = nI + nY + nX; nl = (nI + nY) / 2
    # connected components (crossings + Y abutments)
    par = list(range(n))
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j in pr:
        par[f(i)] = f(j)
    for e in np.where(onto)[0]:
        par[f(own[e])] = f(int(partner[e]))
    roots = np.array([f(i) for i in range(n)])
    lens = seglen(L)
    lcc = max(lens[roots == r].sum() for r in set(roots)) / lens.sum()
    return dict(nX=nX, pI=nI / tot if tot else np.nan, pY=nY / tot if tot else np.nan, pX=nX / tot if tot else np.nan,
                CL=2 * (nY + nX) / nl if nl else np.nan, lcc=lcc)


def p21_map(L, H, cell=4.0, step=0.05):
    nx, ny = int(round(M / cell)), max(1, int(round(H / cell)))
    cx, cy = M / nx, H / ny
    g = np.zeros((ny, nx))
    for x1, y1, x2, y2 in L:
        l = math.hypot(x2 - x1, y2 - y1); k = max(2, int(l / step))
        t = (np.arange(k) + 0.5) / k
        xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        ix = np.clip((xs / cx).astype(int), 0, nx - 1); iy = np.clip((ys / cy).astype(int), 0, ny - 1)
        np.add.at(g, (iy, ix), l / k)
    return g / (cx * cy)


def metrics(L, H, gt_cache=None):
    L = np.asarray(L, float).reshape(-1, 4)
    A = M * H
    ln = seglen(L); th = segdir(L)
    t = topo(L, H)
    return dict(n=len(L), P21=ln.sum() / A, med_len=float(np.median(ln)) if len(ln) else np.nan,
                p90_len=float(np.percentile(ln, 90)) if len(ln) else np.nan,
                I100=100 * t['nX'] / A, CL=t['CL'], pI=t['pI'], pY=t['pY'], pX=t['pX'], lcc=t['lcc'],
                _len=ln, _th=th, _map=p21_map(L, H))


SCORES = ['count_err_pct', 'P21_err_pct', 'med_len_err', 'p90_len_err', 'len_W1', 'dir_W1', 'I100_err', 'I100_ratio',
          'CL_err', 'pI_err', 'pX_err', 'lcc_err', 'map_rmse']


def score(m, g):
    def ae(a, b):
        return abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
    from scipy.stats import wasserstein_distance as wd
    return dict(count_err_pct=100 * abs(m['n'] - g['n']) / max(g['n'], 1),
                P21_err_pct=100 * abs(m['P21'] - g['P21']) / g['P21'],
                med_len_err=ae(m['med_len'], g['med_len']), p90_len_err=ae(m['p90_len'], g['p90_len']),
                len_W1=float(wd(m['_len'], g['_len'])) if len(m['_len']) else np.nan,
                dir_W1=common.w1_circular_doubled(m['_th'], g['_th']) if len(m['_th']) > 1 else np.nan,
                I100_err=ae(m['I100'], g['I100']), I100_ratio=m['I100'] / g['I100'] if g['I100'] > 0 else np.nan,
                CL_err=ae(m['CL'], g['CL']), pI_err=ae(m['pI'], g['pI']), pX_err=ae(m['pX'], g['pX']),
                lcc_err=ae(m['lcc'], g['lcc']),
                map_rmse=float(np.sqrt(np.mean((m['_map'] - g['_map']) ** 2))) if m['_map'].shape == g['_map'].shape else np.nan)
