# -*- coding: utf-8 -*-
"""Shared code of the Section 1 comparison: the box library, the three held-out schemes and their folds, the fold
split (fitting / validation / test, no shared fractures), the context boxes of a test box, fold data for EVAE
training, EVAE posterior generation, a batched ADFNE runner, network files and seeds.

Taken from the study code (cv_common.py) with paths made repository-relative; the computations are unchanged.
EVAE = learned (trained with the learned orientation clusters as guidance). ADFNE and KDE = fitted distributions
(standard 2-set orientation fit, fitted lengths, mean count), fitted on the fitting boxes of the fold only."""
import os, sys, json, csv, math, subprocess
from pathlib import Path
import numpy as np

for k, v in (('ROUTEB_ISO', '1'), ('EVFIX_WQ', '1.0'), ('EVFIX_WLEN', '0.005')):    # final EVAE settings (Paper 1 L2)
    os.environ[k] = v
os.environ.setdefault('LAMBDA_FREE', '0.22')
_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC / 'evaluation'), str(_SRC / 'study')]
import repo_paths as RP  # noqa: E402
os.environ.setdefault('ADFNE_ROOT', str(RP.ADFNE_ROOT))
os.environ.setdefault('ADFNE_WORK', str(RP.WORK_DIR / 'adfne'))
import common            # noqa: E402  metric helpers, 2-set orientation fit, clipping (Paper 1 release, unchanged)
import generators        # noqa: E402  EVAE loader, ADFNE and KDE generators (Paper 1 release, unchanged)
generators.BRANCH = RP.MODEL_DIR

SEEDS = (1337, 20260903, 7)       # run seeds: each EVAE is trained once per seed; every method draws NDRAW networks per seed
NDRAW = 8                         # 3 seeds x 8 draws = 24 networks per test box and method
M = 20.0                          # 1 box unit = 20 m (boxes are normalised by their longest side)
METHODS = ('EVAE', 'ADFNE', 'KDE')
ORDER = ['W00', 'W01', 'W02', 'W03', 'W04', 'W05', 'W06']
SCHEMES = ('S1', 'S2', 'S3')
SCHEME_NAME = {'S1': 'Neighbourhood (fill a gap)', 'S2': 'Leave-one-bench-out (new bench level)', 'S3': 'Leave-one-stretch-out (new part of the wall)'}
# the internal variant names are part of every generator seed key, so they are kept exactly as used in the study
SEED_VARIANT = {'EVAE': 'geobg', 'ADFNE': 'main', 'KDE': 'main'}


# ---------------------------------------------------------------- data
def load_library():
    boxes = {}
    for r in csv.DictReader(open(RP.BOX_LIBRARY_DIR / 'box_manifest.csv')):
        if r['usable'] != 'True':
            continue
        d = json.load(open(RP.BOX_LIBRARY_DIR / 'boxes' / (r['box'] + '.json')))
        s = float(max(d['imageWidth'], d['imageHeight']))
        sh = d['shapes']
        L = np.array([[x['points'][0][0] / s, x['points'][0][1] / s, x['points'][1][0] / s, x['points'][1][1] / s] for x in sh], float).reshape(-1, 4)
        boxes[r['box']] = dict(name=r['box'], row=r['row'], ri=int(r['row_index']), col=int(r['col']), lines=L,
                               ids=np.array([x['description'] for x in sh], dtype=object), y_max=d['imageHeight'] / s,
                               w_px=d['imageWidth'], h_px=d['imageHeight'], cu=float(r['centre_u']), cs=float(r['centre_stack']),
                               u_left=float(r['u_left']), y_top=float(r['y_top']), h_m=float(r['height_m']))
    return boxes


def folds(scheme, boxes):
    if scheme == 'S1':
        pos = {(b['ri'], b['col']) for b in boxes.values()}
        elig = [k for k, b in boxes.items() if 1 <= b['ri'] <= len(ORDER) - 2 and
                all((b['ri'] + dr, b['col'] + dc) in pos for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)]
        cls = {}
        for k in elig:
            cls.setdefault((boxes[k]['ri'] % 2, boxes[k]['col'] % 2), []).append(k)
        return [sorted(v) for _, v in sorted(cls.items())]
    if scheme == 'S2':
        return [sorted(k for k in boxes if boxes[k]['row'] == r) for r in ORDER]
    if scheme == 'S3':
        return [sorted(k for k in boxes if boxes[k]['col'] in (c, c + 1)) for c in (1, 3, 5, 7)]
    raise ValueError(scheme)


def split(scheme, fi, boxes, check=True):
    """test / validation / fitting boxes of one fold; every fracture in a test box is removed from all non-test boxes.
    With check=True the result is compared with the stored data/splits/<scheme>_f<fi>.json."""
    test = folds(scheme, boxes)[fi]
    held = set(np.concatenate([boxes[k]['ids'] for k in test])) if test else set()
    nontest = sorted(k for k in boxes if k not in test)
    val = nontest[fi % 10::10]
    fit = [k for k in nontest if k not in val]
    panels = {}
    for k, b in boxes.items():
        if k in test:
            panels[k] = b
        else:
            keep = np.array([i not in held for i in b['ids']], bool)
            panels[k] = dict(b, lines=b['lines'][keep], ids=b['ids'][keep])
    if check:
        st = json.load(open(RP.SPLITS_DIR / ('%s_f%d.json' % (scheme, fi))))
        assert st['test'] == test and st['validation'] == val and st['fitting'] == fit, 'split differs from data/splits'
    return dict(scheme=scheme, fold=fi, test=test, val=val, fit=fit, held_ids=held, panels=panels)


def fold_of(scheme, box, boxes):
    return [k for k, f in enumerate(folds(scheme, boxes)) if box in f][0]


def context(sp, bid, n=8):
    """the n non-test boxes nearest to test box bid (centre distance along the wall and down the face)"""
    P = sp['panels']; b = P[bid]
    cand = [k for k in P if k not in sp['test']]
    near = sorted(cand, key=lambda k: (math.hypot(P[k]['cu'] - b['cu'], P[k]['cs'] - b['cs']), k))[:n]
    if sp['scheme'] == 'S1':                         # in S1 these are exactly the 8 neighbours
        nb = {(b['ri'] + dr, b['col'] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc}
        assert {(P[k]['ri'], P[k]['col']) for k in near} == nb, (bid, near)
    return near


def family_rule(sp):
    """the standard 2-set orientation fit of the baselines (fitting boxes only)"""
    ang = np.concatenate([common.geom(sp['panels'][k]['lines'])[2] for k in sp['fit'] if len(sp['panels'][k]['lines'])])
    return common.fit_set_rule(ang, k=2)


def gen_seed(*parts):
    """generator seed of one network: crc32 of 'METHOD|scheme|variant|fold|box|run seed|draw' (unsigned)"""
    return common.gen_seed(*[str(p) for p in parts])


def network_seed(method, scheme, fi, box, seed, draw):
    return gen_seed(method, scheme, SEED_VARIANT[method], fi, box, seed, draw)


# ---------------------------------------------------------------- fold data for EVAE training
def write_labelme(path, lines, ids, w_px, h_px, extra=None):
    s = float(max(w_px, h_px))
    shapes = [dict(label='fracture', points=[[l[0] * s, l[1] * s], [l[2] * s, l[3] * s]], group_id=None, description=str(i),
                   shape_type='line', flags={}) for l, i in zip(lines, ids)]
    json.dump(dict(version='6.3.1', flags={}, shapes=shapes, imagePath=Path(path).stem + '.jpg', imageData=None,
                   imageHeight=int(h_px), imageWidth=int(w_px), cv=extra or {}), open(path, 'w'))


def fold_dir(scheme, fi):
    return RP.WORK_DIR / 'fold_data' / ('%s_f%d' % (scheme, fi))


def write_fold_data(scheme, fi):
    """fitting and validation boxes (held-out fractures removed) as LabelMe JSON for the EVAE training code"""
    d = fold_dir(scheme, fi)
    if (d / 'split.json').exists():
        return d
    sp = split(scheme, fi, load_library())
    for sub, keys in (('fit', sp['fit']), ('val', sp['val'])):
        (d / sub).mkdir(parents=True, exist_ok=True)
        for k in keys:
            p = sp['panels'][k]
            write_labelme(d / sub / (k + '.json'), p['lines'], p['ids'], p['w_px'], p['h_px'], dict(box=k, role=sub))
    json.dump(dict(scheme=scheme, fold=fi, test=sp['test'], val=sp['val'], fit=sp['fit'], held_fractures=len(sp['held_ids'])),
              open(d / 'split.json', 'w'), indent=1)
    return d


# ---------------------------------------------------------------- EVAE generation (posterior sampling, Paper 1 release)
def aggregate_posterior(model, dev, panels, windows):
    import torch
    t = generators._training()
    out = []
    for w in windows:
        P = np.zeros((1, model.max_lines, 4), np.float32)
        L = panels[w]["lines"][: model.max_lines]
        P[0, : len(L)] = L
        with torch.no_grad():
            c, l, a = t.to_centers_lengths_angles(torch.from_numpy(P).to(dev))
            sm, sl, gm, gl = model.encode(c, torch.stack([l, a], -1))
        out.append((sm, (0.5 * sl).exp(), gm, (0.5 * gl).exp()))
    return out


def generate(model, dev, y_max, seed_gen, post, draw):
    """one EVAE network: latent drawn around the posterior of context box (draw mod 8), decoded, clipped to the box"""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed_gen)
    rng = np.random.default_rng(seed_gen)
    sm, ss, gm, gs = post[draw % len(post)]
    with torch.no_grad():
        eps_s = torch.randn(1, 24, device=dev)
        eps_g = torch.randn(1, 32, device=dev)
        sz = sm + eps_s * ss
        gz = gm + eps_g * gs
        m = model
        centers = torch.sigmoid(m.spatial_decoder(sz)).view(-1, m.max_lines, 2)[0]
        lengths = torch.sigmoid(m.length_decoder(gz)).view(-1, m.max_lines)[0]
        logits = m.angle_logits_head(gz).view(-1, m.max_lines, m.K_angle)[0]
        mu_raw = m.angle_mu_head(gz).view(-1, m.max_lines, m.K_angle)[0]
        kappa_raw = m.angle_kappa_head(gz).view(-1, m.max_lines, m.K_angle)[0]
        alpha = F.softmax(logits, dim=-1).cpu().numpy()
        mu = (math.pi * torch.tanh(mu_raw)).cpu().numpy()
        kappa = (F.softplus(kappa_raw) + 1e-4).cpu().numpy()
        exist = m.existence_head(torch.cat([sz, gz], 1)).view(-1, m.max_lines)[0]
        keep = (torch.sigmoid(exist) >= 0.5).cpu().numpy()
        if keep.sum() < 2:
            keep = np.zeros(m.max_lines, bool)
            keep[torch.topk(exist, 10).indices.cpu().numpy()] = True
        c = centers.cpu().numpy()[keep]
        L = lengths.cpu().numpy()[keep]
        aa, am, ak = alpha[keep], mu[keep], kappa[keep]
    th = []
    for j in range(len(c)):
        k = rng.choice(m.K_angle, p=aa[j] / aa[j].sum())
        th.append((rng.vonmises(2.0 * am[j, k], max(ak[j, k], 1e-4)) / 2.0) % math.pi)
    th = np.asarray(th)
    dx, dy = np.cos(th) * L / 2, np.sin(th) * L / 2
    segs = np.column_stack([c[:, 0] - dx, c[:, 1] - dy, c[:, 0] + dx, c[:, 1] + dy])
    return common.clip_lines(segs, y_max)


# ---------------------------------------------------------------- ADFNE, batched (one Octave session per test box)
def adfne_batch(stats, y_max, jobs, tag):
    """Identical to generators.generate_adfne for every (seed_gen, key) in jobs, but all networks are written in ONE
    Octave run: each network block starts with its own rng(seed_gen) exactly as the one-network script does after
    adfne_init. Host-side filtering and subsampling use the same seed, per network, unchanged."""
    work = Path(os.environ['ADFNE_WORK']); work.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ['ADFNE_ROOT'])
    init_path = str(root / 'adfne_init.m').replace('\\', '/')
    rows, plan = [f"run('{init_path}');"], []
    for sg, key in jobs:
        rows.append(f"rng({sg % (2**31)});")
        outs = []
        for s in stats['sets']:
            q = generators.quota(s, stats['mean_ymax'], y_max)
            requested = max(100, int(math.ceil(q / max(y_max, 1e-4) * 8)))
            low, mu, high = s['length_fit']
            op = work / f"{tag}_{key}_set{s['set']}.csv"; outs.append((op, q))
            rows += [f"o = DFN('dim',2,'n',{requested},'dir',{s['centre_deg']:.10f},'ddir',{-s['kappa_dir']:.10f},"
                     f"'minl',{low:.10f},'mu',{mu:.10f},'maxl',{high:.10f},'bbx',[0,0,1,1]);", "allx=o.Orig;",
                     f"dlmwrite('{str(op).replace(chr(92), '/')}',allx,',');"]
        plan.append((sg, key, outs))
    m_path = work / f"adfne_{tag}.m"; m_path.write_text("\n".join(rows), encoding='ascii')
    launcher = root / ('run_adfne.bat' if os.name == 'nt' else 'run_adfne.sh')
    done = subprocess.run([str(launcher), str(m_path)], cwd=str(root), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=3600)
    res = {}
    for sg, key, outs in plan:
        if done.returncode != 0 or any(not p.exists() for p, _ in outs):
            raise RuntimeError(f"ADFNE failed for {tag}/{key}:\n{done.stdout[-2000:]}")
        rng = np.random.default_rng(sg); sel = []
        for p, q in outs:
            raw = np.atleast_2d(np.loadtxt(p, delimiter=','))
            cy = (raw[:, 1] + raw[:, 3]) / 2
            raw = raw[(cy >= 0) & (cy <= y_max)]
            if len(raw) == 0:
                continue
            sel.extend(raw[rng.choice(len(raw), size=q, replace=len(raw) < q)])
        res[key] = common.clip_lines(sel, y_max)
        for p, _ in outs:
            p.unlink(missing_ok=True)
    m_path.unlink(missing_ok=True)
    return res


# ---------------------------------------------------------------- network files
def net_path(scheme, method, fi, box, seed, d, root=None):
    return Path(root or RP.NETWORKS_DIR) / scheme / method / ('f%d_%s_s%d_d%02d.csv' % (fi, box, seed, d))


def save_net(p, L):
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, np.asarray(L, float).reshape(-1, 4), delimiter=',')


def load_net(p):
    raw = np.loadtxt(p, delimiter=',')
    return np.atleast_2d(raw) if raw.size else np.zeros((0, 4))


# ---------------------------------------------------------------- orientation clusters
def cluster_model(scheme, fi):
    import setbg
    return setbg.load(RP.CLUSTERS_DIR / ('%s_f%d.json' % (scheme, fi)))


def axd(a, b):
    d = np.abs(np.mod(a - b, np.pi)); return np.minimum(d, np.pi - d)


def axial_mean(th):
    return (np.angle(np.exp(2j * th).mean()) / 2) % np.pi
