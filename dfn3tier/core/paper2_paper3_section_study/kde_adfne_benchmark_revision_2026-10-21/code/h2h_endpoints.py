"""Endpoints for every arm, computed identically from saved case outputs (frozen protocol section 5). Runs after generation;
reads eval-stage truth (window_trace_manifest.csv) and saved predictions.csv only. Writes results/ENDPOINTS_<fold>.csv.

Evaluation sets
  full : the complete output network vs all truth traces of the window (every arm; framework arms keep observations).
  sf   : strip-free subset. Output traces that do not cross any observed strip vs hidden truth (truth traces not crossing
         any strip). The same geometric filter for every arm; evaluation only. For framework arms sf = inferred traces.
Units: rows (cx, cy, length, angle) in window units (metres / 20), angle in radians.
Unit of analysis: (arm, window, training seed); the three generation seeds are pooled for energy distances (as Plan D)
and averaged for per-realisation metrics.
"""
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

HP = Path(__file__).resolve().parent.parent; STUDY = HP.parent
ST2 = STUDY / 'plan_d_step2_staging_2026-10-21'; STG = ST2 / 'stages'
S3 = STUDY / 'plan_d_step3_training_evaluation_2026-10-21'; HY = STUDY / 'post_plan_d_hybrid_2026-10-21'
sys.path.insert(0, str(ST2)); import analysis_metrics as M  # noqa: frozen Step 2 module
import os
SMOKE = os.environ.get('H2H_SMOKE') == '1'   # code test on Plan D's smoke stage and smoke outputs only
if SMOKE: STG = S3 / 'gates/smoke_tree/stages'
COLS = ['cx', 'cy', 'length', 'angle']; TOL_PRES = 0.002; TOL_NODE = 0.02; TOL_OBS = 0.02


def segment_graph(segments, height, tol=.02):
    """Verbatim copy of stepwise_execution_2026-10-19/engine.py::segment_graph (the Plan D network metric)."""
    a = np.asarray(segments, dtype=float).reshape(-1, 2, 2); n = len(a); parent = list(range(n)); crossings = 0
    def root(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def cross(u, v): return u[0] * v[1] - u[1] * v[0]
    def distance(p, line):
        v = line[1] - line[0]; t = np.clip(np.dot(p - line[0], v) / max(np.dot(v, v), 1e-16), 0, 1)
        return np.linalg.norm(p - line[0] - t * v)
    for i in range(n):
        for j in range(i):
            u = a[i, 1] - a[i, 0]; v = a[j, 1] - a[j, 0]; den = cross(u, v); hit = False
            if abs(den) > 1e-12:
                d = a[j, 0] - a[i, 0]; s = cross(d, v) / den; t = cross(d, u) / den
                if -1e-10 <= s <= 1 + 1e-10 and -1e-10 <= t <= 1 + 1e-10: hit = True; crossings += 1
            if hit or min(distance(a[i, 0], a[j]), distance(a[i, 1], a[j]), distance(a[j, 0], a[i]), distance(a[j, 1], a[i])) <= tol:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(n): groups.setdefault(root(i), []).append(i)
    spanning = any(a[g, :, 1].min() <= tol and a[g, :, 1].max() >= height - tol for g in groups.values())
    return dict(graph_components=len(groups), graph_largest_fraction=max([len(v) for v in groups.values()] + [0]) / max(n, 1), graph_top_bottom=int(spanning), graph_crossings=crossings)


def _pt_seg(p, a, b):
    v = b - a; t = np.clip(((p - a) * v).sum(-1) / np.maximum((v * v).sum(-1), 1e-16), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * v), axis=-1)


def nodes(arr, h):
    """I / Y / X node counts (metres): X = proper interior crossings; an endpoint within 2 cm of another trace = Y; else I.
    Endpoints on the window boundary (within 2 cm) are censored and not classified."""
    a, b = M.ends(arr); a = a * 20; b = b * 20; n = len(a); X = Y = I = 0
    for i in range(n):
        for j in range(i):
            u = b[i] - a[i]; v = b[j] - a[j]; den = u[0] * v[1] - u[1] * v[0]
            if abs(den) > 1e-12:
                d = a[j] - a[i]; s = (d[0] * v[1] - d[1] * v[0]) / den; t = (d[0] * u[1] - d[1] * u[0]) / den
                if 1e-6 < s < 1 - 1e-6 and 1e-6 < t < 1 - 1e-6: X += 1
    for i in range(n):
        for p in (a[i], b[i]):
            if p[0] <= TOL_NODE or p[0] >= 20 - TOL_NODE or p[1] <= TOL_NODE or p[1] >= h - TOL_NODE: continue
            others = np.r_[0:i, i + 1:n]
            if len(others) and _pt_seg(p[None], a[others], b[others]).min() <= TOL_NODE: Y += 1
            else: I += 1
    tot = max(I + Y + X, 1); return dict(I=I, Y=Y, X=X, fI=I / tot, fY=Y / tot, fX=X / tot)


def crossing_strip(arr, st):
    if not st or not len(arr): return np.zeros(len(arr), bool)
    _, par = M.observe(arr, st); return np.isin(np.arange(len(arr)), np.unique(par))


def preservation(obs, out):
    """Fraction of observed clips whose two endpoints lie within 2 mm of one output trace."""
    if not len(obs): return math.nan
    if not len(out): return 0.0
    oa, ob = M.ends(obs); pa, pb = M.ends(out); oa, ob, pa, pb = oa * 20, ob * 20, pa * 20, pb * 20; ok = 0
    for i in range(len(obs)):
        d = np.maximum(_pt_seg(oa[i][None], pa, pb), _pt_seg(ob[i][None], pa, pb)); ok += bool(d.min() <= TOL_PRES)
    return ok / len(obs)


def conflicts(obs, out, st):
    """Output traces whose strip clips do not coincide (2 cm) with an observed clip: crossings the observations deny."""
    if not st or not len(out): return 0
    clips, par = M.observe(out, st)
    if not len(clips): return 0
    if not len(obs): return int(len(np.unique(par)))
    ca, cb = M.ends(clips); oa, ob = M.ends(obs); bad = set()
    for i in range(len(clips)):
        d1 = np.minimum(np.linalg.norm((oa - ca[i]) * 20, axis=1) + np.linalg.norm((ob - cb[i]) * 20, axis=1), np.linalg.norm((oa - cb[i]) * 20, axis=1) + np.linalg.norm((ob - ca[i]) * 20, axis=1))
        if d1.min() > 2 * TOL_OBS: bad.add(int(par[i]))
    return len(bad)


def validity(out, h):
    if not len(out): return dict(invalid=0, duplicates=0)
    a, b = M.ends(out); z = np.r_[a, b]; ym = h / 20
    inv = int((~np.isfinite(out).all(1) | (out[:, 2] <= 1e-8)).sum() + ((z[:, 0] < -1e-4) | (z[:, 0] > 1 + 1e-4) | (z[:, 1] < -1e-4) | (z[:, 1] > ym + 1e-4)).reshape(2, -1).any(0).sum())
    e = np.c_[a, b] * 20; e2 = np.c_[b, a] * 20; dup = 0
    for i in range(len(out)):
        for j in range(i):
            if min(np.abs(e[i] - e[j]).max(), np.abs(e[i] - e2[j]).max()) < 0.01: dup += 1
    return dict(invalid=inv, duplicates=dup)


def truth(fold):
    ev = STG / fold / 'eval'; inv = pd.read_csv(ev / 'window_inventory.csv'); tm = pd.read_csv(ev / 'window_trace_manifest.csv'); W = {}
    for r in inv[inv.role == 'validation'].itertuples():
        W[r.window_id] = dict(arr=tm[tm.window_id == r.window_id][COLS].to_numpy(np.float32), h=float(r.height_m), x0=float(r.x0_m))
    return W


def strips_for(layout, x0): return M.strips('5m') if layout == 'L0' else ([((10 - .05) / 20, (10 + .05) / 20)] if int(round(x0)) % 60 == 0 else [])


def case_dirs(fold, layout):
    """(source, arm label, directory). Plan D arms B,C,D,E; hybrid arms G0,HY1,HY2 (L0 only); benchmark arms."""
    out = []
    if SMOKE: return [('benchmark', Path(d).name.split('__')[0], d) for d in glob.glob(str(HP / 'runs' / 'smoke' / layout / f'{layout}_on' / 'cases' / '*'))]
    for d in glob.glob(str(S3 / 'evaluation' / fold / 'recon' / f'{layout}_on' / 'cases' / '*')):
        lab = Path(d).name.split('__')[0]
        if lab in ('B', 'C', 'D', 'E'): out.append(('planD', lab, d))
    if layout == 'L0':
        for d in glob.glob(str(HY / 'recon' / fold / 'L0_on' / 'cases' / '*')):
            lab = Path(d).name.split('__')[0]
            if lab in ('HY_G0', 'HY1', 'HY2'): out.append(('hybrid', lab.replace('HY_G0', 'G0'), d))
    for d in glob.glob(str(HP / 'runs' / fold / layout / f'{layout}_on' / 'cases' / '*')):
        out.append(('benchmark', Path(d).name.split('__')[0], d))
    return out


def _ed(P, T, prop):
    return M.energy_distance(M.property_features(P, prop), M.property_features(T, prop)) if len(P) >= 2 and len(T) >= 2 else math.nan


def _es(P, T, prop):
    if len(P) < 2 or len(T) < 2: return math.nan
    X, Y = M.property_features(P, prop), M.property_features(T, prop); return 2 * M._pair_mean(X, Y) - M._within_u(X)


def _matched(T, P):
    if not len(T) or not len(P): return None
    ii, jj = linear_sum_assignment(cdist(T[:, :2] * 20, P[:, :2] * 20)); gt, pr = T[ii], P[jj]
    ang = np.degrees(np.abs((gt[:, 3] - pr[:, 3] + np.pi / 2) % np.pi - np.pi / 2))
    return dict(length_MAE=float(np.abs(gt[:, 2] - pr[:, 2]).mean() * 20), angle_MAE=float(ang.mean()), position_err=float(np.hypot(*((gt[:, :2] - pr[:, :2]) * 20).T).mean()))


def endpoints(fold, layout):
    W = truth(fold); graph_cache = {}; node_cache = {}; units = defaultdict(list)
    for src, lab, d in case_dirs(fold, layout):
        ci = json.loads(Path(d, 'case_inputs.json').read_text()); parts = Path(d).name.split('__'); wid = '__'.join(parts[1:3])
        pr = pd.read_csv(Path(d) / 'predictions.csv')
        units[(lab, wid, int(ci.get('training_seed', 0)))].append(dict(src=src, gen=int(ci['generation_seed']), final=pr[COLS].to_numpy(np.float32),
                                                                       inf=pr[pr.inferred.astype(str).str.lower().eq('true')][COLS].to_numpy(np.float32), ci=ci, dir=d))
    rows = []
    for (lab, wid, ts), cases in units.items():
        w = W[wid]; st = strips_for(layout, w['x0']); T = w['arr']; h = w['h']
        obs, _ = M.observe(T, st) if st else (np.zeros((0, 4), np.float32), np.zeros(0, int))
        Tsf = T[~crossing_strip(T, st)]
        if wid not in graph_cache:
            ga, gb = M.ends(T); graph_cache[wid] = segment_graph(np.stack([ga, gb], 1) * 20, h); node_cache[wid] = nodes(T, h)
        per = defaultdict(list); Pf, Ps = [], []
        for c in cases:
            F = c['final']; S = F[~crossing_strip(F, st)]; Pf.append(F); Ps.append(S)
            pa, pb = M.ends(F); g = segment_graph(np.stack([pa, pb], 1) * 20, h); nd = nodes(F, h)
            per['count_MAE'].append(abs(len(F) - len(T))); per['count_bias'].append(len(F) - len(T))
            per['count_MAE_sf'].append(abs(len(S) - len(Tsf))); per['count_bias_sf'].append(len(S) - len(Tsf))
            per['P21_err'].append(abs(F[:, 2].sum() - T[:, 2].sum()) / h if len(F) else T[:, 2].sum() / h)
            for k in g: per[k.replace('graph_', '') + '_err'].append(abs(g[k] - graph_cache[wid][k]))
            for k in ('fI', 'fY', 'fX'): per[k + '_err'].append(abs(nd[k] - node_cache[wid][k]))
            per['preservation'].append(preservation(obs, F)); per['obs_conflicts'].append(conflicts(obs, F, st))
            v = validity(F, h); per['invalid'].append(v['invalid']); per['duplicates'].append(v['duplicates'])
            for tag, TT, PP in (('full', T, F), ('sf', Tsf, S)):
                mm = _matched(TT, PP)
                if mm:
                    for k, x in mm.items(): per[f'{k}_{tag}'].append(x)
        Pf = np.vstack([p for p in Pf if len(p)]) if any(len(p) for p in Pf) else np.zeros((0, 4)); Ps = np.vstack([p for p in Ps if len(p)]) if any(len(p) for p in Ps) else np.zeros((0, 4))
        row = dict(fold=fold, layout=layout, arm=lab, source=cases[0]['src'], window=wid, training_seed=ts, n_gen=len(cases), n_truth=len(T), n_hidden=len(Tsf),
                   n_obs_clips=len(obs), unique_observed=len(T) - len(Tsf), n_strips=len(st), height_m=h)
        for prop, key in (('log_length', 'loglen'), ('direction', 'dir'), ('position', 'pos')):
            row[f'ED_{key}_full'] = _ed(Pf, T, prop); row[f'ES_{key}_full'] = _es(Pf, T, prop)
            row[f'ED_{key}_sf'] = _ed(Ps, Tsf, prop); row[f'ES_{key}_sf'] = _es(Ps, Tsf, prop)
        row.update({k: float(np.nanmean(v)) if len(v) else math.nan for k, v in per.items()})
        row['mean_loglen_out'] = float(np.log(Pf[:, 2] * 20).mean()) if len(Pf) else math.nan; row['mean_loglen_truth'] = float(np.log(T[:, 2] * 20).mean())
        row['sd_loglen_out'] = float(np.log(Pf[:, 2] * 20).std()) if len(Pf) > 1 else math.nan; row['sd_loglen_truth'] = float(np.log(T[:, 2] * 20).std())
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_rows(fold, layout):
    """Within-fold orientation-cluster secondaries (full window): truth and output assigned with the fold's training centres."""
    W = truth(fold); S = json.loads((STG / fold / 'eval' / 'training_only_statistics.json').read_text()); fc = np.asarray(S['fc']); out = []
    for src, lab, d in case_dirs(fold, layout):
        ci = json.loads(Path(d, 'case_inputs.json').read_text()); wid = '__'.join(Path(d).name.split('__')[1:3]); w = W[wid]
        F = pd.read_csv(Path(d) / 'predictions.csv')[COLS].to_numpy(np.float32); T = w['arr']
        gf = M.assign(T, fc) if len(T) else np.zeros(0, int); af = M.assign(F, fc) if len(F) else np.zeros(0, int)
        for k in range(4):
            gt, pr = T[gf == k], F[af == k]; mm = _matched(gt, pr)
            out.append(dict(fold=fold, layout=layout, arm=lab, window=wid, training_seed=int(ci.get('training_seed', 0)), generation_seed=int(ci['generation_seed']),
                            cluster='C' + str(k + 1) if k < 3 else 'Random', n_truth=len(gt), n_out=len(pr), length_MAE=mm['length_MAE'] if mm else math.nan,
                            angle_MAE=mm['angle_MAE'] if mm else math.nan, ED_loglen=_ed(pr, gt, 'log_length')))
    return pd.DataFrame(out)


if __name__ == '__main__':
    fold, layout = sys.argv[1], sys.argv[2]; out = HP / ('runs/smoke/endpoints' if SMOKE else 'results/endpoints'); out.mkdir(parents=True, exist_ok=True)
    endpoints(fold, layout).to_csv(out / f'ENDPOINTS_{fold}_{layout}.csv', index=False)
    cluster_rows(fold, layout).to_csv(out / f'CLUSTERS_{fold}_{layout}.csv', index=False); print('done', fold, layout)
