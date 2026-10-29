"""Geological-similarity metric suite, computed from SAVED networks only (no generation, no model calls). Truth = eval-stage mapped traces of the
held-out test windows; orientation clusters assigned with each fold's training-only centres (eval stage training_only_statistics.json).
Units: rows (cx, cy, length, angle) in window units (x / 20 m, y / 20 m, length / 20 m), angle axial in [0, pi). Metres used in all reported values.

Arms (identical windows; S10o5 primary, also S05ref, S15o7.5, S20o10 where the run exists):
  conditioned pipeline (same observations, Tier 2 rules, count rule, budget, gates, thinning, scorer):
    B0_stat = submitted B_thin; B1_height = H_thin (phase1); EVAE_cont = submitted C_thin; TF_only = submitted D_thin; Hybrid = submitted E_thin;
    Hybrid_nothin = submitted E0 (original without crossing avoidance); EVAE_dec = E0eq_thin, E2_nat = E2nat_thin, E2_eq = E2eq_thin (S10o5 only)
  unconditioned generators (no scanline evidence): KDE = N_KDE, KDE_spatial = N_KDEs, ADFNE = N_ADFNE (benchmark L0 outputs; layout independent)
  reference: Mapped_replica = mapped traces of the other test window(s) of the same face (a real network that is not position-matched)
Per case (one generation realisation): counts, P21, local P21 on 5 m cells, occupancy, topology (frozen h2h_endpoints functions), preservation, matched
diagnostics (hidden subset). Per unit (arm, window, training seed; generation seeds pooled): cluster proportions, per-cluster axial mean / dispersion,
per-cluster log-length W1 and P90, cluster spacing, centre-position ED. Raw candidates (proposal ledgers / raw_candidates) summarised separately.
Writes cache/UNIT_<fold>_<layout>.csv, cache/RAW_<fold>_<layout>.csv."""
import glob, json, math, sys
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; STUDY = HERE.parent; SP = STUDY / 'dfn_spacing_and_manuscript_review_2026-10-22'; BM = STUDY / 'kde_adfne_benchmark_revision_2026-10-21'
PH1 = STUDY / 'PAPER2_EVAE_REPAIR_CONTROLLED_STUDY_2026-10-27' / 'phase1_height'; E2D = STUDY / 'PAPER2_E2_FULL_PIPELINE_VALIDATION_2026-10-27'; STG = STUDY / 'plan_d_step2_staging_2026-10-21' / 'stages'
sys.path.insert(0, str(SP / 'code')); sys.path.insert(0, str(BM / 'code'))
import layouts as LY  # noqa
import h2h_endpoints as H  # noqa (frozen: segment_graph, nodes, preservation, _matched, M)
M = H.M; FOLDS = ['P1', 'P2', 'P3', 'P4']; LAYOUTS = ['S10o5', 'S05ref', 'S15o7.5', 'S20o10']; CELL = 5.0
SPMAP = {'B_thin': 'B0_stat', 'C_thin': 'EVAE_cont', 'D_thin': 'TF_only', 'E_thin': 'Hybrid', 'E0': 'Hybrid_nothin'}; NAT = {'N_KDE': 'KDE', 'N_KDEs': 'KDE_spatial', 'N_ADFNE': 'ADFNE'}
E2MAP = {'E0eq_thin': 'EVAE_dec', 'E2nat_thin': 'E2_nat', 'E2eq_thin': 'E2_eq'}


def sources(fold, lay):
    out = [(SPMAP[Path(d).name.split('__')[0]], d) for d in glob.glob(str(SP / 'runs' / fold / lay / f'{lay}_on' / 'cases' / '*')) if Path(d).name.split('__')[0] in SPMAP]
    out += [('B1_height', d) for d in glob.glob(str(PH1 / 'runs' / fold / lay / f'{lay}_on' / 'cases' / 'H_thin__*'))]
    out += [(NAT[Path(d).name.split('__')[0]], d) for d in glob.glob(str(BM / 'runs' / fold / 'L0' / 'L0_on' / 'cases' / '*')) if Path(d).name.split('__')[0] in NAT]
    if lay == 'S10o5': out += [(E2MAP[Path(d).name.split('__')[0]], d) for d in glob.glob(str(E2D / 'runs' / fold / 't*' / 'S10o5_on' / 'cases' / '*')) if Path(d).name.split('__')[0] in E2MAP]
    return out


def clip_box(a, x0, x1, y0, y1):
    """Length of each segment inside an axis-aligned box (metres)."""
    p, q = M.ends(a); p = p * 20; q = q * 20; v = q - p; t0 = np.zeros(len(a)); t1 = np.ones(len(a))
    for k, lo, hi in ((0, x0, x1), (1, y0, y1)):
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (lo - p[:, k]) / v[:, k]; tb = (hi - p[:, k]) / v[:, k]
        par = np.abs(v[:, k]) < 1e-12; inside = (p[:, k] >= lo) & (p[:, k] <= hi)
        lo_t = np.where(par, np.where(inside, 0, 2), np.minimum(ta, tb)); hi_t = np.where(par, np.where(inside, 1, -1), np.maximum(ta, tb))
        t0 = np.maximum(t0, lo_t); t1 = np.minimum(t1, hi_t)
    return np.clip(t1 - t0, 0, None) * np.hypot(v[:, 0], v[:, 1])


def local_p21(a, h):
    nx = int(round(20 / CELL)); ny = max(1, int(h // CELL)); hy = h / ny; out = np.zeros((ny, nx))
    if len(a):
        for i in range(nx):
            for j in range(ny): out[j, i] = clip_box(a, i * CELL, (i + 1) * CELL, j * hy, (j + 1) * hy).sum() / (CELL * hy)
    return out.ravel()


def w1(x, y):
    if len(x) < 1 or len(y) < 1: return np.nan
    q = np.linspace(0, 1, 101)[1:-1]; return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def axial_stats(ang):
    z = np.exp(2j * np.asarray(ang)); m = z.mean(); return float((np.angle(m) / 2) % np.pi), float(abs(m))


def spacing(a, h, theta):
    """Spacing (m) between consecutive traces of one cluster along virtual scanlines normal to the cluster mean direction theta, lines every 1 m."""
    if len(a) < 2: return np.zeros(0)
    nrm = np.array([-math.sin(theta), math.cos(theta)]); d = np.array([math.cos(theta), math.sin(theta)]); p, q = M.ends(a); p = p * 20; q = q * 20; out = []
    corners = np.array([[0, 0], [20, 0], [0, h], [20, h]]); offs = corners @ d
    for c in np.arange(offs.min() + .5, offs.max(), 1.0):
        o = c * d; s = []
        for i in range(len(a)):
            u = q[i] - p[i]; den = u @ d
            if abs(den) < 1e-12: continue
            t = ((o - p[i]) @ d) / den
            if 0 <= t <= 1: s.append((p[i] + t * u) @ nrm)
        if len(s) > 1: out += list(np.diff(np.sort(s)))
    return np.asarray(out)


def case_metrics(F, T, h, st, obs):
    r = dict(n=len(F), count_err_pct=100 * abs(len(F) - len(T)) / max(len(T), 1), count_bias=len(F) - len(T),
             P21_rel_err_pct=100 * abs(F[:, 2].sum() - T[:, 2].sum()) / max(T[:, 2].sum(), 1e-9) if len(F) else 100.0)
    lp, lt = local_p21(F, h), local_p21(T, h)
    r.update(localP21_cell_MAE=float(np.abs(lp - lt).mean()), localP21_W1=w1(lp, lt), localP21_CV_diff=float(lp.std() / max(lp.mean(), 1e-9) - lt.std() / max(lt.mean(), 1e-9)),
             occupancy_diff=float((lp > 0).mean() - (lt > 0).mean()))
    pa, pb = M.ends(F); g = H.segment_graph(np.stack([pa, pb], 1) * 20, h) if len(F) else dict(graph_components=0, graph_largest_fraction=0, graph_top_bottom=0, graph_crossings=0)
    ta, tb = M.ends(T); gt = H.segment_graph(np.stack([ta, tb], 1) * 20, h); nd = H.nodes(F, h) if len(F) else dict(fI=0, fY=0, fX=0); nt = H.nodes(T, h); area = 20 * h
    r.update(crossing_density_bias=(g['graph_crossings'] - gt['graph_crossings']) / area, crossing_density_abs=abs(g['graph_crossings'] - gt['graph_crossings']) / area,
             components_rel_err=abs(g['graph_components'] - gt['graph_components']) / max(gt['graph_components'], 1), largest_fraction_err=abs(g['graph_largest_fraction'] - gt['graph_largest_fraction']),
             spanning_mismatch=abs(g['graph_top_bottom'] - gt['graph_top_bottom']), node_IYX_TV=0.5 * (abs(nd['fI'] - nt['fI']) + abs(nd['fY'] - nt['fY']) + abs(nd['fX'] - nt['fX'])),
             preservation=H.preservation(obs, F) if len(obs) else np.nan)
    hid_t = T[~H.crossing_strip(T, st)]; hid_p = F[~H.crossing_strip(F, st)]; mm = H._matched(hid_t, hid_p)
    if mm: r.update(matched_centre_err_m=mm['position_err'], matched_length_mae_m=mm['length_MAE'], matched_angle_err_deg=mm['angle_MAE'])
    return r


def unit_metrics(Fs, T, h, fc):
    P = np.vstack([f for f in Fs if len(f)]) if any(len(f) for f in Fs) else np.zeros((0, 4), np.float32); r = {}
    kp = M.assign(P, fc) if len(P) else np.zeros(0, int); kt = M.assign(T, fc); sp = np.bincount(kp, minlength=4) / max(len(P), 1); stt = np.bincount(kt, minlength=4) / max(len(T), 1)
    r['cluster_share_TV'] = float(.5 * np.abs(sp - stt).sum()); r['position_ED'] = M.energy_distance(P[:, :2] * 20, T[:, :2] * 20) if len(P) > 1 else np.nan
    r['loglen_W1_pooled'] = w1(np.log(P[:, 2] * 20), np.log(T[:, 2] * 20)) if len(P) else np.nan
    r['P90_len_logratio_pooled'] = float(np.log(np.quantile(P[:, 2], .9) / np.quantile(T[:, 2], .9))) if len(P) > 1 else np.nan
    for k in range(4):
        c = f'C{k + 1}' if k < 3 else 'Bg'; a, b = P[kp == k], T[kt == k]; r[f'share_{c}_pred'] = sp[k]; r[f'share_{c}_true'] = stt[k]
        if len(a) >= 2 and len(b) >= 2:
            r[f'loglen_W1_{c}'] = w1(np.log(a[:, 2] * 20), np.log(b[:, 2] * 20)); r[f'P90_logratio_{c}'] = float(np.log(np.quantile(a[:, 2], .9) / np.quantile(b[:, 2], .9)))
            if k < 3:
                ma, ra = axial_stats(a[:, 3]); mb, rb = axial_stats(b[:, 3]); r[f'mean_dir_err_deg_{c}'] = float(np.degrees(abs((ma - mb + np.pi / 2) % np.pi - np.pi / 2))); r[f'R_diff_{c}'] = ra - rb
                th = axial_stats(b[:, 3])[0]; sa = np.concatenate([spacing(f[M.assign(f, fc) == k], h, th) for f in Fs if len(f)] or [np.zeros(0)]); sb = spacing(b, h, th)
                if len(sa) > 2 and len(sb) > 2: r[f'spacing_logW1_{c}'] = w1(np.log(sa), np.log(sb)); r[f'spacing_median_logratio_{c}'] = float(np.log(np.median(sa) / np.median(sb)))
    return r


def truth(fold):
    W = H.truth(fold); S = json.loads((STG / fold / 'eval' / 'training_only_statistics.json').read_text()); return W, np.asarray(S['fc'])


def run(args):
    fold, lay = args; out = HERE / 'cache'; out.mkdir(exist_ok=True); p = out / f'UNIT_{fold}_{lay}.csv'
    if p.exists(): return str(p)
    W, fc = truth(fold); units = defaultdict(list); RAW = []
    for arm, d in sources(fold, lay):
        d = Path(d); parts = d.name.split('__'); wid = '__'.join(parts[1:3]); ci = json.loads((d / 'case_inputs.json').read_text())
        F = pd.read_csv(d / 'predictions.csv')[H.COLS].to_numpy(np.float32); units[(arm, wid, int(ci.get('training_seed', 0)))].append(F)
        T = W[wid]['arr']; st = LY.strips(lay, W[wid]['x0']); hid = T[~H.crossing_strip(T, st)]; R = None
        if (d / 'raw_candidates.csv.gz').exists():
            R = pd.read_csv(d / 'raw_candidates.csv.gz'); R = R[R.valid & R.eligible] if 'eligible' in R else R[R.valid]
        elif (d / 'proposal_ledger.csv.gz').exists():
            R = pd.read_csv(d / 'proposal_ledger.csv.gz'); R = R[~R.disposition.astype(str).str.contains('conflict|invalid|gate|reject', regex=True)]
        if R is not None and len(R) > 1 and len(hid) > 1:
            X = R[H.COLS].to_numpy(np.float32); kx = M.assign(X, fc); kh = M.assign(hid, fc); sx = np.bincount(kx, minlength=4) / len(X); sh = np.bincount(kh, minlength=4) / len(hid)
            RAW.append(dict(fold=fold, layout=lay, arm=arm, window=wid, training_seed=int(ci.get('training_seed', 0)), generation_seed=int(ci['generation_seed']), n_raw=len(X),
                            raw_cluster_share_TV=float(.5 * np.abs(sx - sh).sum()), raw_loglen_W1=w1(np.log(X[:, 2] * 20), np.log(hid[:, 2] * 20)),
                            raw_P90_logratio=float(np.log(np.quantile(X[:, 2], .9) / np.quantile(hid[:, 2], .9)))))
    for wid, w in W.items():   # mapped replica reference: the other test windows of the same face
        face = wid.split('__')[0]; donors = sorted(k for k in W if k != wid and k.split('__')[0] == face and abs(W[k]['h'] - w['h']) < .01)
        for dn in donors[:3]: units[('Mapped_replica', wid, 0)].append(W[dn]['arr'])
    rows = []
    for (arm, wid, ts), Fs in units.items():
        w = W[wid]; T = w['arr']; h = w['h']; st = LY.strips(lay, w['x0']); obs = M.observe(T, st)[0] if st else np.zeros((0, 4), np.float32)
        cm = pd.DataFrame([case_metrics(F, T, h, st, obs) for F in Fs]).mean(numeric_only=True).to_dict()
        rows.append(dict(fold=fold, layout=lay, arm=arm, window=wid, face=wid.split('__')[0], training_seed=ts, n_real=len(Fs), height_m=h, n_truth=len(T), **cm, **unit_metrics(Fs, T, h, fc)))
    pd.DataFrame(rows).to_csv(p, index=False); pd.DataFrame(RAW).to_csv(out / f'RAW_{fold}_{lay}.csv', index=False); return str(p)


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    jobs = [(f, l) for l in LAYOUTS for f in FOLDS]
    with Pool(6) as p:
        for r in p.imap_unordered(run, jobs): print(r, flush=True)
