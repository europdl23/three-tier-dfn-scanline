"""Scores A/B evaluation runs with the audit metric suite against each fold's eval-stage truth (test windows of the held-out wall group), computes the
pre-registered contrasts (SUCCESS_CRITERIA_PREREGISTERED.md) and the Tier 2 true-parent continuation error, and the matched-footprint A vs B comparison.
Writes UNITS_<cfg>.csv, CONTRASTS.csv, MEANS.csv, TIER2_CONTINUATION.csv, AB_MATCHED.csv, AB_MATCHED_CONTRASTS.csv."""
import glob, json, sys
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; P2 = ROOT.parent
sys.path.insert(0, str(P2 / 'PAPER2_DFN_GEOLOGICAL_SIMILARITY_AUDIT_2026-10-27')); import geo_metrics as G
M = G.M; LY = G.LY; C3 = ['C1', 'C2', 'C3']; LAYS = ['S10o5', 'S05ref', 'S15o7.5', 'S20o10']; ARMS = ['cKDE', 'cADFNE', 'eKDE', 'eADFNE', 'D_tf', 'E_hyb']
PRIM = ['cluster_share_TV', 'mean_dir_err_deg', 'abs_R_diff', 'loglen_W1_cluster', 'abs_P90_logratio_cluster', 'P21_rel_err_pct', 'localP21_W1', 'abs_localP21_CV_diff', 'abs_occupancy_diff', 'position_ED', 'spacing_logW1',
        'crossing_density_abs', 'node_IYX_TV', 'components_rel_err', 'largest_fraction_err', 'spanning_mismatch']
SEC = ['count_err_pct', 'matched_centre_err_m', 'matched_length_mae_m', 'matched_angle_err_deg', 'preservation']


def truth(cfg, fold):
    ev = Path(__import__('os').environ['DFN3_DATA']) / 'stages' / fold / 'eval'; inv = pd.read_csv(ev / 'window_inventory.csv'); tm = pd.read_csv(ev / 'window_trace_manifest.csv')
    fc = np.asarray(json.loads((ev / 'training_only_statistics.json').read_text())['fc'])
    W = {r.window_id: dict(arr=tm[tm.window_id == r.window_id][['cx', 'cy', 'length', 'angle']].to_numpy(np.float32), h=float(r.height_m), x0=float(r.x0_m)) for r in inv[inv.role == 'validation'].itertuples()}
    return W, fc


def strips(lay, x0): return M.strips('5m') if lay == 'S05ref' else LY.strips(lay, x0)


def derive(U):
    U['mean_dir_err_deg'] = U[[f'mean_dir_err_deg_{c}' for c in C3]].mean(1); U['abs_R_diff'] = U[[f'R_diff_{c}' for c in C3]].abs().mean(1); U['loglen_W1_cluster'] = U[[f'loglen_W1_{c}' for c in C3]].mean(1)
    U['abs_P90_logratio_cluster'] = U[[f'P90_logratio_{c}' for c in C3]].abs().mean(1); U['spacing_logW1'] = U[[f'spacing_logW1_{c}' for c in C3]].mean(1); U['abs_localP21_CV_diff'] = U.localP21_CV_diff.abs(); U['abs_occupancy_diff'] = U.occupancy_diff.abs(); return U


def run(args):
    cfg, fold, lay = args; W, fc = truth(cfg, fold); units = defaultdict(list); t2 = []
    for d in glob.glob(str(ROOT / 'phase5_eval' / cfg / 'runs' / fold / lay / f'{lay}_on' / 'cases' / '*')):
        d = Path(d); p = d.name.split('__'); arm = p[0]; wid = '__'.join(p[1:3]); pr = pd.read_csv(d / 'predictions.csv'); F = pr[['cx', 'cy', 'length', 'angle']].to_numpy(np.float32); units[(arm, wid)].append(F)
        w = W[wid]; st = strips(lay, w['x0']); T = w['arr']
        if st and len(T):
            obs, par = M.observe(T, st); Tp = T[np.unique(par)] if len(par) else T[:0]; fx = F[~pr.inferred.astype(str).str.lower().eq('true').to_numpy()]
            if len(fx) and len(Tp):
                ii, jj = linear_sum_assignment(cdist(Tp[:, :2] * 20, fx[:, :2] * 20)); t2.append(dict(cfg=cfg, fold=fold, layout=lay, arm=arm, window=wid, tier2_parent_len_mae_m=float(np.abs(Tp[ii, 2] - fx[jj, 2]).mean() * 20), n_tier12=len(fx), n_parents=len(Tp)))
    rows = []
    for (arm, wid), Fs in units.items():
        w = W[wid]; T = w['arr']; st = strips(lay, w['x0']); obs = M.observe(T, st)[0] if st else np.zeros((0, 4), np.float32)
        cm = pd.DataFrame([G.case_metrics(F, T, w['h'], st, obs) for F in Fs]).mean(numeric_only=True).to_dict()
        rows.append(dict(cfg=cfg, fold=fold, layout=lay, arm=arm, window=wid, n_real=len(Fs), height_m=w['h'], **cm, **G.unit_metrics(Fs, T, w['h'], fc)))
    return pd.DataFrame(rows), pd.DataFrame(t2)


def matched(args):
    """A reconstructions cut into the 12 m bands of B (top-down) and scored against the B truth band (fold eval truth of config B); also B itself."""
    fold, lay = args; WB, fc = truth('B', fold); WA, _ = truth('A', fold); rows = []
    for d in glob.glob(str(ROOT / 'phase5_eval' / 'A' / 'runs' / fold / lay / f'{lay}_on' / 'cases' / '*')):
        d = Path(d); p = d.name.split('__'); arm = p[0]; wid = '__'.join(p[1:3]); hA = WA[wid]['h']; F = pd.read_csv(d / 'predictions.csv')[['cx', 'cy', 'length', 'angle']].to_numpy(np.float32)
        for k in range(int(hA // 12)):
            bid = f'{wid}_Y{k}'
            if bid not in WB: continue
            yb, yt = (hA - 12 * (k + 1)) / 20, (hA - 12 * k) / 20; a, b = M.ends(F); seg = []
            for (p_, q_) in zip(a, b):
                dd = q_ - p_; t0, t1 = 0., 1.
                for pv, qv in ((-dd[1], p_[1] - yb), (dd[1], yt - p_[1])):
                    if abs(pv) < 1e-12:
                        if qv < 0: t0, t1 = 1, 0
                    else:
                        t = qv / pv
                        if pv < 0: t0 = max(t0, t)
                        else: t1 = min(t1, t)
                if t1 > t0 + 1e-9:
                    s0, s1 = p_ + t0 * dd - [0, yb], p_ + t1 * dd - [0, yb]; v = s1 - s0; seg.append([*(s0 + s1) / 2, np.hypot(*v), np.arctan2(v[1], v[0]) % np.pi])
            Fb = np.asarray(seg, np.float32).reshape(-1, 4); w = WB[bid]; st = strips(lay, w['x0']); obs = M.observe(w['arr'], st)[0] if st else np.zeros((0, 4), np.float32)
            rows.append(dict(source='A_cut_to_B_band', fold=fold, layout=lay, arm=arm, window=bid, **G.case_metrics(Fb, w['arr'], 12., st, obs), **G.unit_metrics([Fb], w['arr'], 12., fc)))
    for d in glob.glob(str(ROOT / 'phase5_eval' / 'B' / 'runs' / fold / lay / f'{lay}_on' / 'cases' / '*')):
        d = Path(d); p = d.name.split('__'); arm = p[0]; wid = '__'.join(p[1:3]); w = WB[wid]; F = pd.read_csv(d / 'predictions.csv')[['cx', 'cy', 'length', 'angle']].to_numpy(np.float32); st = strips(lay, w['x0']); obs = M.observe(w['arr'], st)[0] if st else np.zeros((0, 4), np.float32)
        rows.append(dict(source='B', fold=fold, layout=lay, arm=arm, window=wid, **G.case_metrics(F, w['arr'], 12., st, obs), **G.unit_metrics([F], w['arr'], 12., fc)))
    return pd.DataFrame(rows)


rng = np.random.default_rng(20261028)


def boot(d, fo, n=2000):
    folds = np.unique(fo); bs = []
    for _ in range(n):
        pick = rng.choice(folds, len(folds)); bs.append(np.concatenate([d[fo == f] for f in pick]).mean())
    bs = np.asarray(bs); return np.percentile(bs, 2.5), np.percentile(bs, 97.5), float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))


CON = [('EVAE', 'eKDE', 'cKDE'), ('EVAE', 'eADFNE', 'cADFNE'), ('TF', 'D_tf', 'cKDE'), ('TF', 'D_tf', 'cADFNE'), ('Hybrid_vs_bench', 'E_hyb', 'cKDE'), ('Hybrid_vs_bench', 'E_hyb', 'cADFNE'),
       ('add_EVAE_to_TF', 'E_hyb', 'D_tf'), ('add_TF_to_EVAE', 'E_hyb', 'eKDE'), ('add_TF_to_EVAE', 'E_hyb', 'eADFNE')]


def contrasts(Wn, keys):
    out = []
    for key, g in Wn.groupby(keys):
        for kind, a, b in CON:
            A = g[g.arm == a].set_index(['fold', 'window']); B = g[g.arm == b].set_index(['fold', 'window']); ix = A.index.intersection(B.index)
            if not len(ix): continue
            for m in PRIM + SEC:
                if m not in A: continue
                d = (A.loc[ix, m] - B.loc[ix, m]).to_numpy(float); ok = ~np.isnan(d)
                if ok.sum() < 4: continue
                fo = ix.get_level_values(0).to_numpy()[ok]; d = d[ok]; lo, hi, pv = boot(d, fo); fm = pd.Series(d).groupby(fo).mean(); ref = float(B.loc[ix, m].to_numpy(float)[ok].mean())
                out.append(dict(**dict(zip(keys, key if isinstance(key, tuple) else (key,))), contrast=kind, model=a, baseline=b, metric=m, family='primary' if m in PRIM else 'secondary', n_windows=int(ok.sum()), n_folds=len(fm),
                                baseline_value=ref, model_value=ref + float(d.mean()), difference=float(d.mean()), change_pct=100 * float(d.mean()) / abs(ref) if (m != 'position_ED' and ref > 0) else np.nan,
                                ci95_lo=lo, ci95_hi=hi, p_boot=pv, folds_lower=int((fm < 0).sum()), folds_higher=int((fm > 0).sum())))
    C = pd.DataFrame(out); C['p_holm'] = np.nan
    for k, gg in C[C.family == 'primary'].groupby(keys + ['model', 'baseline']):
        o = gg.p_boot.sort_values(); n = len(o); C.loc[o.index, 'p_holm'] = np.maximum.accumulate([min(1, (n - i) * p) for i, p in enumerate(o.to_numpy())])
    big = (C.change_pct.abs() >= 3) | C.change_pct.isna(); need = np.ceil(2 * C.n_folds / 3)
    C['supported_better'] = (C.p_holm < .05) & (C.difference < 0) & (C.folds_lower >= need) & big; C['supported_worse'] = (C.p_holm < .05) & (C.difference > 0) & (C.folds_higher >= need) & big
    return C


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    cfgs = [c for c in ('A', 'B') if (ROOT / 'phase5_eval' / c / 'runs').exists()]
    jobs = [(c, f.name, l) for c in cfgs for f in sorted((ROOT / 'phase3_staging' / c / 'stages').iterdir()) for l in LAYS]
    with Pool(8) as p: res = p.map(run, jobs)
    U = derive(pd.concat([r[0] for r in res], ignore_index=True)); T2 = pd.concat([r[1] for r in res], ignore_index=True); U.to_csv(HERE / 'UNITS.csv', index=False); T2.to_csv(HERE / 'TIER2_CONTINUATION.csv', index=False)
    Wn = U.groupby(['cfg', 'layout', 'fold', 'window', 'arm'])[PRIM + SEC].mean().reset_index(); Wn.groupby(['cfg', 'layout', 'arm'])[PRIM + SEC].mean().to_csv(HERE / 'MEANS.csv')
    C = contrasts(Wn, ['cfg', 'layout']); C.to_csv(HERE / 'CONTRASTS.csv', index=False)
    K = C[C.family == 'primary'].groupby(['cfg', 'layout', 'contrast', 'model', 'baseline'])[['supported_better', 'supported_worse']].sum().reset_index(); K.to_csv(HERE / 'COUNTS.csv', index=False)
    if set(cfgs) == {'A', 'B'}:
        with Pool(8) as p: R = pd.concat(p.map(matched, [(f.name, l) for f in sorted((ROOT / 'phase3_staging' / 'B' / 'stages').iterdir()) for l in LAYS]), ignore_index=True)
        R = derive(R); R.to_csv(HERE / 'AB_MATCHED.csv', index=False)
        Rm = R.groupby(['layout', 'arm', 'source', 'fold', 'window'])[PRIM + SEC].mean().reset_index(); out = []
        for (lay, arm), g in Rm.groupby(['layout', 'arm']):
            A = g[g.source == 'A_cut_to_B_band'].set_index(['fold', 'window']); B = g[g.source == 'B'].set_index(['fold', 'window']); ix = A.index.intersection(B.index)
            for m in PRIM + SEC:
                d = (B.loc[ix, m] - A.loc[ix, m]).to_numpy(float); ok = ~np.isnan(d)
                if ok.sum() < 4: continue
                fo = ix.get_level_values(0).to_numpy()[ok]; d = d[ok]; lo, hi, pv = boot(d, fo, 1000); fm = pd.Series(d).groupby(fo).mean(); ref = float(A.loc[ix, m].to_numpy(float)[ok].mean())
                out.append(dict(layout=lay, arm=arm, metric=m, A_value=ref, B_value=ref + float(d.mean()), B_minus_A=float(d.mean()), change_pct=100 * float(d.mean()) / abs(ref) if (m != 'position_ED' and ref > 0) else np.nan,
                                ci95_lo=lo, ci95_hi=hi, p_boot=pv, folds_B_lower=int((fm < 0).sum()), n_folds=len(fm)))
        pd.DataFrame(out).to_csv(HERE / 'AB_MATCHED_CONTRASTS.csv', index=False)
    pd.set_option('display.width', 250); print(K[K.layout == 'S10o5'].to_string(index=False))
