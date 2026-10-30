"""Section 8 test, step 4 (PREREGISTRATION_SECTION8.md): scores every reconstruction from its predictions.csv against the mapped Section 8 traces with
the metric definitions of geo_metrics (as phase14_tier3/verify/verify_tier3.py), then the pre-registered tests: window means (mean over generation
seeds, then over training seeds), paired two-sided t-test over windows, Holm over the 16 primary metrics per comparison; Tier 2 continuation error
(metrics.json tier2_parent_len_mae_m) EVAE (Hybrid) vs statistical (Transformer-only), single test. Directions per bench and per training seed.
Writes S8_UNITS.csv, S8_TESTS.csv, S8_PER_SEED.csv, S8_PER_BENCH.csv, S8_TIER2.csv, S8_MEANS.csv."""
import glob, json, sys, warnings
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; P2 = ROOT.parent
sys.path.insert(0, str(P2 / 'PAPER2_DFN_GEOLOGICAL_SIMILARITY_AUDIT_2026-10-27')); sys.dont_write_bytecode = True; import geo_metrics as G
M, LY = G.M, G.LY; C3 = ['C1', 'C2', 'C3']; LAY = 'S10o5'; SEEDS = (42001, 42002, 42003)
PRIM = ['cluster_share_TV', 'mean_dir_err_deg', 'abs_R_diff', 'loglen_W1_cluster', 'abs_P90_logratio_cluster', 'P21_rel_err_pct', 'localP21_W1', 'abs_localP21_CV_diff',
        'abs_occupancy_diff', 'position_ED', 'spacing_logW1', 'crossing_density_abs', 'node_IYX_TV', 'components_rel_err', 'largest_fraction_err', 'spanning_mismatch']
EV = HERE / 'B' / 'stages' / 'S8' / 'eval'; bench = lambda w: w.split('_P')[0]


def score(args):
    tag, arm, seed = args
    inv = pd.read_csv(EV / 'window_inventory.csv'); tm = pd.read_csv(EV / 'window_trace_manifest.csv'); fc = np.asarray(json.loads((EV / 'training_only_statistics.json').read_text())['fc'])
    W = {r.window_id: (tm[tm.window_id == r.window_id][['cx', 'cy', 'length', 'angle']].to_numpy(np.float32), float(r.height_m), float(r.x0_m)) for r in inv[inv.role == 'validation'].itertuples()}
    U = defaultdict(list); T2 = defaultdict(list)
    for d in glob.glob(str(HERE / 'recon' / tag / 'B' / 'runs' / 'S8' / LAY / f'{LAY}_{tag}' / 'cases' / f'{arm}__*')):
        p = Path(d).name.split('__'); wid = '__'.join(p[1:3])
        pr = pd.read_csv(Path(d) / 'predictions.csv'); F = pr[['cx', 'cy', 'length', 'angle']].to_numpy(np.float32); U[wid].append(F)
        T, h, x0 = W[wid]; st = LY.strips(LAY, x0)   # Tier 2 continuation error exactly as phase6_analysis/analysis_v2.run
        if st and len(T):
            obs, par = M.observe(T, st); Tp = T[np.unique(par)] if len(par) else T[:0]; fx = F[~pr.inferred.astype(str).str.lower().eq('true').to_numpy()]
            if len(fx) and len(Tp):
                ii, jj = linear_sum_assignment(cdist(Tp[:, :2] * 20, fx[:, :2] * 20)); T2[wid].append(float(np.abs(Tp[ii, 2] - fx[jj, 2]).mean() * 20))
    rows = []
    for wid, Fs in U.items():
        T, h, x0 = W[wid]; st = LY.strips(LAY, x0); obs = M.observe(T, st)[0] if st else np.zeros((0, 4), np.float32)
        cm = pd.DataFrame([G.case_metrics(F, T, h, st, obs) for F in Fs]).mean(numeric_only=True).to_dict()
        t2 = float(np.mean(T2[wid])) if T2[wid] else np.nan
        rows.append(dict(window=wid, bench=bench(wid), arm=arm, seed=seed, n_reps=len(Fs), tier2=t2, **cm, **G.unit_metrics(Fs, T, h, fc)))
    return pd.DataFrame(rows)


def derive(U):
    U['mean_dir_err_deg'] = U[[f'mean_dir_err_deg_{c}' for c in C3]].mean(1); U['abs_R_diff'] = U[[f'R_diff_{c}' for c in C3]].abs().mean(1)
    U['loglen_W1_cluster'] = U[[f'loglen_W1_{c}' for c in C3]].mean(1); U['abs_P90_logratio_cluster'] = U[[f'P90_logratio_{c}' for c in C3]].abs().mean(1)
    U['spacing_logW1'] = U[[f'spacing_logW1_{c}' for c in C3]].mean(1); U['abs_localP21_CV_diff'] = U.localP21_CV_diff.abs(); U['abs_occupancy_diff'] = U.occupancy_diff.abs(); return U


def holm(p):
    o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0
    for i, k in enumerate(o): run = max(run, min(1, (n - i) * p[k])); adj[k] = run
    return adj


def wmean(U, arm, m, seed=None):
    x = U[U.arm == arm] if seed is None or arm in ('cKDE', 'cADFNE') else U[(U.arm == arm) & (U.seed == seed)]
    return x.groupby('window')[m].mean()   # learned arms: mean over the three training seeds (each already a generation-seed mean)


def test(a, b, U, metrics, label):
    rows = []
    for m in metrics:
        A, B = wmean(U, a, m), wmean(U, b, m); d = pd.DataFrame(dict(a=A, b=B)).dropna(); dd = d.a - d.b; t = stats.ttest_rel(d.a, d.b)
        ci = stats.t.interval(.95, len(dd) - 1, loc=dd.mean(), scale=stats.sem(dd)) if len(dd) > 2 else (np.nan, np.nan)
        sh = stats.shapiro(dd).pvalue if len(dd) >= 3 else np.nan
        try: wx = stats.wilcoxon(d.a, d.b).pvalue
        except ValueError: wx = np.nan
        bn = d.assign(dd=dd, bench=[bench(w) for w in d.index]).groupby('bench').dd.mean()
        seeds = [] if a in ('cKDE', 'cADFNE') else [float((wmean(U, a, m, s) - B).dropna().mean()) for s in SEEDS]
        rows.append(dict(comparison=label, metric=m, n_windows=len(d), mean_a=d.a.mean(), mean_b=d.b.mean(), change_pct=100 * dd.mean() / abs(d.b.mean()) if m != 'position_ED' else np.nan,
                         diff=dd.mean(), ci_lo=ci[0], ci_hi=ci[1], p=t.pvalue, windows_lower=int((dd < 0).sum()), shapiro_p=sh, wilcoxon_p=wx,
                         benches_lower=f'{int((bn < 0).sum())}/{len(bn)}', seeds_lower=f'{sum(x < 0 for x in seeds)}/{len(seeds)}' if seeds else ''))
    if len(rows) > 1:
        for r, x in zip(rows, holm(np.array([r['p'] for r in rows]))): r['p_holm'] = x
    else: rows[0]['p_holm'] = rows[0]['p']
    return rows


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    jobs = [(('roll1off' if s == 42001 else f'roll1off_s{s}'), a, s) for s in SEEDS for a in ('E_hyb', 'D_tf')]
    jobs += [(('evaeoff' if s == 42001 else f'evaeoff_s{s}'), 'V_evae', s) for s in SEEDS] + [('scoreoff', a, 0) for a in ('cKDE', 'cADFNE')]
    with Pool(4) as p: U = derive(pd.concat(p.map(score, jobs), ignore_index=True))
    U.to_csv(HERE / 'S8_UNITS.csv', index=False); print(U.groupby(['arm', 'seed']).agg(windows=('n_reps', 'size'), reps=('n_reps', 'mean')).to_string())
    R = test('E_hyb', 'D_tf', U, ['tier2'], 'Tier2_EVAE_vs_statistical')
    for a, b, lab in [('D_tf', 'cKDE', 'Tier3_Transformer_vs_cKDE'), ('D_tf', 'cADFNE', 'Tier3_Transformer_vs_cADFNE'), ('E_hyb', 'cKDE', 'Hybrid_vs_cKDE'), ('E_hyb', 'cADFNE', 'Hybrid_vs_cADFNE')]:
        R += test(a, b, U, PRIM, lab)
    R = pd.DataFrame(R); R.to_csv(HERE / 'S8_TESTS.csv', index=False)
    means = pd.DataFrame({a: [wmean(U, a, m).mean() for m in ['tier2'] + PRIM] for a in ('V_evae', 'D_tf', 'E_hyb', 'cKDE', 'cADFNE')}, index=['tier2'] + PRIM); means.to_csv(HERE / 'S8_MEANS.csv')
    pd.set_option('display.width', 250); print(R[['comparison', 'metric', 'n_windows', 'mean_a', 'mean_b', 'change_pct', 'p', 'p_holm', 'windows_lower', 'benches_lower', 'seeds_lower', 'shapiro_p', 'wilcoxon_p']].round(4).to_string(index=False))
