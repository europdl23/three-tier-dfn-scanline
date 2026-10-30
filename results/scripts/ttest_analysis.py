"""Agreed statistics (author's method, 2026-10-29): paired two-sided t-tests on the nine wall-group means, errors averaged over the three
training seeds within each wall group, Holm over the 16 primary metrics within each comparison; Tier 2 continuation error as a single test.
Inputs: phase11_seeds/results/SEED_UNITS.csv (window means over generation seeds 72001-72003, scored with the paper's code) and SEED_TIER2.csv.
Outputs: TTEST_TABLE.csv, TTEST_TABLE_42001.csv, SEED_SD_TABLE.csv, TTEST_RESULTS.md. CPU only."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent; R = HERE.parent / 'phase11_seeds' / 'results'
PRIM = ['cluster_share_TV', 'mean_dir_err_deg', 'abs_R_diff', 'loglen_W1_cluster', 'abs_P90_logratio_cluster', 'P21_rel_err_pct', 'localP21_W1', 'abs_localP21_CV_diff',
        'abs_occupancy_diff', 'position_ED', 'spacing_logW1', 'crossing_density_abs', 'node_IYX_TV', 'components_rel_err', 'largest_fraction_err', 'spanning_mismatch']
CON = [('Hybrid vs EVAE-only', 'Hybrid', 'EVAE_only'), ('Hybrid vs Transformer-only', 'Hybrid', 'Transformer_only'),
       ('Hybrid vs cADFNE', 'Hybrid', 'cADFNE'), ('Hybrid vs cKDE', 'Hybrid', 'cKDE')]
U = pd.read_csv(R / 'SEED_UNITS.csv'); T2 = pd.read_csv(R / 'SEED_TIER2.csv')


def fold_pairs(seeds, a, b, m):
    """Per wall group: mean of a and b over the windows where both are defined, per seed, then mean over seeds."""
    rows = []
    for s in seeds:
        A = U[(U.seed == s) & (U.arm == a)].set_index(['fold', 'window'])[m]; B = U[(U.seed == s) & (U.arm == b)].set_index(['fold', 'window'])[m]
        ix = A.index.intersection(B.index); d = pd.DataFrame(dict(a=A.loc[ix], b=B.loc[ix])).dropna()
        f = d.groupby(level=0).mean(); f['seed'] = s; rows.append(f)
    return pd.concat(rows).groupby(level=0)[['a', 'b']].mean()


def test(a, b):
    d = a - b; n = len(d); t = stats.ttest_rel(a, b); se = d.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(.975, n - 1)
    sh = stats.shapiro(d).pvalue if n >= 3 else np.nan; med = np.median(d); mad = np.median(np.abs(d - med))
    outl = int((np.abs(d - med) > 3 * mad).sum()) if mad > 0 else 0
    try: w = stats.wilcoxon(a, b).pvalue
    except ValueError: w = np.nan
    return dict(n_groups=n, mean_model=a.mean(), mean_baseline=b.mean(), diff=d.mean(), ci_lo=d.mean() - q * se, ci_hi=d.mean() + q * se,
                p=t.pvalue, groups_lower=int((d < 0).sum()), shapiro_p=sh, outliers_3MAD=outl, wilcoxon_p=w)


def holm(p):
    o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0
    for i, k in enumerate(o): run = max(run, min(1, (n - i) * p[k])); adj[k] = run
    return adj


def table(seeds):
    out = []
    for name, a, b in CON:
        rows = []
        for m in PRIM:
            f = fold_pairs(seeds, a, b, m); r = test(f.a.to_numpy(), f.b.to_numpy()); r.update(comparison=name, metric=m)
            r['change_pct'] = np.nan if m == 'position_ED' else 100 * r['diff'] / abs(r['mean_baseline']); rows.append(r)
        p = np.array([r['p'] for r in rows]); adj = holm(p)
        for r, x in zip(rows, adj): r['p_holm'] = x; r['significant'] = x < .05; r['shapiro_flag'] = r['shapiro_p'] < .05
        out += rows
    t2 = T2[T2.seed.isin(seeds)].groupby(['seed', 'arm', 'fold']).tier2_parent_len_mae_m.mean().unstack('arm').groupby(level='fold').mean()
    r = test(t2.E_hyb.to_numpy(), t2.D_tf.to_numpy()); r.update(comparison='Tier 2 continuation: EVAE vs statistical', metric='tier2_parent_len_mae_m',
        change_pct=100 * r['diff'] / r['mean_baseline'], p_holm=np.nan, significant=r['p'] < .05, shapiro_flag=r['shapiro_p'] < .05); out.append(r)
    cols = ['comparison', 'metric', 'n_groups', 'mean_model', 'mean_baseline', 'diff', 'change_pct', 'ci_lo', 'ci_hi', 'p', 'p_holm', 'significant',
            'groups_lower', 'shapiro_p', 'shapiro_flag', 'outliers_3MAD', 'wilcoxon_p']
    return pd.DataFrame(out)[cols]


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    T = table([42001, 42002, 42003]); T.to_csv(HERE / 'TTEST_TABLE.csv', index=False)
    T1 = table([42001]); T1.to_csv(HERE / 'TTEST_TABLE_42001.csv', index=False)
    # reproducibility: all-window means per seed, mean and SD over the three training seeds
    M = U.groupby(['seed', 'arm'])[PRIM].mean(); S = M.groupby(level='arm').agg(['mean', 'std'])
    S.columns = [f'{m}_{k}' for m, k in S.columns]; S.to_csv(HERE / 'SEED_SD_TABLE.csv')
    ident = {b: bool((U[U.arm == b].sort_values(['seed', 'fold', 'window']).groupby('seed')[PRIM].apply(lambda g: g.to_numpy()).apply(lambda x: x.tobytes()).nunique()) == 1) for b in ('cKDE', 'cADFNE')}
    sig = T[T.significant]
    md = ['# Paired t-tests on wall-group means (three training seeds averaged)', '', f'cKDE/cADFNE identical across training seeds: {ident}', '',
          '## Significant results (Holm-adjusted p < 0.05; Tier 2 unadjusted)', '', sig.round(4).to_string(index=False), '',
          '## Significant results with Shapiro-Wilk p < 0.05 or outliers', '', sig[sig.shapiro_flag | (sig.outliers_3MAD > 0)].round(4).to_string(index=False), '',
          '## Full table', '', T.round(4).to_string(index=False), '', '## Reference: seed 42001 only, significant', '', T1[T1.significant].round(4).to_string(index=False)]
    (HERE / 'TTEST_RESULTS.md').write_text('\n'.join(md), encoding='utf-8')
    print('identical', ident); print(sig.round(4).to_string(index=False)); print('--- 42001 only'); print(T1[T1.significant].round(4).to_string(index=False))
    print('--- near misses (p_holm<0.2)'); print(T[(T.p_holm < .2) & ~T.significant].round(4)[['comparison', 'metric', 'change_pct', 'p', 'p_holm', 'groups_lower']].to_string(index=False))
