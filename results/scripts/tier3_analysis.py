"""Tier 3 component comparison (2026-10-29): Transformer-only (D_tf@roll1off) vs cKDE and cADFNE (@scoreoff). All three use the statistical
Tier 2 continuation, so they differ in the Tier 3 generator. Method identical to phase13_ttest/ttest_analysis.py (imported, not copied):
wall-group means over windows and generation seeds, then over training seeds 42001-42003; paired two-sided t-test on the nine wall groups;
Holm over the 16 primary metrics within each comparison. Inputs: phase11_seeds/results/SEED_UNITS.csv. CPU only."""
import sys, hashlib, platform
from pathlib import Path
import numpy as np, pandas as pd, scipy
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'phase13_ttest'))
import ttest_analysis as TT
CON = [('Transformer-only vs cKDE', 'Transformer_only', 'cKDE'), ('Transformer-only vs cADFNE', 'Transformer_only', 'cADFNE')]
SEEDS = [42001, 42002, 42003]


def table(seeds):
    out = []
    for name, a, b in CON:
        rows = []
        for m in TT.PRIM:
            f = TT.fold_pairs(seeds, a, b, m); r = TT.test(f.a.to_numpy(), f.b.to_numpy()); r.update(comparison=name, metric=m)
            r['change_pct'] = np.nan if m == 'position_ED' else 100 * r['diff'] / abs(r['mean_baseline']); rows.append(r)
        adj = TT.holm(np.array([r['p'] for r in rows]))
        for r, x in zip(rows, adj): r['p_holm'] = x; r['significant'] = x < .05; r['shapiro_flag'] = r['shapiro_p'] < .05
        out += rows
    cols = ['comparison', 'metric', 'n_groups', 'mean_model', 'mean_baseline', 'diff', 'change_pct', 'ci_lo', 'ci_hi', 'p', 'p_holm', 'significant',
            'groups_lower', 'shapiro_p', 'shapiro_flag', 'outliers_3MAD', 'wilcoxon_p']
    return pd.DataFrame(out)[cols]


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    T = table(SEEDS); T.to_csv(HERE / 'TIER3_TABLE.csv', index=False)
    P = []
    for s in SEEDS:
        t = table([s])[['comparison', 'metric', 'mean_model', 'mean_baseline', 'change_pct', 'p_holm', 'groups_lower']]; t.insert(0, 'seed', s); P.append(t)
    P = pd.concat(P); P.to_csv(HERE / 'TIER3_PER_SEED.csv', index=False)
    W = P.assign(lower=P.mean_model < P.mean_baseline).groupby(['comparison', 'metric']).lower.sum().rename('seeds_lower')
    T = T.merge(W, on=['comparison', 'metric'])
    U = TT.U; ident = {b: bool(U[U.arm == b].sort_values(['seed', 'fold', 'window']).groupby('seed')[TT.PRIM].apply(lambda g: g.to_numpy().tobytes()).nunique() == 1) for b in ('cKDE', 'cADFNE')}
    src = TT.R / 'SEED_UNITS.csv'; sha = hashlib.sha256(src.read_bytes()).hexdigest()
    print('identical across training seeds', ident)
    print(T.round(4)[['comparison', 'metric', 'mean_model', 'mean_baseline', 'change_pct', 'ci_lo', 'ci_hi', 'p', 'p_holm', 'significant', 'groups_lower', 'seeds_lower', 'shapiro_p', 'outliers_3MAD', 'wilcoxon_p']].to_string(index=False))
    print('INPUT', src, sha); print('VERSIONS', platform.python_version(), 'numpy', np.__version__, 'pandas', pd.__version__, 'scipy', scipy.__version__)
    T.to_csv(HERE / 'TIER3_TABLE.csv', index=False)
