"""Scores the reconstructions in a work folder against the mapped windows and applies the paired statistics.

    python scripts/evaluate.py [--work work] [--out work/results] [--training-seed 42001] [--jobs 4]

Metrics: the 16 primary metrics in five groups (orientation, length, intensity, spatial arrangement, topology) and the secondary measures,
per reconstruction, averaged over the three generation seeds of a window. Paired differences per window are tested with a bootstrap that
resamples whole wall groups (2000 draws, deterministic seed), Holm-corrected over the 16 primary metrics. A difference is supported when the
corrected p < 0.05, it has the same direction in at least two thirds of the wall groups, and it changes the baseline value by at least 3%
(no percentage threshold for the position energy distance). Writes UNITS, WINDOW_UNITS, MEANS, FOLD_MEANS, TIER2, CONTRASTS, COUNTS, RULES (.csv)."""
import argparse, sys, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from _common import P

sys.path.insert(0, str(P.EXP / 'phase9_spatial'))
import phase9_analysis as P9  # noqa: E402  (deterministic bootstrap boot_det installed into the metric module)
A = P9.A; V3 = P9.V3
NAME = {'E_hyb@roll1off': 'Hybrid', 'D_tf@roll1off': 'Transformer_only', 'V_evae@evaeoff': 'EVAE_only', 'cKDE@scoreoff': 'cKDE', 'cADFNE@scoreoff': 'cADFNE'}
CON = [('Hybrid_vs_EVAE_only', 'E_hyb@roll1off', 'V_evae@evaeoff'), ('Hybrid_vs_Transformer_only', 'E_hyb@roll1off', 'D_tf@roll1off'),
       ('Hybrid_vs_cKDE', 'E_hyb@roll1off', 'cKDE@scoreoff'), ('Hybrid_vs_cADFNE', 'E_hyb@roll1off', 'cADFNE@scoreoff')]
KEY = ['P21_rel_err_pct', 'loglen_W1_cluster', 'node_IYX_TV']   # fracture intensity, length and node types (10% loss limit of the decision rules)


def rules(C):
    g = C[C.family == 'primary']; h = {c: g[g.contrast == c] for c, _, _ in CON if (g.contrast == c).any()}; out = []
    big_loss = lambda d: d[d.supported_worse & d.metric.isin(KEY) & (d.change_pct > 10)]
    if {'Hybrid_vs_EVAE_only', 'Hybrid_vs_Transformer_only'} <= set(h):
        ab = [h['Hybrid_vs_EVAE_only'], h['Hybrid_vs_Transformer_only']]
        c1 = all(d.supported_better.any() for d in ab); c2 = all(big_loss(d).empty for d in ab); c3 = all(d.supported_better.sum() > d.supported_worse.sum() for d in ab)
        out.append(dict(rule='combines strengths (Hybrid vs EVAE-only and Transformer-only)', met=c1 and c2 and c3, cond1=c1, cond2=c2, cond3=c3))
    for b in ['cKDE', 'cADFNE']:
        c = f'Hybrid_vs_{b}'
        if c in h:
            d = h[c]; nb, nw = int(d.supported_better.sum()), int(d.supported_worse.sum())
            out.append(dict(rule=f'outperforms {b}', met=nb >= 4 and nb > nw and big_loss(d).empty, supported_better=nb, supported_worse=nw))
    return pd.DataFrame(out)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    a = argparse.ArgumentParser(); a.add_argument('--work', default='work'); a.add_argument('--out', default=None); a.add_argument('--training-seed', type=int, default=42001)
    a.add_argument('--jobs', type=int, default=4); x = a.parse_args()
    work = Path(x.work).resolve(); out = Path(x.out).resolve() if x.out else work / 'results'; out.mkdir(parents=True, exist_ok=True)
    sfx = '' if x.training_seed == 42001 else f'_s{x.training_seed}'
    jobs = [('B', f, P.LAYOUT, str(work / (t + sfx)), t + sfx) for t in ['roll1off', 'evaeoff', 'scoreoff'] for f in P.FOLDS if (work / (t + sfx) / 'B' / 'runs' / f / P.LAYOUT).exists()]
    if not jobs: raise SystemExit(f'no reconstructions found in {work}')
    with Pool(x.jobs) as p: res = p.map(V3.job_tag, jobs)
    U = pd.concat([r[0] for r in res], ignore_index=True)
    for c in ['C1', 'C2', 'C3']:
        if f'spacing_logW1_{c}' not in U: U[f'spacing_logW1_{c}'] = np.nan   # undefined when too few windows are scored
    U = A.derive(U); U['arm'] = U.arm + '@' + U.tag.str.replace(sfx, '', regex=False) if sfx else U.arm + '@' + U.tag
    U = U[U.arm.isin(NAME)].reset_index(drop=True); U['framework'] = U.arm.map(NAME); U['wall_group'] = U.fold.map(P.WALL_GROUP); U.to_csv(out / 'UNITS.csv', index=False)
    U = pd.read_csv(out / 'UNITS.csv')   # statistics from the stored units, as in the study (CSV round trip fixes the floating-point values)
    T2 = pd.concat([r[1] for r in res], ignore_index=True); T2['arm'] = T2.arm + '@' + T2.tag.str.replace(sfx, '', regex=False)
    T2 = T2[T2.arm.isin(NAME)]; T2['framework'] = T2.arm.map(NAME); T2.to_csv(out / 'TIER2.csv', index=False)
    T2.groupby(['framework', 'fold']).tier2_parent_len_mae_m.mean().unstack(0).to_csv(out / 'TIER2_FOLD_MEANS.csv')
    Wn = U.groupby(['cfg', 'layout', 'fold', 'window', 'arm'])[A.PRIM + A.SEC].mean().reset_index(); Wn.to_csv(out / 'WINDOW_UNITS.csv', index=False)
    M = Wn.groupby('arm')[A.PRIM + A.SEC].mean(); M.insert(0, 'framework', M.index.map(NAME)); M.to_csv(out / 'MEANS.csv')
    Wn.groupby(['arm', 'fold'])[A.PRIM + A.SEC].mean().to_csv(out / 'FOLD_MEANS.csv')
    A.CON = [c for c in CON if {c[1], c[2]} <= set(Wn.arm)]
    if A.CON and Wn.window.nunique() < 4: print('fewer than 4 windows: contrasts are not computed'); A.CON = []
    if A.CON:
        C = A.contrasts(Wn, ['cfg', 'layout']); C.to_csv(out / 'CONTRASTS.csv', index=False)
        K = C[C.family == 'primary'].groupby(['contrast', 'model', 'baseline'])[['supported_better', 'supported_worse']].sum().reset_index(); K.to_csv(out / 'COUNTS.csv', index=False)
        Rr = rules(C); Rr.to_csv(out / 'RULES.csv', index=False)
        pd.set_option('display.width', 200); print(K.to_string(index=False)); print(Rr.to_string(index=False))
    print('written to', out)
