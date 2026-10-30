"""Metric family reduced from 16 to 15 network metrics (position energy distance removed at the authors' decision, 30 October 2026).
Raw paired t-test results are unchanged; only the Holm adjustment over the network metrics of each comparison is recomputed.
Inputs (read-only): phase13_ttest/TTEST_TABLE.csv, phase14_tier3/TIER3_TABLE.csv, phase15_section8/S8_TESTS.csv.
Outputs (this folder): TTEST_TABLE_15.csv, TIER3_TABLE_15.csv, S8_TESTS_15.csv, CHANGES_15.csv."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; OUT = Path(__file__).resolve().parent; DROP = 'position_ED'


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for i, k in enumerate(o): run = max(run, min(1, (m - i) * p[k])); out[k] = run
    return out


changes = []
def redo(df, network, name, sig_col=None):
    keep = df[df.metric != DROP].copy()
    for c in keep.comparison.unique():
        if not network(c): continue
        ix = keep.comparison == c; new = holm(keep.loc[ix, 'p'].values)
        for (i, r), h in zip(keep[ix].iterrows(), new):
            changes.append(dict(table=name, comparison=c, metric=r.metric, p=r.p, p_holm_16=r.p_holm, p_holm_15=h))
        keep.loc[ix, 'p_holm'] = new
        if sig_col: keep.loc[ix, sig_col] = keep.loc[ix, 'p_holm'] < .05
    return keep


TT = pd.read_csv(R / 'phase13_ttest' / 'TTEST_TABLE.csv')
redo(TT, lambda c: c.startswith(('Hybrid vs', 'Transformer-only vs', 'EVAE-only vs')) and (TT[TT.comparison == c].shape[0] > 3), 'TTEST', 'significant').to_csv(OUT / 'TTEST_TABLE_15.csv', index=False)
T3 = pd.read_csv(R / 'phase14_tier3' / 'TIER3_TABLE.csv')
redo(T3, lambda c: T3[T3.comparison == c].shape[0] > 3, 'TIER3', 'significant' if 'significant' in T3 else None).to_csv(OUT / 'TIER3_TABLE_15.csv', index=False)
S8 = pd.read_csv(R / 'phase15_section8' / 'S8_TESTS.csv')
redo(S8, lambda c: c.startswith(('Hybrid_vs', 'Tier3_Transformer')), 'S8').to_csv(OUT / 'S8_TESTS_15.csv', index=False)
C = pd.DataFrame(changes); C.to_csv(OUT / 'CHANGES_15.csv', index=False)
flip = C[(C.p_holm_16 < .05) != (C.p_holm_15 < .05)]
print('rows recomputed', len(C)); print('significance changes:'); print(flip.to_string(index=False) if len(flip) else '  none')
