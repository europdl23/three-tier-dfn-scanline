"""Tier 3-only errors: accepted inferred traces (predictions.csv inferred==True) vs the mapped traces crossed by no scanline.
Transformer-only (D_tf roll1off, training seeds 42001-42003) vs cKDE and cADFNE (scoreoff). Aggregation and test as phase14:
window (mean over generation seeds, pooled realisations) -> wall group mean -> mean over training seeds; paired t-test n=9;
Holm over the three Tier 3 metrics within each comparison."""
import glob, json, sys, warnings
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; P2 = ROOT.parent; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(P2 / 'PAPER2_DFN_GEOLOGICAL_SIMILARITY_AUDIT_2026-10-27')); import geo_metrics as G
M, LY = G.M, G.LY; C3 = ['C1', 'C2', 'C3']; LAY = 'S10o5'
MET = ['loglen_W1_cluster', 'mean_dir_err_deg', 'cluster_share_TV']

def score(args):
    fold, base, tag, arm, seed = args
    ev = ROOT / 'phase3_staging' / 'B' / 'stages' / fold / 'eval'; inv = pd.read_csv(ev / 'window_inventory.csv'); tm = pd.read_csv(ev / 'window_trace_manifest.csv')
    fc = np.asarray(json.loads((ev / 'training_only_statistics.json').read_text())['fc'])
    W = {r.window_id: (tm[tm.window_id == r.window_id][['cx', 'cy', 'length', 'angle']].to_numpy(np.float32), float(r.height_m), float(r.x0_m)) for r in inv[inv.role == 'validation'].itertuples()}
    U = defaultdict(list)
    for d in glob.glob(str(Path(base) / 'B' / 'runs' / fold / LAY / f'{LAY}_{tag}' / 'cases' / f'{arm}__*')):
        p = Path(d).name.split('__'); wid = '__'.join(p[1:3]); P = pd.read_csv(Path(d) / 'predictions.csv')
        U[wid].append(P[P.inferred.astype(str).str.lower() == 'true'][['cx', 'cy', 'length', 'angle']].to_numpy(np.float32))
    rows = []
    for wid, Fs in U.items():
        T, h, x0 = W[wid]; st = LY.strips(LAY, x0)
        par = set(M.observe(T, st)[1].tolist()) if st else set(); Th = T[[i for i in range(len(T)) if i not in par]]
        if len(Th) < 2: continue
        r = G.unit_metrics(Fs, Th, h, fc)
        rows.append(dict(fold=fold, window=wid, arm=arm, seed=seed, n_reps=len(Fs), n_hidden=len(Th), n_inf=float(np.mean([len(f) for f in Fs])),
                         loglen_W1_cluster=np.nanmean([r.get(f'loglen_W1_{c}', np.nan) for c in C3]),
                         mean_dir_err_deg=np.nanmean([r.get(f'mean_dir_err_deg_{c}', np.nan) for c in C3]), cluster_share_TV=r['cluster_share_TV']))
    return pd.DataFrame(rows)

def holm(p):
    o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0
    for i, k in enumerate(o): run = max(run, min(1, (n - i) * p[k])); adj[k] = run
    return adj

if __name__ == '__main__':
    warnings.filterwarnings('ignore'); F7 = ROOT / 'phase7_followup'
    folds = sorted(f.name for f in (ROOT / 'phase3_staging' / 'B' / 'stages').iterdir())
    jobs = [(f, str(F7 / ('roll1off' + ('' if s == 42001 else f'_s{s}'))), 'roll1off' + ('' if s == 42001 else f'_s{s}'), 'D_tf', s) for s in (42001, 42002, 42003) for f in folds]
    jobs += [(f, str(F7 / 'scoreoff'), 'scoreoff', a, 0) for a in ('cKDE', 'cADFNE') for f in folds]
    with Pool(4) as p: U = pd.concat(p.map(score, jobs), ignore_index=True)
    U.to_csv(HERE / 'TIER3ONLY_UNITS.csv', index=False); print(U.groupby(['arm', 'seed']).agg(n=('n_reps', 'size'), reps=('n_reps', 'mean'), hidden=('n_hidden', 'mean'), inf=('n_inf', 'mean')))
    out, per = [], []
    for b in ('cKDE', 'cADFNE'):
        rows = []
        for m in MET:
            B = U[U.arm == b].set_index(['fold', 'window'])[m]; fs = []
            for s in (42001, 42002, 42003):
                A = U[(U.arm == 'D_tf') & (U.seed == s)].set_index(['fold', 'window'])[m]; ix = A.index.intersection(B.index)
                d = pd.DataFrame(dict(a=A.loc[ix], b=B.loc[ix])).dropna(); g = d.groupby(level=0).mean(); fs.append(g)
                per.append(dict(baseline=b, metric=m, seed=s, chg=100 * (g.a - g.b).mean() / abs(g.b.mean()), lower=int(((g.a - g.b) < 0).sum())))
            f = pd.concat(fs).groupby(level=0).mean(); dd = f.a - f.b; t = stats.ttest_rel(f.a, f.b); ci = stats.t.interval(.95, len(dd) - 1, dd.mean(), stats.sem(dd))
            rows.append(dict(baseline=b, metric=m, n=len(f), tf=f.a.mean(), base=f.b.mean(), chg=100 * dd.mean() / abs(f.b.mean()), ci_lo=ci[0], ci_hi=ci[1], p=t.pvalue,
                             lower=int((dd < 0).sum()), shapiro=stats.shapiro(dd).pvalue, wilcoxon=stats.wilcoxon(dd).pvalue))
        for r, x in zip(rows, holm(np.array([r['p'] for r in rows]))): r['p_holm'] = x
        out += rows
    R = pd.DataFrame(out); R.to_csv(HERE / 'TIER3ONLY_TABLE.csv', index=False); pd.DataFrame(per).to_csv(HERE / 'TIER3ONLY_PER_SEED.csv', index=False)
    pd.set_option('display.width', 220); print(R.round(4).to_string(index=False)); print(pd.DataFrame(per).round(2).to_string(index=False))
