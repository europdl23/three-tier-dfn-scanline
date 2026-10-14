# -*- coding: utf-8 -*-
"""Is the EVAE (learned) closer to the held-out boxes than ADFNE and KDE (fitted)?

Every network is scored against its held-out test box (error scores: smaller = closer). Box value = median over the
box's 24 networks. EVAE against each baseline: paired two-sided Wilcoxon over the test boxes of a scheme, Holm over
the 2 baselines, alpha 0.05. Three-way verdict per score: the method with the lowest median error is the clear winner
in a scheme if it beats both others (one-sided Wilcoxon, Holm); a method wins the score if it is the clear winner in
at least 2 of the 3 schemes. Orientation clusters come from the fold's learned cluster model (fitting boxes only).

Undefined scores (n.d.): a score is undefined for a realisation that is empty, or that has fewer than 2 members of a
trace-direction cluster the held-out panel has. The panel value is the median over the realisations where the score is
defined; a panel with no defined value is n.d. and is left out of the tests. This is the rule used in the paper.
With --penalty, an undefined score instead counts as the worst value (1e9), as in the first version of the study;
the outputs then end in _penalty.

Writes tables/stats_network_scores.csv, stats_box_medians.csv, stats_decisions.csv, stats_counts.csv, stats_verdict.csv.
Usage:  python scripts/statistical_tests.py [--check] [--penalty]   (--check: compare the counts with tables/reference/)"""
import sys, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'study'))
import study as C            # noqa: E402
import setbg as SB           # noqa: E402
from topology import topology  # noqa: E402
import repo_paths as RP      # noqa: E402

GEN = ('EVAE', 'ADFNE', 'KDE')
KMAX = 4
NAMES_O = ['orient_w1_deg'] + ['oc%d_w1_deg' % j for j in range(1, KMAX + 1)]
NAMES_S = ['oc%d_%s' % (j, n) for j in range(1, KMAX + 1) for n in ('centre_err_deg', 'spread_err_deg', 'share_err')] + ['unclustered_share_err', 'cluster_share_tv']
NAMES_L = ['len_w1_m', 'len_med_err_m', 'len_p90_err_m'] + ['oc%d_len_w1_m' % j for j in range(1, KMAX + 1)]
NAMES_D = ['P20_relerr', 'P21_relerr', 'P22_relerr', 'nn_w1_m', 'cluster_vmr_err'] + ['oc%d_count_relerr' % j for j in range(1, KMAX + 1)]
NAMES_T = ['term_share_err', 'CL_err', 'node_tv']
GROUPS = [('Orientation distribution', NAMES_O), ('Orientation clusters', NAMES_S), ('Length', NAMES_L), ('Density', NAMES_D), ('Topology', NAMES_T)]
NAMES = [n for _, g in GROUPS for n in g]


def classes(L, model):
    L = np.asarray(L, float).reshape(-1, 4)
    if not len(L): return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0, int)
    c, ln, th = C.common.geom(L); return c, ln, th, SB.classify(th, model)


def topo_err(tg, tr):
    return dict(term_share_err=abs(tg['term_share'] - tr['term_share']), CL_err=abs(tg['CL'] - tr['CL']),
                node_tv=0.5 * (abs(tg['pI'] - tr['pI']) + abs(tg['pY'] - tr['pY']) + abs(tg['pX'] - tr['pX'])))


def score(Lg, Lr, ym, model):
    K = model['K']
    cg, lg, tg, kg = classes(Lg, model); cr, lr, tr, kr = classes(Lr, model)
    s = {n: np.nan for n in NAMES}
    if not len(lr): return s
    if not len(lg): return {n: np.inf for n in NAMES}
    sh = lambda k_: np.array([(k_ == j).mean() for j in range(K + 1)])
    pg, pr = sh(kg), sh(kr)
    s.update(orient_w1_deg=C.common.w1_circular_doubled(tg, tr), unclustered_share_err=abs(pg[K] - pr[K]), cluster_share_tv=0.5 * np.abs(pg - pr).sum(),
             len_w1_m=C.common.w1(lg, lr) * C.M, len_med_err_m=abs(np.median(lg) - np.median(lr)) * C.M,
             len_p90_err_m=abs(np.percentile(lg, 90) - np.percentile(lr, 90)) * C.M,
             P20_relerr=abs(len(lg) - len(lr)) / len(lr), P21_relerr=abs(lg.sum() - lr.sum()) / max(lr.sum(), 1e-12))
    xg = len(C.common.crossings_xy(np.asarray(Lg).reshape(-1, 4))) if len(lg) > 1 else 0
    xr = len(C.common.crossings_xy(np.asarray(Lr).reshape(-1, 4))) if len(lr) > 1 else 0
    s['P22_relerr'] = abs(xg - xr) / xr if xr else np.nan
    s['nn_w1_m'] = C.common.w1(C.common.nn_distances(cg), C.common.nn_distances(cr)) * C.M if len(cg) > 1 and len(cr) > 1 else np.nan
    vmr = lambda c_: (lambda h: h.var() / max(h.mean(), 1e-9))(np.histogram2d(c_[:, 0], c_[:, 1], bins=[4, max(1, int(round(ym * 4)))], range=[[0, 1], [0, max(1, int(round(ym * 4))) / 4]])[0])
    s['cluster_vmr_err'] = abs(vmr(cg) - vmr(cr))
    for j in range(min(K, KMAX)):
        g_, r_ = kg == j, kr == j; n = j + 1
        s['oc%d_share_err' % n] = abs(pg[j] - pr[j])
        s['oc%d_count_relerr' % n] = abs(g_.sum() - r_.sum()) / r_.sum() if r_.sum() else np.nan
        if r_.sum() >= 2:
            if g_.sum() >= 2:
                s['oc%d_w1_deg' % n] = C.common.w1_circular_doubled(tg[g_], tr[r_])
                s['oc%d_centre_err_deg' % n] = math.degrees(float(C.axd(C.axial_mean(tg[g_]), C.axial_mean(tr[r_]))))
                s['oc%d_spread_err_deg' % n] = abs(C.common.axial_csd_deg(tg[g_]) - C.common.axial_csd_deg(tr[r_]))
                s['oc%d_len_w1_m' % n] = C.common.w1(lg[g_], lr[r_]) * C.M
            else:
                for m_ in ('w1_deg', 'centre_err_deg', 'spread_err_deg', 'len_w1_m'): s['oc%d_%s' % (n, m_)] = np.inf
    s.update(topo_err(topology(Lg, ym), topology(Lr, ym)))
    return s


def score_fold(job):
    scheme, fi = job
    lib = C.load_library(); sp = C.split(scheme, fi, lib); model = C.cluster_model(scheme, fi); rows = []
    for b in sp['test']:
        real = sp['panels'][b]
        for m in GEN:
            for sd in C.SEEDS:
                for d in range(C.NDRAW):
                    L = C.load_net(C.net_path(scheme, m, fi, b, sd, d))
                    rows.append(dict(scheme=scheme, fold=fi, box=b, method=m, run_seed=sd, draw=d, n=len(L), **score(L, real['lines'], real['y_max'], model)))
    return rows


def paired(bm, a, others):
    res = {}
    for o in others:
        sub = bm[[a, o]].replace([np.inf], 1e9).dropna(); d = (sub[a] - sub[o]).values
        p = 1.0 if len(d) < 3 or np.allclose(d, 0) else float(wilcoxon(d, zero_method='zsplit').pvalue)
        res[o] = (p, float(np.median(d)) if len(d) else np.nan, len(sub))
    ps = sorted((res[o][0], o) for o in others); hol, run = {}, 0.0
    for i, (p, o) in enumerate(ps):
        run = max(run, min(1.0, p * (len(ps) - i))); hol[o] = run
    return {o: dict(n_boxes=res[o][2], median_diff=res[o][1], p_holm=hol[o], result='no difference' if hol[o] >= 0.05 else ('better' if res[o][1] < 0 else 'worse')) for o in others}


def three_way(bm, methods):
    sub = bm[list(methods)].replace([np.inf], 1e9).dropna()
    if len(sub) < 3: return None
    med = {m: float(np.median(sub[m])) for m in methods}; win = min(methods, key=lambda m: med[m]); ps = []
    for o in methods:
        if o == win: continue
        d = sub[win].values - sub[o].values
        ps.append((1.0 if np.allclose(d, 0) else float(wilcoxon(d, alternative='less', zero_method='zsplit').pvalue), o))
    ps.sort(); holm = [min(1.0, p * (len(ps) - i)) for i, (p, o) in enumerate(ps)]; holm = [max(holm[:i + 1]) for i in range(len(holm))]
    return dict(winner=win, clear=all(h < 0.05 for h in holm))


def main(check=False, penalty=False):
    T = RP.TABLES_DIR; T.mkdir(exist_ok=True); sfx = '_penalty' if penalty else ''
    lib = C.load_library(); jobs = [(s, f) for s in C.SCHEMES for f in range(len(C.folds(s, lib)))]
    with Pool(min(12, len(jobs))) as pool:
        res = pool.map(score_fold, jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    if not penalty:                                    # n.d.: undefined values are left out of the panel medians
        df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(T / ('stats_network_scores%s.csv' % sfx), index=False, na_rep='n.d.')
    BM = df.groupby(['scheme', 'fold', 'box', 'method'])[NAMES + ['n']].median().reset_index(); BM.to_csv(T / ('stats_box_medians%s.csv' % sfx), index=False, na_rep='n.d.')
    dd, ver = [], []
    for s in C.SCHEMES:
        sub = BM[BM.scheme == s]
        for grp, names in GROUPS:
            for n in names:
                piv = sub.pivot_table(index='box', columns='method', values=n)
                for o, r in paired(piv, 'EVAE', ('ADFNE', 'KDE')).items():
                    dd.append(dict(scheme=s, group=grp, score=n, against=o, **r))
    for grp, names in GROUPS:
        for n in names:
            r = {s: three_way(BM[BM.scheme == s].pivot_table(index='box', columns='method', values=n), GEN) for s in C.SCHEMES}
            cw = {m: sum(1 for s in C.SCHEMES if r[s] and r[s]['winner'] == m and r[s]['clear']) for m in GEN}
            ver.append(dict(group=grp, score=n, **{'%s_winner' % s: ('%s (%s)' % (r[s]['winner'], 'clear' if r[s]['clear'] else 'tie')) if r[s] else 'n/a' for s in C.SCHEMES},
                            verdict=next((m for m in cw if cw[m] >= 2), 'no method')))
    DD = pd.DataFrame(dd); VR = pd.DataFrame(ver)
    DD.to_csv(T / ('stats_decisions%s.csv' % sfx), index=False, na_rep='n.d.'); VR.to_csv(T / ('stats_verdict%s.csv' % sfx), index=False)
    rows = []
    for ag in ('ADFNE', 'KDE'):
        for grp in [g for g, _ in GROUPS] + ['All']:
            q = DD[DD.against == ag]; q = q if grp == 'All' else q[q.group == grp]; v = q.result.value_counts()
            rows.append(dict(against=ag, group=grp, better=int(v.get('better', 0)), no_difference=int(v.get('no difference', 0)), worse=int(v.get('worse', 0))))
    CNT = pd.DataFrame(rows); CNT.to_csv(T / ('stats_counts%s.csv' % sfx), index=False)
    pd.set_option('display.width', 200)
    print('EVAE against ADFNE and KDE (better / no difference / worse = EVAE closer / no significant difference / EVAE further)')
    print(CNT.to_string(index=False)); print('three-way verdict:', VR.verdict.value_counts().to_dict())
    if check:
        ref = pd.read_csv(RP.TABLES_DIR / 'reference' / ('bg_vs_normal_counts.csv' if penalty else 'stats_counts_nd.csv')).replace({'Sets and background': 'Orientation clusters'})
        same = ref[['against', 'group', 'better', 'no_difference', 'worse']].reset_index(drop=True).equals(CNT[['against', 'group', 'better', 'no_difference', 'worse']])
        rv = pd.read_csv(RP.TABLES_DIR / 'reference' / ('bg_vs_normal_verdict.csv' if penalty else 'stats_verdict_nd.csv'))
        same_v = rv.verdict.value_counts().to_dict() == VR.verdict.value_counts().to_dict()
        print('CHECK counts  ', 'identical' if same else 'DIFFERENT'); print('CHECK verdicts', 'identical' if same_v else 'DIFFERENT')
        if not (same and same_v):
            sys.exit(1)


if __name__ == '__main__':
    main(check='--check' in sys.argv, penalty='--penalty' in sys.argv)
