"""Plan D analysis module (isolated; analysis only, no production code).

Observation and family assignment use the original definitions executed from exp_common.py (split_panel, assign_family_ids),
exactly as run_revision.py::observe / assign. The legacy per-family evaluation replicates r2_frozen_evaluation_2026-10-20/
experiments.py::evaluate (cluster rows) so that saved R2/R3/R4 values can be reproduced exactly.

Primary Plan D endpoints (family-free, matching-free): energy distance between the predicted inferred-tier set (pooled over
generation seeds) and the hidden true set, for axial direction (doubled-angle chord), log length (m) and midpoint position (m).
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, spearmanr
import staging_lib as L

split_panel, assign = L.ENV['split_panel'], L.ENV['assign_family_ids']
W = 20.0


# ---------------------------------------------------------------- observation (original definitions)
def strips(layout, w=W, x0=None, ap=0.1):
    """L0='5m', L1='centre' (run_revision.strips); L2/L3: one 0.1 m strip every 40/60 m of face x, at the window centre of windows
    whose x0 is a multiple of the period (phase6_sparse_feasibility_and_pilot.py::layout_strips). Public geometry only."""
    if layout in ('5m', 'L0'): return L.strips('5m', w)
    if layout in ('centre', 'L1'): return L.strips('centre', w)
    per = {'L2': 40, 'L3': 60}[layout]
    return [((10 - ap / 2) / w, (10 + ap / 2) / w)] if int(round(x0)) % per == 0 else []


def observe(arr, st):
    clips, parents = [], []
    for i, row in enumerate(np.asarray(arr, np.float32)):
        c, _ = split_panel(row[None], st); clips.extend(c); parents.extend([i] * len(c))
    return np.asarray(clips, np.float32).reshape(-1, 4), np.asarray(parents, int)


def hidden_of(arr, st):
    _, par = observe(arr, st) if st else (None, np.zeros(0, int))
    return np.asarray(arr)[~np.isin(np.arange(len(arr)), np.unique(par))]


def ends(a):
    a = np.asarray(a, float).reshape(-1, 4); d = np.c_[np.cos(a[:, 3]), np.sin(a[:, 3])] * a[:, 2:3] / 2
    return a[:, :2] - d, a[:, :2] + d


# ---------------------------------------------------------------- legacy per-family evaluation (exact replication)
def legacy_cluster_rows(truth, final, inf, parents, fc):
    hidden = truth[~np.isin(np.arange(len(truth)), np.unique(parents))]; out = []
    for scope, g, a in [('full_network', truth, final), ('hidden_inferred_only', hidden, inf)]:
        gf = assign(g, fc) if len(g) else np.zeros(0, int); af = assign(a, fc) if len(a) else np.zeros(0, int)
        for k in range(4):
            gt = g[gf == k]; pr = a[af == k]; n = min(len(gt), len(pr))
            row = dict(scope=scope, cluster='C' + str(k + 1) if k < 3 else 'Unassigned', truth_count=len(gt), generated_count=len(pr), matched_count=n,
                       angle_MAE_deg=math.nan, length_MAE_m=math.nan, length_W1_m=math.nan, angle_W1_deg=math.nan)
            if n:
                ii, jj = linear_sum_assignment(cdist(gt[:, :2] * 20, pr[:, :2] * 20)); delta = abs((gt[ii, 3] - pr[jj, 3] + np.pi / 2) % np.pi - np.pi / 2)
                row.update(angle_MAE_deg=float(np.degrees(delta).mean()), length_MAE_m=float(abs(gt[ii, 2] - pr[jj, 2]).mean() * 20),
                           length_W1_m=float(wasserstein_distance(gt[:, 2] * 20, pr[:, 2] * 20)))
                centre = fc[k] if k < 3 else 0; ga = (gt[:, 3] - centre + np.pi / 2) % np.pi - np.pi / 2; aa = (pr[:, 3] - centre + np.pi / 2) % np.pi - np.pi / 2
                row['angle_W1_deg'] = float(np.degrees(wasserstein_distance(ga, aa)))
            out.append(row)
    return out


def legacy_network(truth, final, h):
    return dict(count_MAE=abs(len(final) - len(truth)), count_bias=len(final) - len(truth), P21_abs_error=abs(final[:, 2].sum() - truth[:, 2].sum()) / h)


# ---------------------------------------------------------------- primary: energy distance
def _axial(a): return np.c_[np.cos(2 * np.asarray(a, float)), np.sin(2 * np.asarray(a, float))]


def _pair_mean(X, Y):
    return float(cdist(X, Y).mean())


def _within_u(X):
    n = len(X)
    if n < 2: return math.nan
    d = cdist(X, X); return float(d.sum() / (n * (n - 1)))


def energy_distance(X, Y):
    """D = 2E|X-Y| - E|X-X'| - E|Y-Y'| with U-statistics inside sets; NaN when either set has < 2 members."""
    if len(X) < 2 or len(Y) < 2: return math.nan
    X = np.asarray(X, float).reshape(len(X), -1); Y = np.asarray(Y, float).reshape(len(Y), -1)
    return 2 * _pair_mean(X, Y) - _within_u(X) - _within_u(Y)


def property_features(a, prop, h=None):
    """a: rows (cx, cy, length, angle) in window units (width 20 m). Returns the feature matrix used for the property."""
    a = np.asarray(a, float).reshape(-1, 4)
    if prop == 'direction': return _axial(a[:, 3])                       # chord on the doubled-angle circle = 2|sin(dtheta)|
    if prop == 'log_length': return np.log(np.maximum(a[:, 2] * W, 1e-6))[:, None]
    if prop == 'position': return a[:, :2] * W                             # metres
    raise KeyError(prop)


def primary_endpoints(pred_sets, hidden):
    """pred_sets: list of inferred-tier arrays (one per generation seed) pooled into the predictive ensemble."""
    P = np.vstack([p for p in pred_sets if len(p)]) if any(len(p) for p in pred_sets) else np.zeros((0, 4))
    out = dict(n_pred=len(P), n_hidden=len(hidden))
    for prop in ['direction', 'log_length', 'position']:
        out['ED_' + prop] = energy_distance(property_features(P, prop), property_features(hidden, prop))
    return out


def joint_energy(pred, hidden, scales):
    f = lambda a: np.c_[np.asarray(a)[:, 0] * W / scales['x'], np.asarray(a)[:, 1] * W / scales['y'], _axial(np.asarray(a)[:, 3]) / scales['axial'],
                        np.log(np.maximum(np.asarray(a)[:, 2] * W, 1e-6)) / scales['loglen']]
    return energy_distance(f(pred), f(hidden)) if len(pred) and len(hidden) else math.nan


# ---------------------------------------------------------------- matched secondaries with shrinkage controls
def matched_with_controls(truth, pred, fc, controls, k_family=None):
    """Hungarian on midpoints (metres), as legacy. controls: dict family -> dict(median_length_m, mean_axial_rad, mean_xy_m)."""
    out = []
    gf = assign(truth, fc) if len(truth) else np.zeros(0, int); pf = assign(pred, fc) if len(pred) else np.zeros(0, int)
    fams = range(4) if k_family is None else [k_family]
    for k in fams:
        gt = truth[gf == k]; pr = pred[pf == k]
        if not min(len(gt), len(pr)): continue
        ii, jj = linear_sum_assignment(cdist(gt[:, :2] * W, pr[:, :2] * W)); gl, pl = gt[ii, 2] * W, pr[jj, 2] * W
        ga, pa = gt[ii, 3], pr[jj, 3]; c = controls[k]
        axd = lambda a, b: np.degrees(np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2))
        out.append(dict(family=k, n=len(ii), length_MAE_m=float(np.abs(gl - pl).mean()), length_MAE_control_m=float(np.abs(gl - c['median_length_m']).mean()),
                        angle_MAE_deg=float(axd(ga, pa).mean()), angle_MAE_control_deg=float(axd(ga, c['mean_axial_rad']).mean()),
                        position_err_m=float(np.hypot(*(gt[ii, :2] * W - pr[jj, :2] * W).T).mean()),
                        position_err_control_m=float(np.hypot(*(gt[ii, :2] * W - np.asarray(c['mean_xy_m'])).T).mean()),
                        length_pairs=np.c_[gl, pl]))
    return out


def shrinkage_controls(train_frag_arr, fc):
    """Fitting-only reference values per family (S0)."""
    a = np.asarray(train_frag_arr, float); k = assign(a, fc); out = {}
    for j in range(4):
        s = a[k == j] if (k == j).any() else a
        out[j] = dict(median_length_m=float(np.median(s[:, 2] * W)), mean_axial_rad=float((np.angle(np.exp(2j * s[:, 3]).mean()) / 2) % np.pi),
                      mean_xy_m=(s[:, :2].mean(0) * W).tolist())
    return out


def pair_correlation(pairs):
    p = np.vstack(pairs) if len(pairs) else np.zeros((0, 2))
    if len(p) < 3 or p[:, 0].std() == 0 or p[:, 1].std() == 0: return math.nan
    return float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])


# ---------------------------------------------------------------- information diagnostics
def skill(loss_model, loss_reference):
    """SS = 1 - L_model / L_reference (> 0 means the model beats the reference)."""
    return 1 - float(np.mean(loss_model)) / float(np.mean(loss_reference))


def family_logloss(probs, labels):
    probs = np.asarray(probs, float); labels = np.asarray(labels, int)
    if probs.ndim == 1: probs = np.repeat(probs[None], len(labels), 0)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def dirichlet_observed_predictor(obs_families, prior, a=4.0):
    cnt = np.bincount(obs_families, minlength=4); return (cnt + a * np.asarray(prior)) / (cnt.sum() + a)


def t5_window(truth, st, fc, prior, ym, a=4.0):
    """Model-free information row (T5 definition), for metric I-6."""
    obs, par = observe(truth, st); hid = truth[~np.isin(np.arange(len(truth)), np.unique(par))]
    if not len(hid) or not len(obs): return None
    ho = assign(hid, fc); oo = assign(obs, fc); post = dirichlet_observed_predictor(oo, prior, a); prior = np.asarray(prior)
    return dict(hidden=len(hid), observed=len(obs), logloss_prior=float(-np.log(prior[ho]).mean()), logloss_observed=float(-np.log(post[ho]).mean()),
                hidden_mean_y=float(hid[:, 1].mean() / ym), observed_mean_y=float(obs[:, 1].mean() / ym), hidden_mean_loglen=float(np.log(hid[:, 2]).mean()),
                observed_mean_loglen=float(np.log(obs[:, 2]).mean()), hidden_C3_share=float((ho == 2).mean()), observed_C3_share=float((oo == 2).mean()))


def ridge_skill(Xtr, ytr, Xte, yte, lam=1.0):
    """Fixed-penalty ridge on standardised features, skill vs the training mean (I-6)."""
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1; A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean())); pred = ytr.mean() + B @ w
    return skill((yte - pred) ** 2, (yte - ytr.mean()) ** 2)


def continuation_info(true_m, pred_m, control_m):
    true_m, pred_m, control_m = map(lambda v: np.asarray(v, float), (true_m, pred_m, control_m))
    return dict(n=len(true_m), MAE_model=float(np.abs(pred_m - true_m).mean()), MAE_control=float(np.abs(control_m - true_m).mean()),
                MAE_skill=skill(np.abs(pred_m - true_m), np.abs(control_m - true_m)), spearman=float(spearmanr(pred_m, true_m).correlation) if len(true_m) > 2 else math.nan)


def between_window_correlation(pred_summary, true_summary, folds=None, control_summary=None):
    """Fold-centred correlation across test windows; with control_summary, the partial correlation after regressing both on it."""
    p, t = np.asarray(pred_summary, float), np.asarray(true_summary, float); c = None if control_summary is None else np.asarray(control_summary, float)
    if folds is not None:
        folds = np.asarray(folds)
        for f in np.unique(folds):
            m = folds == f; p[m] -= p[m].mean(); t[m] -= t[m].mean()
            if c is not None: c[m] -= c[m].mean()
    if c is not None:
        def resid(v):
            if c.std() == 0: return v - v.mean()
            b = np.polyfit(c, v, 1); return v - np.polyval(b, c)
        p, t = resid(p), resid(t)
    return float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else math.nan


def sensitivity_ratio(pred_a, pred_b, true_a, true_b):
    dt = np.abs(np.asarray(true_a, float) - np.asarray(true_b, float))
    return float(np.abs(np.asarray(pred_a, float) - np.asarray(pred_b, float)).mean() / dt.mean()) if dt.mean() > 0 else math.nan


def neural_share(network_part, final_value):
    v = np.var(np.asarray(final_value, float)); return float(np.var(np.asarray(network_part, float)) / v) if v > 0 else math.nan


# ---------------------------------------------------------------- uncertainty
def window_bootstrap(per_window, strata=None, reps=10000, seed=20261021):
    """Mean and 95% percentile CI. With strata (folds): windows are resampled within each fold and folds are weighted EQUALLY
    (fold-equal pooling, the Plan D primary aggregation); the point estimate is the mean of fold means."""
    x = np.asarray(per_window, float); ok = ~np.isnan(x); x = x[ok]
    if len(x) < 2: return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    if strata is None:
        b = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    else:
        s = np.asarray(strata)[ok]; groups = [x[s == g] for g in np.unique(s)]; w = np.full(len(groups), 1 / len(groups))
        b = sum(wg * g[rng.integers(0, len(g), (reps, len(g)))].mean(1) for wg, g in zip(w, groups))
        return float(np.mean([g.mean() for g in groups])), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fold_t_interval(fold_means):
    from scipy.stats import t as tdist
    m = np.asarray(fold_means, float); n = len(m)
    if n < 2: return math.nan, math.nan, math.nan
    se = m.std(ddof=1) / math.sqrt(n); q = tdist.ppf(.975, n - 1); return float(m.mean()), float(m.mean() - q * se), float(m.mean() + q * se)
