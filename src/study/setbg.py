# -*- coding: utf-8 -*-
"""Learned joint-set model with BACKGROUND (PROTOCOL_SETS_BG.md): a mixture of K axial von Mises sets (on 2 theta) and
one uniform background component, fitted by EM; K in 1..4 chosen by BIC. Nothing is fixed to 2 sets.
A trace belongs to the component with the highest responsibility: set 1..K (sorted by weight, largest first) or
background."""
import math, json
import numpy as np
from scipy.special import i0e

TWO_PI = 2 * math.pi


def _vm(z, mu, k):
    return np.exp(k * (np.cos(z - mu) - 1.0)) / (TWO_PI * i0e(k))


def _kappa(R):
    R = min(max(float(R), 1e-6), 0.999)
    return R * (2 - R * R) / (1 - R * R)


KAPPA_MIN = 8.0     # geological definition of a set: a tight group (about +-20 deg in 2D); anything wider is background


def fit(theta, weights=None, K_range=(1, 2, 3, 4), restarts=10, iters=400, seed=20261001, kappa_min=KAPPA_MIN):
    z = np.mod(2.0 * np.asarray(theta, float), TWO_PI); n = len(z)
    wts = np.ones(n) if weights is None else np.asarray(weights, float) * n / np.sum(weights)
    rng = np.random.default_rng(seed); fits = []
    for K in K_range:
        best = None
        for _ in range(restarts):
            mu = rng.uniform(0, TWO_PI, K); kap = np.full(K, 4.0); w = np.r_[np.full(K, 0.8 / K), 0.2]; ll_old = -np.inf
            for _ in range(iters):
                dens = np.column_stack([w[j] * _vm(z, mu[j], kap[j]) for j in range(K)] + [np.full(n, w[K] / TWO_PI)])
                tot = dens.sum(1, keepdims=True) + 1e-300; resp = dens / tot; ll = float((wts * np.log(tot[:, 0])).sum())
                rw = resp * wts[:, None]; w = rw.sum(0) / rw.sum()
                for j in range(K):
                    C, S = float((rw[:, j] * np.cos(z)).sum()), float((rw[:, j] * np.sin(z)).sum())
                    mu[j] = math.atan2(S, C) % TWO_PI; kap[j] = min(max(_kappa(math.hypot(C, S) / max(rw[:, j].sum(), 1e-12)), kappa_min), 200.0)
                if abs(ll - ll_old) < 1e-8: break
                ll_old = ll
            if best is None or ll > best['ll']:
                best = dict(ll=ll, mu2=mu.copy(), kappa2=kap.copy(), w=w.copy())
        p = 3 * K                                      # K centres + K concentrations + K free weights (background takes the rest)
        best.update(K=K, bic=-2 * best['ll'] + p * math.log(n)); fits.append(best)
    m = min(fits, key=lambda f: f['bic'])
    o = np.argsort(-m['w'][:-1])                        # sets by weight, largest first
    model = dict(K=int(m['K']), mu2=[float(x) for x in m['mu2'][o]], kappa2=[float(x) for x in m['kappa2'][o]],
                 w_sets=[float(x) for x in m['w'][:-1][o]], w_bg=float(m['w'][-1]),
                 centres_deg=[float(math.degrees(x / 2) % 180) for x in m['mu2'][o]],
                 bic={int(f['K']): float(f['bic']) for f in fits}, n=int(n), kappa_min=float(kappa_min))
    return model


def resp(theta, model):
    """responsibilities: columns = set 1..K, background"""
    z = np.mod(2.0 * np.asarray(theta, float), TWO_PI)
    d = np.column_stack([w * _vm(z, mu, k) for w, mu, k in zip(model['w_sets'], model['mu2'], model['kappa2'])] + [np.full(len(z), model['w_bg'] / TWO_PI)])
    return d / (d.sum(1, keepdims=True) + 1e-300)


def classify(theta, model):
    """0..K-1 = set, K = background"""
    if len(np.atleast_1d(theta)) == 0:
        return np.zeros(0, int)
    return np.argmax(resp(theta, model), 1)


def bin_resp(model, nbins=36):
    grid = (np.arange(nbins) + 0.5) * math.pi / nbins
    return resp(grid, model)                             # nbins x (K+1)


def save(model, path):
    json.dump(model, open(path, 'w'), indent=1)


def load(path):
    return json.load(open(path))
