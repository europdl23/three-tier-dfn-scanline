# -*- coding: utf-8 -*-
"""Fracture-network topology (Sanderson & Nixon 2015 node types) for box networks in box units (1 = 20 m).
Trace ends within EDGE m of the box boundary are censored (the trace leaves the box) and are not counted.
  I node = a free end;  Y node = an end that stops on another trace (within TOL m of it);  X node = a crossing.
Returns counts, node proportions, the share of (uncensored) ends that stop on another trace, and connections per line."""
import numpy as np
import study as C

TOL, EDGE = 0.3, 0.1


def _pt_seg(P, A, B):
    d = B - A; l2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(2) / l2[None], 0, 1)
    return np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * d[None]), axis=2)


def topology(L, y_max, tol=TOL, edge=EDGE):
    L = np.asarray(L, float).reshape(-1, 4) * C.M
    n = len(L); W, H = C.M, y_max * C.M
    out = dict(n_I=0, n_Y=0, n_X=0)
    if n == 0:
        return dict(out, pI=np.nan, pY=np.nan, pX=np.nan, term_share=np.nan, CL=np.nan)
    E = np.vstack([L[:, :2], L[:, 2:]]); own = np.r_[np.arange(n), np.arange(n)]
    inside = (E[:, 0] > edge) & (E[:, 0] < W - edge) & (E[:, 1] > edge) & (E[:, 1] < H - edge)
    if n > 1:
        D = _pt_seg(E, L[:, :2], L[:, 2:]); D[np.arange(2 * n), own] = np.inf
        onto = D.min(1) <= tol
    else:
        onto = np.zeros(2 * n, bool)
    out['n_Y'] = int((inside & onto).sum()); out['n_I'] = int((inside & ~onto).sum())
    out['n_X'] = int(len(C.common.crossings_xy(L / C.M))) if n > 1 else 0
    tot = out['n_I'] + out['n_Y'] + out['n_X']
    nl = (out['n_I'] + out['n_Y']) / 2
    out.update(pI=out['n_I'] / tot if tot else np.nan, pY=out['n_Y'] / tot if tot else np.nan, pX=out['n_X'] / tot if tot else np.nan,
               term_share=out['n_Y'] / (out['n_I'] + out['n_Y']) if out['n_I'] + out['n_Y'] else np.nan,
               CL=2 * (out['n_Y'] + out['n_X']) / nl if nl else np.nan)
    return out
