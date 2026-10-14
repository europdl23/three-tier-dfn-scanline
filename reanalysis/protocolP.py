# -*- coding: utf-8 -*-
r"""protocolP.py: Protocol P analysis library for the JRMGE-D-26-02141 major revision.

Implements the statistics registered as Protocol P (00_PROJECT_CONTROL/REQUIRED_ANALYSES.md Sec. 0, source
09_INDEPENDENT_AUDITS/STATISTICAL_AUDIT.md Sec. 4, author decision AD-10) for paired comparisons of reconstruction
methods ("arms") on non-overlapping 20 m sampling windows.

Dependencies: numpy, scipy (rankdata only) and pandas. No module of the revision study and no module of the Paper 1
release is imported. Paper 1's score names are read from scripts/statistical_tests.py by parsing its source; only the
assignment statements that define KMAX, NAMES_O, NAMES_S, NAMES_L, NAMES_D, NAMES_T, GROUPS and NAMES are evaluated. If
that file cannot be read, the score columns of the row file are used and the fallback is recorded.

Conventions used by every function
----------------------------------
* Arm A is the method under test, arm B the comparator (for example STAT_F, the reference filler of AD-04).
  Paired window difference d_w = value_A(w) - value_B(w), in the raw units of the score column.
* All Protocol P scores are error scores: LOWER IS BETTER. A negative d_w, Hodges-Lehmann estimate or CI limit means
  that arm A is closer to the mapped traces. HIGHER_IS_BETTER is empty; a score placed in it would be sign-reversed
  before labelling (labels are always computed on the harm scale, where positive = worse for arm A).
* Raw units in the row files (Paper 1 scripts/statistical_tests.score plus final_eval.p21map_err):
    len_w1_m       m      Wasserstein-1 distance of in-window trace-piece lengths
    P21_relerr     -      |P21_gen - P21_real| / P21_real (fraction; results tables show percentage points = x 100)
    orient_w1_deg  deg    circular Wasserstein-1 distance of trace directions theta (axial, 0-180 deg, measured
                          anticlockwise from the window x axis along the face), computed on 2 theta and reported in
                          degrees of theta
    node_tv        -      total-variation distance of the I/Y/X node proportions (0-1)
    map_rmse       1/m    RMSE over 4 m cells of the P21 map (m/m2)
    CL_err         -      |C_L,gen - C_L,real|, C_L = 2 (N_Y + N_X) / N_L connections per line
* Practical margins (AD-10), raw units: len_w1_m 0.2 m; P21_relerr 0.02 (2 percentage points); orient_w1_deg 1 deg;
  node_tv 0.02; map_rmse 0.02 1/m; CL_err 0.05 (key secondary).
* n.d. values: Paper 1 marks an undefined score as infinite (written 'n.d.' in its tables); the study rows store it as
  NaN (final_eval.py line 101). load_rows converts 'n.d.' strings and +-inf to NaN and counts them. A window value is
  the median (or mean) over the realisations where the score is defined; a window with no defined realisation has no
  value. In a paired comparison a window without a value in either arm is dropped for that endpoint, and the drops
  are counted by reason.
* Windows are ordered by window identifier (lexicographic) before any resampling, so a fixed seed gives the same
  bootstrap intervals whatever the row order of the input files. The random-number call pattern of the bootstrap
  functions equals that of the phase-1 audit scripts (stats_common.py), so the audit's intervals are reproduced.
* compare() sets paired differences with |d| <= 1e-12 to exactly 0 before any statistic (floating-point noise of
  identical score values, e.g. W1 distances of 2e-14); paired_differences() returns raw differences unless asked.

Decision labels (REQUIRED_ANALYSES.md Sec. 0; evaluated in the listed order, the first that applies is reported)
------------------------------------------------------------------------------------------------------------
  Improvement             adjusted p < 0.05, 95% CI excludes 0 on the favourable side, estimate <= -margin
  Practically equivalent  90% CI inside (-margin, +margin)
  Not practically worse   upper limit of the two-sided 95% CI below +margin
  Worse                   adjusted p < 0.05 and 95% CI excludes 0 on the unfavourable side
  Inconclusive            otherwise
The boolean criteria are also returned separately (improvement, equivalent, noninferior, worse, practically_worse),
so overlaps are visible: a statistically significant difference that lies inside the margin is labelled Practically
equivalent or Not practically worse and flagged in the 'qualifier' column. practically_worse = worse criterion and
estimate >= margin (STATISTICAL_AUDIT.md Sec. 4.7). If the exact Walsh interval cannot reach the stated level with the
available number of windows, the claims that need that interval are not made and the qualifier says so.

Public API (see each docstring)
-------------------------------
score_list, score_groups, group_of, paper1_score_names
load_rows, concat_rows
window_values
paired_differences (returns Paired)
smallest_attainable_p, wilcoxon_exact, signflip_test
walsh_averages, hodges_lehmann, hl_ci
bootstrap_ci, cluster_bootstrap_ci, two_stage_bootstrap_ci, mc_se_of_estimate
holm, bh
decision_flags
compare, method_level_summary
format_value, write_results
file_sha256, environment, write_run_record
"""
from __future__ import annotations

import ast
import hashlib
import math
import platform
import sys
import warnings
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__version__ = '1.0.0 (2026-10-09)'

# ----------------------------------------------------------------------------------------------- paths (read only)
P1_STATS = Path(r'<local path>')
STUDY_RESULTS = Path(r'<local path>')
PROJECT = Path(r'<local path>'
               r'\JRMGE_DFN_Major_Revision')

# ----------------------------------------------------------------------------------------------- Protocol P constants
PRIMARY = ('len_w1_m', 'P21_relerr', 'orient_w1_deg', 'node_tv', 'map_rmse')
KEY_SECONDARY = ('CL_err',)
KEY_ENDPOINTS = PRIMARY + KEY_SECONDARY
MARGINS = {'len_w1_m': 0.2, 'P21_relerr': 0.02, 'orient_w1_deg': 1.0, 'node_tv': 0.02, 'map_rmse': 0.02,
           'CL_err': 0.05}
DISPLAY_SCALE = {'P21_relerr': 100.0}
UNIT_RAW = {'len_w1_m': 'm', 'P21_relerr': 'fraction', 'orient_w1_deg': 'deg', 'node_tv': 'TV (0-1)',
            'map_rmse': '1/m', 'CL_err': 'connections per line'}
UNIT_DISPLAY = dict(UNIT_RAW, P21_relerr='percentage points')
# natural variation (S1 REFERENCE, median over 20 windows of the real-vs-real window medians), raw units;
# STATISTICAL_AUDIT.md Sec. 4.3 / stats_natural_variation_key_scores.csv. For reporting only.
NATURAL_VARIATION = {'len_w1_m': 1.770339, 'P21_relerr': 0.148015, 'orient_w1_deg': 13.317714, 'node_tv': 0.113923,
                     'map_rmse': 0.219954, 'CL_err': 0.441017}
HIGHER_IS_BETTER = frozenset()          # every Protocol P score is an error score
SCHEME_ROLE = {'S2': 'primary', 'S3': 'secondary', 'S1': 'secondary'}
LAYOUT_ROLE = {'5m': 'primary', 'centre': 'secondary'}

IMPROVEMENT = 'Improvement'
EQUIVALENT = 'Practically equivalent'
NONINFERIOR = 'Not practically worse'
WORSE = 'Worse'
INCONCLUSIVE = 'Inconclusive'
NO_LABEL = 'No label (no margin; exploratory)'
LABELS = (IMPROVEMENT, EQUIVALENT, NONINFERIOR, WORSE, INCONCLUSIVE)

ALPHA = 0.05
SEED = 20261008            # analysis seed; equals the phase-1 audit bootstrap seed (stats_common.py)
B_BOOT = 10_000            # percentile bootstrap over windows
B_CLUSTER = 5_000          # cluster bootstrap over bench faces or pieces
B_MC_SE = 2_000            # bootstrap over realisations for the Monte Carlo SE of a window value
N_MC = 1_000_000           # Monte Carlo sign flips (only when exact counting is not used)
EXACT_ENUM_MAX = 20        # explicit enumeration of all 2^m sign patterns up to m = 20
ZERO_TOL = 1e-12           # |d| <= ZERO_TOL is a zero difference
TIE_DECIMALS = 12          # |d| rounded to 12 decimals before ranking (float-noise ties are ties)

KEY_COLUMNS = ('scheme', 'box', 'cfg', 'arm', 'r')
NON_SCORE_COLUMNS = frozenset({'scheme', 'fold', 'box', 'cfg', 'arm', 'r', 'n', 'gt_n', 'sec'})
WINDOW_KEYS = ('scheme', 'layout', 'variant', 'arm', 'window', 'piece')

_NAME_TARGETS = frozenset({'KMAX', 'NAMES_O', 'NAMES_S', 'NAMES_L', 'NAMES_D', 'NAMES_T', 'GROUPS', 'NAMES'})
_GROUP_RENAME = {'Orientation distribution': 'Trace-direction distribution',
                 'Orientation clusters': 'Trace-direction clusters'}


# =============================================================================================== score names
@lru_cache(maxsize=4)
def paper1_score_names(path: str = str(P1_STATS)) -> dict:
    """Read Paper 1's score names from scripts/statistical_tests.py without importing it.

    Only the module-level assignments to KMAX, NAMES_O, NAMES_S, NAMES_L, NAMES_D, NAMES_T, GROUPS and NAMES are
    evaluated. Returns dict(names=tuple of 38 names, groups=tuple of (group, names) with 'Orientation' renamed to
    'Trace-direction', path, sha256 of the file)."""
    p = Path(path)
    raw = p.read_bytes()
    tree = ast.parse(raw.decode('utf-8'))
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _NAME_TARGETS
                                                for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), ns)  # noqa: S102
    names = tuple(ns['NAMES'])
    groups = tuple((_GROUP_RENAME.get(g, g), tuple(n)) for g, n in ns['GROUPS'])
    return dict(names=names, groups=groups, path=str(p), sha256=hashlib.sha256(raw).hexdigest())


def score_list(columns=None, path=P1_STATS):
    """The 39 Protocol P scores: Paper 1's 38 NAMES (statistical_tests.py) followed by 'map_rmse'.

    Returns (list of names, description of the source). If statistical_tests.py cannot be read and `columns` is
    given, the columns that are not key or bookkeeping columns are used instead (fallback, recorded in the source)."""
    try:
        info = paper1_score_names(str(path))
        return list(info['names']) + ['map_rmse'], (f"Paper 1 statistical_tests.py NAMES (sha256 {info['sha256']}) "
                                                    f"+ map_rmse")
    except Exception as exc:  # pragma: no cover - fallback path
        if columns is None:
            raise
        cols = [c for c in columns if c not in NON_SCORE_COLUMNS]
        return cols, f'CSV columns (statistical_tests.py not readable: {exc!r})'


def score_groups(path=P1_STATS):
    """Paper 1 score groups (renamed to trace-direction wording) plus ('Spatial placement', ('map_rmse',))."""
    try:
        groups = list(paper1_score_names(str(path))['groups'])
    except Exception:  # pragma: no cover
        groups = []
    return groups + [('Spatial placement', ('map_rmse',))]


def group_of(endpoint: str) -> str:
    """Score group of an endpoint ('' if unknown)."""
    for g, names in score_groups():
        if endpoint in names:
            return g
    return ''


# =============================================================================================== loading
def file_sha256(path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(path, *, arms=None, layouts=None, boxes=None, rename_arms=None, scheme_map=None, piece_map=None,
              scores=None, hash_file: bool = True) -> pd.DataFrame:
    """Read one per-realisation row file (study *_rows.csv or a Paper 1 score table).

    Parameters
    ----------
    path : CSV with at least the key columns scheme, box, cfg, arm, r and the score columns.
    arms, layouts, boxes : optional filters (layout = part of cfg before '|'; boxes = raw box labels).
    rename_arms : dict old arm label -> new label (applied after the arm filter), e.g. {'EVAE_F': 'EVAE_F_noR5'}.
    scheme_map : dict raw scheme -> analysis scheme, e.g. {'EAST': 'E03+WEST3', 'EASTW3': 'E03+WEST3'}.
    piece_map : dict raw scheme -> piece name, e.g. {'EAST': 'E03', 'EASTW3': 'WEST3', 'EASTC02': 'C02'}.
        Rows whose scheme is not in piece_map get piece = bench face = box prefix before '_' (W00-W06).
    scores : score columns; default score_list() (39 scores).

    Returns
    -------
    DataFrame with the original columns plus scheme_raw, layout, variant, piece, window (unique within an analysis
    scheme: the box label, or '<piece>:<box>' when the piece comes from piece_map) and _order (row order).
    Score columns are float; 'n.d.' strings and +-inf are NaN. df.attrs holds sources (path, sha256, rows),
    scores, score_source and per-score counts nd_strings, inf_values, nan_after (after conversion).
    A duplicated (scheme, window, layout, variant, arm, r) raises ValueError.
    `path` may also be a readable buffer (then no file digest is recorded)."""
    is_file = isinstance(path, (str, Path))
    name = Path(path).name if is_file else '<buffer>'
    df = pd.read_csv(path)
    missing = [k for k in KEY_COLUMNS if k not in df.columns]
    if missing:
        raise ValueError(f'{name}: key columns missing {missing}')
    if scores is None:
        scores, source = score_list(df.columns)
    else:
        scores, source = list(scores), 'given by caller'
    absent = [s for s in scores if s not in df.columns]
    if absent:
        raise ValueError(f'{name}: score columns missing {absent}')
    for k in ('scheme', 'box', 'cfg', 'arm'):
        df[k] = df[k].astype(str)
    if arms is not None:
        df = df[df['arm'].isin(list(arms))]
    if boxes is not None:
        df = df[df['box'].isin(list(boxes))]
    df = df.copy()
    if rename_arms:
        df['arm'] = df['arm'].map(lambda a: rename_arms.get(a, a))
    parts = df['cfg'].str.split('|', n=1)
    df['layout'] = parts.str[0]
    df['variant'] = parts.str[1].fillna('') if parts.map(len).max() > 1 else ''
    if layouts is not None:
        df = df[df['layout'].isin(list(layouts))].copy()
    df['scheme_raw'] = df['scheme']
    if scheme_map:
        df['scheme'] = df['scheme_raw'].map(lambda s: scheme_map.get(s, s))
    bench = df['box'].str.split('_').str[0]
    if piece_map:
        mapped = df['scheme_raw'].map(lambda s: piece_map.get(s))
        df['piece'] = mapped.where(mapped.notna(), bench)
        df['window'] = np.where(mapped.notna(), mapped.fillna('').astype(str) + ':' + df['box'], df['box'])
    else:
        df['piece'] = bench
        df['window'] = df['box']
    df['r'] = df['r'].astype(int)
    nd, inf, nan = {}, {}, {}
    for s in scores:
        col = df[s]
        if col.dtype == object:
            txt = col.astype(str).str.strip().str.lower()
            is_nd = txt.isin(['n.d.', 'n.d', 'nd'])
            nd[s] = int(is_nd.sum())
            col = pd.to_numeric(col.mask(is_nd), errors='raise')
        else:
            nd[s] = 0
        v = col.to_numpy(dtype=float, copy=True)
        isinf = np.isinf(v)
        inf[s] = int(isinf.sum())
        v[isinf] = np.nan
        nan[s] = int(np.isnan(v).sum())
        df[s] = v
    df['_order'] = np.arange(len(df))
    dup = df.duplicated(['scheme', 'window', 'layout', 'variant', 'arm', 'r'], keep=False)
    if dup.any():
        raise ValueError(f'{name}: {int(dup.sum())} duplicated (scheme, window, layout, variant, arm, r) rows')
    df.attrs = dict(sources=[dict(path=str(path) if is_file else name,
                                  sha256=file_sha256(path) if (hash_file and is_file) else '', rows=int(len(df)))],
                    scores=list(scores), score_source=source, nd_strings=nd, inf_values=inf, nan_after=nan)
    return df


def concat_rows(frames) -> pd.DataFrame:
    """Concatenate frames from load_rows (keeps row order), check duplicates across files and merge the records."""
    frames = list(frames)
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(['scheme', 'window', 'layout', 'variant', 'arm', 'r'], keep=False)
    if dup.any():
        raise ValueError(f'{int(dup.sum())} duplicated (scheme, window, layout, variant, arm, r) rows across files')
    df['_order'] = np.arange(len(df))
    scores = frames[0].attrs.get('scores', [])
    att = dict(sources=[s for f in frames for s in f.attrs.get('sources', [])], scores=list(scores),
               score_source=frames[0].attrs.get('score_source', ''))
    for k in ('nd_strings', 'inf_values', 'nan_after'):
        att[k] = {s: int(sum(f.attrs.get(k, {}).get(s, 0) for f in frames)) for s in scores}
    df.attrs = att
    return df


# =============================================================================================== window values
def _nanstat(V: np.ndarray, axis: int, stat: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        has_nan = bool(np.isnan(V).any())
        if stat == 'median':
            return np.nanmedian(V, axis=axis) if has_nan else np.median(V, axis=axis)
        if stat == 'mean':
            return np.nanmean(V, axis=axis) if has_nan else np.mean(V, axis=axis)
    raise ValueError(f'unknown window statistic {stat!r}')


def _key_seed(seed: int, key) -> np.random.Generator:
    """Generator seeded by (seed, CRC-32 of the key text): independent of grouping and call order."""
    crc = zlib.crc32('|'.join(str(k) for k in key).encode('utf-8'))
    return np.random.default_rng([int(seed), int(crc)])


def window_values(rows: pd.DataFrame, endpoints=None, *, stat: str = 'median', mc_se_endpoints=KEY_ENDPOINTS,
                  B_mc: int = B_MC_SE, seed: int = SEED) -> pd.DataFrame:
    """Window value of every (scheme, layout, variant, arm, window) and endpoint.

    value = median over the realisations (stat='median', Protocol P) or mean (stat='mean', sensitivity), NaN skipped
    (n.d. rule); NaN if no realisation is defined. n_real = realisations in the window, n_defined = realisations with
    a defined score. mc_se = Monte Carlo standard error of the window value: standard deviation (ddof 1) of the
    statistic over B_mc bootstrap resamples of the window's realisations (sorted by r), with a generator seeded by
    (seed, CRC-32 of the window key); computed for mc_se_endpoints only (NaN elsewhere or if B_mc = 0).
    Units are those of the score columns. Returns a long table with columns
    scheme, layout, variant, arm, window, piece, endpoint, value, n_real, n_defined, mc_se, stat."""
    if endpoints is None:
        endpoints = rows.attrs.get('scores') or score_list(rows.columns)[0]
    endpoints = [e for e in endpoints if e in rows.columns]
    se_cols = [j for j, e in enumerate(endpoints) if mc_se_endpoints and e in mc_se_endpoints]
    recs = []
    srt = rows.sort_values(list(WINDOW_KEYS) + ['r'], kind='mergesort')
    for key, g in srt.groupby(list(WINDOW_KEYS), sort=True):
        V = g[endpoints].to_numpy(dtype=float)
        n_r = V.shape[0]
        n_def = (~np.isnan(V)).sum(axis=0)
        val = _nanstat(V, 0, stat)
        se = np.full(len(endpoints), np.nan)
        if B_mc and se_cols and n_r > 1:
            rng = _key_seed(seed, key)
            idx = rng.integers(0, n_r, size=(B_mc, n_r))
            boot = _nanstat(V[:, se_cols][idx], 1, stat)            # (B_mc, k)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                se[se_cols] = np.nanstd(boot, axis=0, ddof=1)
        for j, e in enumerate(endpoints):
            recs.append((*key, e, float(val[j]), int(n_r), int(n_def[j]), float(se[j])))
    out = pd.DataFrame(recs, columns=list(WINDOW_KEYS) + ['endpoint', 'value', 'n_real', 'n_defined', 'mc_se'])
    out['stat'] = stat
    return out


# =============================================================================================== paired differences
@dataclass
class Paired:
    """Paired window differences d = A - B for one endpoint (raw units), windows sorted by identifier."""
    endpoint: str
    arm_a: str
    arm_b: str
    scheme: str
    layout: str
    variant: str
    d: np.ndarray
    windows: list
    pieces: np.ndarray
    value_a: np.ndarray
    value_b: np.ndarray
    n_windows_a: int = 0
    n_windows_b: int = 0
    n_common: int = 0
    n_drop_nd_a: int = 0          # common window, no value in A only
    n_drop_nd_b: int = 0          # common window, no value in B only
    n_drop_nd_both: int = 0       # common window, no value in either
    only_a: list = field(default_factory=list)
    only_b: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.d))

    @property
    def n_dropped(self) -> int:
        return self.n_drop_nd_a + self.n_drop_nd_b + self.n_drop_nd_both


def paired_differences(wv: pd.DataFrame, arm_a: str, arm_b: str, endpoint: str, scheme: str, layout: str,
                       variant: str = '', zero_tol: float | None = None) -> Paired:
    """Paired window differences between two arms on their common windows for one endpoint.

    A window is dropped for this endpoint (and counted) when its value is NaN (no defined realisation, n.d.) in arm A,
    in arm B or in both. Windows present for only one arm are listed in only_a / only_b and not used.
    zero_tol : if given, differences with |d| <= zero_tol are set to exactly 0 (floating-point noise of identical
    score values, for example W1 distances of 2e-14 between identical sets); compare() passes ZERO_TOL. Default None
    returns the raw differences (as used by the phase-1 audit)."""
    sel = ((wv['scheme'] == scheme) & (wv['layout'] == layout) & (wv['variant'] == variant)
           & (wv['endpoint'] == endpoint))
    A = wv[sel & (wv['arm'] == arm_a)].set_index('window')
    B = wv[sel & (wv['arm'] == arm_b)].set_index('window')
    if A.index.has_duplicates or B.index.has_duplicates:
        raise ValueError('window identifiers are not unique within the selection (mixed statistics or pieces?)')
    wa, wb = set(A.index), set(B.index)
    common = sorted(wa & wb)
    xa = A.loc[common, 'value'].to_numpy(dtype=float) if common else np.zeros(0)
    xb = B.loc[common, 'value'].to_numpy(dtype=float) if common else np.zeros(0)
    pc = A.loc[common, 'piece'].to_numpy() if common else np.zeros(0, dtype=object)
    na, nb = np.isnan(xa), np.isnan(xb)
    keep = ~(na | nb)
    d = xa[keep] - xb[keep]
    if zero_tol is not None:
        d = np.where(np.abs(d) <= zero_tol, 0.0, d)
    return Paired(endpoint=endpoint, arm_a=arm_a, arm_b=arm_b, scheme=scheme, layout=layout, variant=variant,
                  d=d, windows=[w for w, k in zip(common, keep) if k], pieces=pc[keep],
                  value_a=xa[keep], value_b=xb[keep], n_windows_a=len(wa), n_windows_b=len(wb), n_common=len(common),
                  n_drop_nd_a=int((na & ~nb).sum()), n_drop_nd_b=int((~na & nb).sum()),
                  n_drop_nd_both=int((na & nb).sum()), only_a=sorted(wa - wb), only_b=sorted(wb - wa))


# =============================================================================================== exact tests
def smallest_attainable_p(n: int, alternative: str = 'two-sided') -> float:
    """Smallest p-value an exact sign-flip or signed-rank test can return with n non-zero paired differences:
    2 / 2^n two-sided (6 windows 0.03125, 7 windows 0.015625, 8 windows 0.0078125), 1 / 2^n one-sided; 1 if n <= 0."""
    n = int(n)
    if n <= 0:
        return 1.0
    p = 2.0 ** (-n)
    return min(1.0, 2.0 * p) if alternative == 'two-sided' else p


def _clean(d) -> np.ndarray:
    x = np.asarray(d, dtype=float).ravel()
    return x[~np.isnan(x)]


def _signed_ranks(x: np.ndarray, zero_method: str, zero_tol: float, tie_decimals: int):
    """Doubled integer ranks (2 x midrank) of the non-zero |x| and their signs; zeros: |x| <= zero_tol.

    Pratt: |x| ranked with the zeros included (zeros take the lowest ranks), then the zeros are dropped.
    Wilcox: zeros dropped before ranking."""
    a = np.abs(x)
    zero = a <= zero_tol
    key = np.round(np.where(zero, 0.0, a), tie_decimals)
    if zero_method == 'pratt':
        r = rankdata(key, method='average')[~zero]
    elif zero_method == 'wilcox':
        r = rankdata(key[~zero], method='average')
    else:
        raise ValueError("zero_method must be 'pratt' or 'wilcox'")
    v = np.rint(2.0 * np.asarray(r, float)).astype(np.int64)
    if np.any(np.abs(2.0 * r - v) > 1e-9):
        raise RuntimeError('midranks are not multiples of 0.5')
    return v, np.sign(x[~zero]), int(zero.sum())


def _enumerate_pattern_sums(v: np.ndarray) -> np.ndarray:
    """Explicit list of sum(v_i B_i) over all 2^m patterns B in {0, 1}^m (one entry per pattern), built by doubling:
    after element i the list holds the 2^i partial sums without and with v_i."""
    sums = np.zeros(1, dtype=v.dtype)
    for vi in v:
        sums = np.concatenate([sums, sums + vi])
    return sums


def _counts_enumeration(v: np.ndarray) -> np.ndarray:
    """Null counts of T = sum(v_i B_i), B_i in {0, 1} (B_i = 1: difference i positive), by listing the statistic of
    every one of the 2^m sign patterns (m <= 24) and tabulating it."""
    v = np.asarray(v, dtype=np.int64)
    m, S = len(v), int(v.sum())
    if m > 24:
        raise ValueError('explicit enumeration is limited to m <= 24')
    return np.bincount(_enumerate_pattern_sums(v), minlength=S + 1).astype(np.int64)


def _counts_generating(v: np.ndarray) -> np.ndarray:
    """Same null counts by polynomial multiplication prod_i (1 + z^v_i): every one of the 2^m patterns is counted
    exactly (integer arithmetic; int64 up to m = 62, Python integers above)."""
    v = [int(x) for x in v]
    m, S = len(v), int(sum(v))
    counts = np.zeros(S + 1, dtype=np.int64 if m <= 62 else object)
    counts[0] = 1
    for vi in v:
        nxt = counts.copy()
        nxt[vi:] += counts[:S + 1 - vi]
        counts = nxt
    return counts


def _p_from_counts(counts: np.ndarray, t: int, alternative: str) -> float:
    total = int(sum(int(c) for c in counts)) if counts.dtype == object else int(counts.sum())
    S = len(counts) - 1
    k = np.arange(S + 1)
    if alternative == 'two-sided':
        mask = np.abs(2 * k - S) >= abs(2 * t - S)
    elif alternative == 'less':
        mask = k <= t
    elif alternative == 'greater':
        mask = k >= t
    else:
        raise ValueError(alternative)
    return int(counts[mask].sum()) / total


def _p_monte_carlo(v: np.ndarray, t: int, alternative: str, n_mc: int, seed: int, chunk: int = 50_000) -> float:
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=np.int64)
    S = int(v.sum())
    hit = done = 0
    while done < n_mc:
        b = min(chunk, n_mc - done)
        T = rng.integers(0, 2, size=(b, len(v)), dtype=np.int64) @ v
        if alternative == 'two-sided':
            hit += int((np.abs(2 * T - S) >= abs(2 * t - S)).sum())
        elif alternative == 'less':
            hit += int((T <= t).sum())
        else:
            hit += int((T >= t).sum())
        done += b
    return (hit + 1) / (n_mc + 1)


def wilcoxon_exact(d, *, mu: float = 0.0, alternative: str = 'two-sided', zero_method: str = 'pratt',
                   method: str = 'auto', n_mc: int = N_MC, seed: int = SEED, zero_tol: float = ZERO_TOL,
                   tie_decimals: int = TIE_DECIMALS) -> dict:
    """Exact Wilcoxon signed-rank test of H0: the paired differences are symmetric about mu.

    d : paired differences (NaN removed); the test uses x = d - mu.
    alternative : 'two-sided' (Protocol P), 'less' (x tends to be negative) or 'greater'.
    zero_method : 'pratt' (Protocol P; zeros ranked, then dropped) or 'wilcox'.
    method : 'auto' = explicit enumeration of all 2^m sign patterns of the m non-zero differences if m <= 20,
        otherwise exact counting of the same 2^m patterns by a generating function (no approximation);
        'enumeration', 'generating' force one of them; 'mc' = Monte Carlo with n_mc random sign flips and the given
        seed, p = (1 + hits) / (1 + n_mc).
    Ties: |x| is rounded to tie_decimals decimals before ranking; tied values get midranks and the exact null
    distribution is conditional on these midranks (doubled ranks are integers, so counting is exact).
    Two-sided p = P(|T+ - E| >= |t+ - E|) under the sign-flip null, E = (sum of non-zero ranks) / 2.

    Returns dict(n, n_nonzero, n_zero, ties, statistic = T+ (sum of ranks of positive x, rank units), p,
    alternative, mu, method (text), min_p_n = smallest attainable p for n differences, min_p_data = the same for
    the m non-zero differences)."""
    x = _clean(d) - mu
    n = len(x)
    out = dict(n=n, n_nonzero=0, n_zero=0, ties=False, statistic=np.nan, p=np.nan, alternative=alternative, mu=mu,
               zero_method=zero_method, method='no data', min_p_n=smallest_attainable_p(n, alternative),
               min_p_data=np.nan)
    if n == 0:
        return out
    v, s, nz = _signed_ranks(x, zero_method, zero_tol, tie_decimals)
    m = len(v)
    out.update(n_zero=nz, n_nonzero=m, ties=bool(len(np.unique(v)) < m), min_p_data=smallest_attainable_p(m,
                                                                                                        alternative))
    if m == 0:
        out.update(statistic=0.0, p=1.0, method='all differences zero: p = 1')
        return out
    t = int(v[s > 0].sum())
    out['statistic'] = t / 2.0
    if method == 'mc':
        out.update(p=_p_monte_carlo(v, t, alternative, n_mc, seed),
                   method=f'Monte Carlo: {n_mc} random sign flips, seed {seed}, p = (1 + hits)/(1 + draws)')
        return out
    if method == 'enumeration' or (method == 'auto' and m <= EXACT_ENUM_MAX):
        counts = _counts_enumeration(v)
        txt = f'exact: enumeration of all 2^{m} sign patterns'
    elif method in ('generating', 'auto'):
        counts = _counts_generating(v)
        txt = f'exact: generating-function count of all 2^{m} sign patterns'
    else:
        raise ValueError(f'unknown method {method!r}')
    out.update(p=_p_from_counts(counts, t, alternative), method=txt + (' (midranks for ties)' if out['ties'] else ''))
    return out


def signflip_test(d, *, mu: float = 0.0, alternative: str = 'two-sided', method: str = 'auto',
                  max_exact: int = EXACT_ENUM_MAX, n_mc: int = N_MC, seed: int = SEED, tol: float = 1e-12) -> dict:
    """Sign-flip permutation test on the MEAN paired difference (sensitivity test; RA-07 'exact sign-flip').

    Exact enumeration of all 2^n sign patterns when n <= max_exact (method 'auto' or 'enumeration'; at most 24);
    otherwise Monte Carlo with n_mc flips and a fixed seed, p = (1 + hits) / (1 + n_mc). Two-sided p = share of
    patterns with |mean| >= |observed mean| - tol (the tolerance absorbs floating-point noise; same definition and
    tolerance as stats_common.signflip_p of the phase-1 audit). Returns dict(n, statistic = mean of d - mu, p,
    method, min_p_n)."""
    x = _clean(d) - mu
    n = len(x)
    out = dict(n=n, statistic=float(x.mean()) if n else np.nan, p=np.nan, alternative=alternative, mu=mu,
               method='no data', min_p_n=smallest_attainable_p(n, alternative))
    if n == 0:
        return out
    if np.allclose(x, 0):
        out.update(p=1.0, method='all differences zero: p = 1')
        return out
    obs = float(x.mean())

    def hits(means):
        if alternative == 'two-sided':
            return int((np.abs(means) >= abs(obs) - tol).sum())
        if alternative == 'less':
            return int((means <= obs + tol).sum())
        return int((means >= obs - tol).sum())

    if method in ('auto', 'enumeration') and n <= max(max_exact, 0) and n <= 24:
        # signed sum of every pattern: sum(s_i x_i) = 2 sum(B_i x_i) - sum(x_i), B_i = 1 for s_i = +1
        means = (2.0 * _enumerate_pattern_sums(x.astype(float)) - x.sum()) / n
        out.update(p=hits(means) / (1 << n), method=f'exact: enumeration of all 2^{n} sign patterns')
    else:
        rng = np.random.default_rng(seed)
        h = done = 0
        while done < n_mc:
            b = min(50_000, n_mc - done)
            h += hits(((1.0 - 2.0 * rng.integers(0, 2, size=(b, n))) * x[None, :]).mean(axis=1))
            done += b
        out.update(p=(h + 1) / (n_mc + 1),
                   method=f'Monte Carlo: {n_mc} random sign flips, seed {seed}, p = (1 + hits)/(1 + draws)')
    return out


# =============================================================================================== estimates
def walsh_averages(d) -> np.ndarray:
    """Sorted Walsh averages (d_i + d_j) / 2 for i <= j (n (n + 1) / 2 values)."""
    x = _clean(d)
    i, j = np.triu_indices(len(x))
    return np.sort((x[i] + x[j]) / 2.0)


def hodges_lehmann(d) -> float:
    """One-sample Hodges-Lehmann estimate: median of the Walsh averages of the paired differences (same units)."""
    x = _clean(d)
    return float(np.median(walsh_averages(x))) if len(x) else np.nan


@lru_cache(maxsize=None)
def _noties_cdf(n: int):
    """Exact null CDF counts of the signed-rank statistic T+ for ranks 1..n without ties: (cum counts, 2^n)."""
    counts = _counts_generating(np.arange(1, n + 1))
    return np.cumsum(counts), 2 ** n


@lru_cache(maxsize=None)
def _walsh_k(n: int, level: float):
    """(k, attainable, achieved coverage) of the Walsh-average interval [W(k), W(M + 1 - k)] for n differences."""
    cum, total = _noties_cdf(n)
    a2 = (1.0 - level) / 2.0
    ok = [int(c) <= a2 * total * (1.0 + 1e-12) for c in cum]
    if any(ok):
        t = max(i for i, flag in enumerate(ok) if flag)
        k, attainable = t + 1, True
    else:
        k, attainable = 1, False
    M = n * (n + 1) // 2
    k = min(k, (M + 1) // 2)
    cov = 1.0 - 2.0 * int(cum[k - 1]) / total
    return int(k), bool(attainable), float(cov)


def hl_ci(d, level: float = 0.95) -> dict:
    """Distribution-free confidence interval for the Hodges-Lehmann shift from the Walsh averages.

    With M = n (n + 1) / 2 ordered Walsh averages W(1) <= ... <= W(M), the interval is [W(k), W(M + 1 - k)] where
    k - 1 is the largest value t with P(T+ <= t) <= (1 - level) / 2 under the exact signed-rank null distribution
    for n differences without ties (Hollander and Wolfe). The achieved coverage 1 - 2 P(T+ <= k - 1) is returned.
    If no k reaches the level (n too small: 95% needs n >= 6, 90% needs n >= 5), the widest interval [W(1), W(M)] is
    returned with attainable = False and its own coverage. All n differences are used (zeros included).
    Returns dict(level, lo, hi, coverage, k, attainable, n)."""
    x = _clean(d)
    n = len(x)
    res = dict(level=level, lo=np.nan, hi=np.nan, coverage=np.nan, k=0, attainable=False, n=n)
    if n == 0:
        return res
    W = walsh_averages(x)
    M = len(W)
    k, attainable, cov = _walsh_k(n, float(level))
    res.update(lo=float(W[k - 1]), hi=float(W[M - k]), coverage=cov, k=k, attainable=attainable)
    return res


# =============================================================================================== bootstrap
def _stat_rows(X: np.ndarray, stat: str) -> np.ndarray:
    if stat == 'median':
        return np.median(X, axis=1)
    if stat == 'mean':
        return X.mean(axis=1)
    if stat == 'hl':
        b, n = X.shape
        i, j = np.triu_indices(n)
        out = np.empty(b)
        step = max(1, int(4_000_000 // max(len(i), 1)))
        for s in range(0, b, step):
            out[s:s + step] = np.median((X[s:s + step, i] + X[s:s + step, j]) / 2.0, axis=1)
        return out
    raise ValueError(f'unknown statistic {stat!r}')


def _stat1(x: np.ndarray, stat: str) -> float:
    return float(_stat_rows(np.asarray(x, float)[None, :], stat)[0])


def bootstrap_ci(d, *, stat: str = 'hl', B: int = B_BOOT, seed: int = SEED, levels=(0.95, 0.90)) -> dict:
    """Percentile bootstrap CI over windows (sensitivity).

    Windows (the entries of d, in the given order) are resampled with replacement: one call
    rng.integers(0, n, size=(B, n)) with rng = default_rng(seed); the statistic ('hl' Hodges-Lehmann, 'median',
    'mean') is computed per resample; limits are np.quantile (linear) at (1 - level) / 2 and 1 - (1 - level) / 2.
    Returns dict(ci={level: (lo, hi)}, B, seed, stat, n). NaN limits if n < 2."""
    x = _clean(d)
    n = len(x)
    res = dict(ci={lv: (np.nan, np.nan) for lv in levels}, B=B, seed=seed, stat=stat, n=n)
    if n < 2:
        return res
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    reps = _stat_rows(x[idx], stat)
    res['ci'] = {lv: (float(np.quantile(reps, (1 - lv) / 2)), float(np.quantile(reps, 1 - (1 - lv) / 2)))
                 for lv in levels}
    return res


def cluster_bootstrap_ci(d, clusters, *, stat: str = 'hl', B: int = B_CLUSTER, seed: int = SEED,
                         levels=(0.95, 0.90)) -> dict:
    """Cluster (block) bootstrap CI: bench faces (trained wall) or pieces (new walls) resampled with replacement.

    Clusters are the sorted unique labels; replicate b draws rng.integers(0, G, size=G) (one call per replicate,
    as stats_common.cluster_boot_ci_median of the phase-1 audit) and pools all windows of the drawn clusters.
    Returns dict(ci={level: (lo, hi)}, B, seed, stat, n, n_clusters). NaN limits if fewer than 2 clusters."""
    x = np.asarray(d, dtype=float).ravel()
    c = np.asarray(clusters).ravel()
    ok = ~np.isnan(x)
    x, c = x[ok], c[ok]
    labs = np.unique(c)
    G = len(labs)
    res = dict(ci={lv: (np.nan, np.nan) for lv in levels}, B=B, seed=seed, stat=stat, n=len(x), n_clusters=G)
    if G < 2:
        return res
    groups = [x[c == lab] for lab in labs]
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, size=G)
        reps[b] = _stat1(np.concatenate([groups[i] for i in pick]), stat)
    res['ci'] = {lv: (float(np.quantile(reps, (1 - lv) / 2)), float(np.quantile(reps, 1 - (1 - lv) / 2)))
                 for lv in levels}
    return res


def two_stage_bootstrap_ci(rows: pd.DataFrame, arm_a: str, arm_b: str, endpoint: str, scheme: str, layout: str, *,
                           variant: str = '', windows=None, stat: str = 'hl', window_stat: str = 'median',
                           B: int = 4_000, seed: int = SEED, order: str = 'r', levels=(0.95, 0.90)) -> dict:
    """Two-stage bootstrap (RA-07): resample windows, then realisations inside each drawn window (A and B
    separately); window difference = window_stat(A*) - window_stat(B*); statistic over windows = stat.

    windows : window identifiers to use (default: common windows with at least one defined value in both arms),
        sorted. order : 'r' (realisations sorted by r, default) or 'file' (row order of the loaded files, as the
        phase-1 audit script stats_unseen_window_level.two_stage_boot). Random calls per replicate:
        rng.integers(0, nw, nw), then for each drawn window rng.integers(0, nA, nA) and rng.integers(0, nB, nB).
    Returns dict(ci={level: (lo, hi)}, B, seed, n_windows)."""
    q = rows[(rows['scheme'] == scheme) & (rows['layout'] == layout) & (rows['variant'] == variant)]
    q = q.sort_values('_order' if order == 'file' else ['window', 'r'], kind='mergesort')
    cells, used = [], []
    wins = sorted(set(q['window'])) if windows is None else list(windows)
    for w in wins:
        xa = q[(q['window'] == w) & (q['arm'] == arm_a)][endpoint].to_numpy(dtype=float)
        xb = q[(q['window'] == w) & (q['arm'] == arm_b)][endpoint].to_numpy(dtype=float)
        xa, xb = xa[~np.isnan(xa)], xb[~np.isnan(xb)]
        if len(xa) and len(xb):
            cells.append((xa, xb))
            used.append(w)
    res = dict(ci={lv: (np.nan, np.nan) for lv in levels}, B=B, seed=seed, n_windows=len(cells), windows=used)
    if len(cells) < 2:
        return res
    f = np.median if window_stat == 'median' else np.mean
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    nw = len(cells)
    for b in range(B):
        pick = rng.integers(0, nw, size=nw)
        dw = []
        for j in pick:
            xa, xb = cells[j]
            dw.append(f(xa[rng.integers(0, len(xa), len(xa))]) - f(xb[rng.integers(0, len(xb), len(xb))]))
        reps[b] = _stat1(np.asarray(dw), stat)
    res['ci'] = {lv: (float(np.quantile(reps, (1 - lv) / 2)), float(np.quantile(reps, 1 - (1 - lv) / 2)))
                 for lv in levels}
    return res


def mc_se_of_estimate(rows: pd.DataFrame, arm_a: str, arm_b: str, endpoint: str, scheme: str, layout: str, *,
                      variant: str = '', stat: str = 'hl', window_stat: str = 'median', B: int = 500,
                      seed: int = SEED, zero_tol: float = ZERO_TOL) -> dict:
    """Monte Carlo standard error of the comparison estimate (RA-06 support).

    The windows are kept fixed; inside every common window the realisations of A and of B are resampled with
    replacement (sorted by r), window values are recomputed with window_stat, the paired differences (|d| <=
    zero_tol set to 0) give the statistic (default Hodges-Lehmann). SE = standard deviation (ddof 1) over B
    replicates; the 2.5% and 97.5% quantiles are also returned. It measures how much the reported estimate would
    change with a new set of realisations of the same windows (realisation noise only, no window sampling).
    Random calls per replicate: for each window in sorted order rng.integers(0, nA, nA) then rng.integers(0, nB, nB).
    Returns dict(se, lo, hi, estimate (on the original realisations), B, seed, n_windows)."""
    q = rows[(rows['scheme'] == scheme) & (rows['layout'] == layout) & (rows['variant'] == variant)]
    q = q.sort_values(['window', 'r'], kind='mergesort')
    f = np.median if window_stat == 'median' else np.mean
    cells = []
    for w in sorted(set(q['window'])):
        xa = q[(q['window'] == w) & (q['arm'] == arm_a)][endpoint].to_numpy(dtype=float)
        xb = q[(q['window'] == w) & (q['arm'] == arm_b)][endpoint].to_numpy(dtype=float)
        xa, xb = xa[~np.isnan(xa)], xb[~np.isnan(xb)]
        if len(xa) and len(xb):
            cells.append((xa, xb))
    res = dict(se=np.nan, lo=np.nan, hi=np.nan, estimate=np.nan, B=B, seed=seed, n_windows=len(cells))
    if not cells:
        return res

    def snap(d):
        d = np.asarray(d, float)
        return np.where(np.abs(d) <= zero_tol, 0.0, d)

    res['estimate'] = _stat1(snap([f(a) - f(b) for a, b in cells]), stat)
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for i in range(B):
        dw = [f(a[rng.integers(0, len(a), len(a))]) - f(b[rng.integers(0, len(b), len(b))]) for a, b in cells]
        reps[i] = _stat1(snap(dw), stat)
    res.update(se=float(np.std(reps, ddof=1)) if B > 1 else np.nan, lo=float(np.quantile(reps, 0.025)),
               hi=float(np.quantile(reps, 0.975)))
    return res


# =============================================================================================== multiplicity
def holm(p, nan_as_one: bool = True) -> np.ndarray:
    """Holm step-down adjusted p-values. NaN entries return NaN; with nan_as_one they count in the family size as
    p = 1 (conservative), otherwise they are left out of the family."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    q = np.where(valid, p, 1.0) if nan_as_one else p[valid]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q, kind='mergesort')
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (m - rank) * q[j]))
        adj[j] = running
    if nan_as_one:
        out[valid] = adj[valid]
    else:
        out[valid] = adj
    return out


def bh(p, nan_as_one: bool = True) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up adjusted p-values). NaN handling as in holm()."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    q = np.where(valid, p, 1.0) if nan_as_one else p[valid]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q, kind='mergesort')
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        j = order[rank]
        running = min(running, q[j] * m / (rank + 1))
        adj[j] = min(running, 1.0)
    if nan_as_one:
        out[valid] = adj[valid]
    else:
        out[valid] = adj
    return out


# =============================================================================================== decision labels
def decision_flags(estimate: float, ci95, ci90, p_adj: float, margin: float, *, lower_is_better: bool = True,
                   alpha: float = ALPHA, ci95_attainable: bool = True, ci90_attainable: bool = True) -> dict:
    """Protocol P decision label for one endpoint (all inputs in the same units as margin).

    estimate : Hodges-Lehmann estimate of A - B; ci95, ci90 : (lo, hi) two-sided intervals of the same shift;
    p_adj : adjusted p of the two-sided superiority test (Holm for primaries, BH for secondaries, unadjusted for a
    single confirmatory test). With lower_is_better = False the signs are reversed first (harm scale).
    Criteria: improvement = p_adj < alpha and hi95 < 0 and estimate <= -margin; equivalent = -margin < lo90 and
    hi90 < margin; noninferior = hi95 < margin; worse = p_adj < alpha and lo95 > 0; practically_worse = worse and
    estimate >= margin. Label = first of Improvement, Practically equivalent, Not practically worse, Worse,
    Inconclusive whose criterion holds. Criteria that need an interval whose level is not attainable are False.
    Returns dict(label, qualifier, improvement, equivalent, noninferior, worse, practically_worse)."""
    vals = [estimate, *ci95, *ci90]
    out = dict(label=INCONCLUSIVE, qualifier='', improvement=False, equivalent=False, noninferior=False,
               worse=False, practically_worse=False)
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals) or margin is None:
        out['qualifier'] = 'not estimable'
        return out
    s = 1.0 if lower_is_better else -1.0
    est = s * float(estimate)
    lo95, hi95 = sorted((s * float(ci95[0]), s * float(ci95[1])))
    lo90, hi90 = sorted((s * float(ci90[0]), s * float(ci90[1])))
    sig = p_adj is not None and not (isinstance(p_adj, float) and math.isnan(p_adj)) and float(p_adj) < alpha
    improvement = bool(ci95_attainable and sig and hi95 < 0 and est <= -margin)
    worse = bool(ci95_attainable and sig and lo95 > 0)
    practically_worse = bool(worse and est >= margin)
    equivalent = bool(ci90_attainable and (-margin < lo90) and (hi90 < margin))
    noninferior = bool(ci95_attainable and hi95 < margin)
    if improvement:
        label = IMPROVEMENT
    elif equivalent:
        label = EQUIVALENT
    elif noninferior:
        label = NONINFERIOR
    elif worse:
        label = WORSE
    else:
        label = INCONCLUSIVE
    q = []
    if label in (EQUIVALENT, NONINFERIOR) and sig and ci95_attainable and (hi95 < 0 or lo95 > 0):
        q.append('statistically significant difference inside the margin ('
                 + ('favourable' if hi95 < 0 else 'unfavourable') + ')')
    if label == EQUIVALENT and not noninferior:
        q.append('equivalent at 90% but upper 95% limit not below the margin')
    if practically_worse:
        q.append('practically worse (estimate at least the margin)')
    if label == INCONCLUSIVE and est <= -margin:
        q.append('estimate beyond the margin in the favourable direction, not confirmed')
    if label == INCONCLUSIVE and est >= margin:
        q.append('estimate beyond the margin in the unfavourable direction, not confirmed')
    if not ci95_attainable:
        q.append('95% level not attainable with this number of windows')
    if not ci90_attainable:
        q.append('90% level not attainable with this number of windows')
    out.update(label=label, qualifier='; '.join(q), improvement=improvement, equivalent=equivalent,
               noninferior=noninferior, worse=worse, practically_worse=practically_worse)
    return out


# =============================================================================================== comparison driver
def compare(wv: pd.DataFrame, arm_a: str, arm_b: str, scheme: str, layout: str, *, variant: str = '',
            endpoints=None, analysis: str = '', comparison: str | None = None, role: str | None = None,
            single_tests=(), holm_family=None, margins=None, sensitivity_endpoints=KEY_ENDPOINTS,
            B_boot: int = B_BOOT, B_cluster: int = B_CLUSTER, seed: int = SEED, alpha: float = ALPHA,
            signflip: bool = True, aux_tests: bool = True, zero_tol: float = ZERO_TOL) -> pd.DataFrame:
    """Protocol P comparison of arm A against arm B in ONE scheme and ONE layout (schemes are never pooled).

    endpoints : endpoints to evaluate (default: every score in wv); the five primaries are always included.
    single_tests : primaries tested alone at alpha without adjustment (RA-10 H3: ('len_w1_m',)).
    holm_family : primaries adjusted together by Holm (default: PRIMARY minus single_tests).
    Secondary endpoints (all others, including CL_err) get Benjamini-Hochberg q-values over the secondaries evaluated
    in this call (39 scores give 34 secondaries); their labels (CL_err only, the only secondary with a margin) use the
    q-value as the adjusted p and are exploratory.
    Paired differences with |d| <= zero_tol are set to 0 before any statistic (floating-point noise).
    Per endpoint: n windows used, drop counts, HL estimate, Walsh 95% and 90% CIs (with achieved coverage), exact
    two-sided Wilcoxon signed-rank p (Pratt), smallest attainable p, adjusted p, label and criteria; auxiliary
    one-sided exact tests against the margin (p_noninferiority: H0 shift >= +margin; p_tost = max of the two
    one-sided tests at -margin and +margin) and their Holm adjustment over the primary family (reported, not used by
    the labels); median and mean difference; for sensitivity_endpoints only: sign-flip test of the mean, percentile
    bootstrap and cluster bootstrap CIs of the HL estimate and the labels they would give; Monte Carlo SE of the
    window values of each arm (median and maximum over windows), the share of window values with MC SE below
    margin / 4 and whether all are (mc_se_ok, the RA-06 rule as written).
    Output values are in display units (P21_relerr in percentage points; others raw); see 'unit'."""
    margins = dict(MARGINS if margins is None else margins)
    sel = (wv['scheme'] == scheme) & (wv['layout'] == layout) & (wv['variant'] == variant)
    avail = list(dict.fromkeys(wv.loc[sel, 'endpoint']))
    if not avail:
        raise ValueError(f'no window values for scheme {scheme!r}, layout {layout!r}, variant {variant!r}')
    if endpoints is None:
        order = score_list(avail)[0]
        endpoints = [e for e in order if e in avail] + [e for e in avail if e not in order]
    endpoints = list(dict.fromkeys(list(PRIMARY) + list(endpoints)))
    single_tests = tuple(single_tests)
    if holm_family is None:
        holm_family = tuple(e for e in PRIMARY if e not in single_tests)
    uncovered = [e for e in PRIMARY if e not in holm_family and e not in single_tests]
    if uncovered:
        raise ValueError(f'primaries without a multiplicity rule (holm_family or single_tests): {uncovered}')
    comparison = comparison or f'{arm_a} vs {arm_b}'
    if role is None:
        role = f"scheme {SCHEME_ROLE.get(scheme, 'n/a')}, layout {LAYOUT_ROLE.get(layout, 'n/a')}"
    recs = []
    for e in endpoints:
        pr = paired_differences(wv, arm_a, arm_b, e, scheme, layout, variant, zero_tol=zero_tol)
        d = pr.d
        lib = e not in HIGHER_IS_BETTER
        h = d if lib else -d                              # harm scale: positive = A worse
        rec = dict(analysis=analysis, comparison=comparison, scheme=scheme, layout=layout, endpoint=e,
                   n=pr.n, arm_a=arm_a, arm_b=arm_b, variant=variant, role=role, group=group_of(e),
                   family=('primary' if e in PRIMARY else 'key secondary' if e in KEY_SECONDARY else 'secondary'),
                   lower_is_better=lib, n_windows_a=pr.n_windows_a, n_windows_b=pr.n_windows_b,
                   n_common=pr.n_common, n_dropped_nd=pr.n_dropped, n_dropped_nd_a=pr.n_drop_nd_a,
                   n_dropped_nd_b=pr.n_drop_nd_b, n_dropped_nd_both=pr.n_drop_nd_both,
                   n_only_a=len(pr.only_a), n_only_b=len(pr.only_b), n_clusters=int(len(np.unique(pr.pieces))),
                   n_a_better=int((h < -ZERO_TOL).sum()), n_a_worse=int((h > ZERO_TOL).sum()),
                   n_tied=int((np.abs(d) <= ZERO_TOL).sum()))
        w = wilcoxon_exact(d)
        c95, c90 = hl_ci(d, 0.95), hl_ci(d, 0.90)
        rec.update(estimate=hodges_lehmann(d), ci95_lo=c95['lo'], ci95_hi=c95['hi'], ci95_coverage=c95['coverage'],
                   ci95_attainable=c95['attainable'], ci90_lo=c90['lo'], ci90_hi=c90['hi'],
                   ci90_coverage=c90['coverage'], ci90_attainable=c90['attainable'],
                   median_diff=float(np.median(d)) if pr.n else np.nan,
                   mean_diff=float(np.mean(d)) if pr.n else np.nan,
                   p=w['p'], p_method=w['method'], statistic_T_plus=w['statistic'], n_nonzero=w['n_nonzero'],
                   min_attainable_p=w['min_p_n'], min_attainable_p_nonzero=w['min_p_data'])
        mg = margins.get(e)
        rec['margin'] = mg if mg is not None else np.nan
        if aux_tests and mg is not None and pr.n:
            rec['p_noninferiority'] = wilcoxon_exact(h, mu=mg, alternative='less')['p']
            rec['p_tost'] = max(wilcoxon_exact(h, mu=-mg, alternative='greater')['p'],
                                wilcoxon_exact(h, mu=mg, alternative='less')['p'])
        else:
            rec['p_noninferiority'] = rec['p_tost'] = np.nan
        if signflip and e in sensitivity_endpoints:
            sf = signflip_test(d)
            rec.update(p_signflip_mean=sf['p'], p_signflip_method=sf['method'])
        if e in sensitivity_endpoints and pr.n >= 2:
            bw = bootstrap_ci(d, stat='hl', B=B_boot, seed=seed)
            cb = cluster_bootstrap_ci(d, pr.pieces, stat='hl', B=B_cluster, seed=seed)
            G = int(cb['n_clusters'])
            rec.update(boot95_lo=bw['ci'][0.95][0], boot95_hi=bw['ci'][0.95][1], boot90_lo=bw['ci'][0.90][0],
                       boot90_hi=bw['ci'][0.90][1], cluster95_lo=cb['ci'][0.95][0], cluster95_hi=cb['ci'][0.95][1],
                       cluster90_lo=cb['ci'][0.90][0], cluster90_hi=cb['ci'][0.90][1], boot_B=B_boot,
                       cluster_B=B_cluster, boot_seed=seed,
                       cluster_distinct_resamples=math.comb(2 * G - 1, G) if G >= 1 else 0)
        mc = wv[sel & (wv['endpoint'] == e) & wv['window'].isin(pr.windows) & wv['arm'].isin([arm_a, arm_b])]
        sea = (mc.loc[mc['arm'] == arm_a, 'mc_se'] if 'mc_se' in mc else pd.Series(dtype=float)).dropna()
        seb = (mc.loc[mc['arm'] == arm_b, 'mc_se'] if 'mc_se' in mc else pd.Series(dtype=float)).dropna()
        rec.update(mc_se_median_a=float(sea.median()) if len(sea) else np.nan,
                   mc_se_max_a=float(sea.max()) if len(sea) else np.nan,
                   mc_se_median_b=float(seb.median()) if len(seb) else np.nan,
                   mc_se_max_b=float(seb.max()) if len(seb) else np.nan,
                   n_real_min_a=mc.loc[mc['arm'] == arm_a, 'n_real'].min(),
                   n_real_min_b=mc.loc[mc['arm'] == arm_b, 'n_real'].min())
        se_all = pd.concat([sea, seb])
        if mg is not None and len(se_all):
            rec['mc_se_share_below_quarter_margin'] = float((se_all < mg / 4.0).mean())
            rec['mc_se_ok'] = bool((se_all < mg / 4.0).all())
        else:
            rec['mc_se_share_below_quarter_margin'] = rec['mc_se_ok'] = np.nan
        recs.append(rec)
    R = pd.DataFrame(recs)

    # ---- multiplicity
    R['adjusted_p'] = np.nan
    R['adjustment'] = ''
    fam = R['endpoint'].isin(holm_family)
    if fam.any():
        R.loc[fam, 'adjusted_p'] = holm(R.loc[fam, 'p'].to_numpy())
        R.loc[fam, 'adjustment'] = f'Holm over {int(fam.sum())} primaries'
        R.loc[fam, 'p_noninferiority_holm'] = holm(R.loc[fam, 'p_noninferiority'].to_numpy())
    single = R['endpoint'].isin(single_tests)
    R.loc[single, 'adjusted_p'] = R.loc[single, 'p']
    R.loc[single, 'adjustment'] = 'none (single pre-specified test)'
    sec = ~R['endpoint'].isin(PRIMARY)
    if sec.any():
        R.loc[sec, 'adjusted_p'] = bh(R.loc[sec, 'p'].to_numpy())
        R.loc[sec, 'adjustment'] = f'Benjamini-Hochberg q over {int(sec.sum())} secondaries (exploratory)'
    if 'p_noninferiority_holm' not in R:
        R['p_noninferiority_holm'] = np.nan

    # ---- labels (raw units)
    lab_cols = {k: [] for k in ('label', 'qualifier', 'improvement', 'equivalent', 'noninferior', 'worse',
                                'practically_worse', 'label_window_boot', 'label_cluster_boot')}
    for _, r in R.iterrows():
        mg = margins.get(r['endpoint'])
        if mg is None:
            f = dict(label=NO_LABEL, qualifier='', improvement=np.nan, equivalent=np.nan, noninferior=np.nan,
                     worse=np.nan, practically_worse=np.nan)
            lw = lc = NO_LABEL
        else:
            f = decision_flags(r['estimate'], (r['ci95_lo'], r['ci95_hi']), (r['ci90_lo'], r['ci90_hi']),
                               r['adjusted_p'], mg, lower_is_better=bool(r['lower_is_better']), alpha=alpha,
                               ci95_attainable=bool(r['ci95_attainable']), ci90_attainable=bool(r['ci90_attainable']))
            lw = lc = ''
            if 'boot95_lo' in R and not pd.isna(r.get('boot95_lo', np.nan)):
                lw = decision_flags(r['estimate'], (r['boot95_lo'], r['boot95_hi']), (r['boot90_lo'], r['boot90_hi']),
                                    r['adjusted_p'], mg, lower_is_better=bool(r['lower_is_better']),
                                    alpha=alpha)['label']
                lc = decision_flags(r['estimate'], (r['cluster95_lo'], r['cluster95_hi']),
                                    (r['cluster90_lo'], r['cluster90_hi']), r['adjusted_p'], mg,
                                    lower_is_better=bool(r['lower_is_better']), alpha=alpha)['label']
        for k in ('label', 'qualifier', 'improvement', 'equivalent', 'noninferior', 'worse', 'practically_worse'):
            lab_cols[k].append(f[k])
        lab_cols['label_window_boot'].append(lw)
        lab_cols['label_cluster_boot'].append(lc)
    for k, v in lab_cols.items():
        R[k] = v

    # ---- display units
    R['unit'] = [UNIT_DISPLAY.get(e, '') for e in R['endpoint']]
    R['display_scale'] = [DISPLAY_SCALE.get(e, 1.0) for e in R['endpoint']]
    R['natural_variation'] = [NATURAL_VARIATION.get(e, np.nan) for e in R['endpoint']]
    scale_cols = ['estimate', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi', 'median_diff', 'mean_diff', 'margin',
                  'natural_variation', 'mc_se_median_a', 'mc_se_max_a', 'mc_se_median_b', 'mc_se_max_b',
                  'boot95_lo', 'boot95_hi', 'boot90_lo', 'boot90_hi', 'cluster95_lo', 'cluster95_hi', 'cluster90_lo',
                  'cluster90_hi']
    for c in scale_cols:
        if c in R:
            R[c] = R[c] * R['display_scale']
    R['CI'] = [f"{format_value(lo)} to {format_value(hi)}" if not (pd.isna(lo) or pd.isna(hi)) else ''
               for lo, hi in zip(R['ci95_lo'], R['ci95_hi'])]
    core = ['analysis', 'comparison', 'scheme', 'layout', 'endpoint', 'n', 'estimate', 'CI', 'p', 'adjusted_p',
            'label']
    return R[core + [c for c in R.columns if c not in core]]


def method_level_summary(res: pd.DataFrame, primary=PRIMARY) -> dict:
    """Method-level reading of one compare() result (REQUIRED_ANALYSES.md Sec. 0, last sentence of the labels row).

    better_than_comparator = Improvement on at least one primary and no practically worse primary (in this one
    scheme and layout; the protocol requires it on the trained wall AND on new pieces, which the caller combines).
    all_noninferior = every primary has upper 95% limit below +margin (H2 reading of RA-02)."""
    q = res[res['endpoint'].isin(primary)]
    impr = q.loc[q['label'] == IMPROVEMENT, 'endpoint'].tolist()
    pw = q.loc[q['practically_worse'].astype(bool), 'endpoint'].tolist()
    return dict(n_primaries=int(len(q)), improvement=impr, practically_worse=pw,
                worse_label=q.loc[q['label'] == WORSE, 'endpoint'].tolist(),
                inconclusive=q.loc[q['label'] == INCONCLUSIVE, 'endpoint'].tolist(),
                all_noninferior=bool(len(q) == len(primary) and q['noninferior'].astype(bool).all()),
                better_than_comparator=bool(impr) and not pw)


# =============================================================================================== output
def format_value(x, sig: int = 3) -> str:
    """Compact number: 3 significant digits (no scientific notation for |x| >= 1e-4)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    x = float(x)
    if x == 0:
        return '0'
    if abs(x) < 1e-4:
        return f'{x:.2e}'
    digits = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    return f'{x:.{digits}f}'


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ''
    return f'{p:.4f}' if p >= 1e-3 else f'{p:.2e}'


CORE_COLUMNS = ['analysis', 'comparison', 'scheme', 'layout', 'endpoint', 'n', 'estimate', 'CI', 'p', 'adjusted_p',
                'label']


def write_results(df: pd.DataFrame, out_stem, *, title: str = '', notes=(), md_columns=None):
    """Write a result table as <out_stem>.csv (all columns) and <out_stem>.md (core columns plus unit, CI coverage
    and qualifier, numbers formatted). Columns of the core table: analysis, comparison, scheme, layout, endpoint, n,
    estimate, CI, p, adjusted p, label. Returns (csv path, md path)."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = Path(str(out_stem) + '.csv'), Path(str(out_stem) + '.md')
    df.to_csv(csv_path, index=False)
    cols = md_columns or (CORE_COLUMNS + [c for c in ('unit', 'ci95_coverage', 'min_attainable_p', 'qualifier')
                                          if c in df.columns])
    head = {'adjusted_p': 'adjusted p', 'ci95_coverage': 'CI coverage', 'min_attainable_p': 'smallest attainable p'}
    lines = [f'# {title}' if title else '# Protocol P results', '']
    lines += [str(n) for n in notes] + ([''] if notes else [])
    lines.append('| ' + ' | '.join(head.get(c, c) for c in cols) + ' |')
    lines.append('|' + '|'.join('---' for _ in cols) + '|')
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, '')
            if c in ('p', 'adjusted_p', 'min_attainable_p'):
                s = _fmt_p(v)
            elif c in ('estimate',):
                s = format_value(v)
            elif c == 'ci95_coverage':
                s = '' if pd.isna(v) else f'{float(v):.4f}'
            elif isinstance(v, float) and math.isnan(v):
                s = ''
            else:
                s = str(v)
            cells.append(s.replace('|', '\\|'))
        lines.append('| ' + ' | '.join(cells) + ' |')
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_path, md_path


# =============================================================================================== run records
def environment() -> dict:
    """Interpreter and package versions for a run record."""
    import scipy
    return dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                platform=platform.platform(), executable=sys.executable,
                dont_write_bytecode=bool(sys.dont_write_bytecode), protocolP=__version__)


def write_run_record(path, *, title: str, purpose: str, scripts=(), inputs=(), outputs=(), seeds=None,
                     command: str = '', notes=()) -> Path:
    """Write RUN_RECORD.md: purpose, command, environment, SHA-256 of scripts and inputs, seeds, outputs, notes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = environment()

    def table(files):
        rows = ['| File | Bytes | SHA-256 |', '|---|---|---|']
        for f in files:
            f = Path(f)
            if f.exists():
                rows.append(f'| `{f}` | {f.stat().st_size} | `{file_sha256(f)}` |')
            else:
                rows.append(f'| `{f}` | missing | NOT VERIFIABLE |')
        return rows

    lines = [f'# {title}', '', f'**Purpose:** {purpose}', '', f'**Command:** `{command}`', '', '## Environment', '']
    lines += [f'- {k}: {v}' for k, v in env.items()]
    lines += ['', '## Scripts and parsed modules', ''] + table(scripts)
    lines += ['', '## Inputs', ''] + table(inputs)
    if seeds:
        lines += ['', '## Seeds', ''] + [f'- {k}: {v}' for k, v in seeds.items()]
    if outputs:
        lines += ['', '## Outputs', ''] + [f'- `{o}`' for o in outputs]
    if notes:
        lines += ['', '## Notes', ''] + [f'- {n}' for n in notes]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path
