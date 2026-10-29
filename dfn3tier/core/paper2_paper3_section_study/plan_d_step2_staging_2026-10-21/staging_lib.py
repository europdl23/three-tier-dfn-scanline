"""Plan D Step 2 staging library (no training, no reconstruction, no model calls).

Original definitions are executed from source text exactly as in the F4 freeze (family_freeze_F4_2026-10-20/build_f4_freeze.py):
  exp_common.py    -> split_line_by_strips, split_panel, assign_family_ids, ...
  run_ablation_v3  -> _train_geology_stats, _calibrate_scorer_weights
  build_f4_freeze  -> strips, hidden, windows, statistics  (k-means k=3 on doubled axial angles, random_state 42001, n_init 20;
                      random multi-start, never initialised from any stored centre)
R3 count-rule fitting follows r3_observation_conditioned_count_2026-10-20/step2_lofo_calibration.py::fit (a = 2, the HYB_FIX2 default).
Engine-input writing follows r1_corrected_reference_2026-10-20/prepare_r1_inputs.py (schema and pool layout).
"""
import ast, contextlib, csv, hashlib, io, json, math, sys, types
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
EXP = ROOT / 'Updated__file' / 'improved_dfn_experiments'
D3 = STUDY / 'option3a_dataset_freeze_2026-10-20'
F4DIR = STUDY / 'family_freeze_F4_2026-10-20'
R1 = STUDY / 'r1_corrected_reference_2026-10-20'
PREVIEW = STUDY / 'authorised_execution_2026-10-19' / 's1_inputs' / 'run_local_preview.py'
OVERLAP = STUDY / 'plan_d_face_audit_2026-10-21' / 'OVERLAP_MATRIX.csv'
SOURCES = {'fragments': D3 / 'WINDOW_FRAGMENTS_3A.csv', 'inventory': D3 / 'WINDOW_INVENTORY_3A.csv', 'traces': D3 / 'TRACE_MANIFEST_3A.csv',
           'photos': D3 / 'FACE_PHOTOS_3A.json', 'overlap': OVERLAP, 'exp_common': EXP / 'exp_common.py', 'run_ablation_v3': EXP / 'run_ablation_v3.py',
           'build_f4_freeze': F4DIR / 'build_f4_freeze.py', 'run_local_preview': PREVIEW}

EXCLUDED_WINDOWS = sorted({f'W01_RL1066-1102__X0{k}' for k in (4, 5, 6, 7)} | {f'W02_RL1039-1066__X0{k}' for k in (6, 7)})
EXCLUDED_FACES = {'E03'}
FACE = {'W00': 'W00_RL1102-1120', 'W01': 'W01_RL1066-1102', 'W02': 'W02_RL1039-1066', 'W03': 'W03_RL1012-1039', 'W04': 'W04_RL0985-1012',
        'W05': 'W05_RL0958-0985', 'W06': 'W06_RL0938-0958', 'WEST3': 'WEST3_RL0904-0922_100-160m',
        'S2_B00': 'S2_B00_RL0849-0876_P1', 'S2_B01': 'S2_B01_RL0876-0903_P1'}
SHORT = {v: k for k, v in FACE.items()}
INV_COLS = ['window_id', 'parent_face', 'role', 'section', 'x0_m', 'x1_m', 'width_m', 'height_m', 'traces', 'boundary_clipped_traces', 'discarded_right_remainder_m']
FRAG_COLS = ['window_id', 'parent_face', 'role', 'trace_id', 'boundary_clipped', 'cx', 'cy', 'length', 'angle']

# Outer folds (approved). Buffers are never staged for the fold.
FOLDS = {
    'P1': dict(kind='primary', test=['W00', 'W01'], buffer=['W02'], train=['W03', 'W04', 'W05', 'W06', 'WEST3']),
    'P2': dict(kind='primary', test=['W02'], buffer=['W01', 'W03', 'W04'], train=['W00', 'W05', 'W06', 'WEST3']),
    'P3': dict(kind='primary', test=['W03'], buffer=['W02', 'W04'], train=['W00', 'W01', 'W05', 'W06', 'WEST3']),
    'P4': dict(kind='primary', test=['W05', 'W06'], buffer=['W04'], train=['W00', 'W01', 'W02', 'W03', 'WEST3']),
    'WEST3': dict(kind='secondary', test=['WEST3'], buffer=[], train=['W00', 'W01', 'W02', 'W03', 'W04', 'W05', 'W06']),
    'S2': dict(kind='stress', test=['S2_B00', 'S2_B01'], buffer=[], train=['W00', 'W01', 'W02', 'W03', 'W04', 'W05', 'W06', 'WEST3']),
}
INNER_MIN_WINDOWS, INNER_MIN_TRACES, INNER_MIN_VAL_WINDOWS = 15, 150, 3
# Candidate inner validation blocks: every training face alone, plus photo-linked pairs that are both in the fold's training set.
PAIR_BLOCKS = [('W00', 'W01'), ('W05', 'W06'), ('W02', 'W03')]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _extract(path, names):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    return '\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)


# ---- original functions, executed from source text (same extraction as the F4 freeze) -------------------
ENV = dict(np=np, math=math, Sequence=list, Tuple=tuple, List=list)
exec(compile(_extract(SOURCES['exp_common'], ['_endpoints', 'line_endpoints', 'segments_intersect', 'intersection_density',
                                              'split_line_by_strips', 'split_panel', 'assign_family_ids']), 'exp_common_extract', 'exec'), ENV)
_fake = types.ModuleType('exp_common'); _fake.__dict__.update(ENV); sys.modules.setdefault('exp_common', _fake)
exec(compile(_extract(SOURCES['run_ablation_v3'], ['_train_geology_stats', '_calibrate_scorer_weights']), 'ablation_extract', 'exec'), ENV)
F4ENV = dict(np=np, math=math, KMeans=KMeans, contextlib=contextlib, io=io, env=ENV, assign=ENV['assign_family_ids'],
             train_geology_stats=ENV['_train_geology_stats'])
_F4SRC = _extract(SOURCES['build_f4_freeze'], ['strips', 'hidden', 'windows', 'statistics'])
exec(compile(_F4SRC, 'f4_extract', 'exec'), F4ENV)
assign, strips, hidden_traces, f4_windows = ENV['assign_family_ids'], F4ENV['strips'], F4ENV['hidden'], F4ENV['windows']
STATS_FN = {'legacy': F4ENV['statistics']}
# Plan D family rule: the same objective (k-means inertia, k = 3, doubled axial angles) and the same code, optimised with a
# deterministic 1000-start search instead of 20 starts. This is the single deliberate change to the F4 code; it was needed because
# the 20-start search stops at a local optimum in fold P3 (inertia 0.1% above the minimum). No stored centre is ever used.
PLAN_D_N_INIT = 1000
assert _F4SRC.count('KMeans(3, random_state=42001, n_init=20)') == 1
_PDENV = dict(F4ENV)
exec(compile(_F4SRC.replace('KMeans(3, random_state=42001, n_init=20)', f'KMeans(3, random_state=42001, n_init={PLAN_D_N_INIT})'), 'f4_extract_plan_d', 'exec'), _PDENV)
STATS_FN['plan_d'] = _PDENV['statistics']


# ---- data -------------------------------------------------------------------------------------------------
def load():
    frag = rows(SOURCES['fragments']); inv = rows(SOURCES['inventory']); tr = rows(SOURCES['traces'])
    keep_inv = [r for r in inv if r['window_id'] not in EXCLUDED_WINDOWS and r['parent_face'] in SHORT]
    keep = {r['window_id'] for r in keep_inv}
    keep_frag = [r for r in frag if r['window_id'] in keep]
    excluded = [r for r in frag if r['window_id'] in EXCLUDED_WINDOWS]
    return dict(frag=keep_frag, inv=keep_inv, traces=tr, excluded_frag=excluded, all_frag=frag, all_inv=inv)


def photo_edges():
    ov = rows(SOURCES['overlap']); e = set()
    for r in ov:
        a, b = SHORT.get(r['face_a'], r['face_a']), SHORT.get(r['face_b'], r['face_b'])
        if int(r['shared_photos']) > 0 and 'E03' not in (a, b):
            e.add(frozenset((a, b)))
    return e


def neighbours(face, edges):
    return {g for x in edges if face in x for g in x if g != face}


def windows_of(D, faces, window_filter=None):
    names = {FACE[f] for f in faces}
    return [r['window_id'] for r in D['inv'] if r['parent_face'] in names and (window_filter is None or r['window_id'] in window_filter)]


def size(D, wids, drop_traces=frozenset()):
    w = set(wids); fr = [r for r in D['frag'] if r['window_id'] in w and r['trace_id'] not in drop_traces]
    return dict(windows=len(w), unique_traces=len({r['trace_id'] for r in fr}), fragments=len(fr))


# ---- statistics (original F4 code) ---------------------------------------------------------------------
def fit_statistics(D, train_wids, drop_traces=frozenset(), mode='plan_d'):
    """Run the original F4 statistics() on exactly these windows (and without the purged traces)."""
    w = set(train_wids)
    inv = [r for r in D['inv'] if r['window_id'] in w]
    frag = [r for r in D['frag'] if r['window_id'] in w and r['trace_id'] not in drop_traces]
    faces = sorted({r['parent_face'] for r in inv})
    with contextlib.redirect_stdout(io.StringIO()):
        s = STATS_FN[mode](f4_windows(frag, inv, faces))
    # identical serialisation path to F4_FREEZE.json (floats via json round trip)
    s = json.loads(json.dumps(s, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)))
    return s, frag, inv


def engine_statistics(S, frag, inv, train_wids):
    """prepare_r1_inputs.py pool construction on the training windows."""
    train_w = [r['window_id'] for r in inv if r['window_id'] in set(train_wids)]
    arr = np.vstack([np.array([[float(x['cx']), float(x['cy']), float(x['length']), float(x['angle'])] for x in frag if x['window_id'] == wid], np.float32).reshape(-1, 4) for wid in train_w])
    fc = np.array(S['fc'])
    d = np.abs(arr[:, 3:4] - fc[None]) % np.pi; d = np.minimum(d, np.pi - d)
    fam = d.argmin(1); fam[d.min(1) > np.radians(20)] = 3
    assert np.allclose(np.bincount(fam, minlength=4) / len(fam), S['shares'], atol=1e-12)
    phys = arr[:, 2] * 20.0; pools = {}
    for k in range(4):
        ids = fam == k; sub = arr[ids] if ids.any() else arr
        pools[str(k)] = dict(arr=sub.tolist(), lengths_m=(phys[ids] if ids.any() else phys).tolist(), angles=sub[:, 3].tolist(), centres=sub[:, :2].tolist())
    return dict(fc=S['fc'], gs=S['gs'], pools=pools, shares=S['shares'], rate=S['rate'], logmu=S['logmu'], logsig=S['logsig'], yscale=S['yscale'])


STRIPS_5M = [((x - .05) / 20, (x + .05) / 20) for x in (5., 10., 15.)]


def count_rule_parameters(D, train_wids, drop_traces=frozenset(), trace_scope='windows'):
    """R3 count rule fitted on the training windows only (step2_lofo_calibration.py::fit).
    trace_scope='face' is the legacy R3 definition (every trace of the fitting faces, including traces outside the 20 m windows);
    Plan D uses 'windows' (only traces with a fragment in a fitting window), so excluded windows and purged traces never enter e_v."""
    w = set(train_wids)
    inv = {r['window_id']: r for r in D['inv'] if r['window_id'] in w}
    frag = [r for r in D['frag'] if r['window_id'] in w and r['trace_id'] not in drop_traces]
    area = sum(20 * float(r['height_m']) for r in inv.values())
    n_rate = len(frag) / area
    prior = sum(float(r['length']) * 20 for r in frag) / area
    ids = {r['trace_id'] for r in frag}
    if trace_scope == 'face':
        faces = {r['parent_face'] for r in inv.values()}; tr = [r for r in D['traces'] if r['face_id'] in faces]
    else:
        tr = [r for r in D['traces'] if r['trace_id'] in ids]
    th = np.radians([float(r['direction_deg_axial']) for r in tr]); ln = np.array([float(r['end_to_end_length_m']) for r in tr])
    ev = float(np.sum(ln / ln.sum() * np.abs(np.cos(th))))
    return dict(n_rate=n_rate, prior=prior, a_prior=2.0, strips_5m=3, e_v=ev, fitted_on_windows=len(inv), fitted_on_traces=len(tr),
                rule='P21_hat = (n_hits + a)/(k*H*e_v + a/prior); T = max(1, round(n_rate*20*H*P21_hat/prior)); k = strips in window')


def write_stage(D, dest, roles, S_engine, extra=None):
    """Write an engine input folder: roles = {window_id: 'train'|'validation'}."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    inv = [r for r in D['inv'] if r['window_id'] in roles]
    drop = set((extra or {}).get('purged_traces', []))
    frag = [r for r in D['frag'] if r['window_id'] in roles and not (roles[r['window_id']] == 'train' and r['trace_id'] in drop)]
    for name, data, cols in [('window_inventory.csv', inv, INV_COLS), ('window_trace_manifest.csv', frag, FRAG_COLS)]:
        with open(dest / name, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in data:
                w.writerow({**{k: r[k] for k in cols}, 'role': roles[r['window_id']]})
    (dest / 'training_only_statistics.json').write_text(json.dumps(S_engine), encoding='utf-8')
    (dest / 'run_local_preview.py').write_bytes(Path(PREVIEW).read_bytes())
    return {p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()}


# ---- family diagnostics --------------------------------------------------------------------------------
def doubled(a):
    return np.c_[np.cos(2 * a), np.sin(2 * a)]


def centres_of(km):
    return np.sort((np.arctan2(km.cluster_centers_[:, 1], km.cluster_centers_[:, 0]) / 2) % np.pi)


def family_diagnostics(frag_rows, seeds=(1, 2, 3, 4, 5), n_boot=50):
    a = np.array([float(r['angle']) for r in frag_rows]) % np.pi
    rule = KMeans(3, random_state=42001, n_init=PLAN_D_N_INIT).fit(doubled(a))
    legacy = KMeans(3, random_state=42001, n_init=20).fit(doubled(a))
    alts = [KMeans(3, random_state=s, n_init=PLAN_D_N_INIT).fit(doubled(a)) for s in seeds]
    best = min(alts + [rule], key=lambda m: m.inertia_)
    unique_shift = max(float(max(abs(((np.degrees(centres_of(m)) - np.degrees(centres_of(rule))) + 90) % 180 - 90))) for m in alts)
    # the competing k=3 solution: best inertia among solutions whose centres differ from the rule by > 10 deg
    sols = {}
    for s in range(60):
        m = KMeans(3, init='random', n_init=1, random_state=1000 + s).fit(doubled(a))
        key = tuple(np.round(np.degrees(centres_of(m)), 0)); sols[key] = min(sols.get(key, math.inf), m.inertia_)
    rc = np.degrees(centres_of(rule))
    alt = [(i, k) for k, i in sols.items() if max(abs(((np.array(k) - rc) + 90) % 180 - 90)) > 10]
    alt = min(alt) if alt else (math.nan, ())
    wins = sorted({r['window_id'] for r in frag_rows}); bywin = {w: np.array([float(r['angle']) for r in frag_rows if r['window_id'] == w]) % np.pi for w in wins}
    rng = np.random.default_rng(20261021); boots = []
    for _ in range(n_boot):
        aa = np.concatenate([bywin[w] for w in rng.choice(wins, len(wins))])
        boots.append(np.degrees(centres_of(KMeans(3, random_state=42001, n_init=100).fit(doubled(aa)))))
    boots = np.array(boots)
    return dict(n=len(a), rule_centres_deg=rc.tolist(), rule_inertia=float(rule.inertia_), global_best_inertia=float(best.inertia_),
                legacy20_centres_deg=np.degrees(centres_of(legacy)).tolist(), legacy20_inertia=float(legacy.inertia_),
                max_centre_shift_other_random_states_deg=unique_shift,
                rule_is_global_best=bool(rule.inertia_ <= best.inertia_ + 1e-9), global_best_centres_deg=np.degrees(centres_of(best)).tolist(),
                competing_centres_deg=list(alt[1]), competing_inertia=float(alt[0]), inertia_gap_pct=float(100 * (alt[0] - rule.inertia_) / rule.inertia_) if alt[1] else math.nan,
                boot_ranges_deg=[[float(boots[:, j].min()), float(boots[:, j].max())] for j in range(3)],
                boot_middle_iqr_deg=[float(np.percentile(boots[:, 1], 25)), float(np.percentile(boots[:, 1], 75))])
