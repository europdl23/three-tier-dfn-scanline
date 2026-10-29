"""F4 family and fitting-statistics freeze (JRMGE-D-26-02141 revision). No training, no model calls.

Fitting faces: W05, W06, WEST3 only (S2 excluded from training and calibration).
Original definitions are executed from the historical source text (AST-extracted function bodies from
run_ablation_v3.py and exp_common.py; the modules themselves are not imported):
  _train_geology_stats + _calibrate_scorer_weights (geology references and active scorer weights),
  assign_family_ids, split_panel, intersection_density.
Family centres follow run_revision.py::fit_stats (k-means k=3 on doubled axial angles, random_state 42001,
n_init 20); count rule follows experiments.py (T = max(1, round(rate * w * h))); hidden-family priors follow
training.py (family mix of strip-unobserved targets per layout). The same code reproduces the saved
W02-W06 statistics as a check. Writes only into this folder.
"""
import ast, contextlib, csv, hashlib, io, json, math, sys, types
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ROOT = Path((__import__('os').environ['DFN3_CORE'] + r''.replace(chr(92), '/')))
STUDY = ROOT / 'paper2_paper3_section_study'
EXP = ROOT / 'Updated__file' / 'improved_dfn_experiments'
D3 = STUDY / 'option3a_dataset_freeze_2026-10-20'
OLD = STUDY / 'authorised_execution_2026-10-19'
FIT = ('W05_RL0958-0985', 'W06_RL0938-0958', 'WEST3_RL0904-0922_100-160m')
ROLE_F4 = {'W05_RL0958-0985': 'training', 'W06_RL0938-0958': 'training', 'WEST3_RL0904-0922_100-160m': 'training',
           'W03_RL1012-1039': 'selection', 'W00_RL1102-1120': 'validation', 'W01_RL1066-1102': 'validation',
           'W02_RL1039-1066': 'excluded', 'W04_RL0985-1012': 'excluded',
           'S2_B00_RL0849-0876_P1': 'excluded_F4', 'S2_B01_RL0876-0903_P1': 'excluded_F4'}
INPUTS = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def rows(p):
    INPUTS[str(p)] = sha(p)
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def extract(path, names):
    INPUTS[str(path)] = sha(path)
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    return '\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)


# ---- original functions, executed from source text ------------------------------------------
env = dict(np=np, math=math, Sequence=list, Tuple=tuple, List=list)
exec(compile(extract(EXP / 'exp_common.py', ['_endpoints', 'line_endpoints', 'segments_intersect', 'intersection_density',
                                              'split_line_by_strips', 'split_panel', 'assign_family_ids']), 'exp_common_extract', 'exec'), env)
fake = types.ModuleType('exp_common'); fake.__dict__.update(env); sys.modules['exp_common'] = fake
exec(compile(extract(EXP / 'run_ablation_v3.py', ['_train_geology_stats', '_calibrate_scorer_weights']), 'ablation_extract', 'exec'), env)
assign, split_panel, train_geology_stats = env['assign_family_ids'], env['split_panel'], env['_train_geology_stats']


def strips(layout, w=20.0):  # run_revision.py::strips
    centres = [w / 2] if layout == 'centre' else np.arange(float(layout[:-1]), w, float(layout[:-1]))
    return [(max(0, (x - .05) / w), min(1, (x + .05) / w)) for x in centres]


def hidden(arr, st):  # observe(): a trace is observed if any clipped piece has its midpoint in a strip
    obs = [i for i, row in enumerate(arr) if len(env['split_line_by_strips'](row, st)[0])]
    return arr[[i for i in range(len(arr)) if i not in set(obs)]]


def windows(frag, inv, faces):
    out = []
    for w in (r for r in inv if r['parent_face'] in faces):
        a = np.array([[float(x['cx']), float(x['cy']), float(x['length']), float(x['angle'])] for x in frag if x['window_id'] == w['window_id']], np.float32).reshape(-1, 4)
        out.append(dict(id=w['window_id'], face=w['parent_face'], h=float(w['height_m']), arr=a))
    return out


def statistics(wins):
    arr = np.vstack([w['arr'] for w in wins])
    km = KMeans(3, random_state=42001, n_init=20).fit(np.c_[np.cos(2 * arr[:, 3]), np.sin(2 * arr[:, 3])])
    fc = np.sort((np.arctan2(km.cluster_centers_[:, 1], km.cluster_centers_[:, 0]) / 2) % np.pi)
    fam = assign(arr, fc)
    with contextlib.redirect_stdout(io.StringIO()):
        gs = train_geology_stats([w['arr'] for w in wins], fc)
        canon = []
        for w in wins:
            a = w['arr'].astype(float).copy(); a[:, :3] *= 20 / 36; canon.append(a)
        gsc = train_geology_stats(canon, fc)
    phys = arr[:, 2] * 20
    logmu, logsig, pools = {}, {}, {}
    for k in range(4):
        sub = arr[fam == k] if (fam == k).any() else arr
        logmu[k] = float(np.log(sub[:, 2]).mean()); logsig[k] = max(float(np.log(sub[:, 2]).std()), .1)
        lm = phys[fam == k] if (fam == k).any() else phys
        pools[k] = dict(n=int((fam == k).sum()), length_m_p50=float(np.median(lm)), length_m_p90=float(np.quantile(lm, .9)),
                        length_m_p995=float(np.quantile(lm, .995)))
        if k == 3:  # run_revision.py::fit_stats background extension
            gs['spacing_floor'][3] = .008; gs['count_std'][3] = 1.
            gs['centre_mass_ref'][3] = dict(cx_mean=float(sub[:, 0].mean()), cx_std=max(float(sub[:, 0].std()), .02),
                                            cy_mean=float(sub[:, 1].mean()), cy_std=max(float(sub[:, 1].std()), .02))
            gs['fam_length_ref'][3] = dict(p90=float(np.quantile(sub[:, 2], .9)), p95=float(np.quantile(sub[:, 2], .95)),
                                           mean=float(sub[:, 2].mean()), std=max(float(sub[:, 2].std()), .01))
    hp = {}
    for layout in ['centre', '5m']:
        f = np.concatenate([assign(hidden(w['arr'], strips(layout)), fc) for w in wins if len(hidden(w['arr'], strips(layout)))])
        c = np.bincount(f, minlength=4); hp[layout] = (c / c.sum()).tolist()
    rate = sum(len(w['arr']) for w in wins) / sum(20 * w['h'] for w in wins)
    return dict(fc=fc.tolist(), fc_deg=np.degrees(fc).tolist(), shares=(np.bincount(fam, minlength=4) / len(fam)).tolist(),
                family_fragments=np.bincount(fam, minlength=4).tolist(), rate=rate, yscale=max(w['h'] / 20 for w in wins),
                logmu=logmu, logsig=logsig, pools=pools, gs=gs, SPACING_METRES={**{k: float(gsc['spacing_floor'][k]) * 36 for k in range(3)}, 3: .288},
                TRAIN_X_STD=float(arr[:, 0].std()), TRAIN_Y_STD=float(arr[:, 1].std()), hidden_priors=hp,
                count_rule='T = max(1, round(rate * 20 * window_height_m))', windows=len(wins), fragments=int(len(arr)))


frag = rows(D3 / 'WINDOW_FRAGMENTS_3A.csv')
inv = rows(D3 / 'WINDOW_INVENTORY_3A.csv')
f4 = statistics(windows(frag, inv, FIT))
old_faces = ('W02_RL1039-1066', 'W03_RL1012-1039', 'W04_RL0985-1012', 'W05_RL0958-0985', 'W06_RL0938-0958')
repro = statistics(windows(frag, inv, old_faces))
saved = json.loads((OLD / 's1_inputs' / 'training_only_statistics.json').read_text())
INPUTS[str(OLD / 's1_inputs' / 'training_only_statistics.json')] = sha(OLD / 's1_inputs' / 'training_only_statistics.json')
saved_hp = json.loads((OLD / 'fit_hidden_priors.json').read_text())
INPUTS[str(OLD / 'fit_hidden_priors.json')] = sha(OLD / 'fit_hidden_priors.json')


def close(a, b, tol=1e-6):
    if isinstance(a, dict):
        return all(close(a[k], b[str(k)] if str(k) in b else b[k], tol) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)):
        return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)
    return abs(float(a) - float(b)) <= tol


repro_check = {}
for k in ['target_counts', 'count_std', 'spacing_floor', 'intersection_limit', 'intersection_mean', 'intersection_std',
          'centre_mass_ref', 'xbin_ref', 'xbin_std', 'fam_length_ref']:
    repro_check['gs.' + k] = bool(close(repro['gs'][k], saved['gs'][k]))
repro_check['gs.scorer_weights'] = bool(close({k: v for k, v in repro['gs']['scorer_weights'].items() if isinstance(v, (int, float))},
                                              {k: v for k, v in saved['gs']['scorer_weights'].items() if isinstance(v, (int, float))}))
for k in ['fc', 'shares', 'rate', 'yscale']:
    repro_check[k] = bool(close(repro[k], saved[k]))
repro_check['logmu'] = bool(close(repro['logmu'], saved['logmu']))
repro_check['hidden_priors'] = bool(close(repro['hidden_priors'], saved_hp, 1e-9))

# Frozen family assignment for every non-excluded fragment (applying frozen centres is not fitting)
fc = np.array(f4['fc'])
assigned = []
for r in frag:
    role = ROLE_F4[r['parent_face']]
    if role.startswith('excluded'):
        continue
    k = int(assign(np.array([[float(r['cx']), float(r['cy']), float(r['length']), float(r['angle'])]]), fc)[0])
    assigned.append(dict(window_id=r['window_id'], parent_face=r['parent_face'], role=role, trace_id=r['trace_id'],
                         angle_deg=round(math.degrees(float(r['angle'])), 4), family=['C1', 'C2', 'C3', 'background'][k]))
with open(HERE / 'F4_FRAGMENT_FAMILIES.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(assigned[0].keys())); w.writeheader(); w.writerows(assigned)
with open(HERE / 'F4_ROLES.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['face_id', 'role_F4']); w.writeheader(); w.writerows([dict(face_id=k, role_F4=v) for k, v in ROLE_F4.items()])

freeze = dict(stage='F4 family and fitting-statistics freeze; no training', fitting_faces=list(FIT), roles=ROLE_F4,
              settings=dict(kmeans_k=3, random_state=42001, n_init=20, assignment_threshold_deg=20, window_m=20, scale='x, y, length / 20 m; angle radians axial'),
              statistics=f4, reproduction_check_on_saved_W02_W06=repro_check,
              not_computable_without_trained_models=['trained_horizons.json and training_position_residuals.npz (from Transformer predictions, e.calibrate)',
                                                     'continuation-head input normaliser X_mean/X_std (depends on EVAE latents)',
                                                     'corrected F7 intersection recalibration (earlier status: uninformative, not promoted)'])
(HERE / 'F4_FREEZE.json').write_text(json.dumps(freeze, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)), encoding='utf-8')
prov = dict(inputs=INPUTS, outputs={o: sha(HERE / o) for o in ['F4_FREEZE.json', 'F4_FRAGMENT_FAMILIES.csv', 'F4_ROLES.csv']})
(HERE / 'F4_PROVENANCE.json').write_text(json.dumps(prov, indent=1), encoding='utf-8')
print('F4 centres', np.round(f4['fc_deg'], 2), 'shares', np.round(f4['shares'], 3), 'rate', round(f4['rate'], 5), 'yscale', round(f4['yscale'], 3))
print('hidden priors', {k: np.round(v, 3).tolist() for k, v in f4['hidden_priors'].items()})
print('spacing m', {k: round(v, 3) for k, v in f4['SPACING_METRES'].items()}, 'target counts', f4['gs']['target_counts'])
print('scorer weights', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f4['gs']['scorer_weights'].items()})
print('reproduction', repro_check)
