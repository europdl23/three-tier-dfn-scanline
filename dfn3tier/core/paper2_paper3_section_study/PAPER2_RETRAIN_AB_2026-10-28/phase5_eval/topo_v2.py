"""V2 COPY of topology_correction_validation_2026-10-22/code/topo_evaluate.py: imports pde_v2 and reads rho from V2_CONST."""
"""Topology-correction validation job (guarded EVALUATE mode on a Plan D eval stage; frozen models; no training).
Mirrors plan_d_evaluate.task_recon exactly for the original arms (B0, E0) and adds three variants per arm:
  *_len   Tier 2 continuation-length correction. Statistical (B): Change 1, the per-side extension is drawn from the fold's
          training continuation distances of the clip's cluster (inverse-CDF at the SAME random index as the original draw, so the
          random stream and hence the Tier 3 candidate pool are unchanged). EVAE (E): Change 2, predictive median exp(mu) instead
          of exp(mu + 0.67 sigma); floor and caps unchanged.
  *_thin  Change 3, crossing thinning in Tier 3 greedy selection: a candidate with n > 0 proper crossings against all placed traces
          (Tier 1/2 + already selected) is kept only if u < rho**n; u is a keyed uniform of (window, generation seed, candidate);
          rho is the fold's training-only constant.
  *_both  Changes 1 (or 2) + 3.
PLAN_D_TASK JSON: fold, training_seed, layout ('L0'|'L3'), arms (subset of the 8 labels)."""
import os, sys, json, math, hashlib, inspect, ast, re
from pathlib import Path
TC = Path(__file__).resolve().parent.parent; STUDY = TC.parent; S3 = STUDY / 'plan_d_step3_training_evaluation_2026-10-21'
sys.path.insert(0, str(S3)); sys.path.insert(0, str(S3 / 'engine'))
sys.path.insert(0, os.environ['V2_DIR']); import pde_v2 as PE  # noqa: V2 copy
from pde_v2 import *  # noqa
TASK = PE.TASK; OUTD = PE.OUTD; FOLD = PE.FOLD
CONST = json.loads((Path(os.environ['V2_CONST']) / f'{FOLD}.json').read_text())   # V2 copy: training-only rho of the new fold
FRZ = {}
pass   # V2 copy: no Paper 2 freeze for the new folds
ARMS = ['B0', 'B_len', 'B_thin', 'B_both', 'E0', 'E_len', 'E_thin', 'E_both']


# ------------------------------------------------------------------ Change 1 / 2: Tier 2 continuation
def make_indicated_len(e, cont, checked):
    src = ast.unparse(next(n for n in ast.parse((STUDY / 's1_s2_repair_trial_2026-10-18/repair_trial.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'indicated'))
    old_s = "extension = float(pool[int(rng.integers(len(pool)))]) / face['w']"; old_e = "extension = math.exp(min(float(m) + 0.67 * max(float(l.exp()), 0.05), 2.0))"
    assert src.count(old_s) == 1 and src.count(old_e) == 1, 'Tier 2 targets not found exactly once'
    ind = re.search(r'\n( *)' + re.escape(old_s), src).group(1)
    src = src.replace(old_s, "_ix = int(rng.integers(len(pool)))\n" + ind + "extension = float(np.quantile(CONT_POOL[fam], (_ix + 0.5) / len(pool))) / face['w']")
    src = src.replace(old_e, 'extension = math.exp(min(float(m), 2.0))').replace('def indicated(', 'def indicated_len(')
    pools = {k: np.asarray(cont[str(k)] if str(k) in cont and len(cont[str(k)]) >= 20 else cont['all'], float) for k in range(4)}
    g = dict(e.env, CONT_POOL=pools); exec(src, g); fn = g['indicated_len']

    def with_layout(obs, face, strips, mode, models, rng):
        previous = getattr(checked, 'strips', None); checked.strips = strips
        try: return fn(obs, face, strips, mode, models, rng)
        finally:
            if previous is None: del checked.strips
            else: checked.strips = previous
    return with_layout, hashlib.sha256(src.encode()).hexdigest()


# ------------------------------------------------------------------ Change 3: crossing thinning
def n_cross(row, placed):
    """Strict proper crossings (interior of both segments) of one candidate against placed traces (window units)."""
    if not len(placed): return 0
    a, b = e_ends(row[None]); P, Q = e_ends(placed); u = b[0] - a[0]; v = Q - P
    den = u[0] * v[:, 1] - u[1] * v[:, 0]; ok = np.abs(den) > 1e-12; d = P - a[0]; dd = np.where(ok, den, 1)
    s = (d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0]) / dd; t = (d[:, 0] * u[1] - d[:, 1] * u[0]) / dd
    return int((ok & (s > 1e-9) & (s < 1 - 1e-9) & (t > 1e-9) & (t < 1 - 1e-9)).sum())


def e_ends(a):
    a = np.asarray(a, float).reshape(-1, 4); d = np.c_[np.cos(a[:, 3]), np.sin(a[:, 3])] * a[:, 2:3] / 2
    return a[:, :2] - d, a[:, :2] + d


def make_thin(rho, fid, g, log):
    def thin(row, placed):
        n = n_cross(np.asarray(row, float), np.asarray(placed, float).reshape(-1, 4)) if len(placed) else 0
        if n == 0: return None
        h = hashlib.sha256(f'{fid}|{g}|thin|'.encode() + np.asarray(row, np.float32).tobytes()).digest()
        u = int.from_bytes(h[:8], 'little') / 2 ** 64; keep = u < rho ** n; log.append(dict(n=n, keep=bool(keep)))
        return None if keep else 'crossing_thinned'
    return thin


def thin_patch(src):
    lines = src.split('\n'); idx = [i for i, l in enumerate(lines) if l.strip().startswith("reason = 'family_quota' if pc[f] >= need[f] else duplicate_or_spacing(")]
    assert len(idx) == 1, 'thinning insertion point not found exactly once'
    ind = re.match(r' *', lines[idx[0]]).group(0)
    lines[idx[0] + 1:idx[0] + 1] = [ind + 'if reason is None:', ind + '    reason = THIN(row, placed)']
    return '\n'.join(lines)


def make_replay_thin():
    src = inspect.getsource(X.replay)
    a = "    oldfeat=r.scorer['extract_features']"; b = "REPLAY_WEIGHT_FACTOR=weight[1] if weight else 1)"
    assert src.count(a) == 1 and src.count(b) == 1 and src.startswith('def replay(')
    src = src.replace(a, "    src=_THIN_PATCH(src)\n" + a).replace(b, "REPLAY_WEIGHT_FACTOR=weight[1] if weight else 1,THIN=_THIN_FN)").replace('def replay(', 'def replay_thin(_THIN_FN,', 1)
    g = dict(vars(X), _THIN_PATCH=thin_patch); exec(src, g); return g['replay_thin'], hashlib.sha256(src.encode()).hexdigest()


# ------------------------------------------------------------------ job
def task(e):
    r = e.r; seed = int(TASK['training_seed']); layout = TASK['layout']; arms = TASK['arms']; assert set(arms) <= set(ARMS)
    R, stats, _ = PE.refs('outer_full'); cp = R['count_rule']; checked, _ = install(e); models, nets = PE.load_models(e, 'outer_full', seed); X.HERE = OUTD
    rho = float(CONST['rho']); cont = CONST['cont_' + layout]
    ind_len, ind_sha = make_indicated_len(e, cont, checked); replay_thin, rep_sha = make_replay_thin()
    phase = f'{layout}_on'; Mt, C, A = [], [], []
    wins = sorted(k for k, f in e.faces.items() if f['role'] == 'validation')
    for fid in wins:
        face = e.faces[fid]; pub = public_face(face)
        for g in [72001, 72002, 72003]:
            if layout == 'L0': obs, parents, st, _ = e.observe(face, '5m', 'clean', g)
            else:
                st = [((10 - .05) / 20, (10 + .05) / 20)] if int(round(face['x0'])) % 60 == 0 else []
                obs, parents = r.observe(face['arr'], st) if st else (np.zeros((0, 4), np.float32), np.zeros(0, int))
            e.s.corrected_references(r, e.base, pub, e.cm)
            k = len(st); p21 = (len(obs) + cp['a_prior']) / (k * face['h'] * cp['e_v'] + cp['a_prior'] / cp['prior']); T = max(1, int(round(cp['n_rate'] * 20 * face['h'] * (p21 / cp['prior']))))
            budget = 24 * min(128, math.ceil(1.5 * T)) + 10 * T; m = math.ceil(budget / 24)
            # original Tier 2 (task_recon order and streams)
            rng = np.random.default_rng(g); fixed_e = e.env['indicated'](obs, pub, st, 'evae', models, rng); stat_c = r.statistical_proposals(pub, budget, 'stat', rng)
            rng2 = np.random.default_rng(g); fixed_s = e.env['indicated'](obs, pub, st, 'stat', models, rng2); stat_b = r.statistical_proposals(pub, budget, 'stat', rng2)
            # corrected Tier 2 (same streams; the Tier 3 statistical pool must be identical)
            rngL = np.random.default_rng(g); fixed_e_len = ind_len(obs, pub, st, 'evae', models, rngL); stat_cL = r.statistical_proposals(pub, budget, 'stat', rngL)
            rngL2 = np.random.default_rng(g); fixed_s_len = ind_len(obs, pub, st, 'stat', models, rngL2); stat_bL = r.statistical_proposals(pub, budget, 'stat', rngL2)
            assert np.array_equal(stat_b, stat_bL) and np.array_equal(stat_c, stat_cL), 'Tier 3 pool changed by the Tier 2 correction'
            base = dict(face=face, pub=pub, obs=obs, parents=parents, st=st, fid=fid, seed=g, budget=budget, training_seed=seed)
            qs = {}
            for lab in arms:
                arm = lab[0]; lenfix = lab.endswith('_len') or lab.endswith('_both'); thin = lab.endswith('_thin') or lab.endswith('_both')
                b = dict(base, fixed_e=fixed_e_len if lenfix else fixed_e, fixed_s=fixed_s_len if lenfix else fixed_s)
                if arm == 'E': q = PE.tf_pool(e, b, models, nets['tf_latent'], True, lab, 'E')
                else:
                    fx = b['fixed_s']; q = PE.make_q(e, b, lab, 'B', stat_b, fx, np.ones(len(stat_b), bool), r.assign(stat_b, e.base['fc']), np.full(len(stat_b), -1), ['stat'] * len(stat_b))
                    gg = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{fid}|{g}|B|equal_m'.encode()).digest()[:8], 'little'))
                    ids = np.flatnonzero(q['valid']); keep = gg.choice(ids, min(m, len(ids)), replace=False); el = np.zeros(len(stat_b), bool); el[keep] = True; q['eligible'] = el
                qs[lab] = (q, thin, lenfix)
            checked.strips = st
            try:
                for lab, (q, thin, lenfix) in qs.items():
                    q['meta']['label'] = lab; q['meta']['case_key'] = f"{fid}__{lab}__t{seed}__g{g}"; tlog = []
                    if thin: f_, i_, l_, bud = replay_thin(make_thin(rho, fid, g, tlog), e, q, policy='slotsafe', eligibility='native', tail=True, keyed=True, target=T, scoring=True)
                    else: f_, i_, l_, bud = X.replay(e, q, policy='slotsafe', eligibility='native', tail=True, keyed=True, target=T, scoring=True)
                    met, cl = X.save_case(e, q, lab, phase, f_, i_, l_, bud)
                    dest = OUTD / phase / 'cases' / (lab + '__' + q['meta']['case_key'])
                    if 'network_log' in q: pd.DataFrame(q['network_log']).to_csv(dest / 'network_steps.csv', index=False)
                    fx = q['fixed']; ext = dict(n_fixed=len(fx), fixed_len_sum_m=float(fx[:, 2].sum() * 20) if len(fx) else 0.)
                    info = dict(T=T, raw_budget=budget, m=m, strips=len(st), n_obs=len(obs), unique_observed=int(len(np.unique(parents))), layout=layout, scoring=True,
                                eligible_valid=int((q['valid'] & q['eligible']).sum()), arm=lab[0], variant=lab, training_seed=seed, generation_seed=g, rho=rho if thin else None,
                                tier2_correction=lenfix, thinning=thin, thin_tests=len(tlog), thin_rejected=int(sum(not x['keep'] for x in tlog)),
                                stop_reason=str(l_.attrs.get('stop_reason', '')) if hasattr(l_, 'attrs') else '', **ext,
                                indicated_len_sha=ind_sha, replay_thin_sha=rep_sha, accepted=int((l_.disposition == 'accepted').sum()), thinned=int((l_.disposition == 'crossing_thinned').sum()))
                    dump(dest / 'case_inputs.json', info)
                    met.update(label=lab, arm=lab[0], variant=lab, layout=layout, T_target=T, raw_budget=budget, m=m, eligible_valid=info['eligible_valid'], n_obs=len(obs),
                               unique_observed=info['unique_observed'], n_strips=len(st), thinned=info['thinned'], accepted=info['accepted'], **ext); Mt.append(met)
                    for x in cl: x.update(label=lab, arm=lab[0], layout=layout)
                    C.extend(cl)
            finally:
                del checked.strips
        print(fid, flush=True)
    OUTD.mkdir(parents=True, exist_ok=True); pd.DataFrame(Mt).to_csv(OUTD / f'CASE_METRICS_{phase}_t{seed}.csv', index=False); pd.DataFrame(C).to_csv(OUTD / f'CLUSTER_METRICS_{phase}_t{seed}.csv', index=False)


if __name__ == '__main__':
    e = build_engine()
    assert all(f['role'] in ('train', 'validation') for f in e.faces.values())
    task(e)
    dump(OUTD / f"COMPLETE_{TASK['training_seed']}_{TASK['layout']}_{'-'.join(TASK['arms'])}.json", dict(task=TASK, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
