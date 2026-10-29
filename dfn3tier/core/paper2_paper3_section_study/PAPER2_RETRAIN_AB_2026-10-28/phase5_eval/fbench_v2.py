"""V2 copy of PAPER2_FINAL_COMPONENT_BENCHMARK_2026-10-28/fbench.py for the A/B retraining: evaluation modules topo_v2/pde_v2 (new runs, epochs, rho), new ADFNE pools, extra Hybrid arm E_hyb. PLAN_D_TASK adds cfg."""
"""Factorial Tier 2 x Tier 3 job (guarded EVALUATE mode, eval stage, frozen outer checkpoints; EXPERIMENT_PLAN.md).
Arms (labels): cKDE, cKDEs, cADFNE (statistical Tier 2 + benchmark Tier 3), eKDE, eKDEs, eADFNE (EVAE Tier 2 + the SAME benchmark pools), D_tf (statistical
Tier 2 + Transformer without latent, seeds != 42001 only; seed 42001 = submitted D_thin), T12 (Tier 1+2 statistical, no Tier 3 candidates; playground).
Benchmark samplers copied verbatim from kde_adfne_benchmark_revision_2026-10-21/code/h2h_generate.py (kde_rows, adfne_rows, training-only fits).
Observation layouts and thinning exactly as dfn_spacing_and_manuscript_review_2026-10-22/code/spacing_evaluate.py (submitted rho).
PLAN_D_TASK JSON: fold, training_seed, layout, arms."""
import sys, math, hashlib, json
from pathlib import Path
sys.path.insert(0, (__import__('os').environ['DFN3_CORE'] + r'/paper2_paper3_section_study/dfn_spacing_and_manuscript_review_2026-10-22/code'.replace(chr(92), '/')))
sys.path.insert(0, (__import__('os').environ['DFN3_CORE'] + r'/paper2_paper3_section_study/topology_correction_validation_2026-10-22/code'.replace(chr(92), '/')))
sys.path.insert(0, (__import__('os').environ['DFN3_CORE'] + r'/paper2_paper3_section_study/PAPER2_RETRAIN_AB_2026-10-28/phase5_eval'.replace(chr(92), '/'))); import topo_v2 as TE  # noqa (V2 copy)
from topo_v2 import *  # noqa
import layouts as LY  # noqa
from scipy.stats import gaussian_kde  # noqa
PE = TE.PE; TASK = PE.TASK; OUTD = PE.OUTD; FOLD = PE.FOLD; BM = Path((__import__('os').environ['DFN3_CORE'] + r'/paper2_paper3_section_study/kde_adfne_benchmark_revision_2026-10-21'.replace(chr(92), '/')))


def task(e):
    r = e.r; seed = int(TASK['training_seed']); spec = TASK['layout']; arms = TASK['arms']
    R, stats, _ = PE.refs('outer_full'); cp = R['count_rule']; checked, _ = install(e); models, nets = PE.load_models(e, 'outer_full', seed); X.HERE = OUTD
    rho = float(TE.CONST['rho']); replay_thin, rep_sha = TE.make_replay_thin(); phase = f'{spec}_on'; Mt = []
    # ---- training-only fits, verbatim from h2h_generate.py
    P = e.base['pools']; SH = np.asarray(e.base['shares'], float); FC = np.asarray(e.base['fc'], float); ADF = pd.read_csv(Path(__import__('os').environ['DFN3_ADFNE']) / f"{TASK['cfg']}_{FOLD}_lines.csv.gz")
    def axial_mean(a): return float((np.angle(np.exp(2j * np.asarray(a, float)).mean()) / 2) % np.pi)
    def axial_dev(a, mu): return (np.asarray(a, float) - mu + np.pi / 2) % np.pi - np.pi / 2
    MU = {k: axial_mean(P[k]['angles']) for k in range(4)}; ASD = {k: max(float(np.std(axial_dev(P[k]['angles'], MU[k]))), .01) for k in range(4)}
    TRAIN = [f for f in e.faces.values() if f['role'] == 'train']; KDEPOS = {}
    for k in range(4): KDEPOS[k] = gaussian_kde(np.vstack([f['arr'][r.assign(f['arr'], FC) == k][:, :2] / [1, f['ym']] for f in TRAIN]).T)
    def kde_rows(n, ym, rng, spatial):
        fam = rng['family'].choice(4, n, p=SH); rows = []
        for k in fam:
            pr = P[int(k)]; m_ = len(pr['angles']); h = m_ ** (-1 / 5); ix = int(rng['angle'].integers(m_))
            A = (float(pr['angles'][ix]) + rng['angle'].normal(0, ASD[int(k)] * h)) % np.pi
            L = math.exp(math.log(float(pr['lengths_m'][ix])) + rng['length'].normal(0, max(float(np.std(np.log(pr['lengths_m']))), .1) * h))
            if spatial:
                pt = None
                for _ in range(40):
                    c = KDEPOS[int(k)].resample(1, seed=rng['position']).reshape(2)
                    if 0 <= c[0] <= 1 and 0 <= c[1] <= 1: pt = c; break
                if pt is None: pt = np.array([rng['position'].uniform(), rng['position'].uniform()])
                cx, cy = float(pt[0]), float(pt[1]) * ym
            else: cx, cy = rng['position'].uniform(), rng['position'].uniform(0, ym)
            rows.append([cx, cy, L / 20, A])
        return np.asarray(rows, np.float32).reshape(-1, 4), fam
    def adfne_rows(n, ym, fid, g, purpose, rng):
        fam = rng['family'].choice(4, n, p=SH); sub = ADF[(ADF.window_id == fid) & (ADF.gen == g) & (ADF.purpose == purpose)]; rows = []; used = {k: 0 for k in range(4)}
        for k in fam:
            pool = sub[sub.cluster == int(k)].sort_values('idx'); j = used[int(k)]; used[int(k)] += 1
            if j >= len(pool): raise RuntimeError(f'ADFNE pool exhausted {fid} {g} {purpose} k={k}')
            x1, y1, x2, y2 = pool.iloc[j][['x1', 'y1', 'x2', 'y2']].to_numpy(float)
            rows.append([(x1 + x2) / 2, (y1 + y2) / 2 * ym, math.hypot(x2 - x1, y2 - y1), math.atan2(y2 - y1, x2 - x1) % np.pi])
        return np.asarray(rows, np.float32).reshape(-1, 4), fam
    # ----
    wins = sorted(k for k, f in e.faces.items() if f['role'] == 'validation')
    for fid in wins:
        face = e.faces[fid]; pub = public_face(face); ym = face['h'] / 20
        for g in [72001, 72002, 72003]:
            if spec == 'S05ref': obs, parents, st, _ = e.observe(face, '5m', 'clean', g)
            else:
                st = LY.strips(spec, float(face['x0'])); obs, parents = r.observe(face['arr'], st) if st else (np.zeros((0, 4), np.float32), np.zeros(0, int))
            e.s.corrected_references(r, e.base, pub, e.cm)
            k_ = len(st); p21 = (len(obs) + cp['a_prior']) / (k_ * face['h'] * cp['e_v'] + cp['a_prior'] / cp['prior']); T = max(1, int(round(cp['n_rate'] * 20 * face['h'] * (p21 / cp['prior']))))
            budget = 24 * min(128, math.ceil(1.5 * T)) + 10 * T; m = math.ceil(budget / 24)
            rng = np.random.default_rng(g); fixed_e = e.env['indicated'](obs, pub, st, 'evae', models, rng)
            rng2 = np.random.default_rng(g); fixed_s = e.env['indicated'](obs, pub, st, 'stat', models, rng2)
            b = dict(face=face, pub=pub, obs=obs, parents=parents, st=st, fid=fid, seed=g, budget=budget, fixed_e=fixed_e, fixed_s=fixed_s, training_seed=seed); qs = {}
            pools = {}
            for samp in ('KDE', 'KDEs', 'ADFNE'):
                if not any(a.endswith(samp) and a[0] in 'ce' for a in arms): continue
                rs = streams((fid, g, 'frame', 'F_' + samp))
                raw, fam = kde_rows(budget, ym, rs, samp == 'KDEs') if samp != 'ADFNE' else adfne_rows(budget, ym, fid, g, 'frame', rs)
                raw = e.s.clip_outer(r, raw, pub).astype(np.float32); valid = np.array([r.candidate_gate(x, pub, st) is None for x in raw])
                gg = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{fid}|{g}|F_{samp}|equal_m'.encode()).digest()[:8], 'little'))
                ids = np.flatnonzero(valid); keep = gg.choice(ids, min(m, len(ids)), replace=False); el = np.zeros(len(raw), bool); el[keep] = True; pools[samp] = (raw, valid, el)
            for lab in arms:
                if lab == 'D_tf': q = PE.tf_pool(e, b, models, nets['tf_no_latent'], False, lab, 'D')
                elif lab == 'E_hyb': q = PE.tf_pool(e, b, models, nets['tf_latent'], True, lab, 'E')
                elif lab == 'T12':
                    raw = np.zeros((0, 4), np.float32); q = PE.make_q(e, b, lab, 'B', raw, fixed_s, np.ones(0, bool), np.zeros(0, int), np.zeros(0, int), []); q['valid'] = np.zeros(0, bool); q['eligible'] = np.zeros(0, bool)
                else:
                    raw, valid, el = pools[lab[1:]]; fx = fixed_s if lab[0] == 'c' else fixed_e
                    q = dict(obs=obs, parents=parents, st=st, pub=pub, face=face, fixed=fx, raw=raw, eligible=el.copy(), sampled=r.assign(raw, FC), step=np.full(len(raw), -1), source=['stat'] * len(raw),
                             raw_sha=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(), valid=valid.copy(), rng=None, meta=dict(window_id=fid, generation_seed=g, arm='B', case_key='', training_seed=seed, label=lab))
                qs[lab] = q
            checked.strips = st
            try:
                for lab, q in qs.items():
                    q['meta']['label'] = lab; q['meta']['case_key'] = f"{fid}__{lab}__t{seed}__g{g}"; tlog = []
                    f_, i_, l_, bud = replay_thin(TE.make_thin(rho, fid, g, tlog), e, q, policy='slotsafe', eligibility='native', tail=True, keyed=True, target=T, scoring=True)
                    met, cl = X.save_case(e, q, lab, phase, f_, i_, l_, bud); dest = OUTD / phase / 'cases' / (lab + '__' + q['meta']['case_key'])
                    if len(q['raw']):
                        rd = pd.DataFrame(q['raw'], columns=['cx', 'cy', 'length', 'angle']); rd['valid'] = q['valid']; rd['eligible'] = q['eligible']; rd.to_csv(dest / 'raw_candidates.csv.gz', index=False)
                    dump(dest / 'case_inputs.json', dict(T=T, raw_budget=budget, m=m, variant=lab, training_seed=seed, generation_seed=g, rho=rho, layout=spec, n_fixed=len(q['fixed']),
                                                         accepted=int((l_.disposition == 'accepted').sum()) if len(l_) else 0, thinned=int((l_.disposition == 'crossing_thinned').sum()) if len(l_) else 0))
                    met.update(label=lab, variant=lab, layout=spec); Mt.append(met)
            finally:
                del checked.strips
        print(fid, flush=True)
    pd.DataFrame(Mt).to_csv(OUTD / f'CASE_METRICS_{phase}_t{seed}.csv', index=False)


if __name__ == '__main__':
    e = build_engine(); assert all(f['role'] in ('train', 'validation') for f in e.faces.values()); task(e)
    dump(OUTD / f"COMPLETE_{TASK['training_seed']}_{TASK['layout']}.json", dict(task=TASK, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
