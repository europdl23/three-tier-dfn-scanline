"""Follow-up evaluation driver (2026-10-28): runs phase5_eval/fbench_v2.py UNCHANGED except for three declared switches, each applied by exact
string patching (assert-checked) so the code path is otherwise identical to the A/B evaluation:
  V3_ALPHA      Transformer angular-contrast weight in angle_sampler (fbench_v2/pde_v2 default 0.5; 0 = family centre only, 1 = network angle weighted equally)
  V3_SCORING    1/0: geological scorer ON (default, as fbench_v2) or OFF in the acceptance replay
  V3_TSCALE     JSON {arm: factor}: per-arm multiplier on the count target T at acceptance (raw candidate budget unchanged, so pools are identical)
  V3_TAG        output phase name: cases go to <out>/<layout>_<tag>/cases, marker COMPLETE_<seed>_<layout>_<tag>.json
  V3_ROLL       (added for phase8_tf_position, PREREGISTRATION_TF_POSITION.md) Transformer rollout restart: the left-to-right sequence is reset to the start
                token every L = max(1, round(V3_ROLL * (T - n_fixed))) generated steps (in addition to the original HORIZON reset). 'inf' = never (regression
                test: must reproduce fbench_v2 byte-identically). Unset = original sampler.
Models, epochs, seeds (training 42001, generation 72001-3), stages, rho and benchmark pools are those of the A/B evaluation."""
import os, sys, json, types
from pathlib import Path
HERE = Path(__file__).resolve().parent; V2 = HERE.parent / 'phase5_eval'
ALPHA = float(os.environ.get('V3_ALPHA', '0.5')); SCORING = os.environ.get('V3_SCORING', '1') == '1'; TS = json.loads(os.environ.get('V3_TSCALE', '{}')); TAG = os.environ['V3_TAG']
src = (V2 / 'fbench_v2.py').read_text()
rep = [("phase = f'{spec}_on'", "phase = f'{spec}_' + V3_TAG"),
       ("keyed=True, target=T, scoring=True)", "keyed=True, target=max(1, int(round(T * V3_TS.get(lab, 1.0)))), scoring=V3_SCORING)"),
       ("dump(OUTD / f\"COMPLETE_{TASK['training_seed']}_{TASK['layout']}.json\"", "dump(OUTD / f\"COMPLETE_{TASK['training_seed']}_{TASK['layout']}_{V3_TAG}.json\""),
       ("budget=budget, fixed_e=fixed_e", "budget=budget, T=T, fixed_e=fixed_e")]
for a, b in rep:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
src = src.replace("if __name__ == '__main__':", "def _v3_main():", 1)
sys.path.insert(0, str(V2)); mod = types.ModuleType('fbench_v3_body'); mod.__file__ = str(V2 / 'fbench_v2.py')
mod.__dict__.update(V3_TAG=TAG, V3_TS=TS, V3_SCORING=SCORING); exec(compile(src, 'fbench_v3_body', 'exec'), mod.__dict__)
_orig = mod.PE.angle_sampler
mod.PE.angle_sampler = lambda e, alpha, case: _orig(e, ALPHA, case)   # tf_pool calls angle_sampler(e, .5, ...): only the weight changes
TDIR = os.environ.get('V3_T')
if TDIR:
    # phase 9 (PREREGISTRATION_SPATIAL_LEARNING.md): Transformer nets from phase9_spatial/<cfg>/runs/<tag>/<fold>/outer_full (epochs V3_TSEL); EVAE / head unchanged.
    # space_vec and SpatialTransformer are exec'd from t_train.py (single definition); residuals / horizon of the evaluated layout (S05ref -> 5m, S20o10 -> centre).
    import ast, torch
    TT = Path(__file__).resolve().parents[1] / 'phase9_spatial' / 't_train.py'; tree = ast.parse(TT.read_text()); defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    import math; tenv = dict(np=mod.np, torch=torch, nn=torch.nn, math=math, NB=40, K_MIX=8, LOG2PI=math.log(2 * math.pi)); exec(compile(ast.Module([defs['space_vec'], defs['spatial_net'], defs['mdn_draws']], []), str(TT), 'exec'), tenv)
    LAYKEY = {'S05ref': '5m', 'S20o10': 'centre'}.get(mod.TASK['layout'], mod.TASK['layout']); TSEL = json.loads(Path(os.environ['V3_TSEL']).read_text())['selected'][mod.FOLD]
    _lm = mod.PE.load_models
    def load_models_t(e, treatment, seed, epochs=None):
        models, nets = _lm(e, treatment, seed, epochs)
        for c, lat in [('tf_latent', True), ('tf_no_latent', False)]:
            d = Path(TDIR) / mod.FOLD / 'outer_full' / f'{c}_s{seed}'
            if not d.exists(): continue
            sd = torch.load(d / f'epoch_{TSEL[c]:03d}.pt', map_location=e.r.DEV, weights_only=True)['state_dict']   # T1-MDN detected from its mixture head
            n = tenv['spatial_net'](e, any(k.startswith('mdn_head') for k in sd))().to(e.r.DEV); n.latent = lat; n.load_state_dict(sd); n.eval()
            with mod.np.load(d / 'residuals.npz') as z: errs = {k: z[k] for k in z.files}
            nets[c] = dict(net=n, errors=errs, horizons=json.loads((d / 'CALIBRATION.json').read_text())['horizons'])
        return models, nets
    mod.PE.load_models = load_models_t
    src = ast.unparse(next(n for n in ast.parse(Path(mod.PE.__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'tf_pool'))
    for a, b in [("net.aspect = torch.tensor([b['pub']['ym']], device=r.DEV)", "net.aspect = torch.tensor([b['pub']['ym']], device=r.DEV)\n    net.space = torch.tensor(V3_SV(b['st'], b['pub']['ym'])[None], device=r.DEV)"),
                 ("errors=netinfo['errors'], layout='5m'", "errors=V3_ERR(netinfo['errors']), layout=V3_LAYKEY"), ("e.env['posterior_draws'] = logging_posterior(", "e.env['posterior_draws'] = V3_MDN(e, net, b, logging_posterior("), ("(b['fid'], b['seed'], 'angular_contrast')), log)", "(b['fid'], b['seed'], 'angular_contrast')), log))"), ("netinfo['horizons']['5m']", "netinfo['horizons'][V3_LAYKEY]")]:
        assert src.count(a) == 1, a; src = src.replace(a, b)
    def V3_MDN(e, net, b, post):
        if not getattr(net, 'mdn_on', False): return post
        rng = mod.np.random.default_rng(int.from_bytes(hashlib.sha256(f"{b['fid']}|{b['seed']}|mdn".encode()).digest()[:8], 'little')); ym = b['pub']['ym']
        def p2(position, loglength, angle, logits, face, rr, n):
            pos = mod.np.asarray(position) / [1, ym]; dr = tenv['mdn_draws'](net, rng, 256) - pos; e.current['errors'] = {LAYKEY: dr, '5m': dr}
            return post(position, loglength, angle, logits, face, rr, n)
        return p2
    import hashlib
    V3_ERR = lambda er: dict(er, **{'5m': er[LAYKEY]}) if os.environ.get('V3_LAYRES', '1') == '1' else er   # V3_LAYRES=0 reproduces the first T1 evaluation (5m residuals)
    mod.PE.__dict__.update(V3_SV=tenv['space_vec'], V3_LAYKEY=LAYKEY, V3_MDN=V3_MDN, V3_ERR=V3_ERR); exec(compile(src, str(mod.PE.__file__) + ' [V3_T]', 'exec'), mod.PE.__dict__)
ROLL = os.environ.get('V3_ROLL')
if ROLL:
    GEN = Path((__import__('os').environ['DFN3_CORE'] + r'/paper2_paper3_section_study/spatial_placement_fix_trial_2026-10-19/supported_sampled_generator.py'.replace(chr(92), '/')))
    _tf = mod.PE.tf_pool

    def tf_pool_roll(e, b, models, netinfo, latent, label, arm):
        fn = e.env['sample_neural']; g = fn.__globals__
        assert fn.__name__ == 'sample_neural_sampled' and Path(fn.__code__.co_filename).name == GEN.name, 'unexpected sampler'
        if 'sample_neural_rollout' not in g:
            s = GEN.read_text(); a = 'if seq.size(1) > HORIZON:'; assert s.count(a) == 1 and s.count('def sample_neural_sampled(') == 1
            exec(compile(s.replace(a, 'if seq.size(1) > HORIZON or seq.size(1) > V3_ROLL_L:').replace('def sample_neural_sampled(', 'def sample_neural_rollout('), str(GEN) + ' [V3_ROLL]', 'exec'), g)
        nf = len(b['fixed_e'] if latent else b['fixed_s'])
        g['V3_ROLL_L'] = 10 ** 9 if ROLL == 'inf' else max(1, int(round(float(ROLL) * (b['T'] - nf))))
        e.env['sample_neural'] = g['sample_neural_rollout']
        try: return _tf(e, b, models, netinfo, latent, label, arm)
        finally: e.env['sample_neural'] = fn
    mod.PE.tf_pool = tf_pool_roll
if __name__ == '__main__':
    mod._v3_main()
