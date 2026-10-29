"""Plan D Step 3 training job (runs in its own guarded process; PLAN_D_STAGE / PLAN_D_OUT set by the orchestrator).

Uses the R1 training functions (train_evae, train_head, train_tf) unchanged except for two declared Plan D adaptations:
  (1) epoch_job is replaced by plan_d_epoch_job: identical optimiser, data order, RNG and loss code; it can (a) run with
      validation each epoch and save every epoch's state (inner selection), or (b) run a fixed number of epochs with no
      validation (outer / half refits). Validation never touches the global RNG (EVAE validation forks the RNG), so a fixed
      run is identical to the same number of epochs of a selection run.
  (2) arm D: train_tf with latent=False (sequences(bundle, False) -> statistical context; net.latent = False).
Job spec (JSON in PLAN_D_JOB): component in {evae, head, tf_latent, tf_no_latent}, seed, mode in {select, fixed}, epochs,
paths of the upstream EVAE / head states, and save_every.
"""
import os, sys, json, math, time, hashlib, ast, random
from pathlib import Path
S3 = Path(__file__).resolve().parent; sys.path.insert(0, str(S3 / 'engine'))
import training as TR  # noqa: guarded runtime import happens here
from training import *  # noqa
import numpy as np, pandas as pd, torch  # noqa

JOB = json.loads(os.environ['PLAN_D_JOB']); OUTDIR = Path(os.environ['PLAN_D_OUT']).resolve()


def statesha(sd):
    h = hashlib.sha256()
    for k in sorted(sd): h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def plan_d_epoch_job(run, e, name, net, train_epoch, validation, epochs, lr, seed, settings, diagnostic=None, wd=1e-5):
    epochs = int(JOB['epochs']); mode = JOB['mode']; dest = OUTDIR; dest.mkdir(parents=True, exist_ok=True)
    conf = dict(job=JOB, original_name=name, epochs=epochs, lr=lr, weight_decay=wd, seed=seed, settings={k: v for k, v in settings.items() if k not in ('Xmean', 'Xstd')},
                initial_state=statesha(net.state_dict()), sources={p.name: sha(p) for p in sorted((S3 / 'engine').glob('*.py'))} | {'plan_d_train.py': sha(Path(__file__))},
                inputs={p.name: sha(p) for p in sorted(STAGE.iterdir()) if p.is_file()})
    if 'Xmean' in settings: conf['head_normaliser'] = dict(Xmean=settings['Xmean'], Xstd=settings['Xstd'])
    dump(dest / 'config.json', conf)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd); logs = []; best = math.inf; selected = 0
    torch.save(dict(state_dict=TR.cpu_state(net), epoch=0), dest / 'epoch_000.pt'); hashes = {0: sha(dest / 'epoch_000.pt')}
    t0 = time.monotonic()
    for epoch in range(1, epochs + 1):
        net.train(); tv = train_epoch(net, opt, epoch); net.eval(); row = dict(epoch=epoch, **{k: v for k, v in tv.items()})
        if mode == 'select':
            with torch.no_grad(): v, comp = validation(net, epoch)
            if not math.isfinite(v): raise FloatingPointError('Nonfinite validation loss')
            row.update(common_validation=v, **{'validation_' + k: float(x) for k, x in comp.items()})
            if v < best: best, selected = v, epoch
        logs.append(row); pd.DataFrame(logs).to_csv(dest / 'curve.csv', index=False)
        if JOB.get('save_every', True) or epoch == epochs:
            p = dest / f'epoch_{epoch:03d}.pt'; torch.save(dict(state_dict=TR.cpu_state(net), epoch=epoch), p); hashes[epoch] = sha(p)
        if epoch % 10 == 0: print(JOB['name'], epoch, flush=True)
    rec = dict(job=JOB['name'], mode=mode, epochs=epochs, selected_epoch_within_run=selected if mode == 'select' else None, best_validation=best if mode == 'select' else None,
               final_state_sha=statesha(net.state_dict()), checkpoint_sha=hashes, elapsed_s=time.monotonic() - t0)
    dump(dest / 'COMPLETE.json', rec); return net, rec


TR.epoch_job = plan_d_epoch_job
# arm D: train_tf with latent=False (only the two latent literals change; everything else is the R1 function)
_src = ast.unparse(next(n for n in ast.parse((S3 / 'engine/training.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'train_tf'))
assert _src.count('e.sequences(bundle, True)') == 1 and _src.count('net.latent = True') == 1
_src = _src.replace('def train_tf(', 'def train_tf_no_latent(').replace('e.sequences(bundle, True)', 'e.sequences(bundle, False)').replace('net.latent = True', 'net.latent = False')
exec(compile(_src, 'train_tf_no_latent', 'exec'), TR.__dict__); train_tf_no_latent = TR.__dict__['train_tf_no_latent']


class Run:
    def __init__(self): self.state = dict(phase='plan_d', progress='', training_jobs=0)
    def guard(self, estimate=0): pass
    def log(self, *a, **k): pass


def load_evae(e, path):
    ev = make_evae(e, True); ev.load_state_dict(torch.load(path, map_location=e.r.DEV, weights_only=True)['state_dict']); ev.eval(); return ev


def head_normaliser(e, ev):
    r = e.r; X = []
    for z in e.train:
        latz = torch.cat(r.tensor_encode(ev, z['obs']), -1).detach().cpu().numpy()[0]
        for members in e.env['anchor_groups'](z['obs'], public_face(z['face']), z['strips']):
            for i, side in e.env['group_outer_observations'](z['obs'], members):
                feat, a, u, bound = e.env['continuation_features'](z['obs'], i, side, z['face'])
                if any(abs(a[0] - edge) < 1e-6 for st in z['strips'] for edge in st) and bound > 1e-8: X.append(np.r_[latz, feat])
    X = torch.tensor(np.asarray(X), device=r.DEV); return X.mean(0), X.std(0).clamp_min(1e-3)


def main():
    e = build_engine(); r = e.r; run = Run(); comp = JOB['component']; seed = int(JOB['seed'])
    if JOB['mode'] == 'fixed' and not e.valid:
        e.valid = e.train[:2]   # placeholder so the R1 functions can build their (unused) validation closures; never evaluated in fixed mode
    if comp == 'evae':
        train_evae(run, e, 'E_LB', seed); return
    ev = load_evae(e, JOB['evae_state'])
    if comp == 'head':
        train_head(run, e, ev, seed); return
    hn = r.gen['EVAEContHead']().to(r.DEV); hn.load_state_dict(torch.load(JOB['head_state'], map_location=r.DEV, weights_only=True)['state_dict']); hn.eval()
    mu, sd = head_normaliser(e, ev)
    bundle = dict(seed=seed + 1000000 + (0 if comp == 'tf_latent' else 500000), dest=OUTDIR, models=dict(evae=ev, head=(hn, mu, sd))); e.bundles[bundle['seed']] = bundle
    latent = comp == 'tf_latent'
    (train_tf if latent else train_tf_no_latent)(run, e, bundle, 'T0', seed)
    if JOB.get('calibrate_epoch'):
        net = e.Net().to(r.DEV); net.latent = latent
        net.load_state_dict(torch.load(OUTDIR / f"epoch_{int(JOB['calibrate_epoch']):03d}.pt", map_location=r.DEV, weights_only=True)['state_dict']); net.eval()
        errors, horizons = e.calibrate(bundle, net, latent, OUTDIR / 'residuals.npz')
        dump(OUTDIR / 'CALIBRATION.json', dict(epoch=int(JOB['calibrate_epoch']), horizons=horizons, residual_sha=sha(OUTDIR / 'residuals.npz'), latent=latent,
                                                note='teacher-forced position residuals on the stage training sequences only (engine.calibrate)'))


if __name__ == '__main__':
    main()
