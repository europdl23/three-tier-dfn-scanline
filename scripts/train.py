"""Trains the learned components of one or more wall groups in the order EVAE -> continuation head -> transformers.

    python scripts/train.py [--folds all|S1_EAST,...] [--seed 42001] [--work work/models] [--select] [--epochs N] [--windows N]

Default: the final models are retrained on the full training stage of each wall group for the selected epochs in models/SELECTED_EPOCHS.json
(EVAE: selected epoch; continuation head and transformers: 50 epochs, the selected epoch is used and the transformer position residuals are
calibrated at it). --select first repeats the epoch selection on the inner block of each wall group (earliest minimum of the inner validation
curve; caps 150 / 50 / 50 / 50 epochs) and writes work/models/SELECTED_EPOCHS.json. --epochs N and --windows N are for quick tests only.
The result folder has the layout of models/ and can be passed to reconstruct.py with --models."""
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from _common import P, subset_stage, run, folds, dump

CAP = dict(evae=150, head=50, tf_latent=50, tf_no_latent=50)
JOB = P.ENGINE / 'plan_d_train.py'


def job(stage, out, comp, mode, epochs, save_every, seed, log, **kw):
    if (Path(out) / 'COMPLETE.json').exists() and (not kw.get('calibrate_epoch') or (Path(out) / 'CALIBRATION.json').exists()): return
    spec = dict(name=str(out), component=comp, seed=seed, mode=mode, epochs=int(epochs), save_every=save_every, **kw)
    run(JOB, dict(PLAN_D_STAGE=stage, PLAN_D_OUT=out, PLAN_D_MODE='train', PLAN_D_JOB=json.dumps(spec), PLAN_D_GPU_FRACTION='0.1'), log)


def ck(d, ep): return (Path(d) / f'epoch_{int(ep):03d}.pt').as_posix()


def stage_of(f, sub, n, work):
    st = P.DATA / 'stages' / f / sub
    return subset_stage(st, n, work / '_stages' / f / sub, 'train') if n else st


def select(f, seed, work, n):
    ib = pd.read_csv(P.DATA / 'INNER_BLOCKS.csv'); ib = ib[ib.selected & (ib.fold == f)]; blk = ib.sort_values('validation_windows', ascending=False).iloc[0].block
    st = P.DATA / 'stages' / f / 'inner' / blk; R = work / 'runs' / f / 'inner' / blk; L = work / 'logs'; sel = {}
    if n: st = subset_stage(st, n, work / '_stages' / f / 'inner' / blk, 'train')
    curve = lambda c: int(np.argmin(pd.read_csv(R / c / 'curve.csv').common_validation.to_numpy()) + 1)
    job(st, R / 'evae', 'evae', 'select', CAP['evae'], True, seed, L / f'{f}_inner_evae.log'); sel['evae'] = curve('evae')
    job(st, R / 'head', 'head', 'select', CAP['head'], True, seed, L / f'{f}_inner_head.log', evae_state=ck(R / 'evae', sel['evae'])); sel['head'] = curve('head')
    for c in ['tf_latent', 'tf_no_latent']:
        job(st, R / c, c, 'select', CAP[c], False, seed, L / f'{f}_inner_{c}.log', evae_state=ck(R / 'evae', sel['evae']), head_state=ck(R / 'head', sel['head'])); sel[c] = curve(c)
    return sel


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--folds', default='all'); a.add_argument('--seed', type=int, default=42001); a.add_argument('--work', default='work/models')
    a.add_argument('--select', action='store_true'); a.add_argument('--epochs', type=int, default=0); a.add_argument('--windows', type=int, default=0); x = a.parse_args()
    work = Path(x.work).resolve(); SEL = json.loads((P.MODELS / 'SELECTED_EPOCHS.json').read_text())
    out_sel = json.loads((work / 'SELECTED_EPOCHS.json').read_text()) if (work / 'SELECTED_EPOCHS.json').exists() else dict(selected={})
    for f in folds(x.folds):
        sel = select(f, x.seed, work, x.windows) if x.select else dict(SEL['selected'][f])
        cap = dict(CAP)
        if x.epochs: sel = {c: x.epochs for c in sel}; cap = {c: x.epochs for c in cap}
        st = stage_of(f, 'outer_full', x.windows, work); R = work / 'runs' / f / 'outer_full'; L = work / 'logs'; s = x.seed
        job(st, R / f'evae_s{s}', 'evae', 'fixed', sel['evae'], False, s, L / f'{f}_evae_s{s}.log')
        job(st, R / f'head_s{s}', 'head', 'fixed', cap['head'], True, s, L / f'{f}_head_s{s}.log', evae_state=ck(R / f'evae_s{s}', sel['evae']))
        for c in ['tf_latent', 'tf_no_latent']:
            job(st, R / f'{c}_s{s}', c, 'fixed', cap[c], True, s, L / f'{f}_{c}_s{s}.log', evae_state=ck(R / f'evae_s{s}', sel['evae']),
                head_state=ck(R / f'head_s{s}', sel['head']), calibrate_epoch=sel[c])
        if not (R / 'references').exists(): shutil.copytree(P.MODELS / 'runs' / f / 'outer_full' / 'references', R / 'references')   # training-data references (seed independent)
        out_sel['selected'][f] = sel; dump(work / 'SELECTED_EPOCHS.json', out_sel); print('trained', f, sel, flush=True)
