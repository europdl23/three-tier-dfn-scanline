"""Reconstructs the held-out windows of each wall group with the five compared frameworks.

    python scripts/reconstruct.py [--arms hybrid,transformer,evae,ckde,cadfne] [--folds all|S1_EAST,...] [--training-seed 42001]
                                  [--models models] [--work work] [--windows N]

Each window is reconstructed with generation seeds 72001, 72002 and 72003. Output: work/<tag>/B/runs/<fold>/S10o5/S10o5_<tag>/cases/<case>/
(predictions.csv = reconstructed traces with their tier, metrics.json, proposal ledger and raw candidates). --windows N keeps only the first N
held-out windows of each wall group (quick test)."""
import argparse, json
from pathlib import Path
from _common import P, subset_stage, run, folds

# name -> (code label, output tag, driver, extra environment). Arms of the same tag run in one job, as in the study.
ARMS = {'hybrid': ('E_hyb', 'roll1off', 'fbench_v3', dict(V3_ROLL='1')),
        'transformer': ('D_tf', 'roll1off', 'fbench_v3', dict(V3_ROLL='1')),
        'evae': ('V_evae', 'evaeoff', 'fbench_v10', {}),
        'ckde': ('cKDE', 'scoreoff', 'fbench_v3', {}),
        'cadfne': ('cADFNE', 'scoreoff', 'fbench_v3', {})}
DRIVER = dict(fbench_v3=P.EXP / 'phase7_followup' / 'fbench_v3.py', fbench_v10=P.EXP / 'phase10_ablation' / 'fbench_v10.py')

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--arms', default='hybrid,transformer,evae,ckde,cadfne'); a.add_argument('--folds', default='all')
    a.add_argument('--training-seed', type=int, default=42001); a.add_argument('--models', default=str(P.MODELS)); a.add_argument('--work', default='work')
    a.add_argument('--windows', type=int, default=0); x = a.parse_args()
    work = Path(x.work).resolve(); models = Path(x.models).resolve(); groups = {}
    for name in x.arms.split(','):
        lab, tag, drv, extra = ARMS[name]; groups.setdefault((tag, drv, tuple(sorted(extra.items()))), []).append(lab)
    for f in folds(x.folds):
        stage = P.DATA / 'stages' / f / 'eval'
        if x.windows: stage = subset_stage(stage, x.windows, work / '_stages' / f / 'eval', 'validation')
        for (tag, drv, extra), labs in groups.items():
            tag_s = tag if x.training_seed == 42001 else f'{tag}_s{x.training_seed}'
            out = work / tag_s / 'B' / 'runs' / f / P.LAYOUT
            task = dict(fold=f, training_seed=x.training_seed, layout=P.LAYOUT, cfg='B', arms=labs)
            env = dict(PLAN_D_STAGE=stage, PLAN_D_OUT=out, PLAN_D_MODE='evaluate', PLAN_D_TASK=json.dumps(task), PLAN_D_GPU_FRACTION='0.1',
                       V2_RUNS=models / 'runs', V2_SEL=models / 'SELECTED_EPOCHS.json', V2_CONST=P.DATA / 'constants', V2_DIR=P.EXP / 'phase5_eval',
                       V3_TAG=tag_s, V3_SCORING='0', **dict(extra))
            print(f, tag_s, ','.join(labs), flush=True); run(DRIVER[drv], env, work / 'logs' / f'{f}_{tag_s}.log')
    print('done', flush=True)
