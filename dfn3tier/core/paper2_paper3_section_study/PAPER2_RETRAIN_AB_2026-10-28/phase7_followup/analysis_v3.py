"""Scores runs with phase6_analysis/analysis_v2.py's own functions (truth, case_metrics, unit_metrics, derive, contrasts: the pre-registered rule).
analysis_v2.run only reads phase5_eval/<cfg>/runs/<fold>/<lay>/<lay>_on; run_tag reads any (base dir, tag) with otherwise identical code.
Usage: python analysis_v3.py OUTNAME CFG:TAG[:LAYOUTS] [CFG:TAG[:LAYOUTS] ...]   (TAG 'on' = the A/B evaluation)
Writes <OUTNAME>_UNITS.csv, <OUTNAME>_CONTRASTS.csv (keys cfg, tag, layout), <OUTNAME>_COUNTS.csv, <OUTNAME>_TIER2.csv in this folder."""
import sys, inspect, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / 'phase6_analysis')); import analysis_v2 as A

_src = inspect.getsource(A.run)
_old = "glob.glob(str(ROOT / 'phase5_eval' / cfg / 'runs' / fold / lay / f'{lay}_on' / 'cases' / '*'))"
assert _src.count(_old) == 1
_src = _src.replace('def run(args):', 'def run_tag(args):').replace('cfg, fold, lay = args', 'cfg, fold, lay, base, tag = args').replace(_old, "glob.glob(str(Path(base) / cfg / 'runs' / fold / lay / f'{lay}_{tag}' / 'cases' / '*'))")
exec(compile(_src, 'run_tag', 'exec'), A.__dict__); run_tag = A.__dict__['run_tag']


def base_of(tag): return str(ROOT / 'phase5_eval') if tag == 'on' else str(HERE / tag)


def job_tag(args):
    U, T2 = run_tag(args[:5]); U['tag'] = args[4]; T2['tag'] = args[4]; return U, T2


if __name__ == '__main__':
    warnings.filterwarnings('ignore'); name = sys.argv[1]; jobs = []
    for spec in sys.argv[2:]:
        p = spec.split(':'); cfg, tag = p[0], p[1]; lays = p[2].split(',') if len(p) > 2 else A.LAYS
        jobs += [(cfg, f.name, l, base_of(tag), tag) for f in sorted((ROOT / 'phase3_staging' / cfg / 'stages').iterdir()) for l in lays]
    with Pool(int(__import__('os').environ.get('POOL', '4'))) as pool: res = pool.map(job_tag, jobs)
    U = A.derive(pd.concat([r[0] for r in res], ignore_index=True)); T2 = pd.concat([r[1] for r in res], ignore_index=True)
    U.to_csv(HERE / f'{name}_UNITS.csv', index=False); T2.to_csv(HERE / f'{name}_TIER2.csv', index=False)
    Wn = U.groupby(['cfg', 'tag', 'layout', 'fold', 'window', 'arm'])[A.PRIM + A.SEC].mean().reset_index()
    Wn.groupby(['cfg', 'tag', 'layout', 'arm'])[A.PRIM + A.SEC].mean().to_csv(HERE / f'{name}_MEANS.csv')
    C = A.contrasts(Wn, ['cfg', 'tag', 'layout']); C.to_csv(HERE / f'{name}_CONTRASTS.csv', index=False)
    K = C[C.family == 'primary'].groupby(['cfg', 'tag', 'layout', 'contrast', 'model', 'baseline'])[['supported_better', 'supported_worse']].sum().reset_index(); K.to_csv(HERE / f'{name}_COUNTS.csv', index=False)
    pd.set_option('display.width', 250); print(K.to_string(index=False))
