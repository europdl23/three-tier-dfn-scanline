# -*- coding: utf-8 -*-
"""v3 observation-error runs (15 Oct 2026): the RA-15 scenarios of ra1417_run_window.py (b2: single error types at 5 m;
b3: combined moderate/severe at 5 m and the centre line), rerun with
  - the current statistical reference STAT_FIT (fitted von Mises / lognormal per cluster; pool of 60 uniform centres,
    as hybfix_ext/statfit_run.py) in place of the older resampled STAT_F, and
  - the v3 indicated-trace rule (v3patch.py; detection-aware relaxation when r1 = 'da').
Scenario definitions, observation seeds and arm seeds are those of ra1417_run_window.run_scen (imported, not edited).
The error-free reference is the v3 main run of STAT_FIT (run_main_v3.py, job 'stat').
Output: v3/out_obs/<batch>/<box>_rows.csv and v3/out_obs/ra1417v3_<batch>_rows.csv. Usage: python -B run_obs_v3.py b2|b3 [n]"""
import os, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / 'reanalysis'
HX = HERE.parent / 'hybrid'
for p in (str(SCRIPTS), str(HX), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import ra1417_run_window as RW          # scenario runner (read only)
RC = RW.RC
import statfit_run as SF                # fit / sample / segs (read only)
import v3patch                          # noqa: E402
v3patch.install()
_FAMS = {}


def statfit_pool(C, H, rng):
    key = id(C)
    if key not in _FAMS:
        _FAMS[key] = SF.fit(C)
    fams = _FAMS[key]
    import p2lib as P

    def pool(k):
        c = np.column_stack([rng.uniform(0, P.M, 60), rng.uniform(0, H, 60)]); th, L = SF.sample(fams, 60, rng)
        return SF.segs(c, th, L)
    return pool


RC.stat_pool = statfit_pool
OUTV = HERE.parent / 'work' / 'v3_out_obs'
RW.OUT = OUTV
RW.PARTS = OUTV
import shutil as _sh
OUTV.mkdir(parents=True, exist_ok=True)
_sh.copy2(HERE.parent / 'data' / 'observation_errors' / 'RA15_precomputed_parameters.json', OUTV / 'RA15_precomputed_parameters.json')

if __name__ == '__main__':
    OUTV.mkdir(exist_ok=True)
    RW.main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 22)
