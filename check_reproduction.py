# -*- coding: utf-8 -*-
"""Rerun test window W02_C3 (fold 2) for every run type and compare with the stored score rows in results/scores.

Usage: python -B check_reproduction.py [type ...]
types: hyb hyblay hybnoR5 stat fill noscan obs_b2 obs_b3 (default: all; obs_b2 and obs_b3 take longest)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'v3'))
FI, BOX = 2, 'W02_C3'
STORED = {'hyb': 'hyb', 'hyblay': 'hyblay', 'hybnoR5': 'hybnoR5', 'stat': 'stat', 'fill': 'fill', 'noscan': 'noscan',
          'obs_b2': 'observation_errors_b2', 'obs_b3': 'observation_errors_b3'}


def run(kind):
    if kind.startswith('obs'):
        import run_obs_v3 as O
        return pd.DataFrame(O.RW.worker((kind.split('_')[1], FI, BOX))[2])
    if kind == 'noscan':
        import run_noscan_v3 as N
        return pd.DataFrame(N.job_noscan(FI, BOX)[0])
    import run_main_v3 as M
    return pd.DataFrame(M.JOBS[kind](FI, BOX)[0])


def main(kinds):
    ok = True
    for kind in kinds:
        new = run(kind)
        new['cfg'] = new['cfg'].astype(str).str.split('|').str[0]
        old = pd.read_csv(ROOT / 'results' / 'scores' / f'{STORED[kind]}.csv.gz')
        old = old[(old['fold'] == FI) & (old['box'] == BOX)]
        key = [c for c in ('cfg', 'arm', 'r', 'scen', 'scenario') if c in new.columns and c in old.columns]
        new = new.sort_values(key).reset_index(drop=True)
        old = old.sort_values(key).reset_index(drop=True)
        cols = [c for c in new.columns if c in old.columns and c != 'sec'
                and pd.api.types.is_numeric_dtype(new[c]) and pd.api.types.is_numeric_dtype(old[c])]
        if len(new) != len(old):
            print(f'{kind}: row count {len(new)} against {len(old)} stored: DIFFERENT'); ok = False; continue
        d = np.nanmax(np.abs(new[cols].to_numpy(float) - old[cols].to_numpy(float)))
        nan_mis = int((new[cols].isna().to_numpy() != old[cols].isna().to_numpy()).sum())
        same = d < 1e-6 and nan_mis == 0
        ok &= same
        print(f'{kind}: {len(new)} rows, {len(cols)} columns, largest difference {d:.2g}, '
              f'NaN mismatches {nan_mis}: {"REPRODUCED" if same else "DIFFERENT"}')
    print('all reproduced' if ok else 'differences found')


if __name__ == '__main__':
    main(sys.argv[1:] or list(STORED))
