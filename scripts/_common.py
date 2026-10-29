"""Shared helpers of the command-line scripts."""
import os, sys, json, shutil, subprocess
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfn3tier import paths as P  # noqa: E402  (sets DFN3_* variables)


def subset_stage(stage, n, dest, role):
    """Copy of a stage folder whose inventory keeps all windows except the first n of the given role (quick tests).
    role 'validation': keeps every training window and the first n validation windows; role 'train': keeps the first n windows."""
    dest = Path(dest)
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(stage, dest); inv = pd.read_csv(dest / 'window_inventory.csv')
    if role == 'validation':
        keep = sorted(inv[inv.role == 'validation'].window_id)[:n]; inv = inv[(inv.role != 'validation') | inv.window_id.isin(keep)]
    else: inv = inv[inv.window_id.isin(sorted(inv.window_id)[:n])]
    inv.to_csv(dest / 'window_inventory.csv', index=False); return dest


def run(script, env, log):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as fh:
        p = subprocess.run([sys.executable, '-u', str(script)], env=dict(os.environ, **{k: str(v) for k, v in env.items()}), stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode: raise SystemExit(f'failed ({p.returncode}): {script.name}, see {log}')


def folds(arg): return P.FOLDS if arg in (None, 'all') else arg.split(',')


def dump(p, x): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(x, indent=1), encoding='utf-8')
