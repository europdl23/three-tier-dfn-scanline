"""Reruns the five frameworks on the first three held-out windows of wall group G6 (fold S1_EAST) with the released models and compares
every reconstructed network with the study reconstructions (SHA-256 of predictions.csv).

    python tests/check_reproduction.py [--work work/check]

Identical networks are expected on a CUDA GPU. On a CPU, cKDE and cADFNE are identical, while the learned frameworks give different but
equally valid draws, because the neural networks and their random streams then run on another device."""
import argparse, json, hashlib, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--work', default=str(REPO / 'work' / 'check')); x = a.parse_args(); work = Path(x.work).resolve()
    subprocess.run([sys.executable, '-B', str(REPO / 'scripts' / 'reconstruct.py'), '--folds', 'S1_EAST', '--windows', '3', '--work', str(work)], check=True)
    ref = json.loads((HERE / 'reference_S1_EAST_3windows.json').read_text())['predictions_sha256']; same = diff = 0
    for k, h in ref.items():
        tag, case = k.split('/'); p = work / tag / 'B' / 'runs' / 'S1_EAST' / 'S10o5' / f'S10o5_{tag}' / 'cases' / case / 'predictions.csv'
        ok = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h; same += ok; diff += not ok
        if not ok: print('differs:', k)
    print(f'{same} of {len(ref)} reconstructions identical to the study')
