# -*- coding: utf-8 -*-
"""Train one EVAE for one fold and run seed, exactly as for the released checkpoints.

The model and training are the Paper 1 release (final L2 settings) with two guidance terms added in
src/model/training.py, both weight 1.0:
  EVGEO_W  circular W1 between the directions the model would generate for a window and the window's real directions;
  EVBG_W   L1 difference between the orientation-cluster shares (each learned cluster and the unclustered share) of
           the generated and the real directions, under the fold's learned cluster model (data/orientation_clusters).
Data: the fitting boxes of the fold (training) and its validation boxes (early stopping), with every fracture of a
test box removed. CPU, 4 threads; python, numpy and torch seeded with the run seed; deterministic algorithms.

Usage:  python scripts/train_evae.py <S1|S2|S3> <fold> <run seed> [--out DIR]
Output goes to work/models_retrained/<scheme>_f<fold>_s<seed>/ unless --out is given (the released models/ are never
overwritten). Training one model takes about 2-3 minutes on a desktop CPU."""
import os, sys, json, time, random, hashlib, argparse, csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('scheme', choices=('S1', 'S2', 'S3')); ap.add_argument('fold', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--out', default=None)
A = ap.parse_args()
os.environ['LAMBDA_FREE'] = '0.22'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'       # CPU: for this small model ~2.5x faster per epoch than a GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'study'))
import study as C            # noqa: E402
import repo_paths as RP      # noqa: E402

data = C.write_fold_data(A.scheme, A.fold)
OUT = Path(A.out).resolve() if A.out else RP.WORK_DIR / 'models_retrained' / ('%s_f%d_s%d' % (A.scheme, A.fold, A.seed))
for d in ('checkpoints', 'results', 'figures'):
    (OUT / d).mkdir(parents=True, exist_ok=True)
os.chdir(OUT)
os.environ['EVGEO_W'] = '1.0'; os.environ['EVBG_W'] = '1.0'
os.environ['EVBG_MODEL'] = str(RP.CLUSTERS_DIR / ('%s_f%d.json' % (A.scheme, A.fold)))
sys.path.insert(0, str(RP.MODEL_DIR))
import numpy as np           # noqa: E402
import torch                 # noqa: E402
import training              # noqa: E402

torch.set_num_threads(4)
assert training.LAMBDA_FREE == 0.22 and training.ROUTEB_ISO is True
training.BASE_SAVE_DIR = str(OUT / 'never_used_generation_outputs')
random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed); torch.cuda.manual_seed_all(A.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
t0 = time.time()
device = training.device_select()
fit_samples = training.load_labelme(str(data / 'fit'))
val_samples = training.load_labelme(str(data / 'val'))
split = json.load(open(data / 'split.json'))
assert not set(split['test']) & {s['file'][:-5] for s in fit_samples + val_samples}   # no test box in training
for i, s in enumerate(fit_samples):
    s['sample_id'] = i
for i, s in enumerate(val_samples):
    s['sample_id'] = i
training.per, priors = training.extract_targets_and_priors(fit_samples)
priors['constraints'] = training.compute_data_driven_constraints_local(fit_samples)
train_dataset = training.EnhancedDataset(fit_samples, augment=False)
val_dataset = training.EnhancedDataset(val_samples, augment=False)
RUN = '%s_geobg_f%d_s%d' % (A.scheme, A.fold, A.seed)          # internal run name, as for the released checkpoints
model, history = training.train_model_upgraded(train_dataset, val_dataset, priors, training.per, device,
                                               epochs=100, K_angle=3, run_name=RUN)
with torch.no_grad():
    batch = [train_dataset[i] for i in range(min(len(train_dataset), 400))]
    out = model(torch.stack([b['lines_norm'] for b in batch]).to(device), torch.stack([b['valid_mask'] for b in batch]).to(device),
                torch.stack([b['img_size'] for b in batch]).to(device))
    kls = float((0.5 * (out['smu'].pow(2) + out['slog'].exp() - out['slog'] - 1)).mean())
    klg = float((0.5 * (out['gmu'].pow(2) + out['glog'].exp() - out['glog'] - 1)).mean())
ckpt = OUT / 'checkpoints' / ('evai_best_%s.pt' % RUN)
(OUT / 'checkpoint.pt').write_bytes(ckpt.read_bytes())
json.dump(dict(scheme=A.scheme, fold=A.fold, seed=A.seed, lambda_free=training.LAMBDA_FREE, device=str(device), evgeo_w='1.0', evbg_w='1.0',
               model_code='src/model/training.py', n_fit=len(fit_samples), n_val=len(val_samples), traces_fit=sum(len(s['lines']) for s in fit_samples),
               epochs=len(history), elapsed_s=round(time.time() - t0, 1), best_val_loss=min(h['val_loss'] for h in history),
               kl_spatial=kls, kl_geometry=klg, classification='LATENT_COLLAPSE' if min(kls, klg) < 0.01 else 'STABLE',
               checkpoint='checkpoint.pt', checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(), internal_run_name=RUN),
          open(OUT / 'run_status.json', 'w'), indent=1, default=float)
with open(OUT / 'train_history.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss']); w.writeheader()
    w.writerows([{k: h[k] for k in ('epoch', 'train_loss', 'val_loss')} for h in history])
rel = RP.MODELS_DIR / ('%s_f%d_s%d' % (A.scheme, A.fold, A.seed)) / 'run_status.json'
same = rel.exists() and json.load(open(rel))['checkpoint_sha256'] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
print('DONE', RUN, 'epochs', len(history), 'best val %.4f' % min(h['val_loss'] for h in history), '%.0fs' % (time.time() - t0),
      '| checkpoint identical to the released one' if same else '| checkpoint differs from the released one (see docs/REPRODUCIBILITY.md)')
