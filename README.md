# Three-tier reconstruction of fracture trace networks from scanline observations

Code, labelled trace data and trained models of the study:

> Paudyal E, Tang Y, Lamei Ramandi H, Zhai H, Yang S-Q, Si G. A Three-Tier Evidence Hierarchy with Conditioned Generative Models for Discrete Fracture Network Reconstruction from Scanline Observations. *Journal of Rock Mechanics and Geotechnical Engineering* (under review).

The framework reconstructs the two-dimensional fracture trace network of a bench-face window from vertical scanline observations. Every reconstructed trace belongs to one of three evidence tiers:

| Tier | Traces | How they are produced |
|---|---|---|
| 1 Measured | fracture clips recorded inside the 10 cm scanlines | kept exactly as observed |
| 2 Indicated | unobserved continuations of the measured clips | continuation length predicted by the enhanced variational autoencoder (EVAE) continuation head |
| 3 Inferred | traces in the regions that no scanline observes | generated one after another by a sequential autoregressive transformer conditioned on the known traces |

The EVAE (dual spatial and geometric latents, 12 and 20 dimensions, attention pooling) and the transformer together form the **Hybrid framework**. The repository also contains the four compared frameworks of the study, all of which share the same observations, trace count, candidate budget and acceptance rules.

This repository holds code, data and models only. It does not contain manuscript figures or tables, photographs, orthophotos or point clouds, and it contains no plotting or image-processing code: functions of the study code that drew figures or read photographs are replaced by stubs that raise an error if called, and none of them is used by the framework. The geological scorer of earlier versions is present only as dormant code and is not used.

## Contents

| Folder | What it holds |
|---|---|
| `scripts/` | Command-line entry points: `train.py`, `reconstruct.py`, `evaluate.py` |
| `tests/` | `check_reproduction.py` and the reference hashes of 45 study reconstructions |
| `dfn3tier/paths.py` | Repository paths; sets the `DFN3_*` environment variables read by the core code |
| `dfn3tier/core/` | The code of the framework, exactly as it produced the study results (see *Code layout*) |
| `data/stages/<fold>/` | Labelled traces of each leave-one-wall-out fold: `outer_full` (training windows), `eval` (training and held-out windows), `inner/<block>` (epoch selection) |
| `data/FOLDS.csv`, `INNER_BLOCKS.csv`, `STAGE_INDEX.csv` | Fold definitions: tested wall group, excluded neighbouring groups that share photographs, training groups and window counts |
| `data/constants/` | Crossing retention ρ of each fold, calibrated on its training windows |
| `data/adfne_pools/` | Pre-generated ADFNE candidate traces of the cADFNE benchmark (fitted parameters in `*_params.json`), so that no Octave installation is needed |
| `models/runs/<fold>/outer_full/` | Trained models of training seeds 42001, 42002 and 42003: `evae_s<seed>`, `head_s<seed>` (continuation head), `tf_latent_s<seed>` (Hybrid transformer, with EVAE latent token), `tf_no_latent_s<seed>` (Transformer-only), with the transformer position residuals (`residuals.npz`, `CALIBRATION.json`) and the training-data references |
| `unseen_benches/` | Test on four bench lines not used in training, epoch selection or any design choice: frozen models trained on all nine wall groups (three training seeds, epochs fixed in advance), labelled traces of the 33 test windows, crossing constant and ADFNE pool |
| `results/wall_groups/` | Leave-one-wall-out results: paired t-tests with Holm correction over the 15 metrics, Tier 3 tests, seed spread |
| `results/unseen_benches/` | Unseen-bench results: paired t-tests of all 15 metrics with Holm correction, means, and the errors of every window (including the number of windows with a lower error) |
| `results/scripts/` | The analysis scripts that computed the result tables (paths refer to the study workstation) |
| `models/SELECTED_EPOCHS.json` | The training epoch of each model and fold |

### Data format

`window_trace_manifest.csv`: one row per trace in a 20 m × 12 m window; `cx`, `cy` (centre), `length` in window units (1 = 20 m), `angle` (radians on the face, 0 horizontal), `boundary_clipped` (the trace was truncated at a window side). `window_inventory.csv`: one row per window with its role (`train`, `validation` = held out), position along the face and height. `training_only_statistics.json`: cluster centres, shares and length and orientation pools fitted on the training windows only.

## Wall groups and validation

Validation is leave-one-wall-out over nine wall groups; each fold tests one group, and groups that share source photographs with the tested group are excluded from training.

| Wall group | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 |
|---|---|---|---|---|---|---|---|---|---|
| Fold code | S1W_P1 | S1W_P2 | S1W_P3 | S1W_P4 | S1W_W04 | S1_EAST | S1_WEST | S2_LOW | S2_HIGH |

Scanline layout `S10o5`: vertical scanlines 10 cm wide and 10 m apart, the first 5 m from the left window side. Training seeds 42001, 42002 and 42003; generation seeds 72001, 72002 and 72003 for every window.

## Compared frameworks

| Option of `reconstruct.py` | Code label | Tier 2 | Tier 3 |
|---|---|---|---|
| `hybrid` | `E_hyb@roll1off` | EVAE continuation | transformer with the EVAE latent token |
| `transformer` | `D_tf@roll1off` | lengths drawn from the training traces of the cluster | transformer without the latent token |
| `evae` | `V_evae@evaeoff` | EVAE continuation | EVAE decoder |
| `ckde` | `cKDE@scoreoff` | as Transformer-only | conditioned kernel density estimate of the training traces |
| `cadfne` | `cADFNE@scoreoff` | as Transformer-only | conditioned ADFNE: Fisher orientation, truncated exponential length |

`roll1` is the restart sampler of the transformer: the sequence is restarted from the known traces each time the number of traces the window still needs has been generated. The suffix `off` means that the geological scorer of earlier versions is switched off; the scorer is **not** part of the method, and its code is present only because the shared candidate replay still calls it with scoring disabled.

Shared acceptance rules (no learned component): a candidate is rejected if it leaves the window, crosses a scanline, duplicates an accepted trace, exceeds the share of its cluster in the trace count or lies too close to an accepted trace of the same cluster; a candidate that would cross n accepted traces is retained with probability ρⁿ; collinear segments of one cluster that meet end to end are merged. The trace count of each window comes from a prior-weighted estimator of the scanline intensity.

## Installation

Python 3.10 with the packages in `requirements.txt`. A CUDA GPU reproduces the study networks exactly; a CPU runs everything (see *Reproducibility*).

```
pip install -r requirements.txt
```

## Usage

```
python -B tests/check_reproduction.py                      # 3 windows of G6, 5 frameworks, compared with the study (a few minutes on a GPU)
python -B scripts/reconstruct.py                           # all frameworks, all 9 wall groups -> work/
python -B scripts/evaluate.py --work work                  # per-window metrics (and an earlier bootstrap analysis) -> work/results/
python -B scripts/train.py --folds S1_EAST --work work/models            # retrain the models of one wall group
python -B scripts/reconstruct.py --models work/models --work work_new    # reconstruct with retrained models
```

Options: `--folds S1_EAST,S2_LOW` selects wall groups, `--arms hybrid,ckde` selects frameworks, `--windows N` keeps the first N held-out windows (quick tests), `--training-seed` selects the model seed. `train.py --select` repeats the epoch selection on the inner blocks; `--epochs N` is for smoke tests only.

### Outputs

`reconstruct.py` writes one folder per window, framework and generation seed under `work/<tag>/B/runs/<fold>/S10o5/S10o5_<tag>/cases/`. `predictions.csv` lists the reconstructed traces (`cx`, `cy`, `length`, `angle`, and `inferred` = Tier 3; Tier 1 and 2 traces have `inferred` False); `metrics.json`, `proposal_ledger.csv.gz` and `raw_candidates.csv.gz` record the per-case scores, every candidate decision and the candidate pool.

`evaluate.py` writes `UNITS.csv` (one row per window and framework), `MEANS.csv`, `FOLD_MEANS.csv` (per wall group), `TIER2.csv` and `TIER2_FOLD_MEANS.csv` (Tier 2 continuation error), `CONTRASTS.csv`, `COUNTS.csv` and `RULES.csv`.

## Metrics and statistics

The paper reports 15 metrics in five groups, computed per reconstruction against the mapped traces of the window (definitions: Supplementary Table S4 of the paper):

* **Orientation:** cluster shares (including the random joints C4); orientation error and orientation dispersion of clusters C1 to C3.
* **Length:** trace length (1-Wasserstein distance of log trace length) and long-trace length (90th percentile) error of C1 to C3.
* **Intensity:** P21 error; local P21 on 5 m × 6 m cells and its spatial heterogeneity.
* **Spatial arrangement:** cell coverage (fraction of 5 m × 6 m cells containing traces); spacing of each cluster.
* **Topology:** intersection density; I, Y and X node types; number of connected components; largest connected component; top-to-bottom connectivity.

Statistics of the paper: for every framework, metric and wall group, errors are averaged over windows and generation seeds and then over the three training seeds, giving nine paired wall-group means per comparison. Differences are tested with paired two-sided t-tests (n = 9), with Holm correction over the 15 metrics (or over the three Tier 3 inferred-trace measures) within each comparison; the Tier 2 continuation error is a single unadjusted test. The result tables are in `results/` and the scripts that computed them in `results/scripts/` (`ttest_analysis.py`, `tier3_analysis.py`, `tier3only.py`, `unseen_bench_analysis.py`, and `recompute_15.py`, which applies the Holm correction over the 15 metrics; their paths refer to the study workstation).

`evaluate.py` computes the per-window metric values used by these tests. It also writes an energy distance of trace-centre positions, which is not part of the 15 metrics of the paper, and an earlier wall-group bootstrap with decision rules, which the paper does not use.

## Training settings

| Model | Epochs | Learning rate | Batch | Other |
|---|---|---|---|---|
| EVAE | selected by inner validation, cap 150 | 3 × 10⁻⁴ | 8 | Kullback-Leibler weight β rising from 0 to 0.001 over 50 epochs, free bits 0.1, gradient clip 1 |
| Continuation head | 50 (selected epoch used) | 10⁻³ | 8 | trained on the frozen EVAE |
| Transformers | 50 (selected epoch used) | 3 × 10⁻⁴ | 8 | dropout 0.1, gradient clip 1; position residuals calibrated at the selected epoch |

Adam optimiser with weight decay 10⁻⁵ for all models; every model is trained on the training windows of its fold only. Epochs are the earliest minimum of the validation curve on the inner block of the fold (`train.py --select`).

## Reproducibility

`tests/check_reproduction.py` reproduces the study networks bit for bit on a CUDA GPU (verified for all 45 reference cases), and `train.py` reproduces the study training exactly for the same inputs and epochs. `evaluate.py` applied to the study reconstructions gives the same metric values as the study. On a CPU, cKDE and cADFNE are identical; the learned frameworks give different but equally valid draws.

## Code layout

`dfn3tier/core/` keeps the folder structure in which the study was run, because the code locates its modules relative to that structure and adds the final settings to earlier modules by exact, assert-checked text substitution. The main entry files are:

* `PAPER2_RETRAIN_AB_2026-10-28/phase7_followup/fbench_v3.py`: reconstruction of Hybrid, Transformer-only, cKDE and cADFNE (restart sampler `V3_ROLL`, scorer switch `V3_SCORING`), built on `phase5_eval/fbench_v2.py`, `topo_v2.py` and `pde_v2.py`;
* `PAPER2_RETRAIN_AB_2026-10-28/phase10_ablation/fbench_v10.py`: EVAE-only (EVAE decoder for Tier 3);
* `plan_d_step3_training_evaluation_2026-10-21/plan_d_train.py` and `engine/training.py`: training of the EVAE, continuation head and transformers;
* `PAPER2_RETRAIN_AB_2026-10-28/phase6_analysis/analysis_v2.py`, `phase9_spatial/phase9_analysis.py` and `PAPER2_DFN_GEOLOGICAL_SIMILARITY_AUDIT_2026-10-27/geo_metrics.py`: metrics and statistics;
* `Updated__file/improved_dfn_experiments/`: the EVAE (`model_v4.py`), transformer (`transformer_v4.py`) and continuation rules (`hybrid_v6.py`) on which the engine builds.

Changes from the study code are limited to absolute paths of the research machine, which now come from `dfn3tier/paths.py`, the ADFNE pools being read as compressed files, the analysis reading the data folder, the removal of the file-access guards used during the study, and the stubbing of plotting and image-processing functions together with their matplotlib, OpenCV and Pillow imports.

## Licence and citation

Code: MIT (`LICENSE`). Data and trained models: CC BY 4.0 (`LICENSE-DATA`). Please cite the paper (`CITATION.cff`).
