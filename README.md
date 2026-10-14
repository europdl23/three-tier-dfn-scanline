# Three-tier reconstruction of fracture trace maps from scanline observations

Code, traced data and trained models of the study:

> Paudyal E, Tang Y, Lamei Ramandi H, Zhai H, Yang S-Q, Si G. A Three-Tier Evidence Hierarchy with Conditioned Generative Models for Discrete Fracture Network Reconstruction from Scanline Observations. *Journal of Rock Mechanics and Geotechnical Engineering* (under review).

The repository reconstructs two-dimensional fracture trace maps of test windows on a mapped open-pit wall from simulated vertical scanlines. It does this with a three-tier evidence hierarchy and several generators for the unobserved area. It holds:

* the code that produced every reported reconstruction;
* the traced data and the trained models it uses;
* one score row for every realisation, plus the reconstructed maps.

Figure and table plotting code is not included.

## Contents

| Folder | What it holds |
|---|---|
| `data/box_library/` | The traced data: the fracture traces of the 50 test windows (20 m wide) on seven bench faces, W00 to W06, as two-point lines in window coordinates (LabelMe JSON, 2 cm pixels), and `box_manifest.csv` with the window grid. 735 trace pieces of 629 mapped traces |
| `data/splits/` | The seven leave-one-bench-out folds (`S2_f0` to `S2_f6`): training, validation and test windows |
| `data/orientation_clusters/` | The orientation-cluster model of each fold (C1 to C4 and random joints), fitted on the training windows of the fold |
| `data/observation_errors/` | The pre-computed parameters of the observation-error scenarios |
| `models/evae/` | The trained enhanced variational autoencoder (EVAE): 7 folds x 3 training runs |
| `study/tr_models/` | The trained sequential transformer (`gast2_*`) and evidence network (`evn_*`): 7 folds x 3 training runs |
| `models/termination/` | Termination networks (`term2_*`): 7 folds x 3 training runs |
| `networks/S2/` | The stored candidate realisations of the EVAE-only, KDE and ADFNE generators (24 per test window) |
| `src/`, `scripts/` | Trace-map representation, the EVAE (`src/model`, training script `scripts/train_evae.py`), the generators and the error scores (`scripts/statistical_tests.py`) |
| `study/` | Simulated scanlines, the three evidence tiers, the placement rules, the sequential transformer (`gast.py`, with its training function) and the window runner |
| `reanalysis/` | Shared composition of generators and rules (`ra01abl_run.py`), model loading (`learned_arm.py`), observation-error scenarios (`ra1417_*.py`) and the paired statistical tests (`protocolP.py`) |
| `hybrid/` | The hybrid generator (transformer position and cluster, EVAE trace of that cluster, length range) and fitted statistical sampling |
| `v3/` | The runners of all reported results and `v3patch.py`, the indicated-trace rule (an indicated trace may not reach a scanline that recorded none of its hits) |
| `results/scores/` | One row per realisation: every generator, scanline layout and test window (gzip-compressed CSV) |
| `results/realisations/` | The reconstructed maps (`.npz`; key `gt` = mapped traces of the window, `ARM__LAYOUT__rNN` = realisation NN) |

The evidence and termination networks are loaded by the shared model loader. The reported generators do not use their outputs: indicated traces follow the rule of `v3patch.py`, and no termination is applied.

## Names used in the code

| Code | In the paper |
|---|---|
| `STAT_FIT` | Statistical sampling (reference): fitted von Mises orientation and lognormal length per cluster |
| `TR_FIT` | Sequential transformer only |
| `EVAE_F` | EVAE only |
| `KDE_F`, `ADFNE_F` | KDE, ADFNE |
| `HYB_FIX2` | Hybrid |
| `*_noR5` | Without map adjustment |
| `STAT_FIT_NOSCAN`, `STAT_FIT_FREE` | Statistical sampling without scanlines, with and without the placement rules |
| `STAT_F` | Statistical sampling with resampled training traces (earlier form, sensitivity check) |
| R1, R2, R3, R4, R5 | Scanline rule, intensity budget, crossing limit, duplicate removal, map adjustment |
| `centre`, `10m`, `7m`, `5m`, `3m` | Scanline layouts: one scanline at the window centre, and scanlines 10, 7, 5 and 3 m apart |
| `S2`, fold 0 to 6 | Leave-one-bench-out validation; fold k tests bench face W0k |

## Measures in `results/scores`

Per realisation:

* **Orientation error:** mean of `oc1_w1_deg` to `oc4_w1_deg`, the first Wasserstein distance of trace orientation in each cluster. Empty values mark clusters with fewer than two traces in the mapped window or the reconstruction.
* **Trace length error:** mean of `oc1_len_w1_m` to `oc4_len_w1_m`.
* **Intensity error:** `abs(P21_relerr) x 100`.
* **Node-type error:** `node_tv`.

The value of a window is the median over its 24 realisations (3 training runs x 8 draws); reported values are medians over the 50 test windows.

Files:

* `observation_errors_b2` and `observation_errors_b3` hold the single and combined observation-error scenarios.
* The error-free reference for those scenarios is `stat.csv.gz`.

## Reproducing the results

Python 3.10 or newer with the packages in `requirements.txt`. A CPU is sufficient; no ADFNE or Octave installation is needed, because the ADFNE candidates are stored in `networks/`.

```
pip install -r requirements.txt
python -B check_reproduction.py            # reruns test window W02_C3 for every run type and compares with results/scores
python -B v3/run_main_v3.py all 8          # all generators and layouts, 50 windows, 8 processes (output in work/)
python -B v3/run_noscan_v3.py 8            # statistical sampling without scanlines
python -B v3/run_obs_v3.py b2 8            # observation errors, single scenarios (b3: combined scenarios)
```

Every run uses fixed seeds and reproduces the stored score rows exactly (observation-error runs to within 1e-13). The check script confirms this for one window in a few minutes.

## Not included

* Photographs, orthoimages and the photogrammetric model of the pit.
* Mine-grid coordinates. The traced data are in window coordinates.
* Data of other parts of the pit.
* The ADFNE 1.5 software (third party). Only its stored candidate realisations are included.

## Licence

* **Code:** MIT (`LICENSE`).
* **Traced data, trained models, stored realisations and result tables:** CC BY 4.0 (`LICENSE-DATA`).
