"""Phase 9 analysis (PREREGISTRATION_SPATIAL_LEARNING.md). Scores new tags with analysis_v2's metric code (analysis_v3.run_tag), merges them with the
phase-8 units (P8_UNITS.csv: tags on / scoreoff / roll1 / roll1off / roll1.5), and applies analysis_v2.contrasts with a DETERMINISTIC bootstrap: each
contrast's generator is seeded by SHA-256 of its own paired differences and fold labels, so a comparison gives identical numbers in any run or list order.
Usage: P9_NEW="t1:S10o5;..." [P9_CFGS=B] [P9_OUT=name] python phase9_analysis.py   Arms of new tags are <arm>@<tag>. Contrasts: CON below (pairs absent are skipped).
Writes <out>_UNITS, _MEANS, _FOLD_MEANS, _CONTRASTS, _COUNTS, _POSITION, _POSITION_FOLDS, _LEDGER, _TIER2 (.csv)."""
import sys, glob, os, hashlib, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; FU = ROOT / 'phase7_followup'; P8 = ROOT / 'phase8_tf_position'
sys.path.insert(0, str(FU)); sys.path.insert(0, str(ROOT / 'phase6_analysis')); sys.path.insert(0, str(P8))
import analysis_v3 as V3; import analysis_v2 as A; import phase8_analysis as P8A


def boot_det(d, fo, n=2000):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(np.ascontiguousarray(d, float).tobytes() + '|'.join(map(str, fo)).encode()).digest()[:8], 'little'))
    folds = np.unique(fo); bs = []
    for _ in range(n):
        pick = rng.choice(folds, len(folds)); bs.append(np.concatenate([d[fo == f] for f in pick]).mean())
    bs = np.asarray(bs); return np.percentile(bs, 2.5), np.percentile(bs, 97.5), float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))


A.boot = boot_det
NEW = [(t.split(':')[0], t.split(':')[1].split(',')) for t in os.environ.get('P9_NEW', '').split(';') if t]; CFGS = list(os.environ.get('P9_CFGS', 'AB')); OUT = os.environ.get('P9_OUT', 'P9')
TS = [t for t, _ in NEW]


def con():
    c = [('T0_vs_original', 'D_tf@roll1', 'D_tf'), ('T0_vs_bench', 'D_tf@roll1', 'cKDE'), ('T0_vs_bench', 'D_tf@roll1', 'cADFNE'),
         ('H0_vs_bench', 'E_hyb@roll1', 'cKDE'), ('H0_vs_bench', 'E_hyb@roll1', 'cADFNE'), ('H0_vs_original', 'E_hyb@roll1', 'E_hyb'),
         ('EVAE_only_vs_bench', 'eKDE', 'cKDE'), ('EVAE_only_vs_bench', 'eADFNE', 'cADFNE')]
    for t in TS:   # a scorer-off tag (name ends in 'off') is compared only with scorer-off references: T0 = roll1off, original / benchmarks / EVAE-only = scoreoff
        o = t.endswith('off'); t0 = 'roll1off' if o else 'roll1'; s = '@scoreoff' if o else ''
        c += [(f'{t}_vs_T0', f'D_tf@{t}', f'D_tf@{t0}'), (f'{t}_vs_original', f'D_tf@{t}', 'D_tf' + s), (f'{t}_vs_bench', f'D_tf@{t}', 'cKDE' + s), (f'{t}_vs_bench', f'D_tf@{t}', 'cADFNE' + s),
              (f'H{t}_vs_H0', f'E_hyb@{t}', f'E_hyb@{t0}'), (f'H{t}_vs_original', f'E_hyb@{t}', 'E_hyb' + s), (f'H{t}_vs_bench', f'E_hyb@{t}', 'cKDE' + s), (f'H{t}_vs_bench', f'E_hyb@{t}', 'cADFNE' + s),
              (f'add_EVAE_to_{t}', f'E_hyb@{t}', f'D_tf@{t}'), (f'add_{t}_to_EVAE', f'E_hyb@{t}', 'eKDE' + s), (f'add_{t}_to_EVAE', f'E_hyb@{t}', 'eADFNE' + s)]
    c += [('T0off_vs_bench', 'D_tf@roll1off', 'cKDE@scoreoff'), ('T0off_vs_bench', 'D_tf@roll1off', 'cADFNE@scoreoff'), ('H0off_vs_bench', 'E_hyb@roll1off', 'cKDE@scoreoff'), ('H0off_vs_bench', 'E_hyb@roll1off', 'cADFNE@scoreoff'),
          ('EVAE_only_off_vs_bench', 'eKDE@scoreoff', 'cKDE@scoreoff'), ('EVAE_only_off_vs_bench', 'eADFNE@scoreoff', 'cADFNE@scoreoff')]
    for t in TS:   # pairwise among new tags (e.g. T2 vs T1, restart vs no restart, scorer on vs off)
        for u in TS:
            if t < u: c += [(f'{u}_vs_{t}', f'D_tf@{u}', f'D_tf@{t}'), (f'H{u}_vs_H{t}', f'E_hyb@{u}', f'E_hyb@{t}')]
    return c


if __name__ == '__main__':
    warnings.filterwarnings('ignore'); POOL = int(os.environ.get('POOL', '4'))
    jobs = [(c, f.name, l, V3.base_of(t), t) for c in CFGS for t, lays in NEW for f in sorted((ROOT / 'phase3_staging' / c / 'stages').iterdir()) for l in lays
            if (Path(V3.base_of(t)) / c / 'runs' / f.name / l).exists()]
    with Pool(POOL) as p: res = p.map(V3.job_tag, jobs)
    Un = A.derive(pd.concat([r[0] for r in res], ignore_index=True)); Un['arm'] = Un.arm + '@' + Un.tag
    U0 = pd.read_csv(P8 / 'P8_UNITS.csv'); U = pd.concat([U0[U0.cfg.isin(CFGS)], Un], ignore_index=True); U.to_csv(HERE / f'{OUT}_UNITS.csv', index=False)
    T2 = pd.concat([r[1] for r in res], ignore_index=True); T2['arm'] = T2.arm + '@' + T2.tag; T2 = pd.concat([pd.read_csv(P8 / 'P8_TIER2.csv'), T2]); T2.to_csv(HERE / f'{OUT}_TIER2.csv', index=False)
    Wn = U.groupby(['cfg', 'layout', 'fold', 'window', 'arm'])[A.PRIM + A.SEC].mean().reset_index()
    Wn.groupby(['cfg', 'layout', 'arm'])[A.PRIM + A.SEC].mean().to_csv(HERE / f'{OUT}_MEANS.csv'); Wn.groupby(['cfg', 'layout', 'arm', 'fold'])[A.PRIM + A.SEC].mean().to_csv(HERE / f'{OUT}_FOLD_MEANS.csv')
    A.CON = con(); C = A.contrasts(Wn, ['cfg', 'layout']); C.to_csv(HERE / f'{OUT}_CONTRASTS.csv', index=False)
    K = C[C.family == 'primary'].groupby(['cfg', 'layout', 'contrast', 'model', 'baseline'])[['supported_better', 'supported_worse']].sum().reset_index(); K.to_csv(HERE / f'{OUT}_COUNTS.csv', index=False)
    pj = [(c, f.name, l, t) for c in CFGS for t, lays in NEW for f in sorted((ROOT / 'phase3_staging' / c / 'stages').iterdir()) for l in lays if (Path(V3.base_of(t)) / c / 'runs' / f.name / l).exists()]
    with Pool(POOL) as p: P = pd.concat(p.map(P8A.position, pj), ignore_index=True)
    P = pd.concat([P[P.src != 'truth_hidden'], P[P.src == 'truth_hidden'].drop_duplicates()]) if len(P) else P
    PS = P.groupby(['cfg', 'layout', 'src', 'stage']).x.agg(['mean', 'std', 'count']).reset_index(); PS.to_csv(HERE / f'{OUT}_POSITION.csv', index=False)
    P[P.stage.isin(['truth', 'accepted_inferred', 'raw', 'valid_eligible'])].groupby(['cfg', 'layout', 'src', 'stage', 'fold']).x.mean().reset_index().to_csv(HERE / f'{OUT}_POSITION_FOLDS.csv', index=False)
    lj = [(c, f.name, 'S10o5', t) for c in CFGS for t, lays in NEW if 'S10o5' in lays for f in sorted((ROOT / 'phase3_staging' / c / 'stages').iterdir())]
    with Pool(POOL) as p: L = pd.concat(p.map(P8A.ledger, lj), ignore_index=True)
    L.to_csv(HERE / f'{OUT}_LEDGER.csv', index=False)
    pd.set_option('display.width', 250); print(K[K.layout == 'S10o5'].to_string(index=False))
