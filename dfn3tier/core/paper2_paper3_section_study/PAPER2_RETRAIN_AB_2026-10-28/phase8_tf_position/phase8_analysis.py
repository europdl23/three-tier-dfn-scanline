"""Phase 8 analysis (PREREGISTRATION_TF_POSITION.md). Scores the corrected-sampler runs with analysis_v2's own metric code (via analysis_v3.run_tag), merges
them with the existing A/B units (phase6_analysis/UNITS.csv, tag 'on') and the scorer-off units (phase7_followup/FOLLOWUP_UNITS.csv), and applies analysis_v2.contrasts
(the pre-registered rule) to the phase-8 contrast list. Arms from follow-up tags are named <arm>@<tag>.
Also: position diagnostics (raw / valid-eligible / accepted inferred mean x vs the true hidden mean x) and the per-case acceptance ledger.
Writes P8_UNITS.csv, P8_MEANS.csv, P8_CONTRASTS.csv, P8_COUNTS.csv, P8_FOLD_MEANS.csv, P8_POSITION.csv, P8_LEDGER.csv."""
import sys, glob, json, os, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; FU = ROOT / 'phase7_followup'
sys.path.insert(0, str(FU)); sys.path.insert(0, str(ROOT / 'phase6_analysis'))
import analysis_v3 as V3; import analysis_v2 as A
LAYS = A.LAYS; NEW = [('roll1', LAYS), ('roll1off', ['S10o5']), ('roll1.5', ['S10o5'])]
CON = [('TF_corrected_vs_original', 'D_tf@roll1', 'D_tf'), ('TF_corrected_vs_bench', 'D_tf@roll1', 'cKDE'), ('TF_corrected_vs_bench', 'D_tf@roll1', 'cADFNE'),
       ('TF_original_vs_bench', 'D_tf', 'cKDE'), ('TF_original_vs_bench', 'D_tf', 'cADFNE'),
       ('Hyb_corrected_vs_original', 'E_hyb@roll1', 'E_hyb'), ('Hyb_corrected_vs_bench', 'E_hyb@roll1', 'cKDE'), ('Hyb_corrected_vs_bench', 'E_hyb@roll1', 'cADFNE'),
       ('Hyb_original_vs_bench', 'E_hyb', 'cKDE'), ('Hyb_original_vs_bench', 'E_hyb', 'cADFNE'),
       ('add_EVAE_to_TFc', 'E_hyb@roll1', 'D_tf@roll1'), ('add_TFc_to_EVAE', 'E_hyb@roll1', 'eKDE'), ('add_TFc_to_EVAE', 'E_hyb@roll1', 'eADFNE'),
       ('EVAE_only_vs_bench', 'eKDE', 'cKDE'), ('EVAE_only_vs_bench', 'eADFNE', 'cADFNE'),
       ('sensitivity_L1.5', 'D_tf@roll1.5', 'D_tf@roll1'), ('sensitivity_L1.5', 'E_hyb@roll1.5', 'E_hyb@roll1'),
       ('scorer_on_vs_off', 'D_tf@roll1', 'D_tf@roll1off'), ('scorer_on_vs_off', 'E_hyb@roll1', 'E_hyb@roll1off'),
       ('scorer_on_vs_off', 'cKDE', 'cKDE@scoreoff'), ('scorer_on_vs_off', 'cADFNE', 'cADFNE@scoreoff'), ('scorer_on_vs_off', 'D_tf', 'D_tf@scoreoff'), ('scorer_on_vs_off', 'E_hyb', 'E_hyb@scoreoff')]


def position(args):
    cfg, fold, lay, tag = args; W, _ = A.truth(cfg, fold); base = Path(V3.base_of(tag)); rows = []
    for wid, w in W.items():
        st = A.strips(lay, w['x0']); T = w['arr']
        if st and len(T): rows += [dict(cfg=cfg, fold=fold, layout=lay, src='truth_hidden', stage='truth', x=float(x)) for x in T[~A.G.H.crossing_strip(T, st)][:, 0]]
    for d in glob.glob(str(base / cfg / 'runs' / fold / lay / f'{lay}_{tag}' / 'cases' / '*__g72001')):
        d = Path(d); arm = d.name.split('__')[0]; name = arm if tag == 'on' else f'{arm}@{tag}'
        if arm not in ('D_tf', 'E_hyb', 'cKDE', 'cADFNE'): continue
        r = pd.read_csv(d / 'raw_candidates.csv.gz'); p = pd.read_csv(d / 'predictions.csv'); inf = p[p.inferred.astype(str).str.lower().eq('true')]
        for stage, xs in (('raw', r.cx), ('valid_eligible', r.cx[r.valid & r.eligible]), ('accepted_inferred', inf.cx)):
            rows += [dict(cfg=cfg, fold=fold, layout=lay, src=name, stage=stage, x=float(x)) for x in xs]
    return pd.DataFrame(rows)


def ledger(args):
    cfg, fold, lay, tag = args; base = Path(V3.base_of(tag)); out = []
    for d in glob.glob(str(base / cfg / 'runs' / fold / lay / f'{lay}_{tag}' / 'cases' / '*')):
        d = Path(d); arm = d.name.split('__')[0]; l = pd.read_csv(d / 'proposal_ledger.csv.gz', usecols=['disposition']); v = l.disposition.value_counts(normalize=True)
        out.append(dict(cfg=cfg, fold=fold, layout=lay, arm=arm if tag == 'on' else f'{arm}@{tag}', n=len(l), **{'share_' + k: float(x) for k, x in v.items()}))
    return pd.DataFrame(out)


if __name__ == '__main__':
    warnings.filterwarnings('ignore'); POOL = int(os.environ.get('POOL', '4'))
    jobs = [(c, f.name, l, V3.base_of(t), t) for c in 'AB' for t, lays in NEW for f in sorted((ROOT / 'phase3_staging' / c / 'stages').iterdir()) for l in lays]
    with Pool(POOL) as p: res = p.map(V3.job_tag, jobs)
    Un = A.derive(pd.concat([r[0] for r in res], ignore_index=True)); T2n = pd.concat([r[1] for r in res], ignore_index=True)
    U0 = pd.read_csv(ROOT / 'phase6_analysis' / 'UNITS.csv'); U0['tag'] = 'on'
    Us = pd.read_csv(FU / 'FOLLOWUP_UNITS.csv'); Us = Us[Us.tag == 'scoreoff']
    U = pd.concat([U0, Us, Un], ignore_index=True); U['arm'] = np.where(U.tag == 'on', U.arm, U.arm + '@' + U.tag); U.to_csv(HERE / 'P8_UNITS.csv', index=False)
    T2 = pd.concat([pd.read_csv(ROOT / 'phase6_analysis' / 'TIER2_CONTINUATION.csv').assign(tag='on'), T2n]); T2['arm'] = np.where(T2.tag == 'on', T2.arm, T2.arm + '@' + T2.tag); T2.to_csv(HERE / 'P8_TIER2.csv', index=False)
    Wn = U.groupby(['cfg', 'layout', 'fold', 'window', 'arm'])[A.PRIM + A.SEC].mean().reset_index()
    Wn.groupby(['cfg', 'layout', 'arm'])[A.PRIM + A.SEC].mean().to_csv(HERE / 'P8_MEANS.csv'); Wn.groupby(['cfg', 'layout', 'arm', 'fold'])[A.PRIM + A.SEC].mean().to_csv(HERE / 'P8_FOLD_MEANS.csv')
    A.CON = CON; C = A.contrasts(Wn, ['cfg', 'layout']); C.to_csv(HERE / 'P8_CONTRASTS.csv', index=False)
    K = C[C.family == 'primary'].groupby(['cfg', 'layout', 'contrast', 'model', 'baseline'])[['supported_better', 'supported_worse']].sum().reset_index(); K.to_csv(HERE / 'P8_COUNTS.csv', index=False)
    pj = [(c, f.name, l, t) for c in 'AB' for t, lays in [('on', LAYS)] + NEW for f in sorted((ROOT / 'phase3_staging' / c / 'stages').iterdir()) for l in lays]
    with Pool(POOL) as p: P = pd.concat(p.map(position, pj), ignore_index=True)
    PS = P.groupby(['cfg', 'layout', 'src', 'stage']).x.agg(['mean', 'std', 'count']).reset_index(); PS.to_csv(HERE / 'P8_POSITION.csv', index=False)
    PF = P[P.stage.isin(['truth', 'accepted_inferred'])].groupby(['cfg', 'layout', 'src', 'fold']).x.mean().reset_index(); PF.to_csv(HERE / 'P8_POSITION_FOLDS.csv', index=False)
    lj = [(c, f.name, 'S10o5', t) for c in 'AB' for t in ['on', 'roll1', 'roll1off', 'scoreoff'] for f in sorted((ROOT / 'phase3_staging' / c / 'stages').iterdir())]
    with Pool(POOL) as p: L = pd.concat(p.map(ledger, lj), ignore_index=True)
    L.to_csv(HERE / 'P8_LEDGER.csv', index=False)
    pd.set_option('display.width', 250); print(K[K.layout == 'S10o5'].to_string(index=False))
