# -*- coding: utf-8 -*-
"""Run all Paper 2 revision tests on the Section 1 data. One process per held-out window.
Output: results/<scheme>_rows.csv (one row per window x realisation x configuration x arm) and
results/<scheme>_obs.csv (what the scanlines saw)."""
import sys, math, zlib, itertools, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import p2lib as P

OUT = Path(__file__).parent / 'results'; OUT.mkdir(exist_ok=True)
BASE = dict(spacing=5.0, width=0.10, miss=0.0, noise=0.0, a_prior=2.0, mode='regular')
PROPOSERS = ('EVAE', 'ADFNE', 'KDE', 'UNIF')


def configs(scheme):
    C = [('main', dict(BASE), ['RAW_EVAE', 'RAW_ADFNE', 'RAW_KDE', 'EVID'] + ['T3_' + p for p in PROPOSERS] +
          [f'T3_{p}_{a}' for p in ('EVAE', 'UNIF') for a in ('nogate', 'nobudget', 'notier2', 'nobias')])]
    if scheme != 'S1':
        return C
    for s in (None, 10.0, 7.0, 3.0):
        C.append((f'sp_{s if s else "centre"}', dict(BASE, spacing=s), ['EVID', 'T3_EVAE', 'T3_UNIF', 'T3_KDE']))
    for k, vals in (('width', (0.05, 0.30)), ('miss', (0.2, 0.4)), ('noise', (5.0, 10.0)), ('a_prior', (0.0, 5.0))):
        for v in vals:
            C.append((f'{k}_{v}', dict(BASE, **{k: v}), ['T3_EVAE', 'T3_UNIF']))
    C.append(('irregular', dict(BASE, mode='irregular'), ['T3_EVAE', 'T3_UNIF']))
    for s in (None, 10.0):
        C.append((f'sp_{s if s else "centre"}_nobudget', dict(BASE, spacing=s), ['T3_EVAE_nobudget', 'T3_UNIF_nobudget']))
    return C


def seed_of(*parts):
    return zlib.crc32('|'.join(map(str, parts)).encode()) & 0x7fffffff


def run_window(job):
    scheme, fi, box = job
    boxes = P.load_library(); sp = P.load_split(scheme, fi)
    tr_th, tr_len = P.train_traces(boxes, sp); prior = P.train_p21(boxes, sp)
    b = boxes[box]; H = b['H']; gt = P.clip(b['L'], H)
    g = P.metrics(gt, H)
    rows, obsrows = [], []
    draws = [(s, d) for s in P.SEEDS for d in range(P.NDRAW)]
    stored = {m: [P.stored_net(scheme, m, fi, box, s, d) for s, d in draws] for m in ('EVAE', 'ADFNE', 'KDE')}
    for cname, cf, arms in configs(scheme):
        for r, (s, d) in enumerate(draws):
            orng = np.random.default_rng(seed_of(scheme, box, r, cf['spacing'], cf['width'], cf['miss'], cf['noise'], cf['mode']))
            ap = P.apertures(cf['spacing'], cf['width'], mode=cf['mode'], rng=orng, jitter=1.5)
            obs = P.observe(gt, ap, H, orng, miss=cf['miss'], ang_noise_deg=cf['noise'])
            groups = P.merge_obs(obs)
            if cname == 'main' or cname.startswith('sp_') or r == 0:
                ap_hits = [sum(o['ap'] == j for o in obs) for j in range(len(ap))]
                true_groups = len({o['tid'] for o in obs})
                pure = np.mean([len({o['tid'] for o in gr}) == 1 for gr in groups]) if groups else np.nan
                seen = sorted({o['tid'] for o in obs})
                obsrows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cname, r=r, n_gt=len(gt), n_hits=len(obs),
                                    n_unique_obs=true_groups, frac_traces_obs=true_groups / max(len(gt), 1),
                                    frac_len_obs=P.seglen(gt[seen]).sum() / P.seglen(gt).sum() if seen else 0.0,
                                    aperture_cover=sum(hi - lo for lo, hi in ap) / P.M,
                                    n_scan=len(ap), zero_hit_scan=sum(h == 0 for h in ap_hits),
                                    n_groups=len(groups), merge_purity=pure))
            for arm in arms:
                rrng = np.random.default_rng(seed_of(scheme, box, r, cname, arm))
                if arm.startswith('RAW_'):
                    rec = P.clip(stored[arm[4:]][r], H)
                else:
                    abl = arm.split('_')[2] if arm.count('_') == 2 else ''
                    prop = arm.split('_')[1] if arm.startswith('T3_') else None
                    t1 = P.tier1_pieces(obs)
                    t2 = P.tier2_extend(groups, tr_th, tr_len, H, rrng, use_bias=(abl != 'nobias')) if abl != 'notier2' else t1
                    rec = t2
                    if prop:
                        budget, _ = P.p21_budget(obs, ap, H, tr_th, tr_len, prior, a_prior=cf['a_prior'],
                                                 use_evidence=(abl != 'nobudget'))
                        for k in range(len(draws)):
                            if P.seglen(rec).sum() >= budget:
                                break
                            if prop == 'UNIF':
                                pool = P.unif_proposals(60, tr_th, tr_len, H, rrng)
                            else:
                                pool = stored[prop][(r + k) % len(draws)]
                            add = P.tier3_fill(pool, ap, rec, budget, H, rrng, gate=(abl != 'nogate'))
                            rec = np.vstack([rec, add]) if len(add) else rec
                m = P.metrics(rec, H)
                sc = P.score(m, g)
                rows.append(dict(scheme=scheme, fold=fi, box=box, cfg=cname, arm=arm, r=r, n=m['n'], P21=m['P21'],
                                 I100=m['I100'], gt_n=g['n'], gt_P21=g['P21'], gt_I100=g['I100'], **sc))
    # natural variability reference: the 8 nearest non-test windows scored against this window
    if scheme == 'S1':
        cand = [k for k in boxes if k not in sp['test']]
        near = sorted(cand, key=lambda k: (math.hypot(boxes[k]['cu'] - b['cu'], boxes[k]['cs'] - b['cs']), k))[:8]
        held = set().union(*[set(boxes[t]['ids']) for t in sp['test']])
        for j, k in enumerate(near):
            nb = boxes[k]; keep = np.array([i not in held for i in nb['ids']], bool)
            Ln = P.clip(nb['L'][keep], nb['H'])
            m = P.metrics(Ln, nb['H']); sc = P.score(m, g)
            rows.append(dict(scheme=scheme, fold=fi, box=box, cfg='main', arm='REFERENCE', r=j, n=m['n'], P21=m['P21'],
                             I100=m['I100'], gt_n=g['n'], gt_P21=g['P21'], gt_I100=g['I100'], **sc))
    return rows, obsrows


def main(scheme):
    nfold = {'S1': 4, 'S2': 7, 'S3': 4}[scheme]
    jobs = [(scheme, fi, box) for fi in range(nfold) for box in P.load_split(scheme, fi)['test']]
    t0 = time.time(); R, O = [], []
    with Pool(min(16, len(jobs))) as pool:
        for i, (r, o) in enumerate(pool.imap_unordered(run_window, jobs)):
            R += r; O += o
            print(f'{scheme}: {i + 1}/{len(jobs)} windows  {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(R).to_csv(OUT / f'{scheme}_rows.csv', index=False)
    pd.DataFrame(O).to_csv(OUT / f'{scheme}_obs.csv', index=False)


if __name__ == '__main__':
    for s in sys.argv[1:] or ['S1']:
        main(s)
