# -*- coding: utf-8 -*-
r"""ra1417_run_window.py: window-scale runs of RA-14, RA-15, RA-16 and RA-17 (S2, 50 windows, 24 realisations).

Usage (Python with -B):  python -B ra1417_run_window.py <batch> [n_proc]
  b1  layouts and no-scanline arm (RA-14, RA-16) with and without R5, RA-17 local measures, local natural variation
  b2  RA-15 single-factor error models at 5 m (needs RA15_precomputed_parameters.csv)
  b3  RA-15 combined field-like scenarios CS-M, CS-S (with and without false hits) at 5 m and the centre line
  smoke  one window, timing only (writes nothing to the results folder)
Per-window part files go to results\RA14_15_16_17\_parts\<batch>\ and are concatenated into
ra1417_<batch>_rows.csv (and ra1417_b1_local.csv). Mirrors final_eval.run_window: fold_ctx, gt = clip(L, H),
observation and arm seeds from run_v2.sd, Paper 1 scores + map RMSE, infinite -> NaN."""
import sys, os, time, math, json
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ra1417_common as RC          # noqa: E402
from multiprocessing import Pool    # noqa: E402
import numpy as np, pandas as pd    # noqa: E402

P, V, TC, sd = RC.P, RC.V, RC.TC, RC.sd
OUT = RC.OUT
PARTS = OUT / '_parts'
LAYOUTS_B1 = ['centre', '5m', '10m', '7m', '3m', '5m+h2', 'centre+h2']


def s2_jobs():
    return [(fi, box) for fi in range(7) for box in P.load_split('S2', fi)['test']]


def window_setup(fi, box):
    RC.check_W()
    C = TC.fold_ctx('S2', fi); model = C['model']
    b = C['boxes'][box]; H = b['H']; gt = P.clip(b['L'], H)
    assert len(gt) == len(b['L']), 'clip dropped a piece: full lengths would not align'
    return C, model, b, H, gt


def base_fields(fi, box, cfg, arm, r, obs, gt, H, lines, n2, bud, p21h):
    seen = sorted({o['tid'] for o in obs if o['tid'] >= 0})
    return dict(scheme='S2', fold=fi, box=box, cfg=cfg, arm=arm, r=r, n_hits=len(obs),
                n_false=sum(1 for o in obs if o['tid'] < 0), n_seen=len(seen), frac_seen=len(seen) / max(len(gt), 1),
                n2=n2, bud=bud, p21h=p21h, effort_m=RC.effort(lines, H) if lines else 0.0)


# ============================================================================================ batch 1
def run_b1(job):
    fi, box = job
    t0 = time.time()
    C, model, b, H, gt = window_setup(fi, box)
    prm = dict(RC.BASE); dmin_cache = {}
    rows, local = [], []
    pg = RC.points(gt)
    # no-scanline arm (one reconstruction per r; rows copied under every layout label)
    nos = {}
    for clean, arm in ((True, 'STAT_NOSCAN'), (False, 'STAT_NOSCAN_noR5')):
        for r in range(24):
            rng = np.random.default_rng(sd('S2', box, r, 'noscan', 'final', 'STAT_NOSCAN'))
            rec, n2, bud, p21h = RC.frozen_g(C, [], [], H, rng, RC.stat_pool(C, H, rng), prm, dmin_cache, clean)
            nos[(arm, r)] = rec
            sc = RC.score_row(rec, gt, H, model)
            for lay in LAYOUTS_B1:
                rows.append(dict(base_fields(fi, box, lay, arm, r, [], gt, H, [], n2, bud, p21h), **sc))
    for lay in LAYOUTS_B1:
        lines = RC.layout(lay, H, prm['APERTURE'])
        for r in range(24):
            obs = V.observe(gt, lines, H, np.random.default_rng(sd('S2', box, r, lay, 'final_obs')))
            for clean, arm in ((True, 'STAT_F'), (False, 'STAT_F_noR5')):
                rng = np.random.default_rng(sd('S2', box, r, lay, 'final', 'STAT_F'))
                rec, n2, bud, p21h = RC.frozen_g(C, obs, lines, H, rng, RC.stat_pool(C, H, rng), prm, dmin_cache, clean)
                rows.append(dict(base_fields(fi, box, lay, arm, r, obs, gt, H, lines, n2, bud, p21h),
                                 **RC.score_row(rec, gt, H, model)))
                if clean and lay in ('centre', '5m'):
                    t2 = rec[:n2]
                    for a_name, a_rec in (('STAT_F', rec), ('STAT_NOSCAN', nos[('STAT_NOSCAN', r)])):
                        for d in RC.local_bands(a_rec, gt, H, lines, lay, pg=pg) + RC.local_zones(a_rec, gt, H, lines, t2, pg=pg):
                            local.append(dict(scheme='S2', fold=fi, box=box, cfg=lay, arm=a_name, r=r, **d))
    # local natural variation: 8 nearest other usable windows, shared traces removed
    near = sorted([k for k in C['boxes'] if k != box],
                  key=lambda k: (math.hypot(C['boxes'][k]['cu'] - b['cu'], C['boxes'][k]['cs'] - b['cs']), k))[:8]
    ids_w = set(b['ids'])
    for j, k in enumerate(near):
        nb = C['boxes'][k]; keep = np.array([i not in ids_w for i in nb['ids']], bool)
        Ln = P.clip(nb['L'][keep], nb['H'])
        for lay in ('centre', '5m'):
            lines = RC.layout(lay, H, prm['APERTURE'])
            # neighbour measured on its own height: its points and band areas use nb['H']; the window's on H
            prn = RC.points(Ln)
            dr, dg = RC.nearest_dist(prn[0], lines), RC.nearest_dist(pg[0], lines)
            regs = RC.BANDS_C if lay == 'centre' else [(lo * 2.5, hi * 2.5) for lo, hi in RC.BINS_5]
            for o, (lo, hi) in enumerate(regs):
                wdt = RC.band_width(lines, lo, hi)
                mr, mg = (dr >= lo) & (dr < hi), (dg >= lo) & (dg < hi)
                lr, lg = float(prn[3][mr].sum()), float(pg[3][mg].sum())
                ar, ag = wdt * nb['H'], wdt * H
                d = dict(p21_rec=lr / ar if ar > 0 else np.nan, p21_map=lg / ag if ag > 0 else np.nan)
                d['p21_abs_err'] = abs(d['p21_rec'] - d['p21_map']) if ar > 0 and ag > 0 else np.nan
                d['p21_rel_err'] = d['p21_abs_err'] / d['p21_map'] if (ag > 0 and lg > 0) else np.nan
                d['dir_w1_deg'] = (RC.w1_circ_doubled_w(prn[2][mr], prn[3][mr], pg[2][mg], pg[3][mg])
                                   if lr >= 0.5 and lg >= 0.5 else np.nan)
                d['len_rec'] = lr; d['len_map'] = lg
                if lay == 'centre':
                    reg = f'{lo}-{hi if np.isfinite(hi) else "inf"} m'; kind = 'band'
                else:
                    reg = f'{lo / 2.5:.1f}-{min(hi / 2.5, 1.0):.1f}'; kind = 'bin'
                local.append(dict(scheme='S2', fold=fi, box=box, cfg=lay, arm='NATURAL', r=j, kind=kind, region=reg,
                                  order=o + 1, width_m=wdt, neighbour=k, **d))
    return box, rows, local, time.time() - t0


# ============================================================================================ batch 2 and 3: RA-15
def load_params():
    p = OUT / 'RA15_precomputed_parameters.json'
    return json.loads(p.read_text(encoding='utf-8'))


def b2_scenarios(par):
    """(code, observation settings, list of arms (name, miss_known, r1, m_hat), extra) for the 5 m single factors"""
    a_ang, b_len = par['struct_angle_a'], par['struct_length_b']
    S = []
    for m in (0.1, 0.2, 0.4):
        c = f'M{int(round(m * 100)):02d}'
        arms = [('none', 0.0, 'std', 0.0), ('known', m, 'std', 0.0)]
        if m >= 0.2:
            arms.append(('known+daR1', m, 'da', m))
        S.append((c, dict(miss=m), arms, {}))
    S.append(('SA', dict(struct=('angle', a_ang)), [('none', 0.0, 'std', 0.0), ('known', 0.2, 'std', 0.0)], {}))
    S.append(('SL', dict(struct=('length', b_len)), [('none', 0.0, 'std', 0.0), ('known', 0.2, 'std', 0.0)], {}))
    for g, lo, hi in ((0.1, 1.0, 2.0), (0.2, 1.0, 4.0)):
        c = f'G{int(round(g * 100)):02d}'
        S.append((c, dict(gaps=(g, lo, hi)), [('none', 0.0, 'std', 0.0), ('known', g, 'std', 0.0), ('known+daR1', g, 'da', g)], {}))
    for n in (5.0, 10.0):
        S.append((f'N{int(n):02d}', dict(noise=n), [('std', 0.0, 'std', 0.0)], {}))
    for pe in (0.2, 0.5):
        S.append((f'P{int(round(pe * 10)):02d}', dict(pos_err=pe), [('std', 0.0, 'std', 0.0)], {}))
    for ml in (0.1, 0.2):
        S.append((f'L{int(round(ml * 100)):02d}', dict(mislabel=ml), [('std', 0.0, 'std', 0.0)], {}))
    for a in (10.0, 20.0):
        S.append((f'I{int(a):02d}', {}, [('std', 0.0, 'std', 0.0)], dict(incl=a)))
        S.append((f'U{int(a):02d}', {}, [('std', 0.0, 'std', 0.0)], dict(incl=a, unknown=True)))
    for ap in (0.05, 0.30):
        S.append((f'A{int(round(ap * 100)):02d}', {}, [('std', 0.0, 'std', 0.0)], dict(aperture=ap)))
    S.append(('IRR', {}, [('std', 0.0, 'std', 0.0)], dict(jitter=1.25)))
    return S


def b3_scenarios(par):
    out = []
    for code, m, g, glo, ghi, noise, pos, rfloor, incl, ff in (
            ('CSM', 0.2, 0.1, 1.0, 2.0, 6.0, 0.2, 0.10, 10.0, 0.0),
            ('CSS', 0.4, 0.2, 1.0, 4.0, 12.0, 0.5, 0.20, 20.0, 0.10),
            ('CSSnoFH', 0.4, 0.2, 1.0, 4.0, 12.0, 0.5, 0.20, 20.0, 0.0)):
        mt = 1 - (1 - m) * (1 - g)
        arms = [(f'std:m0', 0.0, 'std', 0.0)]
        for lab, mh in (('mtrue', mt), ('mhalf', 0.5 * mt), ('m1.5', 1.5 * mt)):
            arms.append((f'std:{lab}', mh, 'std', 0.0))
            arms.append((f'da:{lab}', mh, 'da', mh))
        out.append((code, dict(miss=m, gaps=(g, glo, ghi), noise=noise, pos_err=pos, false_frac=ff),
                    arms, dict(incl=incl, rfloor=rfloor, m_true=mt)))
    return out


def run_scen(job):
    """job = (batch, fi, box); runs every scenario and arm of the batch for one window"""
    batch, fi, box = job
    t0 = time.time()
    C, model, b, H, gt = window_setup(fi, box)
    par = load_params()
    prm0 = dict(RC.BASE); dmin_cache = {}
    rows = []
    full = b['full']
    if batch == 'b2':
        scen = b2_scenarios(par); layouts = ['5m']
    else:
        scen = b3_scenarios(par); layouts = ['5m', 'centre']
    for lay in layouts:
        for code, oset, arms, extra in scen:
            prm = dict(prm0)
            if 'aperture' in extra:
                prm['APERTURE'] = extra['aperture']
            for r in range(24):
                orng = np.random.default_rng(sd('S2', box, r, lay, 'ra15', code, 'obs'))
                if extra.get('incl') or extra.get('jitter'):
                    lines = RC.layout(lay, H, prm['APERTURE'], rng=orng, incl_deg=extra.get('incl', 0.0),
                                      jitter=extra.get('jitter', 0.0))
                else:
                    lines = RC.layout(lay, H, prm['APERTURE'])
                o = dict(oset)
                if 'rfloor' in extra:
                    o['mislabel'] = max(par['mislabel_rate_fold'][str(fi)], extra['rfloor'])
                if o.get('mislabel'):
                    o.update(tr_th=C['tr_th'], model=model)
                if o.get('false_frac'):
                    o['false_dirs'] = C['tr_th']
                simple = set(o) <= {'miss', 'noise', 'mislabel', 'pos_err', 'tr_th', 'model'} and all(ln['o'] == 'v' for ln in lines)
                if simple and not extra.get('unknown'):
                    obs = V.observe(gt, lines, H, orng, miss=o.get('miss', 0.0), noise=o.get('noise', 0.0),
                                    mislabel=o.get('mislabel', 0.0), pos_err=o.get('pos_err', 0.0),
                                    tr_th=o.get('tr_th'), model=o.get('model'))
                else:
                    obs = RC.observe_g(gt, lines, H, orng, full=full, unknown_dev=bool(extra.get('unknown')), **o)
                mlines = RC.believed_lines(lines, H, prm['APERTURE']) if extra.get('unknown') else lines
                for aname, mk, r1, mh in arms:
                    arm = f'E:{code}:{aname}'
                    rng = np.random.default_rng(sd('S2', box, r, lay, 'ra15', code, aname))
                    for clean in ((True, False) if (batch == 'b3' and lay == '5m' and aname == 'std:mtrue') else (True,)):
                        if not clean:
                            rng = np.random.default_rng(sd('S2', box, r, lay, 'ra15', code, aname))
                        rec, n2, bud, p21h = RC.frozen_g(C, obs, mlines, H, rng, RC.stat_pool(C, H, rng), prm, dmin_cache,
                                                         clean, miss_known=mk, r1=r1, m_hat=mh)
                        rows.append(dict(base_fields(fi, box, lay, arm + ('' if clean else '|noR5'), r, obs, gt, H, lines,
                                                     n2, bud, p21h), **RC.score_row(rec, gt, H, model)))
    return box, rows, [], time.time() - t0


def worker(job):
    try:
        if job[0] == 'b1':
            return ('ok',) + run_b1(job[1:])
        return ('ok',) + run_scen(job)
    except Exception as exc:  # pragma: no cover
        import traceback
        return ('error', job, traceback.format_exc(), 0.0, 0.0)


def main(batch, nproc=5):
    pdir = PARTS / batch; pdir.mkdir(parents=True, exist_ok=True)
    jobs = [(batch, fi, box) for fi, box in s2_jobs() if not (pdir / f'{box}_rows.csv').exists()]
    print(f'{batch}: {len(jobs)} windows to run, {nproc} processes', flush=True)
    t0 = time.time()
    with Pool(nproc) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, jobs)):
            if res[0] != 'ok':
                print('ERROR', res[1], res[2], flush=True)
                continue
            _, box, rows, local, dt = res
            pd.DataFrame(rows).to_csv(pdir / f'{box}_rows.tmp', index=False)
            os.replace(pdir / f'{box}_rows.tmp', pdir / f'{box}_rows.csv')
            if local:
                pd.DataFrame(local).to_csv(pdir / f'{box}_local.csv', index=False)
            print(f'{batch} {i + 1}/{len(jobs)} {box} {dt:.0f}s (elapsed {time.time() - t0:.0f}s)', flush=True)
    # concatenate in window order
    frames = [pd.read_csv(pdir / f'{box}_rows.csv') for _, box in s2_jobs() if (pdir / f'{box}_rows.csv').exists()]
    if len(frames) == 50:
        pd.concat(frames, ignore_index=True).to_csv(OUT / f'ra1417_{batch}_rows.csv', index=False)
        if batch == 'b1':
            loc = [pd.read_csv(pdir / f'{box}_local.csv') for _, box in s2_jobs()]
            pd.concat(loc, ignore_index=True).to_csv(OUT / 'ra1417_b1_local.csv', index=False)
        print(f'{batch}: all 50 windows written', flush=True)
    else:
        print(f'{batch}: only {len(frames)} of 50 windows present', flush=True)


if __name__ == '__main__':
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    b = sys.argv[1]
    if b == 'smoke':
        fi, box = 4, 'W04_C3'
        t = time.time(); res = run_b1((fi, box)); print('b1 one window', round(time.time() - t, 1), 's', len(res[1]), 'rows', len(res[2]), 'local')
    else:
        main(b, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
