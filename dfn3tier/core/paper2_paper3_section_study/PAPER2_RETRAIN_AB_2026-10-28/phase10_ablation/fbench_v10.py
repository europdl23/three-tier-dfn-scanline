"""Phase 10 (PREREGISTRATION_EVAE_ONLY_ABLATION.md): EVAE-only complete-network arm 'V_evae' = EVAE Tier 2 continuation (fixed_e, same stream as eKDE / E_hyb)
+ EVAE decoder Tier 3 pool. Runs phase7_followup/fbench_v3.py (itself fbench_v2 with assert-checked patches) UNCHANGED except one extra assert-checked patch
that adds the 'V_evae' branch to the arm loop. Scorer off (V3_SCORING=0), same T, budget, m, gates, thinning rho, seeds; nothing else differs.
Decoder pool = PAPER2_E2_FULL_PIPELINE_VALIDATION_2026-10-27/e2_evaluate.py::decoder_pool, applied to the present per-fold EVAE (models['evae'], selected epoch).
Minimal adaptations: tensors moved to the model device (latent noise still drawn from the CPU torch generator keyed as in decoder_pool, then moved);
cluster centres = the fold's training-only e.base['fc'] (e2 used its own inner constants); batch_yscale set to the window aspect and restored afterwards.
Eligibility as in e2/v_evaluate: the first m valid candidates in pool order (decoder ranking)."""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; FU = HERE.parent / 'phase7_followup'
assert os.environ.get('V3_SCORING') == '0' and not os.environ.get('V3_ROLL') and not os.environ.get('V3_T')
src = (FU / 'fbench_v3.py').read_text()
a = "for a, b in rep:"; assert src.count(a) == 1; src = src.replace(a, "rep += V10_REP\n" + a, 1)
V10_REP = [("                elif lab == 'T12':", "                elif lab == 'V_evae': q = V_POOL(e, b, models, lab, m, FC)\n                elif lab == 'T12':")]
ns = dict(__name__='fbench_v3_v10', __file__=str(FU / 'fbench_v3.py'), V10_REP=V10_REP)
exec(compile(src, str(FU / 'fbench_v3.py') + ' [V10]', 'exec'), ns)
mod = ns['mod']; PE = mod.PE; LOG = []


def decoder_pool(e, m, obs, face, budget, fid, g, fc):
    dev = next(m.parameters()).device; ym = float(face['ym']); old = getattr(m, 'batch_yscale', None); m.batch_yscale = torch.tensor([ym], device=dev); o = np.asarray(obs, np.float32).reshape(-1, 4)
    gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{fid}|{g}|decoder'.encode()).digest()[:4], 'little'))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{fid}|{g}|decoder_np'.encode()).digest()[:8], 'little'))
    T_ = lambda x: torch.tensor(x, device=dev)
    try:
        with torch.no_grad():
            if len(o): sm, sl, gm, gl = m.encode(T_(o[None, :, :2]), T_(o[None, :, 2:3]), T_(o[None, :, 3:4]), torch.ones(1, len(o), dtype=torch.bool, device=dev))
            else: sm, sl, gm, gl = m.encode(torch.zeros(1, 1, 2, device=dev), torch.zeros(1, 1, 1, device=dev), torch.zeros(1, 1, 1, device=dev), torch.zeros(1, 1, dtype=torch.bool, device=dev))
        pool, draw, native = [], 0, None
        while len(pool) < budget and draw < 60:
            with torch.no_grad():
                if draw == 0: sz, gz = sm, gm
                else: sz = sm + torch.exp(.5 * sl) * torch.randn(sm.shape, generator=gen).to(dev); gz = gm + torch.exp(.5 * gl) * torch.randn(gm.shape, generator=gen).to(dev)
                c, l, ml, kp, el = m.decode(sz, gz)
            c = c[0].cpu().numpy(); l = l[0, :, 0].cpu().numpy(); pf = torch.softmax(ml[0], -1).cpu().numpy().astype(float); kp = kp[0].cpu().numpy(); pe = torch.sigmoid(el[0]).cpu().numpy()
            if draw == 0: native = int((pe > .5).sum()); pe0 = pe
            P = []
            for i in np.argsort(-pe):
                q = int(rng.choice(4, p=pf[i] / pf[i].sum())); A = (fc[q] + rng.vonmises(0, max(float(kp[i, q]), 1e-3)) / 2) % np.pi if q < 3 else rng.uniform(0, np.pi); P.append([c[i, 0], c[i, 1], l[i], A])
            P = np.asarray(P, np.float32); pool += [x for x in (e.s.clip_outer(e.r, P, face) if len(P) else P)]; draw += 1
    finally:
        m.batch_yscale = old
    return np.asarray(pool[:budget], np.float32).reshape(-1, 4), native, float(pe0.max()), draw


def V_POOL(e, b, models, lab, m, FC):
    r = e.r; raw, nat, pmax, nd = decoder_pool(e, models['evae'], b['obs'], b['pub'], b['budget'], b['fid'], b['seed'], np.asarray(FC, float))
    q = PE.make_q(e, b, lab, 'B', raw, b['fixed_e'], np.ones(len(raw), bool), r.assign(raw, FC), np.full(len(raw), -1), ['decoder'] * len(raw))
    ids = np.flatnonzero(q['valid'])[:m]; el = np.zeros(len(raw), bool); el[ids] = True; q['eligible'] = el
    LOG.append(dict(fold=mod.FOLD, window=b['fid'], generation_seed=b['seed'], native_count=nat, max_existence=pmax, draws=nd, n_pool=len(raw), budget=b['budget'], m=m, n_valid=int(q['valid'].sum()), n_eligible=int(el.sum()), n_fixed=len(b['fixed_e'])))
    return q


mod.__dict__['V_POOL'] = V_POOL
if __name__ == '__main__':
    mod._v3_main()
    pd.DataFrame(LOG).to_csv(mod.OUTD / f"DECODER_POOL_LOG_{mod.TASK['layout']}_{mod.V3_TAG}.csv", index=False)
