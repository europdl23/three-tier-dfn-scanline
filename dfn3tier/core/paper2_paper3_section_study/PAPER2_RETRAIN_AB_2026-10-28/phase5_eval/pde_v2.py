"""V2 COPY of plan_d_step3_training_evaluation_2026-10-21/plan_d_evaluate.py: only RUNS and SELECTED_EPOCHS paths changed (env V2_RUNS, V2_SEL)."""
"""Plan D Step 3 evaluation job (guarded EVALUATE mode on stages/<fold>/eval). Reads frozen models from runs/ and training-only
references from runs/<fold>/<treatment>/references/; reads test truth only from its own eval stage. Never writes to runs/.
Tasks (PLAN_D_TASK JSON):
  info   : Level 1 information metrics I-1 (Transformer teacher-forced skill), I-2 (continuation), I-3 (latent probes),
           I-6 (model-free ceiling) for every outer seed and half model of the fold, plus the oracle-epoch bound (all saved epochs)
  recon  : reconstructions for one (treatment, training seed, layout, scorer) group: arms among B, C, D, E (B only once)
"""
import os, sys, json, math, hashlib, time, copy
from pathlib import Path
S3 = Path((__import__('os').environ['DFN3_CORE'] + r'/paper2_paper3_section_study/plan_d_step3_training_evaluation_2026-10-21'.replace(chr(92), '/'))); sys.path.insert(0, str(S3 / 'engine'))
from training import *  # noqa: guarded runtime (evaluate mode)
import experiments as X  # noqa
from continuity_contract import install  # noqa
from followup import angle_sampler  # noqa
import numpy as np, pandas as pd, torch  # noqa
from scipy.stats import spearmanr  # noqa
TASK = json.loads(os.environ['PLAN_D_TASK']); OUTD = Path(os.environ['PLAN_D_OUT']).resolve(); FOLD = TASK['fold']
SMOKE = os.environ.get('PLAN_D_SMOKE_EVAL') == '1'   # smoke test only: throwaway models on a training-data pseudo-eval stage
RUNS = Path(os.environ['V2_RUNS'])   # V2 copy: runs of the A/B retraining
if SMOKE: SEL = dict(selected={'P1': dict(evae=1, head=1, tf_latent=1, tf_no_latent=1)})
else: SEL = json.loads(Path(os.environ['V2_SEL']).read_text())   # V2 copy: selected epochs of the A/B retraining
EP = SEL['selected'][FOLD]


def keyrestore(stats, e):
    s = e.m.keyrestore(copy.deepcopy(stats))
    for k in ['fc', 'shares']: s[k] = np.asarray(s[k])
    s['pools'] = {k: {n: np.asarray(v) for n, v in p.items()} for k, p in s['pools'].items()}; return s


def use_stats(e, stats):
    e.base = keyrestore(stats, e); e.r.STATS = copy.deepcopy(e.base); e.sequence_cache = {}


def load_models(e, treatment, seed, epochs=None):
    r = e.r; ep = epochs or EP; d = RUNS / FOLD / treatment
    ev = make_evae(e, True); ev.load_state_dict(torch.load(d / f'evae_s{seed}' / f"epoch_{ep['evae']:03d}.pt", map_location=r.DEV, weights_only=True)['state_dict']); ev.eval()
    hn = r.gen['EVAEContHead']().to(r.DEV); hn.load_state_dict(torch.load(d / f'head_s{seed}' / f"epoch_{ep['head']:03d}.pt", map_location=r.DEV, weights_only=True)['state_dict']); hn.eval()
    cfg = json.loads((d / f'head_s{seed}' / 'config.json').read_text())['head_normaliser']
    mu = torch.tensor(cfg['Xmean'], device=r.DEV, dtype=torch.float32); sd = torch.tensor(cfg['Xstd'], device=r.DEV, dtype=torch.float32)
    nets = {}
    for c, lat in [('tf_latent', True), ('tf_no_latent', False)]:
        n = e.Net().to(r.DEV); n.latent = lat; n.load_state_dict(torch.load(d / f'{c}_s{seed}' / f"epoch_{ep[c]:03d}.pt", map_location=r.DEV, weights_only=True)['state_dict']); n.eval()
        with np.load(d / f'{c}_s{seed}' / 'residuals.npz') as z: errs = {k: z[k] for k in z.files}
        nets[c] = dict(net=n, errors=errs, horizons=json.loads((d / f'{c}_s{seed}' / 'CALIBRATION.json').read_text())['horizons'])
    return dict(evae=ev, head=(hn, mu, sd)), nets


def refs(treatment):
    d = RUNS / FOLD / treatment / 'references'
    return json.loads((d / 'REFERENCES.json').read_text()), json.loads((d / 'training_only_statistics.json').read_text()), pd.read_csv(d / 'RIDGE_TRAIN_TABLE.csv')


# ============================================================ Level 1 information
def tf_rows(e, net, seq_items, R, label):
    r = e.r; out = []; tr = R['tf_refs']; step = {int(k): v for k, v in tr['per_step'].items()}; marg = tr['marginal']; fm = {int(k): v for k, v in tr['family_means'].items()}
    tsh = np.asarray(tr['target_family_shares']); axl = lambda a, b: 1 - np.cos(2 * (a - b))
    with torch.no_grad():
        for x in seq_items:
            e.s.set_aspect(net, [x], r.DEV); c, cm, y, mask, fam, sz, gz = r.seq_batch([x]); di = torch.cat([torch.zeros(1, 1, 5, device=r.DEV), y[:, :-1]], 1)
            p, L, a, f = net(c, cm, di, torch.zeros(di.shape[:2], dtype=torch.bool, device=r.DEV), sz, gz)
            p = p[0].cpu().numpy(); L = L[0, :, 0].cpu().numpy(); pa = np.arctan2(a[0, :, 0].cpu().numpy(), a[0, :, 1].cpu().numpy()); lp = torch.log_softmax(f[0], -1).cpu().numpy()
            yy = y[0].cpu().numpy(); ta = np.arctan2(yy[:, 3], yy[:, 4]); fa = fam[0].cpu().numpy(); asp = x['aspect']
            for j in range(len(yy)):
                k = int(fa[j]); pk = int(lp[j].argmax()); s = step.get(j, None) or dict(x=marg['x'], y_frac=marg['y_frac'], loglen=marg['loglen'], angle=marg['angle'])
                fmk = fm.get(pk, marg)
                out.append(dict(model=label, window=x['face_id'], step=j, family=k, pred_family=pk,
                                fam_model=-lp[j, k], fam_ref=-math.log(max(tsh[k], 1e-12)),
                                len_model=(L[j] - yy[j, 2]) ** 2, len_ref_family=(fmk['loglen'] - yy[j, 2]) ** 2, len_ref_step=(s['loglen'] - yy[j, 2]) ** 2, len_ref_marginal=(marg['loglen'] - yy[j, 2]) ** 2,
                                ang_model=axl(pa[j], ta[j]), ang_ref_family=axl(fmk['angle'], ta[j]), ang_ref_step=axl(s['angle'], ta[j]), ang_ref_marginal=axl(marg['angle'], ta[j]),
                                pos_model=float(((p[j] - yy[j, :2]) ** 2).sum()), pos_ref_step=(s['x'] - yy[j, 0]) ** 2 + (s['y_frac'] * asp - yy[j, 1]) ** 2,
                                pos_ref_marginal=(marg['x'] - yy[j, 0]) ** 2 + (marg['y_frac'] * asp - yy[j, 1]) ** 2, pos_ref_family=(fmk['x'] - yy[j, 0]) ** 2 + (fmk['y_frac'] * asp - yy[j, 1]) ** 2,
                                pred_loglen=float(L[j]), true_loglen=float(yy[j, 2]), pred_x=float(p[j, 0]), true_x=float(yy[j, 0]), pred_y=float(p[j, 1] / asp), true_y=float(yy[j, 1] / asp),
                                pred_angle=float(pa[j] % np.pi), true_angle=float(ta[j] % np.pi)))
    return out


def head_rows(e, models, R, label):
    r = e.r; ev = models['evae']; hn, mu, sd = models['head']; ext = {str(k): v for k, v in R['ext_refs'].items()}; out = []
    for z in e.valid:
        if z['layout'] != '5m': continue
        latz = torch.cat(r.tensor_encode(ev, z['obs']), -1).detach().cpu().numpy()[0]
        for members in e.env['anchor_groups'](z['obs'], public_face(z['face']), z['strips']):
            for i, side in e.env['group_outer_observations'](z['obs'], members):
                feat, a, u, bound = e.env['continuation_features'](z['obs'], i, side, z['face'])
                if not any(abs(a[0] - edge) < 1e-6 for st in z['strips'] for edge in st) or bound <= 1e-8: continue
                p, q = r.ends(z['arr'][z['parents'][i]:z['parents'][i] + 1]); true = max(float((p[0] - a) @ u), float((q[0] - a) @ u), 0)
                k = int(r.assign(z['obs'][i:i + 1], e.base['fc'])[0]); cap = min(bound, float(np.quantile(e.base['pools'][k]['lengths_m'], .95)) / 20)
                Xf = (torch.tensor(np.r_[latz, feat], device=r.DEV, dtype=mu.dtype) - mu) / sd
                with torch.no_grad(): m_, l_ = hn(Xf[None])
                m_, l_ = float(m_), float(l_); evae_pred = min(math.exp(min(max(m_, -15), 2)), cap); stat_pred = min(float(np.median(e.base['pools'][k]['lengths_m'])) / 20 / 2, cap)
                y = math.log(max(true, 1e-6)); g = ext.get(str(k), ext['all']); mu_r = g['log_mean'] - math.log(20); sd_r = g['log_sd']
                out.append(dict(model=label, window=z['face']['id'], family=k, true_m=true * 20, evae_m=evae_pred * 20, stat_m=stat_pred * 20, bound_m=bound * 20,
                                nll_model=.5 * ((y - m_) / math.exp(l_)) ** 2 + l_, nll_ref=.5 * ((y - mu_r) / sd_r) ** 2 + math.log(sd_r)))
    return out


def latent_rows(e, net, seq_items, train_seq, models, label):
    r = e.r; out = []; msz = torch.cat([x['sz'] for x in train_seq]).mean(0, keepdim=True); mgz = torch.cat([x['gz'] for x in train_seq]).mean(0, keepdim=True)
    W = dict(e.source_module.REFERENCE)
    with torch.no_grad():
        for k, x in enumerate(seq_items):
            other = seq_items[(k + 3) % len(seq_items)]; y = dict(x); y['sz'], y['gz'] = other['sz'], other['gz']; m0 = dict(x); m0['sz'], m0['gz'] = msz, mgz
            out.append(dict(model=label, window=x['face_id'], own=float(tf_loss(e, net, [x], W, reference=True)[0]), swapped=float(tf_loss(e, net, [y], W, reference=True)[0]),
                            mean_latent=float(tf_loss(e, net, [m0], W, reference=True)[0])))
    # statistical context with the E model (T1 analogue)
    stat_seq = [s for s in e.sequences(dict(seed=models['_key'] + 777, models=models), False)['validation'] if s['layout'] == '5m']
    sl = {s['face_id']: float(tf_loss(e, net, [dict(s, sz=s['sz'], gz=s['gz'])], W, reference=True)[0]) for s in stat_seq}
    for row in out: row['stat_context'] = sl.get(row['window'], math.nan)
    return out


def ridge_rows(e, R, ridge_train, label):
    r = e.r; fc = np.asarray(R['fc']); sh = np.asarray(R['shares']); rows = []
    sys.path.insert(0, str(S3)); from plan_d_references import window_summary  # noqa
    for fid, face in e.faces.items():
        if face['role'] != 'validation': continue
        st = r.strips('5m', 20); obs, par = r.observe(face['arr'], st); hid = face['arr'][~np.isin(np.arange(len(face['arr'])), np.unique(par))]
        f, t = window_summary(r, fc, sh, obs, hid, face['ym']); oo = r.assign(obs, fc) if len(obs) else np.zeros(0, int); ho = r.assign(hid, fc) if len(hid) else np.zeros(0, int)
        post = (np.bincount(oo, minlength=4) + 4 * sh) / (len(oo) + 4)
        rows.append(dict(model=label, window=fid, **f, **t, dir_logloss_obs=float(-np.log(post[ho]).mean()) if len(ho) else np.nan, dir_logloss_prior=float(-np.log(sh[ho]).mean()) if len(ho) else np.nan, n_hid_lab=len(ho)))
    T = pd.DataFrame(rows); feats = ['n_obs', 'obs_mean_y', 'obs_mean_loglen', 'obs_share_0', 'obs_share_1', 'obs_share_2', 'obs_share_3', 'height_m']; res = []
    for tgt in ['hid_mean_y', 'hid_mean_loglen', 'hid_share_0', 'hid_share_1', 'hid_share_2', 'hid_share_3', 'n_hidden']:
        A = ridge_train.dropna(subset=[tgt]); B = T.dropna(subset=[tgt])
        Xa = A[feats].to_numpy(float); Xb = B[feats].to_numpy(float); m_, s_ = Xa.mean(0), Xa.std(0); s_[s_ == 0] = 1
        Za, Zb = (Xa - m_) / s_, (Xb - m_) / s_; ya = A[tgt].to_numpy(float); w = np.linalg.solve(Za.T @ Za + np.eye(len(feats)), Za.T @ (ya - ya.mean()))
        pred = ya.mean() + Zb @ w; yb = B[tgt].to_numpy(float)
        res.append(dict(model=label, target=tgt, n_train=len(A), n_test=len(B), mse_ridge=float(((yb - pred) ** 2).mean()), mse_train_mean=float(((yb - ya.mean()) ** 2).mean()),
                        skill=1 - float(((yb - pred) ** 2).mean()) / float(((yb - ya.mean()) ** 2).mean())))
    res.append(dict(model=label, target='hidden_family_logloss (Dirichlet a=4)', n_train=0, n_test=int(T.n_hid_lab.sum()), mse_ridge=float(np.nansum(T.dir_logloss_obs * T.n_hid_lab) / T.n_hid_lab.sum()),
                    mse_train_mean=float(np.nansum(T.dir_logloss_prior * T.n_hid_lab) / T.n_hid_lab.sum()), skill=1 - float(np.nansum(T.dir_logloss_obs * T.n_hid_lab)) / float(np.nansum(T.dir_logloss_prior * T.n_hid_lab))))
    return T, res


def counterfactual_rows(e, net, items, label, arm):
    """I-4: swap the observed context (ctx, sz, gz) with another test window of the fold (k+3 cyclic); compare the change in
    teacher-forced predicted hidden summaries (mean log length, mean y, family distribution) with the change in the true summaries."""
    r = e.r; out = []
    def summ(x, ctx_from):
        y = dict(x); y['ctx'], y['sz'], y['gz'] = ctx_from['ctx'], ctx_from['sz'], ctx_from['gz']
        with torch.no_grad():
            e.s.set_aspect(net, [y], r.DEV); c, cm, yy, mask, fam, sz, gz = r.seq_batch([y]); di = torch.cat([torch.zeros(1, 1, 5, device=r.DEV), yy[:, :-1]], 1)
            p, L, a, f = net(c, cm, di, torch.zeros(di.shape[:2], dtype=torch.bool, device=r.DEV), sz, gz)
        pr = torch.softmax(f[0], -1).mean(0).cpu().numpy()
        return float(L[0, :, 0].mean()), float(p[0, :, 1].mean() / x['aspect']), pr
    for k, x in enumerate(items):
        o = items[(k + 3) % len(items)]
        if o['face_id'] == x['face_id']: continue
        l0, y0, p0 = summ(x, x); l1, y1, p1 = summ(x, o)
        tl_x = float(np.mean(x['tgt'][:, 2])); tl_o = float(np.mean(o['tgt'][:, 2])); ty_x = float(np.mean(x['tgt'][:, 1]) / x['aspect']); ty_o = float(np.mean(o['tgt'][:, 1]) / o['aspect'])
        fx = np.bincount(x['fam'], minlength=4) / len(x['fam']); fo = np.bincount(o['fam'], minlength=4) / len(o['fam'])
        out.append(dict(model=label, arm=arm, window=x['face_id'], other=o['face_id'], d_pred_loglen=abs(l1 - l0), d_true_loglen=abs(tl_o - tl_x), d_pred_y=abs(y1 - y0), d_true_y=abs(ty_o - ty_x),
                        d_pred_fam_tv=float(.5 * np.abs(p1 - p0).sum()), d_true_fam_tv=float(.5 * np.abs(fo - fx).sum()),
                        dir_loglen=float(np.sign(l1 - l0) * np.sign(tl_o - tl_x)), dir_y=float(np.sign(y1 - y0) * np.sign(ty_o - ty_x))))
    return out


def task_info(e):
    full_stats = copy.deepcopy(e.base); treatments = [('outer_full', s) for s in [42001, 42002, 42003]] + ([('half_A', 42001), ('half_B', 42001)] if FOLD in ['P1', 'P2', 'P3', 'P4'] else [])
    if 'treatments' in TASK: treatments = [tuple(t) for t in TASK['treatments']]
    TF, HD, LT, RW, RG, OR, CF = [], [], [], [], [], [], []
    for treatment, seed in treatments:
        R, stats, rt = refs(treatment); use_stats(e, stats); label = f'{treatment}_s{seed}'
        models, nets = load_models(e, treatment, seed); models['_key'] = seed * 10 + len(TF) % 7
        for c, lat in [('tf_latent', True), ('tf_no_latent', False)]:
            seq = e.sequences(dict(seed=hash((treatment, seed, c)) % 10 ** 8, models=models), lat); items = [x for x in seq['validation'] if x['layout'] == '5m']
            TF += [dict(r_, arm='E' if lat else 'D') for r_ in tf_rows(e, nets[c]['net'], items, R, label)]
            if lat: LT += latent_rows(e, nets[c]['net'], items, [x for x in seq['train'] if x['layout'] == '5m'], models, label)
            CF += counterfactual_rows(e, nets[c]['net'], items, label, 'E' if lat else 'D')
            # oracle-epoch bound: every saved epoch of this model (selected epochs already frozen)
            for ep in range(0, 51):
                pth = RUNS / FOLD / treatment / f'{c}_s{seed}' / f'epoch_{ep:03d}.pt'
                if not pth.exists(): continue
                n = e.Net().to(e.r.DEV); n.latent = lat; n.load_state_dict(torch.load(pth, map_location=e.r.DEV, weights_only=True)['state_dict']); n.eval()
                d = pd.DataFrame(tf_rows(e, n, items, R, label))
                OR.append(dict(model=label, arm='E' if lat else 'D', component=c, epoch=ep, selected=ep == EP[c], **{f'skill_{h}_{ref}': 1 - d[f'{h}_model'].mean() / d[f'{h}_ref_{ref}'].mean()
                               for h in ['len', 'ang', 'pos'] for ref in ['step', 'family', 'marginal']}, skill_fam=1 - d.fam_model.mean() / d.fam_ref.mean()))
        HD += head_rows(e, models, R, label)
        hn0 = models['head'][0]
        for ep in range(0, 51):
            pth = RUNS / FOLD / treatment / f'head_s{seed}' / f'epoch_{ep:03d}.pt'
            if not pth.exists(): continue
            hn = e.r.gen['EVAEContHead']().to(e.r.DEV); hn.load_state_dict(torch.load(pth, map_location=e.r.DEV, weights_only=True)['state_dict']); hn.eval()
            mm = dict(models, head=(hn, models['head'][1], models['head'][2])); d = pd.DataFrame(head_rows(e, mm, R, label))
            OR.append(dict(model=label, arm='C', component='head', epoch=ep, selected=ep == EP['head'], skill_nll_gain=float((d.nll_ref - d.nll_model).mean()),
                           skill_mae=1 - float((d.evae_m - d.true_m).abs().mean()) / float((d.stat_m - d.true_m).abs().mean())))
        T, res = ridge_rows(e, R, rt, label); RW.append(T); RG += res
    use_stats(e, full_stats)
    OUTD.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(TF).to_csv(OUTD / 'I1_TF_TARGET_ROWS.csv.gz', index=False); pd.DataFrame(HD).to_csv(OUTD / 'I2_CONTINUATION_ENDS.csv', index=False)
    pd.DataFrame(LT).to_csv(OUTD / 'I3_LATENT_PROBES.csv', index=False); pd.concat(RW).to_csv(OUTD / 'I6_TEST_WINDOW_SUMMARIES.csv', index=False)
    pd.DataFrame(RG).to_csv(OUTD / 'I6_MODEL_FREE_SKILL.csv', index=False); pd.DataFrame(OR).to_csv(OUTD / 'ORACLE_EPOCH_BOUND.csv', index=False); pd.DataFrame(CF).to_csv(OUTD / 'I4_COUNTERFACTUAL_CONTEXT.csv', index=False)


# ============================================================ reconstruction
def logging_posterior(e, inner, log):
    def post(position, loglength, angle, logits, face, rng, n):
        a, s, fids = inner(position, loglength, angle, logits, face, rng, n)
        pr = np.exp(logits - np.max(logits)); pr /= pr.sum()
        log.append(dict(step=len(log), net_x=float(position[0]), net_y=float(position[1]), net_loglen=float(loglength), net_angle=float(angle), **{f'net_p{k}': float(pr[k]) for k in range(4)}))
        return a, s, fids
    return post


def tf_pool(e, b, models, netinfo, latent, label, arm):
    r = e.r; net = netinfo['net']; net.aspect = torch.tensor([b['pub']['ym']], device=r.DEV)
    e.current.update(errors=netinfo['errors'], layout='5m'); e.env['HORIZON'] = netinfo['horizons']['5m']
    old = e.env['posterior_draws']; log = []
    e.env['posterior_draws'] = logging_posterior(e, angle_sampler(e, .5, (b['fid'], b['seed'], 'angular_contrast')), log)
    fixed = b['fixed_e'] if latent else b['fixed_s']
    try:
        raw, eligible, sampled, step = e.env['sample_neural'](net, fixed, b['pub'], b['budget'], latent, models['evae'], streams((b['fid'], b['seed'], 'local_prefix_choice'))['selection'], b['st'])
    finally:
        e.env['posterior_draws'] = old
    q = make_q(e, b, label, arm, raw, fixed, np.asarray(eligible, bool), sampled, step, ['transformer'] * len(raw)); q['network_log'] = log; return q


def make_q(e, b, label, arm, raw, fixed, eligible, sampled, step, source):
    r = e.r; valid = np.array([r.candidate_gate(x, b['pub'], b['st']) is None for x in raw])
    return dict(obs=b['obs'], parents=b['parents'], st=b['st'], pub=b['pub'], face=b['face'], fixed=fixed, raw=raw, eligible=eligible, sampled=sampled, step=step, source=source,
                raw_sha=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(), valid=valid, rng=None,
                meta=dict(window_id=b['fid'], generation_seed=b['seed'], arm=arm, case_key=f"{b['fid']}__{label}__g{b['seed']}", training_seed=b['training_seed'], label=label))


def task_recon(e):
    r = e.r; treatment = TASK.get('treatment', 'outer_full'); seed = int(TASK['training_seed']); layout = TASK['layout']; scoring = bool(TASK['scoring']); arms = TASK['arms']
    R, stats, _ = refs(treatment)
    # Stage C (O12): half treatments run only on the composed half eval stages (stages_stageC/<fold>/<half>/eval)
    if treatment != 'outer_full' and STAGE.parts[-2] != treatment: raise RuntimeError('half-data reconstruction requires its own Stage C eval stage')
    cp = R['count_rule']; checked, _ = install(e); models, nets = load_models(e, treatment, seed); X.HERE = OUTD
    phase = f'{layout}_{"on" if scoring else "off"}'; M, C, A = [], [], []
    wins = sorted(k for k, f in e.faces.items() if f['role'] == 'validation')
    for fid in wins:
        face = e.faces[fid]; pub = public_face(face)
        for g in [72001, 72002, 72003]:
            if layout == 'L0': obs, parents, st, _ = e.observe(face, '5m', 'clean', g)
            else:
                st = [((10 - .05) / 20, (10 + .05) / 20)] if int(round(face['x0'])) % 60 == 0 else []
                obs, parents = r.observe(face['arr'], st) if st else (np.zeros((0, 4), np.float32), np.zeros(0, int))
            e.s.corrected_references(r, e.base, pub, e.cm)
            k = len(st); p21 = (len(obs) + cp['a_prior']) / (k * face['h'] * cp['e_v'] + cp['a_prior'] / cp['prior']); T = max(1, int(round(cp['n_rate'] * 20 * face['h'] * (p21 / cp['prior']))))
            budget = 24 * min(128, math.ceil(1.5 * T)) + 10 * T; m = math.ceil(budget / 24)
            rng = np.random.default_rng(g); fixed_e = e.env['indicated'](obs, pub, st, 'evae', models, rng); stat_c = r.statistical_proposals(pub, budget, 'stat', rng)
            rng2 = np.random.default_rng(g); fixed_s = e.env['indicated'](obs, pub, st, 'stat', models, rng2); stat_b = r.statistical_proposals(pub, budget, 'stat', rng2)
            b = dict(face=face, pub=pub, obs=obs, parents=parents, st=st, fid=fid, seed=g, budget=budget, fixed_e=fixed_e, fixed_s=fixed_s, training_seed=seed)
            qs = {}
            if 'E' in arms: qs['E'] = (tf_pool(e, b, models, nets['tf_latent'], True, 'E', 'E'), 'native')
            if 'D' in arms: qs['D'] = (tf_pool(e, b, models, nets['tf_no_latent'], False, 'D', 'D'), 'native')
            for arm, raw, fx in [('C', stat_c, fixed_e), ('B', stat_b, fixed_s)]:
                if arm not in arms: continue
                q = make_q(e, b, arm, arm, raw, fx, np.ones(len(raw), bool), r.assign(raw, e.base['fc']), np.full(len(raw), -1), ['stat'] * len(raw))
                gg = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{fid}|{g}|{arm}|equal_m'.encode()).digest()[:8], 'little'))
                ids = np.flatnonzero(q['valid']); keep = gg.choice(ids, min(m, len(ids)), replace=False); el = np.zeros(len(raw), bool); el[keep] = True; q['eligible'] = el
                qs[arm] = (q, 'native')
            checked.strips = st
            try:
                for arm, (q, elig) in qs.items():
                    lab = arm if scoring else arm + '_off'; q['meta']['label'] = lab; q['meta']['case_key'] = f"{fid}__{lab}__t{seed}__g{g}"
                    f_, i_, l_, bud = X.replay(e, q, policy='slotsafe', eligibility=elig, tail=True, keyed=True, target=T, scoring=scoring)
                    met, cl = X.save_case(e, q, lab, phase, f_, i_, l_, bud)
                    dest = OUTD / phase / 'cases' / (lab + '__' + q['meta']['case_key'])
                    if 'network_log' in q: pd.DataFrame(q['network_log']).to_csv(dest / 'network_steps.csv', index=False)
                    dump(dest / 'case_inputs.json', dict(T=T, raw_budget=budget, m=m, strips=len(st), n_obs=len(obs), unique_observed=int(len(np.unique(parents))), layout=layout, scoring=scoring,
                                                     eligible_valid=int((q['valid'] & q['eligible']).sum()), arm=arm, training_seed=seed, generation_seed=g))
                    met.update(label=lab, arm=arm, layout=layout, scoring=scoring, T_target=T, raw_budget=budget, m=m, eligible_valid=int((q['valid'] & q['eligible']).sum()), n_obs=len(obs),
                               unique_observed=int(len(np.unique(parents))), n_strips=len(st)); M.append(met)
                    for x in cl: x.update(label=lab, arm=arm, layout=layout, scoring=scoring)
                    C.extend(cl)
            finally:
                del checked.strips
        print(fid, flush=True)
    OUTD.mkdir(parents=True, exist_ok=True); pd.DataFrame(M).to_csv(OUTD / f'CASE_METRICS_{phase}_t{seed}.csv', index=False); pd.DataFrame(C).to_csv(OUTD / f'CLUSTER_METRICS_{phase}_t{seed}.csv', index=False)


if __name__ == '__main__':
    e = build_engine()
    assert all(f['role'] in ('train', 'validation') for f in e.faces.values())
    {'info': task_info, 'recon': task_recon}[TASK['task']](e)
    dump(OUTD / f"COMPLETE_{TASK['task']}_{TASK.get('training_seed', '')}_{TASK.get('layout', '')}_{TASK.get('scoring', '')}.json", dict(task=TASK, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
