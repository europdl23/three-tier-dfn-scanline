def sample_neural_sampled(mdl, context, face, budget, latent, evae, rng, st):
    ctx = context[np.argsort(context[:, 0])] if len(context) else context
    ct = r.common['tok_array'](ctx)
    if not len(ct):
        ct = np.zeros((1, 5), np.float32)
    c = torch.tensor(ct, device=r.DEV)[None]
    cm = torch.zeros(1, len(ct), device=r.DEV, dtype=torch.bool)
    (sz, gz) = r.tensor_encode(evae, context)
    seq = torch.zeros(1, 1, 5, device=r.DEV)
    rows = []
    eligible = []
    sampled = []
    steps = []
    with torch.no_grad():
        step = 0
        while len(rows) < budget:
            if seq.size(1) > HORIZON:
                seq = torch.zeros(1, 1, 5, device=r.DEV)
            (p, l, a, f) = mdl(c, cm, seq, torch.zeros(1, seq.size(1), device=r.DEV, dtype=torch.bool), sz if latent else None, gz if latent else None)
            pos = p[0, -1].cpu().numpy()
            logL = float(l[0, -1, 0])
            angle = float(torch.atan2(a[0, -1, 0], a[0, -1, 1])) % np.pi
            fl = f[0, -1].cpu().numpy()
            (candidates, scores, fids) = posterior_draws(pos, logL, angle, fl, face, rng, min(24, budget - len(rows)))
            valid = [i for (i, row) in enumerate(candidates) if r.candidate_gate(row, face, st) is None]
            best = int(rng.choice(valid)) if valid else None
            rows.extend(candidates)
            eligible.extend([i == best for i in range(len(candidates))])
            sampled.extend(fids)
            steps.extend([step] * len(candidates))
            if best is not None:
                new = candidates[best]
            else:
                new = np.array([np.clip(pos[0], 0, 1), np.clip(pos[1], 0, face['ym']), math.exp(float(np.clip(logL, -12, 0))), angle], np.float32)
            seq = torch.cat([seq, torch.tensor(r.common['tok_array'](new[None]), device=r.DEV)[None]], dim=1)
            step += 1
    return (np.asarray(rows, np.float32), np.asarray(eligible), np.asarray(sampled), np.asarray(steps))