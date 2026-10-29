import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def make_enc(d, h=4, l=2):
    try:
        el = nn.TransformerEncoderLayer(
            d, h, d * 4, 0.1, batch_first=True, norm_first=True, enable_nested_tensor=False
        )
        return nn.TransformerEncoder(el, l, enable_nested_tensor=False)
    except:
        el = nn.TransformerEncoderLayer(d, h, d * 4, 0.1, batch_first=True, norm_first=True)
        return nn.TransformerEncoder(el, l)


class DualLatentVAE(nn.Module):
    def __init__(self, sd=12, gd=20, N=120, dm=64, yscale=None):
        super().__init__()
        if yscale is None:
            raise ValueError("yscale must be provided from build_physics_stats")
        self.sd = sd
        self.gd = gd
        self.N = N
        self.yscale = float(yscale)
        self.se = nn.Sequential(nn.Linear(2, dm), nn.ReLU(), nn.Linear(dm, dm))
        self.st = make_enc(dm)
        self.sp = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))
        self.smu = nn.Linear(dm, sd)
        self.slv = nn.Linear(dm, sd)
        self.ge = nn.Sequential(nn.Linear(2, dm), nn.ReLU(), nn.Linear(dm, dm))
        self.gt = make_enc(dm)
        self.gp = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))
        self.gmu = nn.Linear(dm, gd)
        self.glv = nn.Linear(dm, gd)
        self.cd = nn.Sequential(
            nn.Linear(sd, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, N * 2),
        )
        self.ld = nn.Sequential(
            nn.Linear(gd, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, N),
        )
        self.am = nn.Sequential(nn.Linear(gd, 128), nn.ReLU(), nn.Linear(128, N * 4))
        self.ak = nn.Sequential(nn.Linear(gd, 128), nn.ReLU(), nn.Linear(128, N * 4))
        self.eh = nn.Sequential(
            nn.Linear(sd + gd, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N),
        )
        self.cd_shift = nn.Linear(sd, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.ld[-1].bias.fill_(-2.0)
            nn.init.xavier_uniform_(self.cd_shift.weight, gain=0.5)
            nn.init.zeros_(self.cd_shift.bias)

    def _pool(self, f, net, pm):
        s = net(f).masked_fill(pm.unsqueeze(-1), float("-inf"))
        w = torch.nan_to_num(torch.softmax(s, 1), 0.0)
        return (f * w).sum(1)

    def encode(self, C, L, A, mask):
        pm = ~mask
        sf = torch.nan_to_num(self.se(C), 0.0)
        se = torch.nan_to_num(self.st(sf), 0.0)
        sp = self._pool(se, self.sp, pm)
        gf = torch.nan_to_num(self.ge(torch.cat([L, A], -1)), 0.0)
        ge = torch.nan_to_num(self.gt(gf), 0.0)
        gp = self._pool(ge, self.gp, pm)
        sm = self.smu(sp)
        sl = self.slv(sp).clamp(-6.0, 6.0)
        gm = self.gmu(gp)
        gl = self.glv(gp).clamp(-6.0, 6.0)
        return sm, sl, gm, gl

    def decode_centers_B(self, sz):
        B = sz.size(0)
        N = self.N
        ys = self.yscale
        c_raw = self.cd(sz).view(B, N, 2)
        base = torch.tanh(c_raw) * torch.tensor(
            [0.5, ys / 2.0], device=sz.device
        ) + torch.tensor([0.5, ys / 2.0], device=sz.device)
        shift_vec = torch.tanh(self.cd_shift(sz)).view(B, 1, 2)
        shift = shift_vec * torch.tensor([0.35, ys * 0.4], device=sz.device)
        cx = (base[..., 0:1] + shift[..., 0:1]).clamp(0.0, 1.0)
        cy = (base[..., 1:2] + shift[..., 1:2]).clamp(0.0, ys)
        c = torch.cat([cx, cy], dim=-1)
        return c

    def decode(self, sz, gz, mode="B"):
        B = sz.size(0)
        N = self.N
        if mode == "B":
            c = self.decode_centers_B(sz)
        else:
            raise ValueError("Only mode 'B' is supported")
        l = torch.sigmoid(self.ld(gz).view(B, N, 1))
        ml = self.am(gz).view(B, N, 4).clamp(-10.0, 10.0)
        rk = self.ak(gz).view(B, N, 4)
        kf = F.softplus(rk[:, :, :3]).clamp(0.5, 30.0)
        kr = (F.softplus(rk[:, :, 3:4]) * 0.3).clamp(0.05, 0.5)
        kp = torch.cat([kf, kr], -1)
        el = self.eh(torch.cat([sz, gz], -1))
        return c, l, ml, kp, el

    def forward(self, C, L, A, mask, mode="B"):
        sm, sl, gm, gl = self.encode(C, L, A, mask)
        sz = sm
        gz = gm + torch.randn_like(gm) * (0.5 * gl).exp()
        c, l, ml, kp, el = self.decode(sz, gz, mode=mode)
        return {
            "pc": c,
            "pl": l,
            "ml": ml,
            "kp": kp,
            "el": el,
            "sm": sm,
            "sl": torch.zeros_like(sm),
            "gm": gm,
            "gl": gl,
        }


def kl_loss(mu, lv, fb=0.1):
    lv = lv.clamp(-6.0, 6.0)
    return (-0.5 * (1.0 + lv - mu.pow(2.0) - lv.exp())).clamp(min=fb).sum(-1).mean()


def angle_nll(ml_r, kp_r, angle, fct):
    p3 = F.softmax(ml_r[:3], 0)
    kpr = kp_r[:3].clamp(0.1, 30.0)
    nll_terms = []
    for k in range(3):
        diff = angle - fct[k]
        nll_k = -kpr[k] * torch.cos(diff) + torch.log(torch.special.i0e(kpr[k])) + kpr[k]
        nll_terms.append(nll_k)
    nll = torch.stack(nll_terms)
    return (p3 * nll).sum()


def var_penalty(pc, mask):
    tot = 0.0
    B = pc.size(0)
    for b in range(B):
        n = int(mask[b].sum())
        if n < 2:
            continue
        vx = pc[b, :n, 0].var()
        vy = pc[b, :n, 1].var()
        tot = tot + F.relu(0.05 - vx) + F.relu(0.001 - vy)
    return tot / float(B)


def encode_panel(model, lines_array, device):
    if lines_array.ndim != 2 or lines_array.shape[1] != 4:
        raise ValueError("lines_array must be (n,4): [cx,cy,L,A_rad]")
    arr = np.asarray(lines_array, dtype=np.float32)
    n = arr.shape[0]
    C = torch.from_numpy(arr[:, 0:2].reshape(1, n, 2)).to(device)
    L = torch.from_numpy(arr[:, 2:3].reshape(1, n, 1)).to(device)
    A = torch.from_numpy(arr[:, 3:4].reshape(1, n, 1)).to(device)
    mask = torch.ones(1, n, dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        sm, sl, gm, gl = model.encode(C, L, A, mask)
    return sm[0].detach().cpu().numpy(), gm[0].detach().cpu().numpy()
