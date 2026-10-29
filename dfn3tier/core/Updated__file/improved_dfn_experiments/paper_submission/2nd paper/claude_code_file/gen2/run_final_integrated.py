"""
gen2: Final Integrated 10 cm DFN Reconstruction Test.

Builds and tests the best EVAE + Transformer + Geological Scorer system.
Retrains EVAE continuation head and transformer from scratch, saves checkpoints,
then evaluates 5 variants across all 45 configurations with figures.
"""
from __future__ import annotations
import copy, json, math, os, sys, time, traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = Path((__import__('os').environ['DFN3_CORE'] + r"\Updated__file\improved_dfn_experiments".replace(chr(92), '/')))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(Path((__import__('os').environ['DFN3_CORE'] + r"\Evae_sequential_hybrid_revised".replace(chr(92), '/')))))

from exp_common import (
    load_panels_raw, get_physics_stats, split_panel, strips_for_interval,
    assign_family_ids, N_MAX_EVAE, SPARSE_INTERVALS_M, VALIDATION_PANELS,
    segments_intersect, intersection_density, TOTAL_WIDTH_M, BOREHOLE_WIDTH_M,
)
from exp_metrics import compute_all_metrics
from model import DualLatentVAE
from transformer_v4 import (SeqFracTransformerV4, generate_v4, line_to_token,
                              sort_by_cx, MAX_CTX, MAX_GEN, D_MODEL, N_HEAD,
                              N_LAYERS, DROPOUT, N_FAM, collate)
from continuation_model import ContinuationLengthPredictor
from random_family_extension import append_random_family
from run_ablation_v3 import _train_geology_stats
from hybrid_v6 import (
    _build_indicated_independent, _merge_fixed_lines, _line_intrudes_strip,
    _line_from_xy, _touch_points, _max_t_to_bounds, _inside_strip,
    segment_crosses_any_borehole, _clamp_extension_before_strip,
    score_panel_quality, FAM_ASSIGN_THRESHOLD_DEG, CANDIDATE_MULT,
)

SEED = 42001
K_SEL = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
EPS = 1e-6

print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED DATA AND MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_all():
    ps = get_physics_stats()
    panels = load_panels_raw()
    fc = np.asarray(ps["fam_centers_rad"], dtype=float)
    gstats = _train_geology_stats(panels[:7], fc)
    # EVAE
    evae = DualLatentVAE(sd=12, gd=20, N=N_MAX_EVAE, yscale=ps["yscale"]).to(DEVICE)
    ck = torch.load(EXP_DIR/"outputs/models/evae_fragment_to_full_v4.pt",
                     map_location=DEVICE, weights_only=False)
    evae.load_state_dict(ck["model_state_dict"]); evae.eval()
    # Continuation MLP
    cck = torch.load(EXP_DIR/"outputs/models/continuation_predictor.pt",
                      map_location="cpu", weights_only=False)
    cont = ContinuationLengthPredictor(input_dim=cck.get("input_dim",11))
    cont.load_state_dict(cck["model_state_dict"]); cont.eval()
    # p05/p95
    al = np.vstack(panels); af = assign_family_ids(al, fc, threshold_deg=20.0)
    p05, p95 = np.zeros(3), np.zeros(3)
    for f in range(3):
        s = al[af==f,2]; s=s[s>1e-6]
        if len(s)>=10: p05[f]=float(np.percentile(s,5)); p95[f]=float(np.percentile(s,95))
    tl = np.vstack(panels[:7]); tf = assign_family_ids(tl, fc, threshold_deg=20.0)
    fl = {}
    for f in range(3):
        s=tl[tf==f,2]; s=s[np.isfinite(s)]; fl[f]=s if len(s)>0 else np.array([0.08])
    return {"ps":ps,"panels":panels,"fam_centers":fc,"gstats":gstats,
            "evae":evae,"cont":cont,"cont_X_mean":cck["X_mean"],"cont_X_std":cck["X_std"],
            "p05":p05,"p95":p95,"fam_lengths":fl,"device":DEVICE}


def encode_evae(evae, frags, device):
    arr = np.asarray(frags, dtype=np.float32).reshape(-1,4)
    if len(arr)==0: return np.zeros(12), np.zeros(20)
    n=min(len(arr),N_MAX_EVAE)
    C=torch.from_numpy(arr[:n,0:2].reshape(1,n,2)).to(device)
    L=torch.from_numpy(arr[:n,2:3].reshape(1,n,1)).to(device)
    A=torch.from_numpy(arr[:n,3:4].reshape(1,n,1)).to(device)
    mk=torch.ones(1,n,dtype=torch.bool,device=device)
    if n<N_MAX_EVAE:
        C=F.pad(C,(0,0,0,N_MAX_EVAE-n));L=F.pad(L,(0,0,0,N_MAX_EVAE-n))
        A=F.pad(A,(0,0,0,N_MAX_EVAE-n));mk=F.pad(mk,(0,N_MAX_EVAE-n),value=False)
    with torch.no_grad():
        sm,_,gm,_=evae.encode(C,L,A,mk)
    return sm[0].cpu().numpy().astype(np.float32), gm[0].cpu().numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# EVAE CONTINUATION HEAD
# ══════════════════════════════════════════════════════════════════════════════
class EVAEContHead(nn.Module):
    def __init__(self, lat=32, feat=11, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(lat+feat,hid),nn.ReLU(),nn.Linear(hid,32),nn.ReLU())
        self.mu = nn.Linear(32,1)
        self.ls = nn.Linear(32,1); self.ls.bias.data.fill_(-0.7)
    def forward(self, x):
        h=self.net(x); return self.mu(h), self.ls(h).clamp(-3,2)
    def predict_length(self, x):
        mu,ls=self.forward(x); return mu, ls.exp()


def build_cont_dataset(m):
    """Build continuation dataset with per-config EVAE latents."""
    panels=m["panels"]; evae=m["evae"]; fc=m["fam_centers"]; dev=m["device"]
    y_max=float(m["ps"].get("yscale",0.315))
    X_tr,Y_tr,X_va,Y_va=[],[],[],[]
    for pid in range(9):
        is_tr=pid<7
        augs=["orig","flip_h","rot180"] if is_tr else ["orig"]
        for iv in SPARSE_INTERVALS_M:
            for aug in augs:
                panel=panels[pid].copy()
                if aug=="flip_h": panel[:,0]=1-panel[:,0]; panel[:,3]=(math.pi-panel[:,3])%math.pi
                elif aug=="rot180": panel[:,0]=1-panel[:,0]; panel[:,1]=y_max-panel[:,1]
                strips=strips_for_interval(iv)
                meas,_=split_panel(panel,strips)
                if len(meas)==0: continue
                sz,gz=encode_evae(evae,meas,dev)
                lat=np.concatenate([sz,gz])
                ma=np.asarray(meas,dtype=np.float32)
                mf=assign_family_ids(ma,fc,threshold_deg=20.0)
                for i,(r,fam) in enumerate(zip(ma,mf)):
                    if int(fam)>2: continue
                    cx,cy,L,A=float(r[0]),float(r[1]),float(r[2]),float(r[3])
                    cA,sA=math.cos(A),math.sin(A)
                    x1,y1=cx-0.5*L*cA,cy-0.5*L*sA; x2,y2=cx+0.5*L*cA,cy+0.5*L*sA
                    for lo,hi in strips:
                        for bx_test in [lo,hi]:
                            if min(abs(x1-bx_test),abs(x2-bx_test))>0.003: continue
                            touch_on_x1=abs(x1-bx_test)<abs(x2-bx_test)
                            tx,ty=(x1,y1) if touch_on_x1 else (x2,y2)
                            sign=-1.0 if tx<cx else 1.0
                            ux,uy=cA,sA
                            if ux*sign<=0: ux,uy=-ux,-uy
                            if ux*sign<=0: continue
                            tb=1.0
                            if abs(ux)>1e-6:
                                tb=min(tb,(1.0-tx)/ux if ux>0 else -tx/ux)
                            if abs(uy)>1e-6:
                                tb=min(tb,(y_max-ty)/uy if uy>0 else -ty/uy)
                            tb=max(tb,0.001)
                            best_d,best_cl=1.0,0.001
                            for gr in panel:
                                if abs(gr[3]-A)>0.15 and abs(gr[3]-A-math.pi)>0.15: continue
                                d=math.sqrt((gr[0]-cx)**2+(gr[1]-cy)**2)
                                if d<best_d: best_d=d; best_cl=max(gr[2]-L,0.001)
                            nn=sum(1 for r2 in ma if r2 is not r and math.sqrt((r2[0]-cx)**2+(r2[1]-cy)**2)<0.1)
                            feat=np.array([math.log(max(L,1e-8)),sA,cA,
                                           1.0 if int(fam)==0 else 0.0,1.0 if int(fam)==1 else 0.0,
                                           1.0 if int(fam)==2 else 0.0,
                                           sign,tx,ty,tb,float(nn)],dtype=np.float32)
                            full=np.concatenate([lat,feat])
                            tgt=math.log(max(best_cl,1e-8))
                            (X_tr if is_tr else X_va).append(full)
                            (Y_tr if is_tr else Y_va).append(tgt)
    Xt=torch.tensor(np.array(X_tr),dtype=torch.float32)
    Yt=torch.tensor(np.array(Y_tr),dtype=torch.float32).unsqueeze(1)
    Xv=torch.tensor(np.array(X_va),dtype=torch.float32)
    Yv=torch.tensor(np.array(Y_va),dtype=torch.float32).unsqueeze(1)
    print(f"  EVAE cont dataset: {len(Xt)} train, {len(Xv)} val")
    return Xt,Yt,Xv,Yv


def train_evae_cont(m, epochs=50):
    Xt,Yt,Xv,Yv=build_cont_dataset(m)
    Xm=Xt.mean(0);Xs=Xt.std(0).clamp(min=1e-6)
    Xtn=(Xt-Xm)/Xs; Xvn=(Xv-Xm)/Xs
    mdl=EVAEContHead(32,11,64); opt=torch.optim.Adam(mdl.parameters(),lr=1e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=8,factor=0.5)
    best_v=float("inf"); best_s=None; pat=15; w=0; log=[]
    for ep in range(1,epochs+1):
        mdl.train(); idx=torch.randperm(len(Xtn)); losses=[]
        for i in range(0,len(idx),64):
            bi=idx[i:i+64]; xb=Xtn[bi]; yb=Yt[bi]
            mu,ls=mdl(xb); sig=ls.exp()
            nll=0.5*((yb-mu)/sig)**2+ls+0.5*math.log(2*math.pi)
            loss=nll.mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        mdl.eval()
        with torch.no_grad():
            mv,lsv=mdl(Xvn); sv=lsv.exp()
            vnll=(0.5*((Yv-mv)/sv)**2+lsv).mean().item()
            vmae=(Yv-mv).abs().mean().item()
        sch.step(vnll)
        log.append({"epoch":ep,"train_nll":np.mean(losses),"val_nll":vnll,"val_mae":vmae})
        if vnll<best_v: best_v=vnll; best_s=copy.deepcopy(mdl.state_dict()); w=0
        else:
            w+=1
            if w>=pat: break
    mdl.load_state_dict(best_s)
    # Save checkpoint
    torch.save({"model_state_dict":best_s,"Xm":Xm,"Xs":Xs,"val_mae":vmae},
               SCRIPT_DIR/"evae_cont_head.pt")
    print(f"  EVAE cont head: best val MAE={log[-1]['val_mae'] if log else 0:.4f}")
    return mdl, Xm, Xs, pd.DataFrame(log)


# ══════════════════════════════════════════════════════════════════════════════
# INDICATED EXTENSION WITH EVAE CONTINUATION HEAD
# ══════════════════════════════════════════════════════════════════════════════
def build_indicated_evae_cont(measured_dicts, fam_lengths, fam_p95, y_max, strips,
                                rng, ind_budget, evae_head, evae_Xm, evae_Xs,
                                evae_latent):
    """Build indicated extensions using EVAE continuation head instead of MLP."""
    indicated = []; n_touch = 0
    for md in measured_dicts:
        fam=int(md["family"])
        if fam>2: continue
        dx=float(md["x2"])-float(md["x1"]); dy=float(md["y2"])-float(md["y1"])
        ln=math.hypot(dx,dy)
        if ln<=1e-9: continue
        ux,uy=dx/ln,dy/ln; parent=str(md["parent_trace_id"])
        touches=_touch_points(float(md["x1"]),float(md["y1"]),float(md["x2"]),float(md["y2"]),strips)
        if not touches: continue
        for side,(bx,by) in touches:
            sign=-1.0 if side=="left" else 1.0
            if abs(ux)<1e-9: continue
            dux,duy=(ux,uy) if ux*sign>0 else (-ux,-uy)
            if dux*sign<=0: continue
            t_bound=_max_t_to_bounds(float(bx),float(by),float(dux),float(duy),y_max)
            max_len=min(float(fam_p95[fam]),float(t_bound))
            if max_len<=1e-9: continue
            # EVAE continuation prediction
            clip_A=float(md["angle"])
            nn_count=sum(1 for m2 in measured_dicts if m2 is not md and
                         math.hypot(float(m2["cx"])-float(md["cx"]),float(m2["cy"])-float(md["cy"]))<0.1)
            feat=np.array([math.log(max(float(md["length"]),1e-8)),
                           math.sin(clip_A),math.cos(clip_A),
                           1.0 if fam==0 else 0.0,1.0 if fam==1 else 0.0,1.0 if fam==2 else 0.0,
                           sign,float(bx),float(by),float(t_bound),float(nn_count)],dtype=np.float32)
            full_feat=np.concatenate([evae_latent,feat])
            ft=torch.tensor(full_feat,dtype=torch.float32).unsqueeze(0)
            ft_n=(ft-evae_Xm)/evae_Xs
            with torch.no_grad():
                mu_pred,sig_pred=evae_head.predict_length(ft_n)
            logL=float(mu_pred[0,0])+0.67*max(float(sig_pred[0,0]),0.05)
            sample_len=min(max(float(np.exp(logL)),0.03),max_len)
            x2_ext=float(bx+sample_len*dux); y2_ext=float(by+sample_len*duy)
            # Anti-leak: clamp at other boreholes
            src_bh=None
            for slo,shi in strips:
                if abs(float(bx)-float(slo))<EPS or abs(float(bx)-float(shi))<EPS:
                    src_bh=(float(slo),float(shi)); break
            crossed=segment_crosses_any_borehole(float(bx),float(by),x2_ext,y2_ext,strips,source_strip=src_bh)
            if crossed is not None:
                x2_ext,y2_ext=_clamp_extension_before_strip(float(bx),float(by),x2_ext,y2_ext,crossed,side)
            line=_line_from_xy(float(bx),float(by),x2_ext,y2_ext,fam,"indicated",parent)
            if float(line["length"])<1e-6: continue
            if _inside_strip(0.5*(float(line["x1"])+float(line["x2"])),strips): continue
            indicated.append(line); n_touch+=1
    return indicated, n_touch


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER M+I DATASET AND TRAINING (with restored losses)
# ══════════════════════════════════════════════════════════════════════════════
class MIDataset(torch.utils.data.Dataset):
    def __init__(self, m, pids, intervals, augment=True):
        self.seqs=[]; fc=m["fam_centers"]
        for pid in pids:
            for iv in intervals:
                for aug in (["orig","flip_h","rot180"] if augment else ["orig"]):
                    panel=m["panels"][pid].copy()
                    ym=float(m["ps"].get("yscale",0.315))
                    if aug=="flip_h": panel[:,0]=1-panel[:,0];panel[:,3]=(math.pi-panel[:,3])%math.pi
                    elif aug=="rot180": panel[:,0]=1-panel[:,0];panel[:,1]=ym-panel[:,1]
                    strips=strips_for_interval(iv)
                    meas_raw,_=split_panel(panel,strips)
                    if len(meas_raw)==0: continue
                    ma=np.asarray(meas_raw,dtype=np.float32)
                    mf=assign_family_ids(ma,fc,threshold_deg=20.0)
                    tc=dict(m["gstats"]["target_counts"])
                    md=[]
                    for i,(r,f) in enumerate(zip(ma,mf)):
                        cx,cy,L,A=float(r[0]),float(r[1]),float(r[2]),float(r[3])
                        cA,sA=math.cos(A),math.sin(A)
                        md.append({"x1":cx-0.5*L*cA,"y1":cy-0.5*L*sA,"x2":cx+0.5*L*cA,"y2":cy+0.5*L*sA,
                                   "cx":cx,"cy":cy,"length":L,"angle":A,"family":int(f),
                                   "semantic_role":"measured","parent_trace_id":f"meas_{i:04d}"})
                    ib={f:max(0,tc[f]-int((mf==f).sum())) for f in range(3)}
                    ind,_=_build_indicated_independent(
                        measured_dicts=md,fam_lengths=m["fam_lengths"],
                        fam_p95_indicated={f:float(m["p95"][f]) for f in range(3)},
                        y_max=ym,strips=strips,rng=np.random.default_rng(SEED+199000+pid*100+int(iv)),
                        ind_budget=ib,cont_model=m["cont"],cont_X_mean=m["cont_X_mean"],cont_X_std=m["cont_X_std"])
                    merged=_merge_fixed_lines(md,ind)
                    for l in merged:
                        fid=int(l["family"])
                        if fid<=2 and float(l["length"])>m["p95"][fid]:
                            l["length"]=float(m["p95"][fid])
                            A2=float(l["angle"]);cx2=float(l["cx"]);cy2=float(l["cy"])
                            h=0.5*float(l["length"])
                            l["x1"]=cx2-h*math.cos(A2);l["y1"]=cy2-h*math.sin(A2)
                            l["x2"]=cx2+h*math.cos(A2);l["y2"]=cy2+h*math.sin(A2)
                    m_arr=np.array([[l["cx"],l["cy"],l["length"],l["angle"]] for l in merged],dtype=np.float32)
                    if len(m_arr)==0: continue
                    # Missing = panel minus merged
                    miss_mask=np.ones(len(panel),dtype=bool)
                    for ml in m_arr:
                        d=np.sqrt((panel[:,0]-ml[0])**2+(panel[:,1]-ml[1])**2)
                        b=np.argmin(d)
                        if d[b]<0.02 and miss_mask[b]: miss_mask[b]=False
                    miss=panel[miss_mask]
                    if len(miss)==0: continue
                    ctx=sort_by_cx(m_arr)
                    tf2=assign_family_ids(miss,fc,threshold_deg=20.0)
                    order=np.lexsort((miss[:,0],tf2))
                    tgt=miss[order]; tgt_f=tf2[order]
                    ct=np.array([line_to_token(l) for l in ctx[:MAX_CTX]],dtype=np.float32)
                    tt=np.array([line_to_token(l) for l in tgt[:MAX_GEN]],dtype=np.float32)
                    cf=assign_family_ids(ctx[:MAX_CTX],fc).astype(np.int64)
                    tf3=tgt_f[:MAX_GEN].astype(np.int64)
                    self.seqs.append({"ctx":ct,"ctx_f":cf,"tgt":tt,"tgt_f":tf3})
    def __len__(self): return len(self.seqs)
    def __getitem__(self,i): return self.seqs[i]


def compute_mi_loss(model,ctx,ctx_m,tgt,tgt_m,tgt_fids,ps,device,sz=None,gz=None):
    pos_p,logL_p,ang_p,fam_p=model(ctx,ctx_m,tgt[:,:-1,:],tgt_m[:,:-1],sz=sz,gz=gz)
    tt=tgt[:,1:,:]; valid=~tgt_m[:,1:]
    if valid.sum()==0: return torch.tensor(0.0,device=device),{}
    pos_loss=F.mse_loss(pos_p[valid],tt[:,:,:2][valid])
    len_loss=F.mse_loss(logL_p[valid],tt[:,:,2:3][valid])
    at=torch.atan2(tt[:,:,3:4],tt[:,:,4:5])
    ap=torch.atan2(ang_p[:,:,0:1],ang_p[:,:,1:2])
    ang_loss=(1.0-torch.cos(at[valid]-ap[valid])).mean()
    ft=tgt_fids[:,1:]; fam_loss=F.cross_entropy(fam_p[valid],ft[valid])
    # Family proportion
    fps=torch.tensor([float(ps.get("fam_shares",{}).get(f,0.33)) for f in range(3)],device=device)
    fp=F.softmax(fam_p[valid],dim=-1)
    fpl=F.mse_loss(fp[:,:3].mean(0),fps) if len(fp)>0 else torch.tensor(0.0,device=device)
    # Crossing surrogate
    it=float(ps.get("intersection_density_mean",0.44))
    tau=max(float(np.hypot(float(ps.get("cx_std",0.28)),float(ps.get("cy_std",0.07)))),1e-3)
    tau_t=torch.tensor(tau,device=device); it_t=torch.tensor(it,device=device)
    ints=[]
    for b in range(pos_p.size(0)):
        vb=valid[b]; nv=int(vb.sum().item())
        if nv<2: continue
        xb=pos_p[b,:,0][vb];yb=pos_p[b,:,1][vb]
        lb=torch.exp(logL_p[b,:,0][vb]).clamp_min(1e-6)
        av=F.normalize(ang_p[b][vb],dim=-1,eps=1e-6)
        thb=torch.atan2(av[:,0],av[:,1])
        dx=xb.unsqueeze(1)-xb.unsqueeze(0);dy=yb.unsqueeze(1)-yb.unsqueeze(0)
        dij=torch.sqrt(dx*dx+dy*dy+1e-6)
        lbar=0.5*(lb.unsqueeze(1)+lb.unsqueeze(0))
        rij=dij/(lbar+1e-6)
        aij=torch.sin(thb.unsqueeze(1)-thb.unsqueeze(0)).pow(2)
        cij=aij*torch.exp(-(rij*rij)/(tau_t*tau_t+1e-6))
        iu=torch.triu_indices(nv,nv,offset=1,device=device)
        if iu.size(1)>0: ints.append(cij[iu[0],iu[1]].mean())
    isl=(torch.stack(ints).mean()-it_t)**2 if ints else torch.tensor(0.0,device=device)
    total=1.0*pos_loss+0.5*len_loss+1.5*ang_loss+1.0*fam_loss+0.3*fpl+0.5*isl
    return total,{"pos":pos_loss.item(),"len":len_loss.item(),"ang":ang_loss.item(),
                  "fam":fam_loss.item(),"fprop":fpl.item(),"int_surr":isl.item(),"total":total.item()}


def train_transformer_mi(m, epochs=50):
    print("  Building M+I dataset...")
    tr_ds=MIDataset(m,list(range(7)),SPARSE_INTERVALS_M,augment=True)
    va_ds=MIDataset(m,[7,8],SPARSE_INTERVALS_M,augment=False)
    print(f"  Train: {len(tr_ds)}, Val: {len(va_ds)}")
    tl=torch.utils.data.DataLoader(tr_ds,batch_size=8,shuffle=True,collate_fn=collate)
    vl=torch.utils.data.DataLoader(va_ds,batch_size=8,shuffle=False,collate_fn=collate)
    mdl=SeqFracTransformerV4(sd=12,gd=20).to(DEVICE)
    opt=torch.optim.Adam(mdl.parameters(),lr=3e-4,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
    ps=m["ps"]; best_v=float("inf"); best_s=None; pat=10; w=0; log=[]
    for ep in range(1,epochs+1):
        mdl.train(); ep_l=[]
        for ctx,cm,tgt,tm,tf in tl:
            ctx,cm,tgt,tm,tf=ctx.to(DEVICE),cm.to(DEVICE),tgt.to(DEVICE),tm.to(DEVICE),tf.to(DEVICE)
            bos=torch.zeros(ctx.size(0),1,5,device=DEVICE)
            di=torch.cat([bos,tgt],dim=1)
            dm=torch.cat([torch.zeros(ctx.size(0),1,dtype=torch.bool,device=DEVICE),tm],dim=1)
            to2=torch.cat([tgt,torch.zeros(ctx.size(0),1,5,device=DEVICE)],dim=1)
            tom=torch.cat([tm,torch.ones(ctx.size(0),1,dtype=torch.bool,device=DEVICE)],dim=1)
            fo=torch.cat([tf,torch.zeros(ctx.size(0),1,dtype=torch.long,device=DEVICE)],dim=1)
            loss,_=compute_mi_loss(mdl,ctx,cm,di,dm,fo,ps,DEVICE)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0);opt.step()
            ep_l.append(loss.item())
        mdl.eval(); vls=[]
        with torch.no_grad():
            for ctx,cm,tgt,tm,tf in vl:
                ctx,cm,tgt,tm,tf=ctx.to(DEVICE),cm.to(DEVICE),tgt.to(DEVICE),tm.to(DEVICE),tf.to(DEVICE)
                bos=torch.zeros(ctx.size(0),1,5,device=DEVICE)
                di=torch.cat([bos,tgt],dim=1)
                dm=torch.cat([torch.zeros(ctx.size(0),1,dtype=torch.bool,device=DEVICE),tm],dim=1)
                fo=torch.cat([tf,torch.zeros(ctx.size(0),1,dtype=torch.long,device=DEVICE)],dim=1)
                vl2,_=compute_mi_loss(mdl,ctx,cm,di,dm,fo,ps,DEVICE)
                vls.append(vl2.item())
        vm=np.mean(vls) if vls else 0
        sch.step(vm); log.append({"epoch":ep,"train":np.mean(ep_l),"val":vm})
        print(f"    Ep {ep}: train={np.mean(ep_l):.4f}, val={vm:.4f}")
        if vm<best_v: best_v=vm; best_s=copy.deepcopy(mdl.state_dict()); w=0
        else:
            w+=1
            if w>=pat: break
    mdl.load_state_dict(best_s)
    torch.save({"model_state_dict":best_s,"val_loss":best_v},SCRIPT_DIR/"transformer_mi.pt")
    return mdl, pd.DataFrame(log)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_variant(m, pid, iv, variant, trans_model=None, evae_head=None, evae_Xm=None, evae_Xs=None,
                cand_mult=80):
    ps=m["ps"]; gstats=m["gstats"]; fc=m["fam_centers"]; p05=m["p05"]; p95=m["p95"]
    real=m["panels"][pid]; strips=strips_for_interval(iv)
    meas_raw,_=split_panel(real,strips)
    ma=np.asarray(meas_raw,dtype=np.float32); mf=assign_family_ids(ma,fc,threshold_deg=20.0)
    y_max=float(ps.get("yscale",0.315))
    tc=dict(gstats["target_counts"]); sf=dict(gstats["spacing_floor"])
    fam_logmu=np.asarray(ps.get("fam_logmu",[-2.33,-2.30,-1.99]))
    fam_logsig=np.asarray(ps.get("fam_logsig",[0.43,0.35,0.58]))
    fam_kappa=np.array([8.69,9.51,10.61])

    # Build measured dicts
    md=[]
    for i,(r,f) in enumerate(zip(ma,mf)):
        cx,cy,L,A=float(r[0]),float(r[1]),float(r[2]),float(r[3])
        cA,sA=math.cos(A),math.sin(A)
        md.append({"x1":cx-0.5*L*cA,"y1":cy-0.5*L*sA,"x2":cx+0.5*L*cA,"y2":cy+0.5*L*sA,
                   "cx":cx,"cy":cy,"length":L,"angle":A,"family":int(f),
                   "semantic_role":"measured","parent_trace_id":f"meas_{i:04d}"})
    ib={f:max(0,tc[f]-int((mf==f).sum())) for f in range(3)}

    # Indicated: choose based on variant
    use_evae_cont = variant in ("V3_EVAE_continuation","V5_combined_best") and evae_head is not None
    if use_evae_cont:
        lat=np.concatenate(encode_evae(m["evae"],ma,m["device"]))
        ind,nt=build_indicated_evae_cont(md,m["fam_lengths"],{f:float(p95[f]) for f in range(3)},
                                          y_max,strips,np.random.default_rng(SEED+199000),ib,
                                          evae_head,evae_Xm,evae_Xs,lat)
    else:
        ind,nt=_build_indicated_independent(
            measured_dicts=md,fam_lengths=m["fam_lengths"],
            fam_p95_indicated={f:float(p95[f]) for f in range(3)},
            y_max=y_max,strips=strips,rng=np.random.default_rng(SEED+199000),ind_budget=ib,
            cont_model=m["cont"] if variant!="V1_geology_only" else None,
            cont_X_mean=m["cont_X_mean"] if variant!="V1_geology_only" else None,
            cont_X_std=m["cont_X_std"] if variant!="V1_geology_only" else None)

    # Merge
    merged=_merge_fixed_lines(md,ind)
    for l in merged:
        fid=int(l["family"])
        if fid<=2 and float(l["length"])>p95[fid]:
            l["length"]=float(p95[fid])
            A2=float(l["angle"]);cx2=float(l["cx"]);cy2=float(l["cy"])
            h=0.5*float(l["length"])
            l["x1"]=cx2-h*math.cos(A2);l["y1"]=cy2-h*math.sin(A2)
            l["x2"]=cx2+h*math.cos(A2);l["y2"]=cy2+h*math.sin(A2)
    merged_arr=np.array([[l["cx"],l["cy"],l["length"],l["angle"]] for l in merged],dtype=np.float32) if merged else ma.copy()
    merged_fams=[int(l["family"]) for l in merged] if merged else list(mf)

    cur={f:0 for f in range(3)}
    for f in merged_fams:
        if f<3: cur[f]+=1
    needed={f:max(0,tc[f]-cur[f]) for f in range(3)}

    # K=5 selection
    best_gen=None;best_sc=np.inf;best_srcs=None
    use_trans=variant in ("V4_transformer_MI","V5_combined_best") and trans_model is not None
    for k in range(K_SEL):
        seed=SEED+k*1000; rng=np.random.default_rng(seed)
        # Transformer candidates
        tcands=[]
        if use_trans:
            tv4=generate_v4(trans_model,m["evae"],merged_arr,int(len(real)*1.5),
                            ps,DEVICE,strips=strips,temperature=1.0,seed=seed)
            tv4f=assign_family_ids(tv4,fc,threshold_deg=20.0)
            for row,fam in zip(tv4,tv4f):
                fi=int(fam)
                if fi>2 or needed.get(fi,0)<=0: continue
                cx,cy,L,A=float(row[0]),float(row[1]),float(row[2]),float(row[3])
                if _line_intrudes_strip(cx,cy,L,A,strips): continue
                L=float(np.clip(L,p05[fi],p95[fi]))
                x1=cx-0.5*L*math.cos(A);x2=cx+0.5*L*math.cos(A)
                y1=cy-0.5*L*math.sin(A);y2=cy+0.5*L*math.sin(A)
                if min(x1,x2)<-0.01 or max(x1,x2)>1.01: continue
                if min(y1,y2)<-0.01 or max(y1,y2)>y_max+0.01: continue
                tcands.append((cx,cy,L,A,fi,"trans"))
        # Statistical candidates
        scands=[]
        for fam in range(3):
            nn2=needed[fam]
            if nn2<=0: continue
            sc2=0;att=0;mx=nn2*cand_mult
            while sc2<mx and att<mx*2:
                att+=1
                cx=float(rng.uniform(0,1));cy=float(rng.uniform(0,y_max))
                L=float(np.exp(rng.normal(float(fam_logmu[fam]),float(fam_logsig[fam]))))
                L=float(np.clip(L,p05[fam],p95[fam]))
                A=float(rng.vonmises(float(fc[fam]),float(fam_kappa[fam]))%math.pi)
                if _line_intrudes_strip(cx,cy,L,A,strips): continue
                x1=cx-0.5*L*math.cos(A);x2=cx+0.5*L*math.cos(A)
                y1=cy-0.5*L*math.sin(A);y2=cy+0.5*L*math.sin(A)
                if min(x1,x2)<-0.01 or max(x1,x2)>1.01: continue
                if min(y1,y2)<-0.01 or max(y1,y2)>y_max+0.01: continue
                scands.append((cx,cy,L,A,fam,"stat"));sc2+=1
        pool=scands+tcands
        # Greedy scorer
        known=[np.array(r,dtype=np.float32) for r in merged_arr]
        kf=list(merged_fams); inf_l=[];inf_f=[];pc={f:0 for f in range(3)};ts=0;ss=0
        ap=list(known);apf=list(kf)
        cm_ref=gstats.get("centre_mass_ref",{f:{"cx_mean":0.5,"cx_std":0.1,"cy_mean":0.15,"cy_std":0.05} for f in range(3)})
        xbin_ref=np.asarray(gstats.get("xbin_ref",np.ones(5)/5.0))
        xbin_std_ref=np.asarray(gstats.get("xbin_std",np.ones(5)*0.05))
        count_std=gstats.get("count_std",{f:3.0 for f in range(3)})
        inter_std=float(gstats.get("intersection_std",0.1))
        fam_len_ref=gstats.get("fam_length_ref",{f:{"p90":0.2,"p95":0.3,"mean":0.1,"std":0.05} for f in range(3)})
        XBINS=np.array([0.0,0.2,0.4,0.6,0.8,1.0])
        pl=list(pool)
        for _ in range(sum(needed.values())):
            if not pl: break
            bi=-1;bs=-np.inf
            ids=list(range(len(pl)))
            if len(ids)>200: ids=list(rng.choice(len(pl),200,replace=False))
            if len(ap)>0:
                cxv=np.array([float(l[0]) for l in ap])
                h=np.histogram(cxv,bins=XBINS)[0].astype(float);s=h.sum()
                cp=h/s if s>0 else np.zeros(5)
            else: cp=np.zeros(5)
            for ci in ids:
                cx,cy,L,A,fam,src=pl[ci]
                if pc[fam]>=needed[fam]: continue
                sc=0.0
                sf2=[e for e,ef in zip(known+inf_l,kf+inf_f) if int(ef)==fam]
                sfv=float(sf[fam])
                if sf2:
                    dm=min(math.sqrt((cx-float(e[0]))**2+(cy-float(e[1]))**2) for e in sf2)
                    if dm<sfv*0.25: continue
                    sc+=2.0*min(dm/max(sfv,1e-6),3.0)
                else: sc+=2.0
                b2=int(np.clip(np.searchsorted(XBINS,cx,side="right")-1,0,4))
                sc+=3.0*np.clip((xbin_ref[b2]-cp[b2])/max(float(xbin_std_ref[b2]),0.02),-2.0,3.0)
                fd=float(needed[fam]-pc[fam])/max(float(count_std[fam]),1.0)
                sc+=2.0*min(fd,3.0)
                rcx=float(cm_ref[fam]["cx_mean"]);rcy=float(cm_ref[fam]["cy_mean"])
                rcxs=float(cm_ref[fam]["cx_std"]);rcys=float(cm_ref[fam]["cy_std"])
                fps=[(float(l[0]),float(l[1])) for l,f2 in zip(known+inf_l,kf+inf_f) if int(f2)==fam]
                ccx=np.mean([p[0] for p in fps]) if fps else 0.5
                ccy=np.mean([p[1] for p in fps]) if fps else 0.15
                ne=max(len(fps),1)
                pcx=(ccx*ne+cx)/(ne+1);pcy=(ccy*ne+cy)/(ne+1)
                cmd=math.sqrt(((pcx-rcx)/rcxs)**2+((pcy-rcy)/rcys)**2)
                sc-=1.5*min(cmd,4.0)
                if fam==2:
                    f2p=float(fam_len_ref[2]["p90"]);f2s=float(fam_len_ref[2]["std"])
                    if L>=f2p: sc+=1.5*min((L-f2p)/max(f2s,0.01),2.0)
                    elif L>=float(fam_len_ref[2]["mean"]): sc+=0.3
                ld=abs(math.log(max(L,1e-8))-float(fam_logmu[fam]))/max(float(fam_logsig[fam]),0.1)
                sc-=0.5*min(ld,3.0)
                if len(ap)>=8:
                    cr=np.array([cx,cy,L,A],dtype=np.float32)
                    nc=sum(1 for e in ap[-20:] if segments_intersect(cr,e))
                    if nc>3: sc-=1.0*(nc-3)/max(inter_std,0.1)
                if sc>bs: bs=sc;bi=ci
            if bi<0: break
            cx,cy,L,A,fam,src=pl.pop(bi)
            inf_l.append(np.array([cx,cy,L,A],dtype=np.float32));inf_f.append(fam)
            ap.append(np.array([cx,cy,L,A],dtype=np.float32));apf.append(fam);pc[fam]+=1
            if src=="trans": ts+=1
            else: ss+=1
        parts=[]
        if len(merged_arr)>0: parts.append(merged_arr)
        if inf_l: parts.append(np.vstack(inf_l))
        fs=np.vstack(parts).astype(np.float32) if parts else ma.copy()
        final,_=append_random_family(fs,ma,tc,ps,strips,rand_fraction=0.06,seed=seed+50)
        sv,_=score_panel_quality(final,gstats,fc)
        if sv<best_sc:
            best_sc=sv;best_gen=final
            best_srcs={"ts":ts,"ss":ss,"tp":len(tcands),"sp":len(scands),"ind":nt}
    met=compute_all_metrics(real,best_gen,ma,strips,ps)
    return best_gen, met, best_srcs, merged_arr, ma


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_three_panel(real, input_arr, generated, strips, ps, fc, title, filepath):
    raise NotImplementedError('plotting / image processing removed from the public release')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0_all=time.time()
    print("Loading data and models...")
    m=load_all()

    # ── TRAIN EVAE CONTINUATION HEAD ────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING EVAE CONTINUATION HEAD")
    print("="*60)
    evae_head, evae_Xm, evae_Xs, evae_log = train_evae_cont(m, epochs=50)
    evae_log.to_csv(SCRIPT_DIR/"evae_continuation_pipeline_results.csv", index=False)

    # Baseline MLP MAE for comparison
    cck2 = torch.load(EXP_DIR/"outputs/data/continuation_dataset.pt", map_location="cpu", weights_only=False)
    Xv=cck2["X_val"]; Yv=cck2["Y_val"]
    m["cont"].eval()
    with torch.no_grad():
        mu_b,_=m["cont"].predict((Xv-m["cont_X_mean"])/m["cont_X_std"])
        baseline_mae=(Yv-mu_b).abs().mean().item()
    print(f"  Baseline MLP MAE: {baseline_mae:.4f}")
    print(f"  EVAE head MAE:    {evae_log['val_mae'].min():.4f}")

    # ── TRAIN TRANSFORMER M+I ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER M+I (RESTORED LOSSES)")
    print("="*60)
    trans_mi, trans_log = train_transformer_mi(m, epochs=50)
    trans_log.to_csv(SCRIPT_DIR/"transformer_candidate_selection_audit.csv", index=False)

    # ── EVALUATE ALL VARIANTS ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATING ALL VARIANTS (45 configs each)")
    print("="*60)

    variants = {
        "V1_geology_only":    {"trans":None,     "evae_h":None,     "cm":80, "desc":"Geology scorer only, no MLP"},
        "V2_current_MLP":     {"trans":None,     "evae_h":None,     "cm":80, "desc":"Current MLP continuation + scorer"},
        "V3_EVAE_continuation":{"trans":None,    "evae_h":evae_head,"cm":80, "desc":"EVAE continuation head + scorer"},
        "V4_transformer_MI":  {"trans":trans_mi, "evae_h":None,     "cm":10, "desc":"Retrained trans M+I 10x + scorer"},
        "V5_combined_best":   {"trans":trans_mi, "evae_h":evae_head,"cm":10, "desc":"EVAE cont + trans M+I 10x + scorer"},
    }

    all_rows=[]; fig_manifest=[]
    for vname,vcfg in variants.items():
        print(f"\n  --- {vname}: {vcfg['desc']} ---")
        for pid in range(9):
            for iv in SPARSE_INTERVALS_M:
                t1=time.time()
                gen,met,srcs,merged,meas = run_variant(
                    m,pid,iv,vname,
                    trans_model=vcfg["trans"],evae_head=vcfg["evae_h"],
                    evae_Xm=evae_Xm,evae_Xs=evae_Xs,cand_mult=vcfg["cm"])
                el=time.time()-t1
                row={"variant":vname,"panel":pid,"spacing_m":iv,
                     "count_error":met.get("count_error_pct",0),
                     "family_l1":met.get("family_prop_l1",0),
                     "angle_similarity":met.get("angle_sim",0),
                     "length_similarity":met.get("length_sim",0),
                     "spatial_similarity":met.get("spatial_sim",0),
                     "int_density_error":met.get("int_density_err",0),
                     "measured_preservation":met.get("measured_preservation_rate",0),
                     "F0_error":met.get("F0_count_err_pct",0),
                     "F1_error":met.get("F1_count_err_pct",0),
                     "F2_error":met.get("F2_count_err_pct",0),
                     "n_measured":len(meas),"n_indicated":srcs.get("ind",0),
                     "n_inferred":srcs.get("ss",0)+srcs.get("ts",0),
                     "n_trans_generated":srcs.get("tp",0),"n_trans_selected":srcs.get("ts",0),
                     "n_stat_pool":srcs.get("sp",0),"elapsed_s":round(el,1)}
                all_rows.append(row)
                ts2=srcs.get("ts",0)
                print(f"    P{pid}/{iv}m: cnt={row['count_error']:.1f}% ang={row['angle_similarity']:.3f} "
                      f"len={row['length_similarity']:.3f} ts={ts2} t={el:.1f}s")

                # Figures for key cases
                if pid in VALIDATION_PANELS and iv==5.0:
                    fn=f"Panel{pid}_{iv}m_{vname}_three_panel.png"
                    plot_three_panel(m["panels"][pid],merged,gen,
                                     strips_for_interval(iv),m["ps"],m["fam_centers"],
                                     vname,FIG_DIR/fn)
                    fig_manifest.append({"filename":fn,"panel":pid,"spacing_m":iv,
                                         "method":vname,"count_error":row["count_error"],
                                         "angle_sim":row["angle_similarity"],
                                         "length_sim":row["length_similarity"]})

    # Also generate figures for 3m and 11m at panel 7
    for iv_extra in [3.0, 11.0]:
        for vn in ["V1_geology_only","V5_combined_best"]:
            vcfg=variants[vn]
            gen,met,srcs,merged,meas=run_variant(m,7,iv_extra,vn,
                trans_model=vcfg["trans"],evae_head=vcfg["evae_h"],
                evae_Xm=evae_Xm,evae_Xs=evae_Xs,cand_mult=vcfg["cm"])
            fn=f"Panel7_{iv_extra}m_{vn}_three_panel.png"
            plot_three_panel(m["panels"][7],merged,gen,strips_for_interval(iv_extra),
                             m["ps"],m["fam_centers"],vn,FIG_DIR/fn)
            fig_manifest.append({"filename":fn,"panel":7,"spacing_m":iv_extra,"method":vn,
                                 "count_error":met.get("count_error_pct",0),
                                 "angle_sim":met.get("angle_sim",0),
                                 "length_sim":met.get("length_sim",0)})

    # ── SAVE ALL OUTPUTS ───────────────────────────────────────────────────
    df=pd.DataFrame(all_rows)
    df.to_csv(SCRIPT_DIR/"final_per_config_results.csv",index=False)

    # Summary
    summary=[]
    for vn in variants:
        vd=df[df["variant"]==vn]
        summary.append({"variant":vn,"description":variants[vn]["desc"],
            "count_error_mean":round(vd["count_error"].mean(),2),
            "count_error_std":round(vd["count_error"].std(),2),
            "angle_sim_mean":round(vd["angle_similarity"].mean(),4),
            "length_sim_mean":round(vd["length_similarity"].mean(),4),
            "spatial_sim_mean":round(vd["spatial_similarity"].mean(),4),
            "int_dens_err":round(vd["int_density_error"].mean(),4),
            "meas_pres":round(vd["measured_preservation"].mean(),4),
            "trans_sel_total":int(vd["n_trans_selected"].sum()),
            "n_configs":len(vd)})
    pd.DataFrame(summary).to_csv(SCRIPT_DIR/"final_variant_summary.csv",index=False)

    # Validation only
    val_df=df[df["panel"].isin(VALIDATION_PANELS)]
    val_summary=[]
    for vn in variants:
        vd=val_df[val_df["variant"]==vn]
        if len(vd)==0: continue
        val_summary.append({"variant":vn,
            "count_error_mean":round(vd["count_error"].mean(),2),
            "angle_sim_mean":round(vd["angle_similarity"].mean(),4),
            "length_sim_mean":round(vd["length_similarity"].mean(),4),
            "spatial_sim_mean":round(vd["spatial_similarity"].mean(),4),
            "meas_pres":round(vd["measured_preservation"].mean(),4),
            "trans_sel_total":int(vd["n_trans_selected"].sum()),
            "n_configs":len(vd)})
    pd.DataFrame(val_summary).to_csv(SCRIPT_DIR/"validation_panel_results.csv",index=False)

    # Figure manifest
    with open(SCRIPT_DIR/"figure_manifest.md","w") as f:
        f.write("# Figure Manifest\n\n")
        for fm in fig_manifest:
            f.write(f"## {fm['filename']}\n")
            f.write(f"- Panel: {fm['panel']}, Spacing: {fm['spacing_m']}m, Method: {fm['method']}\n")
            f.write(f"- Count error: {fm['count_error']:.1f}%, Angle: {fm['angle_sim']:.3f}, Length: {fm['length_sim']:.3f}\n\n")

    # Experiment log
    with open(SCRIPT_DIR/"experiment_log.md","w") as f:
        f.write(f"# Experiment Log\n\n**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Total runtime:** {(time.time()-t0_all)/60:.1f} min\n")
        f.write(f"**Device:** {DEVICE}\n**K:** {K_SEL}\n**Seed:** {SEED}\n\n")
        f.write(f"## EVAE Continuation Head\n- Baseline MLP MAE: {baseline_mae:.4f}\n")
        f.write(f"- EVAE head MAE: {evae_log['val_mae'].min():.4f}\n\n")
        f.write(f"## Transformer M+I\n- Best val loss: {trans_log['val'].min():.4f}\n")

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY (ALL 45 CONFIGS)")
    print("="*60)
    for s in summary:
        print(f"  {s['variant']}: cnt={s['count_error_mean']:.2f} ang={s['angle_sim_mean']:.4f} "
              f"len={s['length_sim_mean']:.4f} spa={s['spatial_sim_mean']:.4f} ts={s['trans_sel_total']}")

    print("\n" + "="*60)
    print("VALIDATION PANELS ONLY")
    print("="*60)
    for s in val_summary:
        print(f"  {s['variant']}: cnt={s['count_error_mean']:.2f} ang={s['angle_sim_mean']:.4f} "
              f"len={s['length_sim_mean']:.4f} spa={s['spatial_sim_mean']:.4f} ts={s['trans_sel_total']}")

    print(f"\nTotal runtime: {(time.time()-t0_all)/60:.1f} min")
    print("ALL COMPLETE")
