"""One S1-only local-window retraining; two fixed S2 preview cases."""
from pathlib import Path
import ast, contextlib, copy, importlib.util, io, json, math, shutil, sys, time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
REPAIR=STUDY/'s1_s2_repair_trial_2026-10-18'
SPATIAL=STUDY/'spatial_placement_fix_trial_2026-10-19'
PREV=STUDY/'minimal_original_method_preview_2026-10-19'
DEST=Path((__import__('os').environ['DFN3_CORE'] + '/_external_not_used'))
SEED=42001
CHECKS=[]

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def check(name,ok,detail=''):
    CHECKS.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    assert ok, name+': '+str(detail)

def windows(r):
    labels=pd.read_csv(STUDY/'data_prep/manual_trace_manifest.csv')
    geom=pd.read_csv(REPAIR/'face_split_geometry.csv')
    result={};inventory=[];records=[]
    for row in geom.sort_values('face_id').itertuples():
        data=labels[labels.face_id.eq(row.face_id)]
        w=float(row.width_m);h=float(row.height_up_face_m)
        p=data[['u1_m','v1_m']].to_numpy()+[0,h]
        q=data[['u2_m','v2_m']].to_numpy()+[0,h]
        for index in range(int(w//20)):
            left=index*20.;right=left+20;v=q-p
            den=v[:,0];parallel=np.abs(den)<1e-12;safe=np.where(parallel,1,den)
            aa=(left-p[:,0])/safe;bb=(right-p[:,0])/safe
            lo=np.maximum(0,np.minimum(aa,bb));hi=np.minimum(1,np.maximum(aa,bb))
            lo[parallel]=0;hi[parallel]=1
            good=(hi>lo+1e-9)&(~parallel|((p[:,0]>=left)&(p[:,0]<right)))
            ix=np.flatnonzero(good)
            pp=p[ix]+lo[ix,None]*v[ix]-[left,0]
            qq=p[ix]+hi[ix,None]*v[ix]-[left,0]
            arr=r.lines(pp/20,qq/20)
            fid=f'{row.face_id}__X{index:02d}'
            result[fid]=dict(id=fid,parent_face=row.face_id,role=row.role,w=20.,h=h,ym=h/20,arr=arr,x0=left)
            inventory.append(dict(window_id=fid,parent_face=row.face_id,role=row.role,section=row.section,x0_m=left,x1_m=right,width_m=20,height_m=h,traces=len(arr),boundary_clipped_traces=int(((lo[ix]>1e-8)|(hi[ix]<1-1e-8)).sum()),discarded_right_remainder_m=w-20*int(w//20)))
            for k,j in enumerate(ix):
                records.append(dict(window_id=fid,parent_face=row.face_id,role=row.role,trace_id=data.iloc[j].trace_id,boundary_clipped=bool(lo[j]>1e-8 or hi[j]<1-1e-8),cx=arr[k,0],cy=arr[k,1],length=arr[k,2],angle=arr[k,3]))
    inv=pd.DataFrame(inventory);inv.to_csv(HERE/'window_inventory.csv',index=False)
    trace=pd.DataFrame(records);trace.to_csv(HERE/'window_trace_manifest.csv',index=False)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        check('parent_face_disjoint:'+a+':'+b,not(set(inv[inv.role==a].parent_face)&set(inv[inv.role==b].parent_face)))
        check('trace_id_disjoint:'+a+':'+b,not(set(trace[trace.role==a].trace_id)&set(trace[trace.role==b].trace_id)))
    for fid,f in result.items():
        p,q=r.ends(f['arr']);z=np.vstack([p,q])
        check('crop_in_bounds:'+fid,not len(z) or (z[:,0].min()>=-1e-5 and z[:,0].max()<=1.00001 and z[:,1].min()>=-1e-5 and z[:,1].max()<=f['ym']+1e-5))
    return result

def make_examples(r,faces,role):
    out=[]
    for face in faces.values():
        if face['role']!=role or not len(face['arr']):continue
        for layout in ['centre','5m']:
            for aug in (['orig','rot180'] if role=='train' else ['orig']):
                a=r.augment(face,aug);st=r.strips(layout,face['w']);obs,parents=r.observe(a,st)
                out.append(dict(face=face,layout=layout,arr=a,obs=obs,parents=parents,strips=st))
    return out

def main():
    raise NotImplementedError('plotting / image processing removed from the public release')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('RUN_FAILURE.json',dict(error=repr(exc)))
        pd.DataFrame(CHECKS).to_csv(HERE/'execution_checks_on_failure.csv',index=False)
        raise
