"""Complete-observation consistency: continuation/merging may not invent strip hits."""
from pathlib import Path
import numpy as np
def segments(a):
    a=np.asarray(a,dtype=float).reshape(-1,4);d=np.column_stack([np.cos(a[:,3]),np.sin(a[:,3])])*a[:,2,None]/2
    return a[:,:2]-d,a[:,:2]+d
def clip_pieces(arr,strips):
    p,q=segments(arr);v=q-p;pieces=[]
    for lo,hi in strips:
        for i in range(len(p)):
            if abs(v[i,0])<1e-12:
                if not lo<=p[i,0]<=hi:continue
                t0,t1=0.,1.
            else:
                aa=(lo-p[i,0])/v[i,0];bb=(hi-p[i,0])/v[i,0];t0=max(0.,min(aa,bb));t1=min(1.,max(aa,bb))
            if t1<=t0+1e-12:continue
            a=p[i]+t0*v[i];b=p[i]+t1*v[i];vec=b-a;length=np.linalg.norm(vec)
            pieces.append((i,np.array([*(a+b)/2,length,np.arctan2(vec[1],vec[0])%np.pi])))
    return pieces
def contains(piece,observed,w,tol=.002):
    a,b=segments(np.asarray(piece)[None]);p,q=segments(np.asarray(observed)[None]);v=q[0]-p[0];length=np.linalg.norm(v)
    if length<1e-12:return False
    u=v/length;d=np.vstack([a[0]-p[0],b[0]-p[0]]);along=d@u;perp=abs(u[0]*d[:,1]-u[1]*d[:,0])*w
    return bool(perp.max()<=tol and along.min()*w>=-tol and (along.max()-length)*w<=tol)
def violations(arr,obs,strips,w=20.,tol=.002):
    return [dict(segment_index=i,unmatched_piece=piece.tolist(),length_m=float(piece[2]*w)) for i,piece in clip_pieces(arr,strips) if piece[2]*w>tol and not any(contains(piece,o,w,tol) for o in obs)]
def install(e):
    original=e.env['continuity'];audit=[]
    def checked(arr,face,obs,stage):
        st=e.r.strips('5m',face['w']) if not hasattr(checked,'strips') else checked.strips
        result=original(arr,face,obs,stage);bad=violations(result,obs,st,face['w'])
        if bad:
            before_bad=violations(arr,obs,st,face['w']);audit.append(dict(window_id=face['id'],stage=stage,violations=len(bad),input_violations=len(before_bad),veto=True))
            if before_bad:raise RuntimeError('Continuity input already violates complete observations')
            # Same conservative all-or-nothing veto used for lost observations.
            if e.env['STAGE_LOG']:
                log=e.env['STAGE_LOG'][-1]
                if log.get('face_id')==face['id'] and log.get('stage')==stage:log.update(after=len(arr),negative_evidence_veto=True,invalid_merged_intersections=len(bad))
            result=np.asarray(arr,np.float32).copy()
        assert not violations(result,obs,st,face['w'])
        return result
    e.env['continuity']=checked
    original_indicated=e.env['indicated']
    def with_layout(obs,face,strips,mode,models,rng):
        previous=getattr(checked,'strips',None);checked.strips=strips
        try:return original_indicated(obs,face,strips,mode,models,rng)
        finally:
            if previous is None:del checked.strips
            else:checked.strips=previous
    e.env['indicated']=with_layout;e.r.indicated=with_layout
    return checked,audit
