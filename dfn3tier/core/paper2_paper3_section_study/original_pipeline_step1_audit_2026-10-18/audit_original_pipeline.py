"""Read-only audit and bounded original-code reproduction; no model training.

Original definitions are compiled from their AST without module-level scripts,
imports, directory writes, automatic legacy validation or training entry points.
Their function/class bodies are not rewritten. Any differences from importing
the complete modules are recorded explicitly in the report.
"""
from __future__ import annotations
import ast,base64,builtins,contextlib,hashlib,io,itertools,json,math,os,re,sys,time,types,typing
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;ROOT=STUDY.parent
EXP=ROOT/'Updated__file/improved_dfn_experiments'
GEN=EXP/'paper_submission/2nd paper/claude_code_file/gen2'
LEGACY=ROOT/'Evae_sequential_hybrid_revised';LOPO=ROOT/'lopo_2d_dfn'
DATA=ROOT/'Updated__file/data/9_original'
REV=Path((__import__('os').environ['DFN3_CORE'] + '/_external_not_used'))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def simple(value,depth=0):
    if isinstance(value,torch.Tensor):return dict(type='tensor',shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,np.ndarray):return dict(type='ndarray',shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    if depth>3:return str(type(value))
    if isinstance(value,dict):return {str(k):simple(v,depth+1) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [simple(v,depth+1) for v in value[:30]]
    return dict(type=f'{type(value).__module__}.{type(value).__name__}')

def definitions(path,extra=None,names=None):
    env={k:v for k,v in vars(typing).items() if not k.startswith('_')}
    env.update(dict(__name__='audit_'+path.stem,__file__=str(path),np=np,pd=pd,torch=torch,nn=nn,F=F,
                    math=math,os=os,sys=sys,json=json,Path=Path,Counter=Counter,copy=__import__('copy'),
                    linear_sum_assignment=linear_sum_assignment,KMeans=KMeans,gaussian_kde=gaussian_kde))
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError,SyntaxError):continue
            for target in node.targets:
                if isinstance(target,ast.Name):env[target.id]=value
    if extra:env.update(extra)
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    nodes += [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
    return env

def restricted_load(path):
    # Existing local research checkpoints include NumPy arrays and cached KDEs.
    # Permit those exact known globals, never arbitrary checkpoint globals.
    allowed=[np.ndarray,np.dtype,gaussian_kde,builtins.getattr]
    multi=np._core.multiarray if hasattr(np,'_core') else np.core.multiarray
    allowed += [multi._reconstruct,multi.scalar]
    allowed += [type(np.dtype(t)) for t in ['float64','float32','int64','int32','bool']]
    allowed += [(multi._reconstruct,'numpy.core.multiarray._reconstruct'),(multi.scalar,'numpy.core.multiarray.scalar'),
                (multi._reconstruct,'numpy._core.multiarray._reconstruct'),(multi.scalar,'numpy._core.multiarray.scalar')]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',weights_only=True)

def evidence(path,function,needle):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    tree=ast.parse('\n'.join(lines));node=next((n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==function),None)
    candidates=range(node.lineno-1,node.end_lineno) if node else range(len(lines))
    found=[dict(line=i+1,text=lines[i].strip()) for i in candidates if needle in lines[i]]
    return dict(path=str(path),symbol=function,search=needle,matches=found)

def main():
    raise NotImplementedError('plotting / image processing removed from the public release')

if __name__=='__main__':main()
