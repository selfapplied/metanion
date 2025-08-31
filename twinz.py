from typing import Any, Dict, List, NamedTuple, Tuple, Optional
import numpy as np, hashlib
from bitstring import ConstBitStream
from aspire import Opix, OpixView, OpixMount

def dct2(x: np.ndarray) -> np.ndarray:
    N=x.size; n=np.arange(N)+0.5; k=np.arange(N)[:,None]
    return x@np.cos(np.pi*n*k/N)

def idct2(X: np.ndarray) -> np.ndarray:
    N=X.size; n=np.arange(N); k=(np.arange(N)+0.5)[:,None]
    s=np.cos(np.pi*k*n/N)@X
    return 2.0*(s+0.5*X[0])/N

class TwinEntry(NamedTuple): i:int; n:int; coeffs:List[Tuple[int,int]]; tail:bytes; anchor:Optional[Tuple[int,str]]
class TwinPhase(NamedTuple): version:int; window:int; K:int; q:float; tail:int; series:List[TwinEntry]; sha1:str

def _qdct_topk(x:np.ndarray,K:int,q:float)->Tuple[List[Tuple[int,int]],float]:
    X=dct2(x); idx=np.argpartition(np.abs(X),-K)[-K:]; idx.sort(); return ([(int(k),int(np.round(X[int(k)]/q))) for k in idx], float(X[0]))

def _kw(prev:Optional[np.ndarray],arr:np.ndarray)->Tuple[int,np.ndarray]:
    if arr.size==0: return 0, (prev if prev is not None else np.zeros(256))
    h=np.bincount(arr,minlength=256).astype(np.float64); h/=max(1,int(h.sum()));
    if prev is None: return 3,h
    M=0.5*(h+prev); eps=1e-12
    KL=lambda a,b: float(np.sum(np.clip(a,eps,1.0)*(np.log2(np.clip(a,eps,1.0))-np.log2(np.clip(b,eps,1.0)))))
    jsd=0.5*(KL(h,M)+KL(prev,M))
    r=np.diff(np.where(np.concatenate(([True],arr[1:]!=arr[:-1],[True])))[0]); L=float(np.mean(r)) if r.size else 1.0; rep=int(np.sum(np.maximum(r-1,0))); lit=1.0-(rep/max(1,int(arr.size)))
    return (2 if jsd>0.15 else (0 if (lit>0.6 and L<8.0) else 1)), h

def _embed_kw(coeffs:List[Tuple[int,int]],qdc:int,kw:int)->List[Tuple[int,int]]:
    if not any(k==0 for k,_ in coeffs):
        if coeffs: coeffs.pop(int(np.argmin([abs(v) for _,v in coeffs])))
        coeffs.append((0,int(qdc)))
    return [(k,((v&~0x3)|int(kw)) if k==0 else v) for (k,v) in coeffs]
def _anchor(i:int,W:int,w:bytes)->Optional[Tuple[int,str]]: return (i,hashlib.sha1(w).hexdigest()[:16]) if ((i//W)%4==0) else None

def build_twinz(data:bytes,W:int=1024,K:int=16,q:float=8.0,T:int=256)->TwinPhase:
    out=TwinPhase(1,W,K,q,T,[],hashlib.sha1(data).hexdigest()); prev=None
    for i in range(0,len(data),W):
        w=data[i:i+W]; x=np.frombuffer(w,dtype=np.uint8).astype(np.float32)
        if x.size==0: return out
        coeffs,dc=_qdct_topk(x,K,q); kw,prev=(3,prev) if i==0 else _kw(prev,np.frombuffer(w,dtype=np.uint8))
        coeffs=_embed_kw(coeffs,int(np.round(dc/q)),kw)
        out.series.append(TwinEntry(i//W,int(x.size),coeffs,(w[-T:] if T else b""),_anchor(i,W,w)))
    return out

def reconstruct(ent:TwinEntry,q:float,T:int)->bytes:
    n=int(ent.n); X=np.zeros(n,dtype=np.float32)
    for k,v in ent.coeffs: X[int(k)]=float((int(v)&~0x3) if int(k)==0 else int(v))*q
    x=idct2(X); x=np.clip(np.round(x),0,255).astype(np.uint8)
    if T and ent.tail: t=min(len(ent.tail),x.size); x[-t:]=np.frombuffer(ent.tail,dtype=np.uint8)[-t:]
    return bytes(x)

def extract_kw(ent:TwinEntry)->int:
    for k,v in ent.coeffs:
        if int(k)==0: return int(v)&0x3
    return (int(ent.coeffs[0][1])&0x3) if ent.coeffs else 3

def seek(u_star:int, idx:TwinPhase)->Tuple[ConstBitStream,bytes]:
    W=int(idx.window); i=max(0,int(u_star//W)); ent=idx.series[min(i,len(idx.series)-1)]
    win=reconstruct(ent, float(idx.q), int(idx.tail)); br=ConstBitStream(bytes=win); br.pos=int((u_star%W)*8); return br, (ent.tail or b"")



