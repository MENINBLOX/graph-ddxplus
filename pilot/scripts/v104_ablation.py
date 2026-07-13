"""All-subset attribute ablation (DDXPlus active 6). Additive scoring: per-channel
IDF² overlap, fixed disease norm → subset score = base + Σ selected attr channels.
Reports best per level (6→0) + leave-one-out importance (each attr's marginal @1/@10)."""
import sys,math,pickle,itertools
from collections import defaultdict,Counter
import numpy as np
D=pickle.load(open("pilot/data/cache/v104_attr_vectors.pkl","rb"))
pats=D["patients"]; dis=D["diseases"]
ATTRS=["location","radiation","severity","onset","character","timing"]
CH=["base"]+ATTRS
dl=sorted(dis)
# IDF per token (disease frequency, per channel namespace to avoid collisions)
df=defaultdict(int)
for d in dl:
    toks=set()
    for c in CH:
        s=dis[d]["base"] if c=="base" else dis[d]["attr"].get(c,set())
        for t in s: toks.add((c,t))
    for t in toks: df[t]+=1
N=len(dl)
idf={t:math.log((N+1)/(df[t]+1))+1.0 for t in df}
# disease channel weight vectors + fixed norm (full set)
dvec={d:{} for d in dl}
for d in dl:
    for c in CH:
        s=dis[d]["base"] if c=="base" else dis[d]["attr"].get(c,set())
        dvec[d][c]={t:idf.get((c,t),1.0) for t in s}
dnorm={d:(math.sqrt(sum(w*w for c in CH for w in dvec[d][c].values()))or 1e-9) for d in dl}
# precompute T[p, d, channel] contribution = Σ_{matched} idf²  / dnorm
P=len(pats)
T=np.zeros((P,len(dl),len(CH)),dtype=np.float32)
truth=np.zeros(P,dtype=np.int32); valid=np.zeros(P,dtype=bool)
didx={d:i for i,d in enumerate(dl)}
for pi,p in enumerate(pats):
    if p["true"] not in didx: continue
    truth[pi]=didx[p["true"]]; valid[pi]=True
    pchan={c:(p["base"] if c=="base" else p["attr"].get(c,set())) for c in CH}
    for di,d in enumerate(dl):
        for ci,c in enumerate(CH):
            dw=dvec[d][c]
            ov=0.0
            for t in pchan[c]:
                if t in dw: ov+=idf.get((c,t),1.0)*dw[t]
            if ov: T[pi,di,ci]=ov/dnorm[d]
T=T[valid]; truth=truth[valid]; P=T.shape[0]
print(f"평가 케이스 N={P}")
def evalsub(chan_idx):  # indices into CH including base(0)
    sc=T[:,:,chan_idx].sum(axis=2)  # (P, ndis)
    order=np.argsort(-sc,axis=1)
    ranks=np.argmax(order==truth[:,None],axis=1)+1
    return (ranks<=1).mean()*100,(ranks<=5).mean()*100,(ranks<=10).mean()*100
base_only=evalsub([0])
print(f"\nbase-only (속성0): @1={base_only[0]:.2f} @5={base_only[1]:.2f} @10={base_only[2]:.2f}")
full=evalsub(list(range(len(CH))))
print(f"FULL 6속성:        @1={full[0]:.2f} @5={full[1]:.2f} @10={full[2]:.2f}")
# all subsets by level
ai=list(range(1,len(CH)))  # attribute indices 1..6
print("\n=== 레벨별 best (속성 개수 k) ===")
results={}
for k in range(len(ATTRS),-1,-1):
    best=None
    for combo in itertools.combinations(ai,k):
        r=evalsub([0]+list(combo))
        results[combo]=r
        if best is None or r[0]>best[1][0]: best=(combo,r)
    names=[CH[i] for i in best[0]]
    print(f"k={k}: @1={best[1][0]:.2f} @5={best[1][1]:.2f} @10={best[1][2]:.2f}  속성={names}")
# leave-one-out from full: marginal importance
print("\n=== leave-one-out (FULL에서 제거 시 @1 하락 = 변별 기여) ===")
fullidx=list(range(len(CH)))
imp=[]
for i in ai:
    r=evalsub([x for x in fullidx if x!=i])
    imp.append((CH[i],full[0]-r[0],r[0],r[2]))
for nm,d1,a1,a10 in sorted(imp,key=lambda x:-x[1]):
    print(f"  -{nm:10s}: @1 {a1:.2f} (Δ{d1:+.2f})  @10 {a10:.2f}")
# single-attribute marginal (base + 1 attr)
print("\n=== 단일 속성 추가 (base+1) ===")
for i in ai:
    r=evalsub([0,i])
    print(f"  +{CH[i]:10s}: @1={r[0]:.2f} (base {base_only[0]:.2f}, Δ{r[0]-base_only[0]:+.2f})  @10={r[2]:.2f}")
