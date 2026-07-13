"""Neo4j-GDS-style graph methods (only-KG): (1) link-prediction heuristics
Adamic-Adar / Resource-Allocation (degree-damped shared-neighbor scoring,
auto-downweights hub phenotypes), (2) FastRP structural embeddings (co-occurring
phenotypes get nearby vectors → generic patient evidence embeds toward the
GT region). Patient = transient node linked to its evidence phenotypes."""
import sys,math,pickle,json,argparse
from collections import defaultdict,Counter
import numpy as np
sys.path.insert(0,"pilot/scripts")
import onlykg_eval_v71_selfaware as V71
from onlykg_eval_v71_selfaware import compute_idf

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--method",default="aa")
    ap.add_argument("--dim",type=int,default=256); ap.add_argument("--seed",type=int,default=1)
    a=ap.parse_args()
    dcs,pats,binary_evs=V71.load_ddxplus_full(3000)
    def L(f): return pickle.load(open(f"pilot/data/cache/{f}","rb"))
    files=["v103h_exh_kg.pkl","v103ddx49_sci_kg.pkl","v103pres_ddx49_sci_kg.pkl","v103sci_ddx49_kg.pkl","v103i_clean_kg.pkl","v103j_exp_kg.pkl"]
    wt={"v103i_clean_kg.pkl":3,"v103j_exp_kg.pkl":3}
    Praw=defaultdict(lambda:defaultdict(float))
    for fn in files:
        G=L(fn)
        for d in dcs:
            if d in G:
                for _,p,e in G.out_edges(d,data=True):
                    if e.get("etype")=="HAS_PHENOTYPE": Praw[d][p]+=wt.get(fn,1)*e.get("n_mentions",0.0)
    P={d:dict((p,x/(x+2.0)) for p,x in Praw[d].items() if x>0) for d in dcs if Praw[d]}
    dl=sorted(P)
    phens=sorted(set().union(*[set(p) for p in P.values()]))
    pid={p:i for i,p in enumerate(phens)}; did={d:i for i,d in enumerate(dl)}
    # phenotype degree (# diseases containing it)
    pdeg=defaultdict(int)
    for pr in P.values():
        for p in pr: pdeg[p]+=1
    idf=compute_idf(P,0.12)
    all_evs=set(phens)
    if a.method in ("aa","ra","cn"):
        # link prediction: patient-disease via shared phenotypes
        atK=Counter(); n=0
        for tc,pos,neg in pats:
            if tc not in P: continue
            posm=[p for p in pos if p in pid]
            if not posm: continue
            sc={}
            for d in dl:
                pr=P[d]; s=0.0
                for p in posm:
                    if p in pr:
                        if a.method=="cn": s+=1
                        elif a.method=="aa": s+=1.0/math.log(1+pdeg[p]+1)
                        elif a.method=="ra": s+=1.0/(pdeg[p])
                sc[d]=s
            rk=sorted(dl,key=lambda d:-sc[d]).index(tc)+1; n+=1
            for K in(1,5,10,20):
                if rk<=K: atK[K]+=1
        print(f"{a.method} N={n}  @1={100*atK[1]/n:.2f} @5={100*atK[5]/n:.2f} @10={100*atK[10]/n:.2f} @20={100*atK[20]/n:.2f}")
        return
    # FastRP
    N=len(dl)+len(phens); nD=len(dl)
    import scipy.sparse as sp
    rows=[];cols=[];vals=[]
    for d,pr in P.items():
        di=did[d]
        for p,w in pr.items():
            pi=nD+pid[p]; ww=w*idf.get(p,1.0)
            rows+=[di,pi];cols+=[pi,di];vals+=[ww,ww]
    A=sp.csr_matrix((vals,(rows,cols)),shape=(N,N))
    deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
    Dinv=sp.diags(1.0/deg); M=Dinv@A  # row-normalized
    rng=np.random.default_rng(a.seed)
    # sparse random projection
    R=rng.choice([-1.0,0,1.0],size=(N,a.dim),p=[1/6,2/3,1/6]).astype(np.float32)*math.sqrt(3)
    weights=[0.0,1.0,1.0,1.0]  # combine hops 1..3
    E=np.zeros((N,a.dim),np.float32); cur=R.copy()
    for k in range(1,len(weights)):
        cur=M@cur
        if weights[k]: E+=weights[k]*cur
    En=E/ (np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
    Demb=En[:nD]  # disease embeddings (normalized)
    atK=Counter(); n=0
    for tc,pos,neg in pats:
        if tc not in P: continue
        posm=[p for p in pos if p in pid]
        if not posm: continue
        pe=np.zeros(a.dim,np.float32)
        for p in posm: pe+=idf.get(p,1.0)*En[nD+pid[p]]
        nn=np.linalg.norm(pe)+1e-9; pe/=nn
        sc=Demb@pe
        order=np.argsort(-sc); rk=list(order).index(did[tc])+1; n+=1
        for K in(1,5,10,20):
            if rk<=K: atK[K]+=1
    print(f"fastrp dim={a.dim} N={n}  @1={100*atK[1]/n:.2f} @5={100*atK[5]/n:.2f} @10={100*atK[10]/n:.2f} @20={100*atK[20]/n:.2f}")

if __name__=="__main__": main()
