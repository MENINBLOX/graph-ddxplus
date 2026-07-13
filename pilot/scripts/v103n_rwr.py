"""Random Walk with Restart (Personalized PageRank) over the bipartite
disease-phenotype KG + phenotype-phenotype co-occurrence edges (원칙#11
evidence-evidence connectivity). Genuine graph-topology diagnosis (vs flat
cosine): seed patient evidence nodes, walk, rank diseases by visit prob.
Multi-hop lets generic patient evidence reach GT via co-occurring specific
phenotypes. Strict KG-only, single algorithm."""
import sys,math,pickle,json,argparse
from collections import defaultdict,Counter
import numpy as np
import scipy.sparse as sp
sys.path.insert(0,"pilot/scripts")
import onlykg_eval_v71_selfaware as V71
from onlykg_eval_v71_selfaware import compute_idf

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--alpha",type=float,default=0.3,help="restart prob")
    ap.add_argument("--ppmi",type=float,default=1.0,help="weight on phen-phen edges")
    ap.add_argument("--iters",type=int,default=30)
    ap.add_argument("--cooc_min",type=int,default=2)
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
    idf=compute_idf(P,0.12)
    dl=sorted(P)
    phens=sorted(set().union(*[set(p) for p in P.values()]))
    pid={p:i for i,p in enumerate(phens)}; did={d:i for i,d in enumerate(dl)}
    nD,nP=len(dl),len(phens)
    N=nD+nP  # node index: 0..nD-1 disease, nD..nD+nP-1 phen
    # build adjacency (undirected weighted)
    rows=[];cols=[];vals=[]
    for d,pr in P.items():
        di=did[d]
        for p,w in pr.items():
            pi=nD+pid[p]
            ww=w*(idf.get(p,1.0))  # IDF-weight disease-phen edge
            rows+=[di,pi]; cols+=[pi,di]; vals+=[ww,ww]
    # phen-phen co-occurrence (same disease)
    if a.ppmi>0:
        co=defaultdict(float)
        for d,pr in P.items():
            ps=list(pr)
            for i in range(len(ps)):
                for j in range(i+1,len(ps)):
                    co[(min(ps[i],ps[j]),max(ps[i],ps[j]))]+=1
        for (p1,p2),c in co.items():
            if c<a.cooc_min: continue
            i1=nD+pid[p1]; i2=nD+pid[p2]
            ww=a.ppmi*math.log(1+c)
            rows+=[i1,i2]; cols+=[i2,i1]; vals+=[ww,ww]
    A=sp.csr_matrix((vals,(rows,cols)),shape=(N,N))
    # column-normalize for random walk (transition)
    deg=np.asarray(A.sum(axis=0)).ravel(); deg[deg==0]=1
    Dinv=sp.diags(1.0/deg)
    W=A@Dinv  # W[:,j] sums to 1
    all_evs=set(phens)
    atK=Counter(); n=0
    for tc,pos,neg in pats:
        if tc not in P: continue
        posm=[p for p in pos if p in pid]
        if not posm: continue
        s=np.zeros(N)
        tot=0.0
        for p in posm:
            w=idf.get(p,1.0); s[nD+pid[p]]=w; tot+=w
        if tot==0: continue
        s/=tot
        x=s.copy()
        for _ in range(a.iters):
            x=(1-a.alpha)*(W@x)+a.alpha*s
        dscore={d:x[did[d]] for d in dl}
        rk=sorted(dl,key=lambda d:-dscore[d]).index(tc)+1; n+=1
        for K in(1,5,10,20):
            if rk<=K: atK[K]+=1
    print(f"alpha={a.alpha} ppmi={a.ppmi} cooc_min={a.cooc_min} N={n}  @1={100*atK[1]/n:.2f} @5={100*atK[5]/n:.2f} @10={100*atK[10]/n:.2f} @20={100*atK[20]/n:.2f}",flush=True)

if __name__=="__main__": main()
