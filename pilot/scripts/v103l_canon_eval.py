"""SapBERT synonym-canonicalization of CUIs, then strict cosine+negative eval.
Bridges patient lay-CUI vs profile clinical-CUI synonym gap (groin bulge vs
'palpable bulge in groin/scrotum') by merging CUIs whose UMLS names are SapBERT-
similar above a threshold into one canonical id. Benchmark-blind (symmetric vocab
normalization, no labels). Hard clustering keeps precision (vs noisy soft-match)."""
import sys,math,pickle,json,argparse
from collections import defaultdict,Counter
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
sys.path.insert(0,"pilot/scripts")
import onlykg_eval_v71_selfaware as V71
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71

MRCONSO="/windows/data/umls_subset/MRCONSO.RRF"

def cui_names(cuis):
    want=set(cuis); out={}
    with open(MRCONSO,encoding="utf-8") as f:
        for line in f:
            p=line.split("|")
            if p[1]=="ENG" and p[0] in want and (p[0] not in out or p[2]=="P"):
                out[p[0]]=p[14]
    return out

def embed(names,bs=512):
    tok=AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    mdl=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda().eval().half()
    out=[]
    with torch.no_grad():
        for i in range(0,len(names),bs):
            b=tok(names[i:i+bs],padding=True,truncation=True,max_length=32,return_tensors="pt").to("cuda")
            e=mdl(**b).last_hidden_state[:,0,:]
            out.append(torch.nn.functional.normalize(e,dim=1).float().cpu())
    return torch.cat(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--thr",type=float,default=0.9)
    ap.add_argument("--lam",type=float,default=1.0); ap.add_argument("--tau",type=float,default=3.0)
    a=ap.parse_args()
    dcs,pats,binary_evs=V71.load_ddxplus_full(3000)
    value_cuis=json.load(open("/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"))
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
    # all CUIs in play
    allc=set()
    for pr in P.values(): allc|=set(pr)
    for _,pos,_ in pats: allc|=pos
    for ev in binary_evs:
        allc|=set(value_cuis.get(ev,{}).get("_question",[]))
    allc|=set(dcs)
    allc=sorted(allc)
    print(f"{len(allc)} unique CUIs → MRCONSO names",flush=True)
    nm=cui_names(allc)
    named=[c for c in allc if c in nm]
    names=[nm[c].lower() for c in named]
    print(f"{len(named)} named, embedding+clustering thr={a.thr}",flush=True)
    E=embed(names).cuda()
    # union-find via thresholded blocks
    parent=list(range(len(named)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def uni(x,y):
        rx,ry=find(x),find(y)
        if rx!=ry: parent[max(rx,ry)]=min(rx,ry)
    bs=2048
    for i in range(0,len(named),bs):
        sims=E[i:i+bs]@E.T  # (bs, N)
        idx=(sims>=a.thr).nonzero(as_tuple=False).cpu().numpy()
        for r,c in idx:
            if c> i+r: uni(i+r,c)
    canon={}
    for i,c in enumerate(named): canon[c]=find(i)  # cluster id
    for c in allc:
        if c not in canon: canon[c]=("raw",c)
    nclust=len({find(i) for i in range(len(named))})
    print(f"{len(named)} CUIs → {nclust} clusters",flush=True)
    # remap profile to canonical, summing weights
    Pc={}
    for d,pr in P.items():
        w=defaultdict(float)
        for c,v in pr.items(): w[canon[c]]=max(w[canon[c]],v)
        Pc[d]=dict(w)
    idf=compute_idf(Pc,0.12); beta=0.75
    Pw=reweight(dict(Pc),idf,1.0,beta)
    all_evs=set().union(*[set(p) for p in Pc.values()])
    # canonical signal: map value_cuis question CUIs to canon
    sig=defaultdict(dict)
    for ev_id in binary_evs:
        cuis={canon[c] for c in value_cuis.get(ev_id,{}).get("_question",[]) if c in canon}
        for d,prof in Pc.items():
            best=0.0
            for c in cuis:
                if c in prof:
                    idf_c=idf.get(c,1.0); factor=1.0/(1.0+math.exp((idf_c-a.tau)/1.0))
                    val=prof[c]*factor
                    if val>best: best=val
            if best>0: sig[d][ev_id]=best
    atK=Counter(); n=0
    for tc,pos,neg in pats:
        if tc not in Pw or not Pw[tc]: continue
        posc={canon[c] for c in pos if c in canon}&all_evs
        if not posc: continue
        patv={e:idf.get(e,1.0)**beta for e in posc}
        pn=math.sqrt(sum(v*v for v in patv.values()))or 1e-9
        sc={}
        for d,prof in Pw.items():
            dn=math.sqrt(sum(v*v for v in prof.values()))or 1e-9
            pos_s=sum(patv[e]*prof[e] for e in posc if e in prof)/(pn*dn)
            s=sig.get(d,{}); npen=sum(s.get(ev,0.0) for ev in neg)
            nn=math.sqrt(len(neg))or 1e-9
            sc[d]=pos_s-a.lam*(npen/(nn*dn))
        rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1; n+=1
        for K in(1,5,10,20):
            if rk<=K: atK[K]+=1
    print(f"\nthr={a.thr} lam={a.lam} tau={a.tau}  N={n}  @1={100*atK[1]/n:.2f} @5={100*atK[5]/n:.2f} @10={100*atK[10]/n:.2f} @20={100*atK[20]/n:.2f}",flush=True)

if __name__=="__main__": main()
