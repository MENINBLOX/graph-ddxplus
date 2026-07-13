"""Semantic-type-gated CUI canonicalization. Anatomy CUIs (T023/T029/T030/T024)
merge aggressively (iliac fossa≈groin≈inguinal region) to bridge DDXPlus
location vocab vs corpus location vocab; finding/symptom CUIs stay strict.
Then cosine+IDF+negative+disease-disease spreading (best only-KG algo)."""
import sys,math,pickle,json,argparse
from collections import defaultdict,Counter
import torch
from transformers import AutoTokenizer,AutoModel
sys.path.insert(0,"pilot/scripts")
import onlykg_eval_v71_selfaware as V71
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71
MRCONSO="/windows/data/umls_subset/MRCONSO.RRF"; MRSTY="/windows/data/umls_subset/MRSTY.RRF"
ANAT={"T023","T029","T030","T024","T018","T025","T026"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--thr_loc",type=float,default=0.78); ap.add_argument("--thr_find",type=float,default=0.93)
    ap.add_argument("--lam",type=float,default=1.0); ap.add_argument("--tau",type=float,default=3.0); ap.add_argument("--gamma",type=float,default=0.2)
    a=ap.parse_args()
    dcs,pats,binary_evs=V71.load_ddxplus_full(3000)
    value_cuis=json.load(open("/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"))
    def L(f): return pickle.load(open(f"pilot/data/cache/{f}","rb"))
    files=["v103h_exh_kg.pkl","v103ddx49_sci_kg.pkl","v103pres_ddx49_sci_kg.pkl","v103sci_ddx49_kg.pkl","v103i_clean_kg.pkl","v103j_exp_kg.pkl","v103q_anat_kg.pkl"]
    wt={"v103i_clean_kg.pkl":3,"v103j_exp_kg.pkl":3,"v103q_anat_kg.pkl":3}
    Praw=defaultdict(lambda:defaultdict(float))
    for fn in files:
        G=L(fn)
        for d in dcs:
            if d in G:
                for _,p,e in G.out_edges(d,data=True):
                    if e.get("etype")=="HAS_PHENOTYPE": Praw[d][p]+=wt.get(fn,1)*e.get("n_mentions",0.0)
    P={d:dict((p,x/(x+2.0)) for p,x in Praw[d].items() if x>0) for d in dcs if Praw[d]}
    allc=set()
    for pr in P.values(): allc|=set(pr)
    for _,pos,_ in pats: allc|=pos
    for ev in binary_evs: allc|=set(value_cuis.get(ev,{}).get("_question",[]))
    allc=sorted(allc)
    # names + TUI
    nm={}; want=set(allc)
    with open(MRCONSO,encoding="utf-8") as f:
        for line in f:
            p=line.split("|")
            if p[1]=="ENG" and p[0] in want and (p[0] not in nm or p[2]=="P"): nm[p[0]]=p[14]
    tui={}
    with open(MRSTY,encoding="utf-8") as f:
        for line in f:
            p=line.split("|")
            if p[0] in want and p[0] not in tui: tui[p[0]]=p[1]
    named=[c for c in allc if c in nm]
    names=[nm[c].lower() for c in named]
    isanat=[tui.get(c,"") in ANAT for c in named]
    print(f"{len(named)} named, {sum(isanat)} anatomy. Embedding...",flush=True)
    tok=AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    mdl=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda().eval().half()
    embs=[]
    with torch.no_grad():
        for i in range(0,len(names),512):
            b=tok(names[i:i+512],padding=True,truncation=True,max_length=32,return_tensors="pt").to("cuda")
            e=mdl(**b).last_hidden_state[:,0,:]
            embs.append(torch.nn.functional.normalize(e,dim=1).float())
    E=torch.cat(embs).cuda()
    anat_t=torch.tensor(isanat,device="cuda")
    parent=list(range(len(named)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def uni(x,y):
        rx,ry=find(x),find(y)
        if rx!=ry: parent[max(rx,ry)]=min(rx,ry)
    for i in range(0,len(named),1024):
        sims=E[i:i+1024]@E.T
        for r in range(sims.shape[0]):
            gi=i+r; ai=isanat[gi]
            thr=a.thr_loc if ai else a.thr_find
            # only merge within same group
            row=sims[r]
            cand=(row>=thr).nonzero(as_tuple=False).flatten().tolist()
            for c in cand:
                if c>gi and isanat[c]==ai: uni(gi,c)
    canon={c:find(i) for i,c in enumerate(named)}
    for c in allc:
        if c not in canon: canon[c]=("raw",c)
    print(f"clusters: {len({find(i) for i in range(len(named))})} (from {len(named)})",flush=True)
    # remap
    Pc={}
    for d,pr in P.items():
        w=defaultdict(float)
        for c,v in pr.items(): w[canon[c]]=max(w[canon[c]],v)
        Pc[d]=dict(w)
    idf=compute_idf(Pc,0.12); beta=0.75
    Pw=reweight(dict(Pc),idf,1.0,beta)
    all_evs=set().union(*[set(p) for p in Pc.values()])
    sig=defaultdict(dict)
    for ev_id in binary_evs:
        cuis={canon[c] for c in value_cuis.get(ev_id,{}).get("_question",[]) if c in canon}
        for d,prof in Pc.items():
            best=0.0
            for c in cuis:
                if c in prof:
                    idf_c=idf.get(c,1.0); fac=1.0/(1.0+math.exp((idf_c-a.tau)/1.0)); v=prof[c]*fac
                    if v>best: best=v
            if best>0: sig[d][ev_id]=best
    dl=list(Pw)
    DN={d:(math.sqrt(sum(v*v for v in Pw[d].values()))or 1e-9) for d in dl}
    ddS={d:{} for d in dl}
    for i,d1 in enumerate(dl):
        x=Pw[d1]
        for d2 in dl[i+1:]:
            y=Pw[d2]; aa,bb=(x,y) if len(x)<len(y) else (y,x)
            s=sum(aa[e]*bb[e] for e in aa if e in bb)/(DN[d1]*DN[d2])
            if s>0.05: ddS[d1][d2]=s; ddS[d2][d1]=s
    atK=Counter(); n=0
    for tc,pos,neg in pats:
        if tc not in Pw or not Pw[tc]: continue
        posc={canon[c] for c in pos if c in canon}&all_evs
        if not posc: continue
        patv={e:idf.get(e,1.0)**beta for e in posc}
        pn=math.sqrt(sum(v*v for v in patv.values()))or 1e-9
        base={}
        for d in dl:
            prof=Pw[d]; dn=DN[d]
            ps=sum(patv[e]*prof[e] for e in posc if e in prof)/(pn*dn)
            s=sig.get(d,{}); npen=sum(s.get(ev,0.0) for ev in neg); nn=math.sqrt(len(neg))or 1e-9
            base[d]=ps-a.lam*(npen/(nn*dn))
        sc={d:base[d]+a.gamma*sum(ddS[d].get(d2,0)*base[d2] for d2 in ddS[d]) for d in dl}
        rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1; n+=1
        for K in(1,5,10,20):
            if rk<=K: atK[K]+=1
    print(f"thr_loc={a.thr_loc} thr_find={a.thr_find}  N={n}  @1={100*atK[1]/n:.2f} @5={100*atK[5]/n:.2f} @10={100*atK[10]/n:.2f} @20={100*atK[20]/n:.2f}",flush=True)

if __name__=="__main__": main()
