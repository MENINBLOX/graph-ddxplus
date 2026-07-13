"""SapBERT embedding soft-match retrieval. Root cause of @10 ceiling = synonym
vocab mismatch (patient 'sore throat' C0242429 vs profile 'pharyngitis' C0031350
get different UMLS CUIs despite being synonyms). Instead of exact-CUI overlap,
embed phenotype NAMES with SapBERT (biomedical synonym encoder) and match by
cosine similarity. score(d)=sum_i idf_i * max_j[ sim(ev_i,phen_j) * w_j ]."""
import sys, json, csv, ast, math, pickle, argparse
from collections import defaultdict
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0,"pilot/scripts")
from onlykg_eval_v71_selfaware import compute_idf

VALUE_CUIS="/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"

def embed(names, bs=256):
    tok=AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    mdl=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda().eval().half()
    out=[]
    with torch.no_grad():
        for i in range(0,len(names),bs):
            b=tok(names[i:i+bs],padding=True,truncation=True,max_length=32,return_tensors="pt").to("cuda")
            e=mdl(**b).last_hidden_state[:,0,:]  # CLS
            out.append(torch.nn.functional.normalize(e,dim=1).float().cpu())
    return torch.cat(out)

def render_patient(evs, evmeta):
    out=[]
    for ev in evs:
        if "_@_" in ev:
            base,val=ev.split("_@_",1); m=evmeta.get(base,{})
            vm=m.get("value_meaning",{}).get(val,{}); en=vm.get("en",val) if isinstance(vm,dict) else val
            q=m.get("question_en","")
            if en and en!="nowhere": out.append(f"{q} {en}".strip())
        else:
            m=evmeta.get(ev,{}); out.append(m.get("question_en",ev))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=3000)
    ap.add_argument("--thr",type=float,default=0.0); a=ap.parse_args()
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond=json.load(open("data/ddxplus/release_conditions_en.json"))
    evmeta=json.load(open("data/ddxplus/release_evidences.json"))
    fr2cui={info.get("cond-name-fr",""):icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs=sorted(set(fr2cui.values()))
    def L(f): return pickle.load(open(f"pilot/data/cache/{f}","rb"))
    files=["v103h_exh_kg.pkl","v103ddx49_sci_kg.pkl","v103pres_ddx49_sci_kg.pkl","v103sci_ddx49_kg.pkl","v103i_clean_kg.pkl","v103j_exp_kg.pkl"]
    wt={"v103i_clean_kg.pkl":3,"v103j_exp_kg.pkl":3}
    # disease -> {phen_name: weight}
    prof=defaultdict(lambda:defaultdict(float))
    for fn in files:
        G=L(fn)
        for d in dcs:
            if d in G:
                for _,p,e in G.out_edges(d,data=True):
                    if e.get("etype")!="HAS_PHENOTYPE": continue
                    nm=e.get("phen_name") or G.nodes[p].get("name")
                    if nm: prof[d][nm.lower().strip()]+=wt.get(fn,1)*e.get("n_mentions",0.0)
    prof={d:{n:x/(x+2.0) for n,x in v.items() if x>0} for d,v in prof.items() if v}
    dl=sorted(prof)
    # IDF over phen names (disease-frequency)
    df=defaultdict(int)
    for v in prof.values():
        for n in v: df[n]+=1
    N=len(prof); pidf={n:math.log((N+1)/(df[n]+1))+1.0 for n in df}
    # global phen vocab
    pvocab=sorted(set().union(*[set(v) for v in prof.values()]))
    pidx={n:i for i,n in enumerate(pvocab)}
    print(f"{len(dl)} diseases, {len(pvocab)} unique phenotype names",flush=True)
    # patients
    tasks=[]
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if len(tasks)>=a.n: break
            tc=fr2cui.get(row["PATHOLOGY"])
            if tc not in prof: continue
            evs=ast.literal_eval(row["EVIDENCES"])
            names=render_patient(evs,evmeta)
            names=[x for x in dict.fromkeys(names) if x]
            if names: tasks.append((tc,names))
    evocab=sorted(set().union(*[set(n.lower().strip() for n in nm) for _,nm in tasks]))
    eidx={n:i for i,n in enumerate(evocab)}
    print(f"{len(tasks)} patients, {len(evocab)} unique evidence strings. Embedding...",flush=True)
    PE=embed(pvocab).cuda()    # (V_p, d)
    EE=embed(evocab).cuda()    # (V_e, d)
    # disease phenotype index lists + weight*pidf vectors
    dmat=[]  # (idxs tensor, w tensor)
    for d in dl:
        items=list(prof[d].items())
        idxs=torch.tensor([pidx[n] for n,_ in items],device="cuda")
        w=torch.tensor([wv*pidf[n] for n,wv in items],device="cuda",dtype=torch.float32)
        dmat.append((idxs,w))
    # eval
    atK=defaultdict(int); n=0
    for tc,names in tasks:
        eids=torch.tensor([eidx[x.lower().strip()] for x in names],device="cuda")
        ev=EE[eids]                      # (n_p, d)
        eidf=torch.tensor([pidf.get(x.lower().strip(),1.0) for x in names],device="cuda")
        sims_all=ev@PE.T                 # (n_p, V_p)
        if a.thr>0: sims_all=torch.clamp(sims_all-a.thr,min=0)/(1-a.thr)
        scores=torch.empty(len(dl),device="cuda")
        for di,(idxs,w) in enumerate(dmat):
            s=sims_all[:,idxs]*w.unsqueeze(0)   # (n_p, n_d)
            best=s.max(dim=1).values            # (n_p,) best phenotype per evidence
            scores[di]=(best*eidf).sum()
        order=torch.argsort(scores,descending=True).tolist()
        rk=order.index(dl.index(tc))+1; n+=1
        for K in (1,5,10,20):
            if rk<=K: atK[K]+=1
    print(f"\nN={n} thr={a.thr}  @1={100*atK[1]/n:.2f} @5={100*atK[5]/n:.2f} @10={100*atK[10]/n:.2f} @20={100*atK[20]/n:.2f}",flush=True)

if __name__=="__main__": main()
