"""Build BOUND attribute structure (professor's qualified-edge design): each
disease phenotype keeps its attributes ATTACHED (pheno_cui, {attr: value set}),
so 'pain[loc=chest]' is distinct from 'rash[loc=chest]'. Disease side via scispaCy
(cached). location/radiation/character -> CUIs; severity/onset/timing -> tokens."""
import json, glob, pickle, re
from collections import defaultdict
import spacy
from scispacy.linking import EntityLinker
ATTRS_CUI=["location","radiation","character"]
def comps(v):
    v=re.sub(r'\(.*?\)','',v.lower())
    out=[]
    for p in re.split(r'[,/;]| and | or | to ',v):
        p=p.strip(" .-")
        if len(p)>2: out.append(p)
    return out
def main():
    files=sorted(glob.glob("pilot/data/cache/v105_grounded_ie/*.json"))
    data=[json.load(open(f)) for f in files]
    strset=set()
    for o in data:
        for f in o["findings"]:
            strset.add(f["name"])
            for at in ATTRS_CUI:
                if f.get(at):
                    for c in comps(f[at]): strset.add(c)
    strs=sorted(strset)
    print(f"scispaCy {len(strs)} strings...",flush=True)
    nlp=spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker",config={"resolve_abbreviations":True,"linker_name":"umls","k":3,"threshold":0.82,"max_entities_per_mention":1})
    s2c={}
    for doc,s in zip(nlp.pipe(strs,batch_size=256),strs):
        best=None;bs=0.0
        for ent in doc.ents:
            for cui,sc in ent._.kb_ents:
                if sc>bs:bs=sc;best=cui
        if best:s2c[s]=best
    def mapv(v):
        out=set()
        for c in comps(v):
            if c in s2c: out.add(s2c[c])
        return out
    dz={}
    for o in data:
        dc=o["cui"]; phs=[]
        for f in o["findings"]:
            pc=s2c.get(f["name"])
            if not pc or pc==dc: continue
            at={}
            for a in ATTRS_CUI:
                if f.get(a): at[a]=mapv(f[a])
            sv=f.get("severity","")
            if sv: at["severity"]={"sev_mild" if "mild" in sv else "sev_moderate" if "moder" in sv else "sev_severe" if "sever" in sv else None}-{None}
            on=f.get("onset","")
            if on: at["onset"]={"onset_sudden" if ("sudden" in on or "acute" in on) else "onset_gradual" if ("gradual" in on or "insid" in on) else None}-{None}
            tm=f.get("timing","")
            if tm and ("noct" in tm or "night" in tm): at["timing"]={"timing_nocturnal"}
            phs.append((pc,{k:v for k,v in at.items() if v}))
        dz[dc]=phs
    pickle.dump({"diseases":dz,"s2c":s2c},open("pilot/data/cache/v105_bound.pkl","wb"))
    nph=sum(len(v) for v in dz.values())
    print(f"saved {len(dz)} diseases, {nph} bound phenotypes")
main()
