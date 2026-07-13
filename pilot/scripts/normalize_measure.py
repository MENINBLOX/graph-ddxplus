"""Phase C — measure REAL normalization rate of extracted attribute atoms via the
frozen scispaCy en_core_sci_lg + UMLS EntityLinker (the actual pipeline linker).
Runs on an IE output dir whose CUI-like attrs are arrays of atoms.
Reports per-attribute: link rate (atom -> UMLS CUI >= threshold) + sample links +
semantic-type sanity (does 'location' link to anatomy? does 'character' link at all?)."""
import argparse, json, glob
from collections import Counter, defaultdict

ap=argparse.ArgumentParser(); ap.add_argument("--in",dest="ind",required=True); ap.add_argument("--thr",type=float,default=0.82)
a=ap.parse_args()
CUI_ATTR=["location","radiation","character","aggravating","relieving","associated"]
# UMLS semantic type groups (TUI) for sanity
ANATOMY={"T023","T029","T030","T024","T017","T018","T025"}

import spacy
from scispacy.linking import EntityLinker
nlp=spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker",config={"resolve_abbreviations":True,"linker_name":"umls","k":3,"threshold":a.thr,"max_entities_per_mention":1})
link=nlp.get_pipe("scispacy_linker")

def link_atom(text):
    d=nlp(text)
    best=None
    for e in d.ents:
        if e._.kb_ents:
            c=e._.kb_ents[0]; ent=link.kb.cui_to_entity[c[0]]
            if best is None or c[1]>best[1]: best=(c[0],c[1],ent.canonical_name,tuple(ent.types))
    return best

atoms=defaultdict(list)
for fp in glob.glob(f"pilot/data/cache/{a.ind}/*.json"):
    if fp.endswith("_stats.json"): continue
    d=json.load(open(fp))
    for f in d.get("findings",[]):
        for at in CUI_ATTR:
            v=f.get(at,[])
            if isinstance(v,str): v=[v] if v else []
            for x in v:
                if x: atoms[at].append(str(x).strip())

res={}
for at in CUI_ATTR:
    uniq=list(dict.fromkeys(atoms[at]))  # dedup preserve order
    linked=0; anat=0; samples=[]
    seen=set()
    for x in uniq:
        b=link_atom(x)
        if b:
            linked+=1
            if ANATOMY & set(b[3]): anat+=1
            if len(samples)<12: samples.append((x,b[2],round(b[1],2)))
        else:
            if len([s for s in samples if s[1]=="—"])<6: samples.append((x,"—UNLINKED—",0))
    res[at]={"n_atoms_total":len(atoms[at]),"n_unique":len(uniq),
             "link_rate_unique":round(linked/max(1,len(uniq)),3),
             "anatomy_share_of_linked":round(anat/max(1,linked),3),
             "samples":samples}

print(f"=== normalization ({a.ind}, thr={a.thr}) ===")
for at in CUI_ATTR:
    r=res[at]; print(f"\n{at}: unique={r['n_unique']} link_rate={r['link_rate_unique']*100:.0f}% anatomy_of_linked={r['anatomy_share_of_linked']*100:.0f}%")
    print("   ", "; ".join(f"{x}->{c}({s})" for x,c,s in r['samples'][:10]))
json.dump(res,open(f"pilot/data/cache/{a.ind}/_normstats.json","w"),indent=2,ensure_ascii=False)
print(f"\nsaved pilot/data/cache/{a.ind}/_normstats.json")
