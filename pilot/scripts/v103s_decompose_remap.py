"""Decomposing scispaCy remap: split verbose IE phrases into clean component
terms (on parens/commas/'and'/'or'/'with'/'across') and map EACH to a CUI, so
'malar rash (butterfly rash) across cheeks and nose' yields rash, cheeks, nose
CUIs that match the patient's separated generic+location vocabulary."""
import sys, json, glob, argparse, pickle, re
from collections import defaultdict
import spacy
from scispacy.linking import EntityLinker
import networkx as nx
def components(name):
    name=name.lower()
    name=re.sub(r'\(.*?\)','',name)  # drop parentheticals
    parts=re.split(r'[,/;]|\band\b|\bor\b|\bwith\b|\bacross\b|\bover\b|\bof the\b|\bon the\b|\bin the\b',name)
    out=[]
    for p in parts:
        p=p.strip(" .-")
        if len(p)>2 and p not in out: out.append(p)
    return out or [name.strip()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--in_dir",required=True); ap.add_argument("--out",required=True); ap.add_argument("--threshold",type=float,default=0.82)
    a=ap.parse_args()
    nlp=spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker",config={"resolve_abbreviations":True,"linker_name":"umls","k":3,"threshold":a.threshold,"max_entities_per_mention":1})
    files=sorted(glob.glob(f"{a.in_dir}/*.json")); data=[]
    comp_set=set()
    for f in files:
        o=json.load(open(f)); data.append(o)
        for nm in o["aggregated"]: comp_set|=set(components(nm))
    comps=sorted(comp_set)
    print(f"{len(comps)} component terms, linking...",flush=True)
    c2cui={}
    for doc,nm in zip(nlp.pipe(comps,batch_size=256),comps):
        best=None;bs=0.0
        for ent in doc.ents:
            for cui,sc in ent._.kb_ents:
                if sc>bs: bs=sc; best=cui
        if best: c2cui[nm]=best
    G=nx.MultiDiGraph(); ne=0
    for o in data:
        dc=o["cui"]; G.add_node(dc,ntype="disease",name=o["disease"])
        for nm,ent in o["aggregated"].items():
            for comp in components(nm):
                pc=c2cui.get(comp)
                if not pc or pc==dc: continue
                if pc not in G: G.add_node(pc,ntype="phenotype",name=comp)
                G.add_edge(dc,pc,etype="HAS_PHENOTYPE",n_mentions=ent["n_mentions"],phen_name=comp); ne+=1
    pickle.dump(G,open(a.out,"wb"))
    print(f"KG: {G.number_of_nodes()} nodes {ne} edges → {a.out}",flush=True)
main()
