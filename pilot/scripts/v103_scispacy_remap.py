"""Systematic CUI-mapping improvement: replace substring matcher with scispaCy
UMLS EntityLinker (abbreviation resolution + synonym + confidence). Tests the
user's point that 72.4% name->CUI mapping is improvable by a better SYSTEM.

High threshold (0.85) keeps precision so we recover SPECIFIC terms, not the
generic low-IDF phrases that the n-gram expansion recovered (which regressed @1).
"""
import sys, json, glob, argparse, pickle
from collections import defaultdict
import spacy
from scispacy.linking import EntityLinker
import networkx as nx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="pilot/data/cache/v103c_attr_ddx49_per_disease")
    ap.add_argument("--out", default="pilot/data/cache/v103sci_ddx49_kg.pkl")
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    print("Loading scispaCy en_core_sci_lg + UMLS linker...", flush=True)
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "k": 3, "threshold": args.threshold, "max_entities_per_mention": 1})
    linker = nlp.get_pipe("scispacy_linker")

    files = sorted(glob.glob(f"{args.in_dir}/*.json"))
    # collect unique names
    names = set()
    data = []
    for f in files:
        o = json.load(open(f)); data.append(o)
        names |= set(o["aggregated"].keys())
    names = sorted(names)
    print(f"Linking {len(names)} unique phenotype names...", flush=True)

    name2cui = {}
    for doc, name in zip(nlp.pipe(names, batch_size=128), names):
        best = None; bestsc = 0.0
        for ent in doc.ents:
            for cui, sc in ent._.kb_ents:
                if sc > bestsc: bestsc = sc; best = cui
        if best: name2cui[name] = best
    print(f"scispaCy mapped: {len(name2cui)}/{len(names)} ({100*len(name2cui)/len(names):.1f}%)", flush=True)

    G = nx.MultiDiGraph()
    nedges = 0
    for o in data:
        dcui = o["cui"]
        G.add_node(dcui, ntype="disease", name=o["disease"])
        for name, ent in o["aggregated"].items():
            pc = name2cui.get(name)
            if not pc or pc == dcui: continue
            if pc not in G: G.add_node(pc, ntype="phenotype", name=name)
            G.add_edge(dcui, pc, etype="HAS_PHENOTYPE",
                       n_mentions=ent["n_mentions"], phen_name=name)
            nedges += 1
    pickle.dump(G, open(args.out, "wb"))
    print(f"KG: {G.number_of_nodes()} nodes, {nedges} edges → {args.out}", flush=True)


if __name__ == "__main__":
    main()
