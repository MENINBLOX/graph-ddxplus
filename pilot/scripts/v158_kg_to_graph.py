"""Convert the v157 attribute-KG (kg.json) into the onlykg NetworkX DiGraph the diagnosis
evaluators consume (onlykg_eval_{rarebench,symcat}.py, DDXPlus).

The evaluators score a patient by, for each disease, summing edge `weight` over its HAS_PHENOTYPE
edges whose phenotype CUI is in the patient's evidence set (onlykg_eval_rarebench.py:24-26). So we
must attach a per-edge `weight`; the principled choice is the phenotype's IDF (Spärck Jones, already
computed by v157) — over-represented hubs contribute ~0, discriminative phenotypes contribute more
(docs/evidence_weighting_dedup.md). The qualified attributes ride on each edge (for the later
leave-one-out attribute ablation; no current scorer reads them yet).

Graph contract (matches onlykg_build_graph.py):
  node d: ntype="Disease", name
  node p: ntype="Phenotype", name, tui, idf
  edge d->p: etype="HAS_PHENOTYPE", weight=idf(p), **attrs   (attrs = location_cui/severity/onset/...)

Usage: python v158_kg_to_graph.py [--kg pilot/cache/prod/kg.json] [--out data/medkg/kg/onlykg_graph_attr.pkl]"""
from __future__ import annotations
import json, pickle, argparse, os
import networkx as nx


def build(kg):
    G = nx.DiGraph()
    for dcui, d in kg["diseases"].items():
        G.add_node(dcui, ntype="Disease", name=d.get("name", ""))
    for pcui, p in kg["phenotypes"].items():
        G.add_node(pcui, ntype="Phenotype", name=p.get("name", ""),
                   tui=p.get("tui", ""), idf=p.get("idf", 0.0))
    for e in kg["edges"]:
        p = e["p"]
        w = kg["phenotypes"].get(p, {}).get("idf", 0.0)
        G.add_edge(e["d"], p, etype="HAS_PHENOTYPE", weight=w, **e.get("attrs", {}))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--out", default="data/medkg/kg/onlykg_graph_attr.pkl")
    a = ap.parse_args()
    kg = json.load(open(a.kg, encoding="utf-8"))
    G = build(kg)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(G, f)
    dz = [n for n, dd in G.nodes(data=True) if dd.get("ntype") == "Disease"]
    ph = [n for n, dd in G.nodes(data=True) if dd.get("ntype") == "Phenotype"]
    ha = sum(1 for *_, dd in G.edges(data=True) if dd.get("etype") == "HAS_PHENOTYPE")
    print(f"graph -> {a.out}")
    print(f"  Disease nodes={len(dz)}  Phenotype nodes={len(ph)}  HAS_PHENOTYPE edges={ha}")


if __name__ == "__main__":
    main()
