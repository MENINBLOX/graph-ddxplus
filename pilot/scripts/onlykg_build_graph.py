#!/usr/bin/env python3
"""only-KG Phase 1: Graph builder.

Build multi-relation knowledge graph from raw-text IE outputs.

Node types:
  - Disease(cui, name)      DDXPlus / SymCat / RareBench diseases + UMLS DISO universe
  - Phenotype(cui, name)    UMLS CUI for symptoms/findings (via scispaCy linker on IE features)
  - Source(id, type)        PMID, NBK_id, wikipedia revid (provenance)

Edge types:
  - Disease -[HAS_PHENOTYPE {weight, sources, freq}]-> Phenotype
  - Phenotype -[ALIAS_OF]-> Phenotype           (CUI-level aliasing)
  - Disease -[MENTIONED_IN]-> Source            (provenance traceability)

Edge weighting (HAS_PHENOTYPE):
  weight = w_freq * w_source_agreement * w_idf
    w_freq            = log1p(occurrence_count_in_pubmed_+_textbook)
    w_source_agreement = n_distinct_sources / max_sources (textbook[4] + pubmed)
    w_idf             = log(N_diseases / DF(phenotype))     # phenotype rarity

Stores in NetworkX MultiDiGraph + saves to pickle.

Equivalent Cypher (documented for Neo4j port):
  // Load Disease nodes
  LOAD CSV WITH HEADERS FROM 'file:///diseases.csv' AS row
  CREATE (:Disease {cui: row.cui, name: row.name})

  // Load Phenotype nodes
  LOAD CSV WITH HEADERS FROM 'file:///phenotypes.csv' AS row
  CREATE (:Phenotype {cui: row.cui, name: row.name})

  // Load HAS_PHENOTYPE edges
  LOAD CSV WITH HEADERS FROM 'file:///has_phenotype.csv' AS row
  MATCH (d:Disease {cui: row.disease_cui}), (p:Phenotype {cui: row.phenotype_cui})
  CREATE (d)-[:HAS_PHENOTYPE {weight: toFloat(row.weight), sources: row.sources, freq: toInteger(row.freq)}]->(p)
"""
from __future__ import annotations
import sys, json, math, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx

GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph.pkl"
META_OUT = MEDKG_ROOT / "kg" / "onlykg_meta.json"


def main():
    print("=" * 70)
    print("only-KG Phase 1: Multi-relation Graph Build")
    print("=" * 70)

    # Load existing dual KG features (raw-text IE only, no curated KG)
    print("\n[1] Loading dual KG features...")
    fc = json.load(open(MEDKG_ROOT / "kg" / "disease_features_dual_by_cui.json"))
    print(f"    {len(fc):,} diseases, {sum(len(v) for v in fc.values()):,} (disease, phenotype) edges")

    # Load phenotype CUIs (scispaCy-linked)
    print("\n[2] Loading phenotype CUIs (scispaCy-linked)...")
    kg_cuis_path = MEDKG_ROOT / "kg" / "disease_kg_cuis.json"
    if not kg_cuis_path.exists():
        sys.exit("Missing disease_kg_cuis.json — run build_kg_cui_index.py first")
    disease_kg_cuis = json.load(open(kg_cuis_path))
    print(f"    {len(disease_kg_cuis):,} diseases with phenotype-CUIs")

    # Map (disease_cui, phenotype_text) -> phenotype_cui by re-linking on-the-fly cache
    # We rebuild via direct scispaCy linker since the prior build only saved disease -> [cuis]
    # Use the disease_kg_cuis as authoritative phenotype-CUI per disease, but we lose
    # per-feature CUI mapping. Solution: load original IE edges + use scispaCy here.
    # For Phase 1 efficiency: trust disease_kg_cuis as Disease -> Set(Phenotype_CUI) and
    # compute weight as count of features mentioning each phenotype.

    print("\n[3] Building Disease -> Phenotype edges...")
    # For each disease, the disease_kg_cuis already gives the unique phenotype CUIs.
    # For weight, count how many KG features map back to each CUI.
    # Approach: reload disease_features and re-link each phenotype to CUI.

    print("    Loading scispaCy linker (this takes ~30s)...")
    import spacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls", "k": 1, "threshold": 0.85
    })

    # Build (disease_cui, phenotype_cui) -> {freq, sources, score_sum}
    print("    Re-linking features to phenotype CUIs + collecting weights...")
    edge_data = defaultdict(lambda: {"freq": 0, "sources": set(), "score_sum": 0.0})
    t0 = time.time()
    n_done = 0
    for disease_cui, feats in fc.items():
        for f in feats:
            text = f.get("phenotype", "")
            if not text: continue
            doc = nlp(text)
            for ent in doc.ents:
                if ent._.kb_ents:
                    pheno_cui = ent._.kb_ents[0][0]
                    key = (disease_cui, pheno_cui)
                    edge_data[key]["freq"] += 1
                    for s in f.get("sources", []):
                        edge_data[key]["sources"].add(s)
                    edge_data[key]["score_sum"] += f.get("score", 0.0)
        n_done += 1
        if n_done % 1000 == 0:
            print(f"    {n_done}/{len(fc)} ({time.time()-t0:.0f}s)")

    print(f"    {len(edge_data):,} unique (disease, phenotype) CUI edges")

    # Compute phenotype IDF (across DDXPlus 49 disease scope for DDXPlus-relevant queries;
    # also compute global IDF over all diseases)
    print("\n[4] Computing IDF weights...")
    pheno_df_all = Counter()
    for (d, p), _ in edge_data.items():
        pheno_df_all[p] += 1
    N_all = len(fc)
    idf_all = {p: math.log((N_all + 1) / (df + 1)) + 1 for p, df in pheno_df_all.items()}

    # Phenotype canonical names (from scispaCy KB)
    print("\n[5] Resolving phenotype canonical names...")
    linker = nlp.get_pipe("scispacy_linker")
    pheno_names = {}
    unique_phenos = set(p for _, p in edge_data.keys())
    for cui in unique_phenos:
        if cui in linker.kb.cui_to_entity:
            pheno_names[cui] = linker.kb.cui_to_entity[cui].canonical_name
        else:
            pheno_names[cui] = cui
    print(f"    {len(pheno_names):,} phenotype CUIs resolved")

    # Build graph
    print("\n[6] Building NetworkX graph...")
    G = nx.MultiDiGraph()
    # Add Disease nodes
    for d_cui in fc.keys():
        G.add_node(d_cui, ntype="Disease")
    # Add Phenotype nodes
    for p_cui, p_name in pheno_names.items():
        G.add_node(p_cui, ntype="Phenotype", name=p_name)
    # Add HAS_PHENOTYPE edges
    n_edges = 0
    for (d_cui, p_cui), info in edge_data.items():
        freq = info["freq"]
        n_src = len(info["sources"])
        w_freq = math.log1p(freq)
        w_src = n_src / 5.0   # max 5 sources (4 textbook + pubmed)
        w_idf = idf_all.get(p_cui, 1.0)
        weight = w_freq * (0.5 + 0.5 * w_src) * w_idf
        G.add_edge(d_cui, p_cui, etype="HAS_PHENOTYPE",
                   weight=weight, freq=freq, n_sources=n_src,
                   w_freq=w_freq, w_src=w_src, w_idf=w_idf)
        n_edges += 1

    print(f"    Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

    # Save graph
    print(f"\n[7] Saving graph to {GRAPH_OUT}...")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)

    # Save metadata
    meta = {
        "n_diseases": len(fc),
        "n_phenotypes": len(unique_phenos),
        "n_has_phenotype_edges": n_edges,
        "phenotype_names": pheno_names,
        "idf_global": idf_all,
        "build_time_sec": time.time() - t0,
    }
    with META_OUT.open("w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"    Meta saved to {META_OUT}")

    # Sample report
    print("\n[8] Sample disease graph (URTI):")
    URTI = "C0041912"
    if URTI in G:
        edges = list(G.out_edges(URTI, data=True))
        edges.sort(key=lambda e: -e[2]["weight"])
        for src, dst, d in edges[:10]:
            print(f"    {pheno_names.get(dst, dst):40s}  weight={d['weight']:.2f}  freq={d['freq']}  src={d['n_sources']}")


if __name__ == "__main__":
    main()
