#!/usr/bin/env python3
"""Textbook-only KG builder (excluding Orphanet, PubMed cache).

Input:
  - $MEDKG_ROOT/processed/edges_normalized.jsonl
    (LLM IE edges from StatPearls/GeneReviews/MedlinePlus/Wikipedia
     with HPO ID normalization where available)

Output:
  - $MEDKG_ROOT/kg/kg_textbook.jsonl              one record per (disease, phenotype)
  - $MEDKG_ROOT/kg/disease_features_textbook.json {disease: [{phenotype, score, sources, n_sources, hpo_id, umls_cui}]}
  - $MEDKG_ROOT/kg/disease_features_textbook_by_cui.json
                                                  {umls_cui: [phenotype list, sorted]}

Score formula:
  score_textbook  = n_textbook_sources / 4
  score_combined  = 0.7 * score_textbook + 0.3 * (n_sources / 4)
                    (collapses to a single signal; PubMed/Orphanet contributions left
                     for the PubMed-side KG and the merge step)

This file is ready to be unioned with the PubMed-IE KG (pending).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

EDGES_NORM = MEDKG_ROOT / "processed" / "edges_normalized.jsonl"
OUT_DIR = MEDKG_ROOT / "kg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_KG = OUT_DIR / "kg_textbook.jsonl"
OUT_FEATURES = OUT_DIR / "disease_features_textbook.json"
OUT_FEATURES_BY_CUI = OUT_DIR / "disease_features_textbook_by_cui.json"

TEXTBOOK_SOURCES = {"statpearls", "genereviews", "medlineplus", "wikipedia"}


def main():
    if not EDGES_NORM.exists():
        sys.exit(f"Input not found: {EDGES_NORM}")

    edges = defaultdict(lambda: {
        "sources": set(),
        "provenance": defaultdict(list),
        "hpo_id": None,
    })
    n_in = 0
    n_skipped_non_textbook = 0
    with EDGES_NORM.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            src = e.get("source")
            if src not in TEXTBOOK_SOURCES:
                n_skipped_non_textbook += 1
                continue
            phen_norm = (e.get("phenotype_normalized") or e.get("phenotype") or "").strip()
            if not phen_norm: continue
            key = (e["disease"], phen_norm.lower())
            edges[key]["sources"].add(src)
            edges[key]["disease_orig"] = e["disease"]
            edges[key]["phenotype_orig"] = e.get("phenotype", phen_norm)
            edges[key]["phenotype_normalized"] = phen_norm.lower()
            edges[key]["umls_cui"] = e.get("umls_cui")
            if e.get("hpo_id"):
                edges[key]["hpo_id"] = e["hpo_id"]
            p_entry = {"source_id": e.get("source_id"), "section": e.get("section_name")}
            for k in ("pmid", "revid", "chapter_title", "topic_title", "title"):
                if e.get(k):
                    p_entry[k] = e[k]
            edges[key]["provenance"][src].append(p_entry)
            n_in += 1
    print(f"Loaded {n_in:,} textbook IE edges → {len(edges):,} unique (disease, phenotype) pairs")
    print(f"Skipped {n_skipped_non_textbook:,} non-textbook edges (e.g., orphanet, pubmed)")

    # Write KG
    n_out = 0
    with OUT_KG.open("w") as out:
        for (disease, phen), e in edges.items():
            sources = sorted(e["sources"])
            n_textbook = len(set(sources) & TEXTBOOK_SOURCES)
            score_textbook = n_textbook / 4.0
            n_sources = len(sources)
            score_combined = 0.7 * score_textbook + 0.3 * (n_sources / 4.0)
            row = {
                "disease": e.get("disease_orig", disease),
                "phenotype": e.get("phenotype_orig", phen),
                "phenotype_normalized": e["phenotype_normalized"],
                "hpo_id": e.get("hpo_id"),
                "umls_cui": e.get("umls_cui"),
                "sources": sources,
                "n_sources": n_sources,
                "n_textbook_sources": n_textbook,
                "score_textbook": round(score_textbook, 3),
                "score_combined": round(score_combined, 3),
                "provenance": {s: e["provenance"][s] for s in sources},
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Wrote {n_out:,} edges → {OUT_KG}")

    # Build disease_features.json: top features per disease, sorted by score desc
    by_disease = defaultdict(list)
    by_cui = defaultdict(list)
    with OUT_KG.open() as f:
        for line in f:
            row = json.loads(line)
            entry = {
                "phenotype": row["phenotype"],
                "phenotype_normalized": row["phenotype_normalized"],
                "score": row["score_combined"],
                "sources": row["sources"],
                "n_sources": row["n_sources"],
                "hpo_id": row.get("hpo_id"),
                "umls_cui": row.get("umls_cui"),
            }
            by_disease[row["disease"]].append(entry)
            if row.get("umls_cui"):
                by_cui[row["umls_cui"]].append(entry)

    for d, lst in by_disease.items():
        lst.sort(key=lambda x: (-x["score"], -x["n_sources"], x["phenotype_normalized"]))
    for c, lst in by_cui.items():
        lst.sort(key=lambda x: (-x["score"], -x["n_sources"], x["phenotype_normalized"]))

    OUT_FEATURES.write_text(json.dumps(by_disease, ensure_ascii=False, indent=1))
    OUT_FEATURES_BY_CUI.write_text(json.dumps(by_cui, ensure_ascii=False, indent=1))
    print(f"Wrote disease_features → {OUT_FEATURES}  ({len(by_disease):,} diseases)")
    print(f"Wrote disease_features_by_cui → {OUT_FEATURES_BY_CUI}  ({len(by_cui):,} CUIs)")

    # Quick stats
    sizes = [len(lst) for lst in by_disease.values()]
    sizes.sort()
    if sizes:
        med = sizes[len(sizes)//2]
        p90 = sizes[int(len(sizes)*0.9)]
        print(f"Per-disease feature counts: median={med}, p90={p90}, max={sizes[-1]}")


if __name__ == "__main__":
    main()
