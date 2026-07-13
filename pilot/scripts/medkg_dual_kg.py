#!/usr/bin/env python3
"""Dual-resource KG builder: textbook IE ∪ PubMed IE.

Inputs:
  - $MEDKG_ROOT/processed/edges_normalized.jsonl   (textbook IE: StatPearls/GeneReviews/MedlinePlus/Wikipedia)
  - $MEDKG_ROOT/processed/edges_pubmed_ie.jsonl    (PubMed IE: 38K seed × top-20 abstracts)

Outputs:
  - $MEDKG_ROOT/kg/kg_dual.jsonl                   one record per (disease, phenotype)
  - $MEDKG_ROOT/kg/disease_features_dual.json
  - $MEDKG_ROOT/kg/disease_features_dual_by_cui.json

Score formula:
  score_textbook = n_textbook_sources / 4   (StatPearls/GeneReviews/MedlinePlus/Wikipedia)
  score_pubmed   = log1p(pubmed_mention_count) / log1p(max_pubmed_count)
  score_combined = 0.4 * score_textbook + 0.4 * score_pubmed + 0.2 * (n_sources / 5)

Phenotype normalization:
  - lowercase trim for matching
  - HPO ID preserved when present (textbook side)

Provenance:
  - per (disease, phenotype): list of source identifiers (NBK*, PMID*, revid*, etc.)
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

EDGES_TEXTBOOK = MEDKG_ROOT / "processed" / "edges_normalized.jsonl"
EDGES_PUBMED = MEDKG_ROOT / "processed" / "edges_pubmed_ie.jsonl"
OUT_DIR = MEDKG_ROOT / "kg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_KG = OUT_DIR / "kg_dual.jsonl"
OUT_FEATURES = OUT_DIR / "disease_features_dual.json"
OUT_FEATURES_BY_CUI = OUT_DIR / "disease_features_dual_by_cui.json"

TEXTBOOK_SOURCES = {"statpearls", "genereviews", "medlineplus", "wikipedia"}


def main():
    if not EDGES_TEXTBOOK.exists():
        sys.exit(f"Input not found: {EDGES_TEXTBOOK}")

    edges = defaultdict(lambda: {
        "sources": set(),
        "provenance": defaultdict(list),
        "hpo_id": None,
        "pubmed_count": 0,
    })

    # 1. Textbook IE
    n_tb = 0
    with EDGES_TEXTBOOK.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            src = e.get("source")
            if src not in TEXTBOOK_SOURCES: continue   # skip orphanet
            phen_norm = (e.get("phenotype_normalized") or e.get("phenotype") or "").strip().lower()
            if not phen_norm: continue
            key = (e["disease"], phen_norm)
            edges[key]["sources"].add(src)
            edges[key]["disease_orig"] = e["disease"]
            edges[key]["phenotype_orig"] = e.get("phenotype", phen_norm)
            edges[key]["phenotype_normalized"] = phen_norm
            edges[key]["umls_cui"] = e.get("umls_cui")
            if e.get("hpo_id"): edges[key]["hpo_id"] = e["hpo_id"]
            p_entry = {"source_id": e.get("source_id"), "section": e.get("section_name")}
            for k in ("pmid", "revid", "chapter_title", "topic_title", "title"):
                if e.get(k): p_entry[k] = e[k]
            edges[key]["provenance"][src].append(p_entry)
            n_tb += 1
    print(f"Textbook IE edges loaded: {n_tb:,} → {len(edges):,} unique pairs so far")

    # 2. PubMed IE (one entry per (disease, phenotype, abstract); dedupe at edge level + count mentions)
    n_pm = 0
    if EDGES_PUBMED.exists():
        with EDGES_PUBMED.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                e = json.loads(line)
                phen_norm = (e.get("phenotype") or "").strip().lower()
                if not phen_norm: continue
                key = (e.get("disease", ""), phen_norm)
                edges[key]["sources"].add("pubmed")
                edges[key]["disease_orig"] = e.get("disease") or edges[key].get("disease_orig", "")
                edges[key]["phenotype_orig"] = e.get("phenotype")
                edges[key]["phenotype_normalized"] = phen_norm
                if e.get("umls_cui"):
                    edges[key]["umls_cui"] = e["umls_cui"]
                pm_entry = {"pmid": e.get("pmid", ""), "title": e.get("title", "")}
                edges[key]["provenance"]["pubmed"].append(pm_entry)
                edges[key]["pubmed_count"] += 1
                n_pm += 1
    else:
        print(f"  (PubMed IE not yet available at {EDGES_PUBMED})")
    print(f"PubMed IE edges loaded: {n_pm:,} → {len(edges):,} unique pairs total")

    if not edges:
        sys.exit("No edges to write")

    max_pm = max(e["pubmed_count"] for e in edges.values()) or 1
    print(f"Max PubMed mention count: {max_pm}")

    n_out = 0
    with OUT_KG.open("w") as out:
        for (disease, phen), e in edges.items():
            sources = sorted(e["sources"])
            n_textbook = len(set(sources) & TEXTBOOK_SOURCES)
            n_sources = len(sources)
            score_textbook = n_textbook / 4.0
            pm = e["pubmed_count"]
            score_pubmed = math.log1p(pm) / math.log1p(max_pm) if max_pm > 0 else 0.0
            score_combined = 0.4 * score_textbook + 0.4 * score_pubmed + 0.2 * (n_sources / 5.0)
            row = {
                "disease": e.get("disease_orig", disease),
                "phenotype": e.get("phenotype_orig", phen),
                "phenotype_normalized": e["phenotype_normalized"],
                "hpo_id": e.get("hpo_id"),
                "umls_cui": e.get("umls_cui"),
                "sources": sources,
                "n_sources": n_sources,
                "n_textbook_sources": n_textbook,
                "pubmed_count": pm,
                "score_textbook": round(score_textbook, 3),
                "score_pubmed": round(score_pubmed, 3),
                "score_combined": round(score_combined, 3),
                "provenance": {s: e["provenance"][s] for s in sources},
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Wrote {n_out:,} edges → {OUT_KG}")

    # disease_features
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
                "pubmed_count": row.get("pubmed_count", 0),
                "hpo_id": row.get("hpo_id"),
                "umls_cui": row.get("umls_cui"),
            }
            by_disease[row["disease"]].append(entry)
            if row.get("umls_cui"):
                by_cui[row["umls_cui"]].append(entry)

    for d, lst in by_disease.items():
        lst.sort(key=lambda x: (-x["score"], -x["n_sources"], -x.get("pubmed_count", 0), x["phenotype_normalized"]))
    for c, lst in by_cui.items():
        lst.sort(key=lambda x: (-x["score"], -x["n_sources"], -x.get("pubmed_count", 0), x["phenotype_normalized"]))

    OUT_FEATURES.write_text(json.dumps(by_disease, ensure_ascii=False, indent=1))
    OUT_FEATURES_BY_CUI.write_text(json.dumps(by_cui, ensure_ascii=False, indent=1))
    print(f"Wrote disease_features → {OUT_FEATURES}  ({len(by_disease):,} diseases)")
    print(f"Wrote disease_features_by_cui → {OUT_FEATURES_BY_CUI}  ({len(by_cui):,} CUIs)")

    sizes = sorted(len(lst) for lst in by_disease.values())
    if sizes:
        print(f"Per-disease feature counts: median={sizes[len(sizes)//2]}, p90={sizes[int(len(sizes)*0.9)]}, max={sizes[-1]}")


if __name__ == "__main__":
    main()
