#!/usr/bin/env python3
"""Dual-resource KG v2: textbook IE ∪ PubMed IE ∪ distinguishing IE.

Adds the third edge type from distinguishing-features IE pass on diagnostic
sections (differential diagnosis, history and physical, signs and symptoms).
Distinguishing features are ranked FIRST in disease_features (highest priority)
because they capture clinical-disambiguation cues missing from generic phenotypes.

Inputs:
  - $MEDKG_ROOT/processed/edges_normalized.jsonl   (textbook IE)
  - $MEDKG_ROOT/processed/edges_pubmed_ie.jsonl    (PubMed IE)
  - $MEDKG_ROOT/processed/edges_distinguishing.jsonl  (distinguishing IE — NEW)

Outputs (suffix "_dual_v2"):
  - $MEDKG_ROOT/kg/kg_dual_v2.jsonl
  - $MEDKG_ROOT/kg/disease_features_dual_v2.json
  - $MEDKG_ROOT/kg/disease_features_dual_v2_by_cui.json

Score formula:
  score_distinguish = 1.0 if edge has distinguishing source
  score_textbook    = n_textbook_sources / 4
  score_pubmed      = log1p(pubmed_mention_count) / log1p(max_pubmed_count)
  score_combined    = 0.5 * score_distinguish + 0.25 * score_textbook + 0.25 * score_pubmed
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

EDGES_TEXTBOOK = MEDKG_ROOT / "processed" / "edges_normalized.jsonl"
EDGES_PUBMED = MEDKG_ROOT / "processed" / "edges_pubmed_ie.jsonl"
EDGES_DISTINGUISH = MEDKG_ROOT / "processed" / "edges_distinguishing.jsonl"
OUT_DIR = MEDKG_ROOT / "kg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_KG = OUT_DIR / "kg_dual_v2.jsonl"
OUT_FEATURES = OUT_DIR / "disease_features_dual_v2.json"
OUT_FEATURES_BY_CUI = OUT_DIR / "disease_features_dual_v2_by_cui.json"

TEXTBOOK_SOURCES = {"statpearls", "genereviews", "medlineplus", "wikipedia"}


def main():
    edges = defaultdict(lambda: {"sources": set(), "provenance": defaultdict(list),
                                  "hpo_id": None, "pubmed_count": 0,
                                  "is_distinguishing": False})

    # 1. Textbook IE
    n_tb = 0
    if EDGES_TEXTBOOK.exists():
        with EDGES_TEXTBOOK.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                e = json.loads(line)
                src = e.get("source")
                if src not in TEXTBOOK_SOURCES: continue
                phen = (e.get("phenotype_normalized") or e.get("phenotype") or "").strip().lower()
                if not phen: continue
                key = (e["disease"], phen)
                edges[key]["sources"].add(src)
                edges[key]["disease_orig"] = e["disease"]
                edges[key]["phenotype_orig"] = e.get("phenotype", phen)
                edges[key]["phenotype_normalized"] = phen
                edges[key]["umls_cui"] = e.get("umls_cui")
                if e.get("hpo_id"): edges[key]["hpo_id"] = e["hpo_id"]
                p_entry = {"source_id": e.get("source_id"), "section": e.get("section_name")}
                for k in ("pmid","revid","chapter_title","topic_title","title"):
                    if e.get(k): p_entry[k] = e[k]
                edges[key]["provenance"][src].append(p_entry)
                n_tb += 1
    print(f"Textbook edges: {n_tb:,} → {len(edges):,} unique pairs")

    # 2. PubMed IE
    n_pm = 0
    if EDGES_PUBMED.exists():
        with EDGES_PUBMED.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                e = json.loads(line)
                phen = (e.get("phenotype") or "").strip().lower()
                if not phen: continue
                key = (e.get("disease",""), phen)
                edges[key]["sources"].add("pubmed")
                edges[key]["disease_orig"] = e.get("disease") or edges[key].get("disease_orig","")
                edges[key]["phenotype_orig"] = e.get("phenotype")
                edges[key]["phenotype_normalized"] = phen
                if e.get("umls_cui"): edges[key]["umls_cui"] = e["umls_cui"]
                edges[key]["provenance"]["pubmed"].append({"pmid": e.get("pmid",""), "title": e.get("title","")})
                edges[key]["pubmed_count"] += 1
                n_pm += 1
    print(f"PubMed edges: {n_pm:,} → {len(edges):,} unique pairs")

    # 3. Distinguishing IE — boost flag, no separate score for source-count
    n_di = 0
    if EDGES_DISTINGUISH.exists():
        with EDGES_DISTINGUISH.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                e = json.loads(line)
                phen = (e.get("phenotype") or "").strip().lower()
                if not phen: continue
                key = (e.get("disease",""), phen)
                edges[key]["sources"].add("distinguishing")
                edges[key]["is_distinguishing"] = True
                edges[key]["disease_orig"] = e.get("disease") or edges[key].get("disease_orig","")
                edges[key]["phenotype_orig"] = e.get("phenotype")
                edges[key]["phenotype_normalized"] = phen
                if e.get("umls_cui"): edges[key]["umls_cui"] = e["umls_cui"]
                edges[key]["provenance"]["distinguishing"].append({
                    "source_id": e.get("source_id"),
                    "section": e.get("section_name"),
                    "underlying_source": e.get("source", "")
                })
                n_di += 1
    print(f"Distinguishing edges: {n_di:,} → {len(edges):,} unique pairs total")

    if not edges: sys.exit("No edges")

    max_pm = max(e["pubmed_count"] for e in edges.values()) or 1

    n_out = 0
    with OUT_KG.open("w") as out:
        for (disease, phen), e in edges.items():
            sources = sorted(e["sources"])
            n_textbook = len(set(sources) & TEXTBOOK_SOURCES)
            n_sources = len(sources)
            score_textbook = n_textbook / 4.0
            score_pubmed = math.log1p(e["pubmed_count"]) / math.log1p(max_pm) if max_pm > 0 else 0
            score_distinguish = 1.0 if e["is_distinguishing"] else 0.0
            score_combined = 0.5 * score_distinguish + 0.25 * score_textbook + 0.25 * score_pubmed
            row = {"disease": e.get("disease_orig", disease),
                   "phenotype": e.get("phenotype_orig", phen),
                   "phenotype_normalized": e["phenotype_normalized"],
                   "hpo_id": e.get("hpo_id"),
                   "umls_cui": e.get("umls_cui"),
                   "sources": sources,
                   "n_sources": n_sources,
                   "n_textbook_sources": n_textbook,
                   "pubmed_count": e["pubmed_count"],
                   "is_distinguishing": e["is_distinguishing"],
                   "score_textbook": round(score_textbook, 3),
                   "score_pubmed": round(score_pubmed, 3),
                   "score_distinguish": score_distinguish,
                   "score_combined": round(score_combined, 3),
                   "provenance": {s: e["provenance"][s] for s in sources}}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Wrote {n_out:,} edges → {OUT_KG}")

    # disease_features: prioritize distinguishing first, then by score_combined
    by_disease = defaultdict(list)
    by_cui = defaultdict(list)
    with OUT_KG.open() as f:
        for line in f:
            row = json.loads(line)
            entry = {"phenotype": row["phenotype"],
                     "phenotype_normalized": row["phenotype_normalized"],
                     "score": row["score_combined"],
                     "sources": row["sources"], "n_sources": row["n_sources"],
                     "is_distinguishing": row["is_distinguishing"],
                     "pubmed_count": row.get("pubmed_count", 0),
                     "hpo_id": row.get("hpo_id"), "umls_cui": row.get("umls_cui")}
            by_disease[row["disease"]].append(entry)
            if row.get("umls_cui"):
                by_cui[row["umls_cui"]].append(entry)

    def _key(x):
        return (-int(x["is_distinguishing"]), -x["score"], -x["n_sources"], x["phenotype_normalized"])
    for d, lst in by_disease.items(): lst.sort(key=_key)
    for c, lst in by_cui.items(): lst.sort(key=_key)

    OUT_FEATURES.write_text(json.dumps(by_disease, ensure_ascii=False, indent=1))
    OUT_FEATURES_BY_CUI.write_text(json.dumps(by_cui, ensure_ascii=False, indent=1))
    print(f"Wrote {OUT_FEATURES}  ({len(by_disease):,} diseases)")
    print(f"Wrote {OUT_FEATURES_BY_CUI}  ({len(by_cui):,} CUIs)")

    sizes = sorted(len(lst) for lst in by_disease.values())
    n_distinguish_per_disease = sorted(
        sum(1 for e in lst if e["is_distinguishing"]) for lst in by_disease.values()
    )
    print(f"Per-disease feature counts: median={sizes[len(sizes)//2]}, p90={sizes[int(len(sizes)*0.9)]}, max={sizes[-1]}")
    print(f"Per-disease DISTINGUISHING counts: median={n_distinguish_per_disease[len(n_distinguish_per_disease)//2]}, p90={n_distinguish_per_disease[int(len(n_distinguish_per_disease)*0.9)]}")


if __name__ == "__main__":
    main()
