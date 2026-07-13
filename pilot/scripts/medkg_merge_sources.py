#!/usr/bin/env python3
"""Multi-source statistical merge of disease–phenotype edges.

Inputs:
  - edges_normalized.jsonl  (LLM IE'd edges from StatPearls/GeneReviews/MedlinePlus/Wikipedia)
  - orphanet_edges.jsonl    (Orphanet structured edges with HPO ID + frequency)
  - kg_v3_cache.json        (PubMed cooccurrence counts) — optional

For each (disease, phenotype) pair, aggregate:
  - source set (which of 5 sources mention it)
  - source_agreement (count of distinct sources)
  - pubmed_cooccurrence (raw count from PubMed cache)
  - frequency (Orphanet, when available)
  - provenance list (NBK IDs, PMIDs, revids, ORPHA codes)

Output edge score formulas:
  score_simple   = source_agreement / 5
  score_textbook = is_in_textbook (StatPearls/MedlinePlus/Wikipedia/GeneReviews) — at least 1
  score_pubmed   = log1p(pubmed_count) / log(max_pubmed_count + 1)
  score_orphanet = orphanet_frequency_score (if present)
  score_combined = α·textbook + β·pubmed + γ·orphanet + δ·source_agreement / Z

Output: kg_merged.jsonl with one record per (disease, phenotype).
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/max/Graph-DDXPlus/data/medkg")
EDGES_NORM = ROOT / "processed" / "edges_normalized.jsonl"
ORPHANET = ROOT / "processed" / "orphanet_edges.jsonl"
PUBMED_CACHE = Path("/home/max/Graph-DDXPlus/pilot/results/kg_v3_cache.json")
OUT = ROOT / "kg" / "kg_merged.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_pubmed_cache():
    """Load PubMed cooccurrence cache (CUI pairs). Returns dict[(cui_a, cui_b)] -> count
    where the keys are sorted tuples to match either direction.
    """
    if not PUBMED_CACHE.exists():
        return {}
    try:
        d = json.loads(PUBMED_CACHE.read_text())
    except Exception:
        return {}
    out = {}
    if isinstance(d, dict) and "pair_counts" in d:
        for pair, cnt in d["pair_counts"]:
            # pair is [cui1, cui2]
            a, b = pair[0], pair[1]
            out[(a, b)] = cnt
            out[(b, a)] = cnt  # symmetric
    return out


def main():
    edges = defaultdict(lambda: {
        "sources": set(),
        "provenance": defaultdict(list),
        "frequency": None, "frequency_score": None,
    })

    # 1. LLM IE edges from textbooks/wikipedia
    n_ie = 0
    if EDGES_NORM.exists():
        with EDGES_NORM.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                e = json.loads(line)
                key = (e["disease"], e.get("phenotype_normalized") or e["phenotype"])
                edges[key]["sources"].add(e["source"])
                edges[key]["disease_orig"] = e["disease"]
                edges[key]["phenotype_orig"] = e["phenotype"]
                edges[key]["umls_cui"] = e.get("umls_cui")
                if e.get("hpo_id"): edges[key]["hpo_id"] = e["hpo_id"]
                # provenance
                p_entry = {"source_id": e.get("source_id"), "section": e.get("section_name")}
                for k in ("pmid", "revid", "chapter_title", "topic_title", "title"):
                    if e.get(k):
                        p_entry[k] = e[k]
                edges[key]["provenance"][e["source"]].append(p_entry)
                n_ie += 1
    print(f"Loaded {n_ie} IE edges → {len(edges)} unique (disease, phenotype) pairs")

    # 2. Orphanet structured edges
    n_orph = 0
    if ORPHANET.exists():
        with ORPHANET.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                e = json.loads(line)
                key = (e["disease"], e["phenotype"].lower())
                edges[key]["sources"].add("orphanet")
                edges[key]["hpo_id"] = e.get("phenotype_id")
                edges[key]["frequency"] = e.get("frequency")
                edges[key]["frequency_score"] = e.get("frequency_score")
                edges[key]["provenance"]["orphanet"].append({"orpha_code": e.get("provenance",{}).get("orpha_code")})
                edges[key]["disease_orig"] = e["disease"]
                edges[key]["phenotype_orig"] = e["phenotype"]
                n_orph += 1
    print(f"Loaded {n_orph} Orphanet edges → {len(edges)} total unique pairs")

    # 3. PubMed CUI-pair cooccurrence (only when both disease and phenotype have CUI)
    pubmed_idx = load_pubmed_cache()
    print(f"Loaded {len(pubmed_idx)} PubMed CUI-pair entries (symmetric expanded)")
    max_pm = max(pubmed_idx.values()) if pubmed_idx else 1
    matched_pm = 0
    for key in edges:
        e = edges[key]
        d_cui = e.get("umls_cui")
        p_cui = e.get("hpo_id")
        # PubMed cache uses CUI pairs (UMLS CUIs). HPO IDs aren't directly CUIs but some
        # phenotypes may have UMLS CUIs separately. For now, only match when disease CUI exists
        # and the phenotype's hpo_id resolves via name lookup later. Skip pubmed alignment.
        cnt = 0
        if d_cui and p_cui:
            cnt = pubmed_idx.get((d_cui, p_cui), 0)
        if cnt > 0:
            edges[key]["pubmed_count"] = cnt
            edges[key]["sources"].add("pubmed")
            matched_pm += 1
        else:
            edges[key]["pubmed_count"] = 0
    print(f"Matched PubMed CUI-pairs: {matched_pm}/{len(edges)} (limited by phenotype CUI availability)")

    # 4. Compute scores + write output
    n_out = 0
    with OUT.open("w") as out:
        for (disease, phen), e in edges.items():
            sources = list(e["sources"])
            n_sources = len(sources)
            textbook_sources = {"statpearls", "genereviews", "medlineplus", "wikipedia"}
            n_textbook = len(set(sources) & textbook_sources)
            score_textbook = n_textbook / 4.0
            pm_count = e.get("pubmed_count", 0)
            score_pubmed = math.log1p(pm_count) / math.log1p(max_pm) if max_pm > 0 else 0
            score_orphanet = e.get("frequency_score") or 0
            # Combined score: weighted average
            score_combined = 0.4 * score_textbook + 0.3 * score_pubmed + 0.2 * score_orphanet + 0.1 * (n_sources / 5.0)
            row = {
                "disease": e.get("disease_orig", disease),
                "phenotype": e.get("phenotype_orig", phen),
                "phenotype_normalized": phen,
                "hpo_id": e.get("hpo_id"),
                "umls_cui": e.get("umls_cui"),
                "sources": sorted(sources),
                "n_sources": n_sources,
                "n_textbook_sources": n_textbook,
                "pubmed_count": pm_count,
                "frequency": e.get("frequency"),
                "frequency_score": e.get("frequency_score"),
                "score_textbook": round(score_textbook, 4),
                "score_pubmed": round(score_pubmed, 4),
                "score_orphanet": round(score_orphanet, 4),
                "score_combined": round(score_combined, 4),
                "provenance": {k: v for k, v in e["provenance"].items()},
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"\nWrote {n_out} merged edges → {OUT}")
    # Sources distribution
    src_count = defaultdict(int)
    for (d, p), e in edges.items():
        for s in e["sources"]:
            src_count[s] += 1
    print(f"Source distribution: {dict(src_count)}")


if __name__ == "__main__":
    main()
