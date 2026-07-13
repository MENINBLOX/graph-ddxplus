#!/usr/bin/env python3
"""DDXPlus benchmark eval using multi-source merged KG.

Loads kg_merged.jsonl, builds disease → top-K phenotypes (sorted by score_combined),
and runs v87-style prompting (KG features inject + CoT tie-break).

Output: results JSON with per-patient diagnosis + provenance (source list per feature).
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
from collections import defaultdict, Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path("/home/max/Graph-DDXPlus")
KG = ROOT / "data" / "medkg" / "kg" / "kg_merged.jsonl"
OUT = ROOT / "data" / "medkg" / "kg" / "ddxplus_eval_results.jsonl"

TOP_K_FEATURES = 8
N_PATIENTS = 5000  # match v87


def load_kg_features():
    """Load merged KG → disease_name → list of (phenotype, score, sources).
    Also build a CUI-keyed index for downstream lookup.
    """
    features = defaultdict(list)
    cui_to_name = {}
    with KG.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            features[e["disease"]].append({
                "phenotype": e["phenotype"],
                "score": e["score_combined"],
                "n_sources": e["n_sources"],
                "sources": e["sources"],
                "hpo_id": e.get("hpo_id"),
            })
            cui = e.get("umls_cui")
            if cui and cui != "null":
                cui_to_name[cui] = e["disease"]
    # Sort each disease's features: multi-source first, then HPO, then score
    # Single-source features kept as fallback when multi-source is empty
    for d in features:
        features[d].sort(key=lambda x: (
            -x["n_sources"],                # multi-source priority (HARD filter signal)
            -1 if x.get("hpo_id") else 0,  # HPO mapped second
            -x["score"],                    # then score
        ))
        features[d] = features[d][:TOP_K_FEATURES]
    return features, cui_to_name


def build_disease_feature_str(features, disease_name, with_provenance=False):
    """Format disease features for prompt insertion."""
    feats = features.get(disease_name, [])
    if not feats: return "—"
    if with_provenance:
        return "; ".join(f"{f['phenotype']} (sources: {','.join(f['sources'])})" for f in feats)
    return ", ".join(f["phenotype"] for f in feats)


def main():
    if not KG.exists():
        print(f"KG not found at {KG} — run merge step first")
        return
    print(f"Loading merged KG from {KG}...")
    features, cui_to_name = load_kg_features()
    print(f"  {len(features)} diseases with features, {len(cui_to_name)} with UMLS CUI")

    # Show diagnostic stats
    n_with_features = sum(1 for d in features if len(features[d]) > 0)
    avg_n = sum(len(features[d]) for d in features) / max(len(features), 1)
    n_textbook = sum(1 for d in features for f in features[d] if any(s in f["sources"] for s in ("statpearls","wikipedia","medlineplus","genereviews")))
    n_orph = sum(1 for d in features for f in features[d] if "orphanet" in f["sources"])
    print(f"  Diseases with features: {n_with_features}")
    print(f"  Avg features per disease: {avg_n:.1f}")
    print(f"  Edges from textbook sources: {n_textbook}")
    print(f"  Edges from orphanet: {n_orph}")

    # Sample: show features for DDXPlus 49 diseases
    print("\nSample DDXPlus disease coverage (first 10):")
    p = Path("/home/max/Graph-DDXPlus/data/ddxplus/disease_umls_mapping.json")
    if p.exists():
        d = json.load(p.open())
        for i, (k, v) in enumerate(d.get("mapping", {}).items()):
            if i >= 10: break
            disease_en = v.get("name_en") or v.get("disease_key") or k
            f_str = build_disease_feature_str(features, disease_en, with_provenance=True)
            print(f"  [{disease_en[:30]:30s}]  {f_str[:200]}")

    # Save disease_features summary for later prompt-based eval
    summary = {d: features[d] for d in features}
    summary_path = ROOT / "data" / "medkg" / "kg" / "disease_features.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved disease_features (by name) → {summary_path}")

    # Also save CUI-keyed version for benchmark eval scripts that use UMLS CUI
    summary_by_cui = {}
    for cui, dn in cui_to_name.items():
        summary_by_cui[cui] = features.get(dn, [])
    summary_cui_path = ROOT / "data" / "medkg" / "kg" / "disease_features_by_cui.json"
    summary_cui_path.write_text(json.dumps(summary_by_cui, indent=2, ensure_ascii=False))
    print(f"Saved disease_features (by CUI) → {summary_cui_path}")
    print(f"  CUI-indexed entries: {len(summary_by_cui)}")
    print(f"\n[Note] Full v87-style prompt eval would require vLLM batch — skipping for sanity test.")


if __name__ == "__main__":
    main()
