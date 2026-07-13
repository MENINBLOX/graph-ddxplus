#!/usr/bin/env python3
"""Quick summary stats of the merged KG: source coverage, agreement distribution, etc."""
import json
from pathlib import Path
from collections import Counter, defaultdict

KG = Path("/home/max/Graph-DDXPlus/data/medkg/kg/kg_merged.jsonl")
DDX_MAP = Path("/home/max/Graph-DDXPlus/data/ddxplus/disease_umls_mapping.json")
SYM_MAP = Path("/home/max/Graph-DDXPlus/data/symcat/disease_umls_mapping.json")
RB_MAP = Path("/home/max/Graph-DDXPlus/data/rarebench/disease_umls_mapping.json")


def load_benchmark_diseases(path):
    if not path.exists(): return set()
    d = json.load(path.open())
    out = set()
    for k, v in d.get("mapping", {}).items():
        name = v.get("name_en") or v.get("disease_name") or v.get("umls_name") or k
        if name: out.add(name)
    return out


def main():
    if not KG.exists():
        print(f"KG not found: {KG}")
        return
    n_edges = 0
    by_source = Counter()
    by_n_sources = Counter()
    diseases = set()
    by_disease_n_features = defaultdict(int)
    score_combined = []
    has_hpo = 0
    by_disease_textbook = defaultdict(int)
    with KG.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            n_edges += 1
            for s in e["sources"]:
                by_source[s] += 1
            by_n_sources[e["n_sources"]] += 1
            diseases.add(e["disease"])
            by_disease_n_features[e["disease"]] += 1
            score_combined.append(e["score_combined"])
            if e.get("hpo_id"): has_hpo += 1
            tb = sum(1 for s in e["sources"] if s in ("statpearls","genereviews","medlineplus","wikipedia"))
            if tb > 0:
                by_disease_textbook[e["disease"]] += 1
    print(f"=== KG Merged Summary ===")
    print(f"Total edges: {n_edges:,}")
    print(f"Unique diseases: {len(diseases):,}")
    print(f"Avg edges/disease: {n_edges/len(diseases):.1f}")
    print(f"Edges with HPO ID: {has_hpo:,} ({100*has_hpo/n_edges:.1f}%)")
    print(f"\nSource counts:")
    for s, c in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {s:12s} {c:>8,}")
    print(f"\nSource agreement distribution:")
    for n in sorted(by_n_sources):
        print(f"  {n} sources: {by_n_sources[n]:>8,} edges")
    print(f"\nScore (combined) percentiles: min={min(score_combined):.3f}, "
          f"p50={sorted(score_combined)[len(score_combined)//2]:.3f}, "
          f"p90={sorted(score_combined)[int(0.9*len(score_combined))]:.3f}, "
          f"max={max(score_combined):.3f}")

    # Benchmark coverage
    print(f"\n=== Benchmark Coverage ===")
    for name, path in [("DDXPlus", DDX_MAP), ("SymCat", SYM_MAP), ("RareBench", RB_MAP)]:
        bench = load_benchmark_diseases(path)
        if not bench: continue
        covered = bench & diseases
        with_textbook = {d for d in covered if by_disease_textbook.get(d, 0) > 0}
        print(f"  {name:12s} {len(bench):>5} diseases | KG covered: {len(covered):>5} ({100*len(covered)/len(bench):.0f}%) | with textbook: {len(with_textbook):>5}")


if __name__ == "__main__":
    main()
