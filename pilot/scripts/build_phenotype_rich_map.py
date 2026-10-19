#!/usr/bin/env python3
"""Build an enriched phenotype relation map {child_cui: [parent_cui,...]} from
UMLS MRREL hierarchical relations (PAR/RB/CHD/RN) — strict §3-safe ontology structure
(no disease-symptom mapping tables). Prunes over-generic parents by fan-in to avoid
over-merging distinct diseases.
"""
from __future__ import annotations
import json, argparse, sys
from collections import defaultdict, Counter

SP = "/tmp/claude-1000/-home-max-thesis/edda7dda-7b76-45bb-a79a-ce62d7e31a1b/scratchpad"


def load_pairs(paths):
    edges = defaultdict(set)  # child -> {parent}
    for p in paths:
        for line in open(p):
            child, parent = line.rstrip("\n").split("\t")
            if child and parent and child != parent:
                edges[child].add(parent)
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_fanin", type=int, default=60,
                    help="drop parents that have more than this many children (over-generic)")
    ap.add_argument("--max_parents", type=int, default=4,
                    help="cap parents kept per child (prefer most specific = lowest fan-in)")
    ap.add_argument("--seed", default="pilot/cache/umls_maps/phenotype_parents.json",
                    help="original map to merge in (kept as-is)")
    ap.add_argument("--out", default="pilot/cache/prod/phenotype_parents_rich.json")
    ap.add_argument("--no_seed", action="store_true")
    a = ap.parse_args()

    edges = load_pairs([f"{SP}/pairs_pass1.tsv", f"{SP}/pairs_pass2.tsv"])

    # global fan-in over all extracted edges
    fanin = Counter()
    for ch, pas in edges.items():
        for pa in pas:
            fanin[pa] += 1

    out = defaultdict(set)
    for ch, pas in edges.items():
        # keep parents below fan-in cap; if all pruned, keep the least-generic ones anyway
        kept = [p for p in pas if fanin[p] <= a.max_fanin]
        if not kept:
            kept = sorted(pas, key=lambda p: fanin[p])[:1]
        # prefer most specific (lowest fan-in) parents
        kept = sorted(set(kept), key=lambda p: fanin[p])[:a.max_parents]
        out[ch].update(kept)

    # merge original seed map (union) unless disabled
    if not a.no_seed:
        seed = json.load(open(a.seed))
        for ch, pas in seed.items():
            out[ch].update(pas)

    final = {ch: sorted(pas) for ch, pas in out.items() if pas}
    json.dump(final, open(a.out, "w"))
    nvals = sum(len(v) for v in final.values())
    print(f"[build] max_fanin={a.max_fanin} max_parents={a.max_parents} no_seed={a.no_seed}")
    print(f"[build] entries={len(final)} edges={nvals} avg={nvals/len(final):.2f} -> {a.out}")


if __name__ == "__main__":
    main()
