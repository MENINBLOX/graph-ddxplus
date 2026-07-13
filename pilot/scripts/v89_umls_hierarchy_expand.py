#!/usr/bin/env python3
"""v89 — UMLS hierarchy expansion of v85 LLM CUI edges.

For each (disease, CUI) edge from LLM IE, expand CUI to include UMLS
direct children (depth 1). Example:
  LLM: Disease X has "Pruritus" (C0033774)
  → UMLS children: "Pruritus of eyelid" (C0xxx), "Pruritus of scrotum", etc.
  → Add edges Disease X → each child with same prob

학술적 정당: UMLS MRREL은 표준 ontology. SymCat CUI 직접 사용 X.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
from collections import defaultdict


def load_mrrel_children(mrrel_path):
    """For each CUI1, find CUI2 such that REL='CHD' or REL='RN' (narrower).
    Returns: dict[parent_cui] -> set[child_cui]
    """
    children = defaultdict(set)
    print(f"  Scanning MRREL for parent → children...", flush=True)
    n = 0
    with open(mrrel_path) as f:
        for line in f:
            n += 1
            parts = line.split("|")
            if len(parts) < 8: continue
            cui1, _, _, rel, cui2 = parts[0], parts[1], parts[2], parts[3], parts[4]
            # CHD/RN: cui1 is broader, cui2 is narrower
            # We want parent_cui → child_cui mapping
            if rel == "CHD":
                # cui1 has child cui2
                children[cui1].add(cui2)
            elif rel == "PAR":
                # cui1's parent is cui2 → cui2 has child cui1
                children[cui2].add(cui1)
            if n % 5_000_000 == 0:
                print(f"    scanned {n//1_000_000}M lines, {len(children):,} parent CUIs",
                      flush=True)
    print(f"  Done: {len(children):,} CUIs with children", flush=True)
    return children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_edges", required=True, help="v85 cui_edges jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mrrel", default="/windows/data/umls_subset/MRREL.RRF")
    ap.add_argument("--depth", type=int, default=1,
                    help="hierarchy depth to expand (1 = direct children only)")
    ap.add_argument("--prob_decay", type=float, default=0.5,
                    help="prob multiplier for expanded children")
    args = ap.parse_args()

    print(f"Loading MRREL hierarchy from {args.mrrel}...", flush=True)
    children = load_mrrel_children(args.mrrel)

    print(f"Loading input edges {args.input_edges}...", flush=True)
    records = []
    with open(args.input_edges) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  {len(records)} disease records loaded", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_orig = n_new = 0
    with open(args.out, "w") as fout:
        for r in records:
            dcui = r["dcui"]
            orig_edges = dict(r["edges"])  # {phen_cui: prob}
            expanded = dict(orig_edges)
            for cui, prob in orig_edges.items():
                # Expand to children at depth 1
                visited = {cui}
                frontier = {cui}
                for d in range(args.depth):
                    next_frontier = set()
                    for c in frontier:
                        for child in children.get(c, []):
                            if child in visited: continue
                            visited.add(child)
                            next_frontier.add(child)
                            # Add with decayed prob, only if not already present
                            new_prob = prob * (args.prob_decay ** (d+1))
                            if child not in expanded:
                                expanded[child] = new_prob
                                n_new += 1
                            elif expanded[child] < new_prob:
                                expanded[child] = new_prob
                    frontier = next_frontier
            n_orig += len(orig_edges)
            fout.write(json.dumps({
                "disease": r["disease"], "dcui": dcui,
                "source": r.get("source",""), "edges": expanded
            }) + "\n")
    print(f"\nOriginal edges: {n_orig}")
    print(f"Added via hierarchy: {n_new} (avg {n_new/len(records):.1f}/disease)")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
