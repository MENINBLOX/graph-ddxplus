#!/usr/bin/env python3
"""v91 — Conservative UMLS hierarchy expansion of v85.

v89 lesson: depth=1 broad expansion (4.9M edges) breaks IDF balance, DDXPlus
regression -3.55%p despite SymCat coverage 80%→88%.

v91 fix:
- Max N children per parent (default 5) — avoid noisy broad CUIs.
- Filter children by semantic type alignment (only T033/T184 phenotype-ish).
- Stronger decay (default 0.3).

학술적 정당: UMLS standard ontology, depth=1, semantic-type filtered.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
from collections import defaultdict


def load_mrsty_phen_cuis(mrsty_path):
    """Phenotype-relevant CUIs: T033 Finding, T184 Sign/Symptom,
    T046 Pathologic Function, T037 Injury, T048 Mental Dis, T049 Cell Func,
    T191 Neoplastic Process, T190 Anatomical Abnormality."""
    phen_tuis = {"T033", "T184", "T046", "T037", "T048", "T049", "T191", "T190"}
    phen_cuis = set()
    with open(mrsty_path) as f:
        for line in f:
            parts = line.split("|")
            if len(parts) >= 2 and parts[1] in phen_tuis:
                phen_cuis.add(parts[0])
    return phen_cuis


def load_mrrel_children_filtered(mrrel_path, phen_cuis, parent_set):
    """Children only if both parent and child are phenotype CUIs.
    parent_set: only collect children for these parent CUIs."""
    children = defaultdict(set)
    print(f"  Scanning MRREL (filtered, phen-only)...", flush=True)
    n = 0
    with open(mrrel_path) as f:
        for line in f:
            n += 1
            parts = line.split("|")
            if len(parts) < 5: continue
            cui1, _, _, rel, cui2 = parts[0], parts[1], parts[2], parts[3], parts[4]
            if cui1 == cui2: continue
            if rel == "CHD":
                if cui1 in parent_set and cui1 in phen_cuis and cui2 in phen_cuis:
                    children[cui1].add(cui2)
            elif rel == "PAR":
                if cui2 in parent_set and cui1 in phen_cuis and cui2 in phen_cuis:
                    children[cui2].add(cui1)
            if n % 10_000_000 == 0:
                print(f"    {n//1_000_000}M lines", flush=True)
    return children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_edges", required=True, help="v85 cui_edges jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mrrel", default="/windows/data/umls_subset/MRREL.RRF")
    ap.add_argument("--mrsty", default="/windows/data/umls_subset/MRSTY.RRF")
    ap.add_argument("--max_children", type=int, default=5)
    ap.add_argument("--prob_decay", type=float, default=0.3)
    args = ap.parse_args()

    print(f"Loading phen-relevant CUIs from MRSTY...", flush=True)
    phen_cuis = load_mrsty_phen_cuis(args.mrsty)
    print(f"  Phen CUIs: {len(phen_cuis):,}", flush=True)

    records = []
    with open(args.input_edges) as f:
        for line in f:
            records.append(json.loads(line))
    parent_set = set()
    for r in records:
        for c in r["edges"]: parent_set.add(c)
    print(f"  Disease records: {len(records)}, parent CUIs in edges: {len(parent_set)}", flush=True)

    children = load_mrrel_children_filtered(args.mrrel, phen_cuis, parent_set)
    n_p = sum(1 for c in children if children[c])
    print(f"  Parents with phen-child: {n_p}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_orig = n_new = 0
    with open(args.out, "w") as fout:
        for r in records:
            orig = dict(r["edges"])
            expanded = dict(orig)
            for cui, prob in orig.items():
                kids = list(children.get(cui, []))[:args.max_children]
                for c in kids:
                    p_new = prob * args.prob_decay
                    if c not in expanded:
                        expanded[c] = p_new; n_new += 1
                    elif expanded[c] < p_new:
                        expanded[c] = p_new
            n_orig += len(orig)
            fout.write(json.dumps({
                "disease": r["disease"], "dcui": r["dcui"],
                "source": r.get("source",""), "edges": expanded
            }) + "\n")
    print(f"\nOriginal edges: {n_orig}")
    print(f"Added (max {args.max_children}/parent, decay {args.prob_decay}): "
          f"{n_new} (avg {n_new/len(records):.1f}/disease)")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
