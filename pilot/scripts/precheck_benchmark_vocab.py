#!/usr/bin/env python3
"""Pre-check: KG vs all benchmarks vocabulary coverage.

CLAUDE.md 원칙 1: IE source 사전 Coverage 검증.

Measures: which benchmark symptom CUIs are currently in our KG profile?
For each benchmark separately so we can target gap.
"""
import json, pickle, argparse, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")


def benchmark_vocab():
    vocabs = {}
    # SymCat
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    vocabs["symcat"] = set(v["umls_cui"] for v in sym_map.values() if v.get("umls_cui"))
    # DDXPlus
    value_cuis = json.load(open("/mnt/medkg/kg/ddxplus_evidence_value_cuis.json"))
    ddx_cuis = set()
    for m in value_cuis.values():
        for k, v in m.items():
            if isinstance(v, list):
                ddx_cuis.update(v)
    vocabs["ddxplus"] = ddx_cuis
    # RareBench (HPO)
    try:
        hpo_map = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
        vocabs["rarebench"] = set(v["umls_cui"] for v in hpo_map.values() if v.get("umls_cui"))
    except: vocabs["rarebench"] = set()
    return vocabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    args = ap.parse_args()
    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open("pilot/data/pr_universe.json")))

    # Collect all CUIs appearing in any disease profile (lay mode)
    profile_cuis = set()
    allowed = {"patient_reportable", "history", "demographic"}
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") != "HAS_PHENOTYPE": continue
        cat = ed.get("category")
        if cat is None:
            if v not in pr: continue
        elif cat not in allowed: continue
        profile_cuis.add(v)

    print(f"KG: {args.graph}")
    print(f"KG profile CUIs (lay mode): {len(profile_cuis):,}\n")

    vocabs = benchmark_vocab()
    for bench, cuis in vocabs.items():
        covered = cuis & profile_cuis
        missing = cuis - profile_cuis
        pct = 100*len(covered)/len(cuis) if cuis else 0
        print(f"  {bench:<10}: {len(covered)}/{len(cuis)} CUIs covered = {pct:.1f}%")
        print(f"    missing: {len(missing)}")
        # Save missing list
        outp = f"pilot/data/cache/missing_{bench}_cuis.json"
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        json.dump(sorted(missing), open(outp, "w"))
        print(f"    saved missing CUIs → {outp}")


if __name__ == "__main__":
    main()
