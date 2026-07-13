#!/usr/bin/env python3
"""Diagnose what SymCat pairs are MISSING from v85 KG.

For low-coverage diseases, examine which specific symptoms are missing.
Is the missing symptom CUI:
 (a) entirely absent from any disease profile (vocabulary gap), or
 (b) present in OTHER disease profiles but not this one (IE incompleteness)?
"""
from __future__ import annotations
import json, pickle, argparse, sys
from pathlib import Path
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/onlykg_graph_v85_s3.pkl")
    ap.add_argument("--max_low_dis", type=int, default=20)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    parsed = json.load(open("data/symcat/symcat_parsed_full.json"))
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    pr = set(json.load(open("pilot/data/pr_universe.json")))
    pairs = parsed["disease_symptom_pairs"]
    sym2cui = {n: v["umls_cui"] for n, v in sym_map.items() if v.get("umls_cui")}
    cui2sym = {v: n for n, v in sym2cui.items()}
    dis2cui = {n: v["umls_cui"] for n, v in dis_map.items() if v.get("umls_cui")}

    allowed = {"patient_reportable", "history", "demographic"}

    # Build all profiles
    profiles = {}
    all_profile_cuis = set()
    for dname, dcui in dis2cui.items():
        if dcui not in G: continue
        prof = set()
        for _, p, ed in G.out_edges(dcui, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr: continue
            elif cat not in allowed: continue
            prof.add(p)
        profiles[dname] = prof
        all_profile_cuis.update(prof)

    # Coverage per disease
    cov = []
    for dname, syms in pairs.items():
        prof = profiles.get(dname, set())
        sym_cuis = [sym2cui[s] for s, _ in syms if sym2cui.get(s)]
        if not sym_cuis: continue
        match = sum(1 for c in sym_cuis if c in prof)
        cov.append((match/len(sym_cuis), dname, sym_cuis, prof))
    cov.sort()

    print(f"\n=== {args.max_low_dis} lowest-coverage diseases (with mapped symptoms) ===")
    n_in_any_other = Counter()
    n_truly_missing = Counter()
    for rate, dname, sym_cuis, prof in cov[:args.max_low_dis]:
        missing = [c for c in sym_cuis if c not in prof]
        n_in_other = sum(1 for c in missing if c in all_profile_cuis)
        n_truly_abs = sum(1 for c in missing if c not in all_profile_cuis)
        print(f"\n  [{rate*100:.1f}%] {dname}: {len(sym_cuis)} symptoms, "
              f"{len(missing)} missing")
        print(f"    in OTHER disease's profile: {n_in_other} (IE gap)")
        print(f"    not in ANY profile: {n_truly_abs} (vocab gap)")
        # Sample missing symptom names
        miss_names = [cui2sym.get(c, c) for c in missing]
        print(f"    sample missing: {miss_names[:5]}")
        n_in_any_other["sum"] += n_in_other
        n_truly_missing["sum"] += n_truly_abs

    print(f"\n=== Summary across {args.max_low_dis} diseases ===")
    print(f"  Missing pairs in OTHER's profile (IE gap): {n_in_any_other['sum']}")
    print(f"  Missing pairs not in ANY profile (vocab gap): {n_truly_missing['sum']}")


if __name__ == "__main__":
    main()
