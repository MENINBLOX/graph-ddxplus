#!/usr/bin/env python3
"""Fix v10 graph: DDXPlus eval CUIs lack content because v7 alias placed
phens on broader CUIs. Now copy those phens to eval CUIs.

For each DDXPlus disease, if its eval CUI != "best alias" CUI with rich content,
copy phens from alias CUI to eval CUI.
"""
from __future__ import annotations
import sys, json, math, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v10.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v13.pkl"

# Eval-CUI → alias CUI (where rich content was added by v7)
# Per audit: eval uses disease_icd10_cui_mapping.json, v7 used different CUIs
EVAL_TO_ALIAS = {
    # NSTEMI: eval=C0010072 (Thrombosis-coronary, wrong), alias=C1304447 (proper)
    "C0010072": "C1304447",
    # HIV (initial infection): eval=C0001175, alias=C0019693 (HIV Infections)
    # — actually v7 alias added to C0001175 directly, so should be fine
    # Spontaneous rib fracture: eval might be C0478237; v7 alias was C0035525
    "C0478237": "C0035525",
    # Localized edema: eval=C0013609, v7 also used C0013609 → no issue
    # Acute COPD exacerbation: eval=?, v7 used C0741421
    # Acute dystonic reactions: eval=C0013362, v7 also C0013362 → ok
    # Pancreatic neoplasm: eval=C0346647, v7 used C0153466
    "C0346647": "C0153466",
    # Larygospasm: check
}


def main():
    G = pickle.load(open(GRAPH_IN, "rb"))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddx_eval_cuis = {info["cui"]: dn for dn, info in icd.items() if "cui" in info}

    print(f"Loaded v10: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"DDXPlus eval CUIs: {len(ddx_eval_cuis)}")

    # For each eval CUI, find a related CUI with more content
    # Search alias relationship: same disease, different concept granularity
    # Manual mapping based on prior audit findings

    print("\nFix plan:")
    added = 0
    for eval_cui, alias_cui in EVAL_TO_ALIAS.items():
        eval_name = ddx_eval_cuis.get(eval_cui, "?")
        n_eval = sum(1 for _,_,e in G.out_edges(eval_cui, data=True) if e.get("etype")=="HAS_PHENOTYPE") if eval_cui in G else 0
        n_alias = sum(1 for _,_,e in G.out_edges(alias_cui, data=True) if e.get("etype")=="HAS_PHENOTYPE") if alias_cui in G else 0
        print(f"  {eval_name}: eval={eval_cui} ({n_eval} phens) ← alias={alias_cui} ({n_alias} phens)")

        if eval_cui not in G:
            G.add_node(eval_cui, ntype="Disease", name=eval_name)
        # Copy edges from alias to eval
        existing_phens = {p for _, p, e in G.out_edges(eval_cui, data=True) if e.get("etype")=="HAS_PHENOTYPE"}
        for _, p, e in list(G.out_edges(alias_cui, data=True)):
            if e.get("etype") != "HAS_PHENOTYPE": continue
            if p in existing_phens: continue
            G.add_edge(eval_cui, p, etype="HAS_PHENOTYPE", weight=e.get("weight", 0), source="alias_copy")
            added += 1
    print(f"\nAdded {added} alias-copied edges")

    # Also check: are there other diseases with name mismatches we missed?
    # Quick scan: for each DDXPlus eval CUI with <30 phens, check possible aliases
    print("\nFinal check on all eval CUIs:")
    n_low = 0
    for cui, name in ddx_eval_cuis.items():
        if cui not in G:
            print(f"  {name}: NOT IN GRAPH ({cui})")
            n_low += 1
            continue
        n = sum(1 for _,_,e in G.out_edges(cui, data=True) if e.get("etype")=="HAS_PHENOTYPE")
        if n < 30:
            print(f"  {name}: {n} phens ({cui})")
            n_low += 1
    print(f"\n{n_low}/49 diseases still have <30 phens")

    print(f"\nSaving v13 to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    main()
