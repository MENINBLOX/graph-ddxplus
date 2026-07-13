#!/usr/bin/env python3
"""only-KG vocabulary gap diagnostic.

Quantify the structural gap between DDXPlus patient evidence CUIs and KG node CUIs:
  - Direct overlap (patient CUI exists as KG node)
  - 1-hop UMLS hierarchy bridge (patient CUI -> KG node via SY/PAR/CHD/RB/RN/RT)
  - 2-hop bridge
  - Unreachable (no UMLS path within 2 hops)

This tells us which lever (multi-candidate linking, multi-SAB, SY-expansion, lay-corpus)
has the most headroom WITHOUT touching benchmark-specific information.
"""
from __future__ import annotations
import sys, json, pickle, time
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_V2 = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"


def main():
    print("=" * 70)
    print("only-KG vocabulary gap diagnostic")
    print("=" * 70)

    print("\n[1] Loading v2 graph...")
    with GRAPH_V2.open("rb") as f:
        G = pickle.load(f)
    pheno_cuis = {n for n, a in G.nodes(data=True) if a.get("ntype") == "Phenotype"}
    print(f"    KG Phenotype nodes: {len(pheno_cuis):,}")

    print("\n[2] Loading patient evidence CUIs...")
    ev_cuis = json.load(open(EVIDENCE_CUI))
    patient_cuis = set()
    for cs in ev_cuis.values():
        patient_cuis.update(cs)
    print(f"    Unique patient evidence CUIs: {len(patient_cuis):,}")

    print("\n[3] Direct overlap analysis...")
    direct_in = patient_cuis & pheno_cuis
    not_in = patient_cuis - pheno_cuis
    print(f"    Patient CUI in KG (direct node): {len(direct_in):,}  ({100*len(direct_in)/len(patient_cuis):.1f}%)")
    print(f"    Patient CUI NOT in KG:           {len(not_in):,}  ({100*len(not_in)/len(patient_cuis):.1f}%)")

    print("\n[4] 1-hop hierarchy bridge analysis (HIERARCHY edges in v2)...")
    # For each missing patient CUI, check if it has any HIERARCHY neighbor in KG
    bridged_1hop = set()
    bridged_via = defaultdict(set)  # patient_cui -> set of KG node CUIs it can bridge to
    for pc in not_in:
        if pc not in G:
            continue
        for _, nb, edata in G.out_edges(pc, data=True):
            if edata.get("etype") == "HIERARCHY" and nb in pheno_cuis:
                bridged_1hop.add(pc); bridged_via[pc].add(nb)
        for nb, _, edata in G.in_edges(pc, data=True):
            if edata.get("etype") == "HIERARCHY" and nb in pheno_cuis:
                bridged_1hop.add(pc); bridged_via[pc].add(nb)
    print(f"    Bridged via 1-hop HIERARCHY: {len(bridged_1hop):,} of {len(not_in):,} missing ({100*len(bridged_1hop)/max(len(not_in),1):.1f}%)")

    print("\n[5] Total reachable (direct + 1-hop bridge):")
    reachable = direct_in | bridged_1hop
    unreachable = patient_cuis - reachable
    print(f"    Direct or 1-hop bridged: {len(reachable):,} ({100*len(reachable)/len(patient_cuis):.1f}%)")
    print(f"    Still unreachable:       {len(unreachable):,} ({100*len(unreachable)/len(patient_cuis):.1f}%)")

    print("\n[6] Per-evidence breakdown (impact analysis)...")
    # For each DDXPlus evidence question, fraction of its CUIs that are reachable
    ev_reachable_frac = {}
    for ev, cs in ev_cuis.items():
        if not cs: ev_reachable_frac[ev] = None; continue
        r = sum(1 for c in cs if c in reachable) / len(cs)
        ev_reachable_frac[ev] = r
    unreach_evs = [e for e, r in ev_reachable_frac.items() if r is not None and r < 0.5]
    fully_unreach = [e for e, r in ev_reachable_frac.items() if r is not None and r == 0]
    print(f"    Total evidences (questions): {len(ev_cuis)}")
    print(f"    Evidences with reachable rate <50%: {len(unreach_evs)}")
    print(f"    Evidences with 0% reachable:        {len(fully_unreach)}")

    print("\n[7] Sample fully-unreachable evidences (top 20):")
    for ev in fully_unreach[:20]:
        cs = ev_cuis.get(ev, [])
        print(f"    {ev}  CUIs={cs}")

    print("\n[8] Headroom estimate (research-safe levers):")
    print(f"    Current coverage:           {100*len(direct_in)/len(patient_cuis):.1f}%")
    print(f"    With UMLS hierarchy (v2):   {100*len(reachable)/len(patient_cuis):.1f}%")
    print(f"    Remaining gap (unreachable): {100*len(unreachable)/len(patient_cuis):.1f}%")
    print(f"")
    print(f"    Interpretation:")
    print(f"      - If unreachable < 30%: UMLS-structural levers (lever 1,2,4) sufficient")
    print(f"      - If unreachable > 50%: need universal lay-corpus expansion (NEW lever)")

    # Save diagnostic
    out = {
        "patient_cuis_total": len(patient_cuis),
        "kg_pheno_nodes": len(pheno_cuis),
        "direct_overlap": len(direct_in),
        "direct_overlap_pct": 100*len(direct_in)/len(patient_cuis),
        "bridged_1hop": len(bridged_1hop),
        "total_reachable": len(reachable),
        "total_reachable_pct": 100*len(reachable)/len(patient_cuis),
        "unreachable_pct": 100*len(unreachable)/len(patient_cuis),
        "fully_unreachable_evidences": fully_unreach,
        "low_reachable_evidences": [e for e, r in ev_reachable_frac.items() if r is not None and r < 0.3],
    }
    out_path = MEDKG_ROOT / "kg" / "onlykg_gap_diagnostic.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n    Saved diagnostic to {out_path}")


if __name__ == "__main__":
    main()
