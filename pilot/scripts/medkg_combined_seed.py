#!/usr/bin/env python3
"""Combine benchmark disease seeds + focused UMLS DISO subset → final seed list.

Output: $MEDKG_ROOT/seeds/combined_seed.jsonl
Each entry: {"cui": ..., "name": ..., "sources": [list of seed sources]}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, DDXPLUS_DIR, SYMCAT_DIR, RAREBENCH_DIR

SEEDS_DIR = MEDKG_ROOT / "seeds"
SEEDS_DIR.mkdir(parents=True, exist_ok=True)
OUT = SEEDS_DIR / "combined_seed.jsonl"


def load_focused():
    p = SEEDS_DIR / "umls_diso_focused.jsonl"
    if not p.exists(): return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out


def load_ddxplus():
    p = DDXPLUS_DIR / "disease_umls_mapping.json"
    if not p.exists(): return []
    d = json.load(p.open())
    out = []
    for k, v in d.get("mapping", {}).items():
        cui = v.get("umls_cui")
        name = v.get("name_en") or v.get("disease_key") or k
        if cui:
            out.append({"cui": cui, "name": name, "source": "ddxplus"})
    return out


def load_symcat():
    p = SYMCAT_DIR / "disease_umls_mapping.json"
    if not p.exists(): return []
    d = json.load(p.open())
    out = []
    for k, v in d.get("mapping", {}).items():
        cui = v.get("umls_cui")
        name = v.get("umls_name") or k
        if cui:
            out.append({"cui": cui, "name": name, "source": "symcat"})
    return out


def load_rarebench():
    p = RAREBENCH_DIR / "disease_umls_mapping.json"
    if not p.exists(): return []
    d = json.load(p.open())
    out = []
    for k, v in d.get("mapping", {}).items():
        cui = v.get("umls_cui")
        name = v.get("disease_name") or v.get("umls_name") or k
        if cui:
            out.append({"cui": cui, "name": name, "source": "rarebench"})
    return out


def main():
    focused = load_focused()
    ddxplus = load_ddxplus()
    symcat = load_symcat()
    rarebench = load_rarebench()
    print(f"Focused DISO: {len(focused):,}")
    print(f"DDXPlus:      {len(ddxplus):,}")
    print(f"SymCat:       {len(symcat):,}")
    print(f"RareBench:    {len(rarebench):,}")

    # Combine, dedup by CUI
    combined = {}  # cui → entry
    for entry in focused:
        cui = entry["cui"]
        combined[cui] = {
            "cui": cui,
            "name": entry["name"],
            "sources": ["umls_diso_focused"],
            "sabs": entry.get("sabs", []),
        }
    for src_list in [ddxplus, symcat, rarebench]:
        for entry in src_list:
            cui = entry["cui"]
            if cui in combined:
                if entry["source"] not in combined[cui]["sources"]:
                    combined[cui]["sources"].append(entry["source"])
            else:
                combined[cui] = {
                    "cui": cui,
                    "name": entry["name"],
                    "sources": [entry["source"]],
                    "sabs": [],
                }

    print(f"\nUnique combined: {len(combined):,}")
    # Source-overlap stats
    n_only_focused = sum(1 for e in combined.values() if e["sources"] == ["umls_diso_focused"])
    n_only_bench = sum(1 for e in combined.values() if "umls_diso_focused" not in e["sources"])
    n_overlap = sum(1 for e in combined.values()
                    if "umls_diso_focused" in e["sources"] and len(e["sources"]) > 1)
    print(f"  Only focused DISO:        {n_only_focused:,}")
    print(f"  Only benchmark:           {n_only_bench:,}  (not in focused DISO — expand seed)")
    print(f"  In both:                  {n_overlap:,}")

    with OUT.open("w") as out:
        for e in sorted(combined.values(), key=lambda x: x["cui"]):
            out.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
