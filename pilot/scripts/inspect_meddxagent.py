#!/usr/bin/env python3
"""Inspector for MEDDxAgent (ACL 2025) bundled data.

MEDDxAgent ships disease/phenotype MAPPING files for iCRAFT-MD, RareBench,
DDXPlus, but pulls the actual patient cases from HuggingFace at runtime
(see ddxdriver/benchmarks/rarebench.py). The only locally bundled patient
case file is iCRAFT-MD (all_craft_md.jsonl).

Reports per-resource stats so we know what is and is not actually downloaded.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

BASE = Path("/home/max/Graph-DDXPlus/data/external_benchmarks/meddxagent/"
            "ddxdriver/benchmarks/data")


def inspect_icraftmd() -> None:
    print("\n=== MEDDxAgent :: iCRAFT-MD (bundled) ===")
    p = BASE / "icraftmd" / "all_craft_md.jsonl"
    if not p.exists():
        print(f"  MISSING: {p}"); return
    records = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    diseases = Counter(r["answer"] for r in records)
    print(f"  Patients (cases)         : {len(records)}")
    print(f"  Unique diseases (answer) : {len(diseases)}")
    print(f"  Evidence format          : free-text vignette + extracted facts")
    print(f"  Sample disease counts    : {dict(diseases.most_common(5))}")
    for i, r in enumerate(records[:3]):
        ans = r["answer"]
        facts = " | ".join(r.get("facts", [])[:3])
        print(f"  Sample {i}: dx={ans!r}; first_facts={facts[:150]}...")


def inspect_rarebench_mapping() -> None:
    print("\n=== MEDDxAgent :: RareBench mappings (patient cases NOT bundled) ===")
    rb = BASE / "rarebench"
    dm = json.load(open(rb / "disease_mapping.json"))
    rdm = json.load(open(rb / "rarebench_disease_mapping.json"))
    phm = json.load(open(rb / "rarebench_phenotype_mapping.json"))
    opts = json.load(open(rb / "diagnosis_options.json"))
    print(f"  Disease subset mapping (per ds) : {sorted(dm.keys())}")
    print(f"  RareBench->canonical disease    : {len(rdm)} ORPHA/OMIM IDs mapped")
    print(f"  Phenotype mapping records       : {len(phm)} entries "
          f"(sample: {list(phm.items())[:1]})")
    print(f"  Diagnosis options per benchmark : "
          f"{ {k: len(v) for k,v in opts.items()} }")
    print(f"  NOTE: actual patient cases pulled from HuggingFace at runtime")


def inspect_ddxplus() -> None:
    print("\n=== MEDDxAgent :: DDXPlus (only fewshot embeddings + dx options) ===")
    dx = BASE / "ddxplus" / "diagnosis_options.txt"
    fs = BASE / "ddxplus" / "fewshot_embeddings.json"
    if dx.exists():
        lines = [l for l in dx.read_text().splitlines() if l.strip()]
        print(f"  Diagnosis options : {len(lines)} (sample: {lines[:3]})")
    if fs.exists():
        d = json.load(open(fs))
        if isinstance(d, list):
            print(f"  Fewshot embeddings (list of [vec, meta]): {len(d)} examples")
            if d and isinstance(d[0], list) and len(d[0]) > 1:
                meta = d[0][1] if isinstance(d[0][1], dict) else None
                if meta:
                    print(f"  Sample meta keys: {list(meta.keys())}")
        else:
            print(f"  Fewshot embeddings type: {type(d).__name__}")
        print(f"  NOTE: full DDXPlus pulled from official source at runtime")


def main() -> None:
    inspect_icraftmd()
    inspect_rarebench_mapping()
    inspect_ddxplus()


if __name__ == "__main__":
    main()
