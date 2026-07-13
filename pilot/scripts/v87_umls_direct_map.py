#!/usr/bin/env python3
"""v87 — Direct UMLS string mapping (bypasses scispaCy NER fragmentation).

scispaCy NER splits "Lip swelling" → "Lip" + "swelling" → 2 generic CUIs.
v87: load all UMLS phenotype CUI strings (from MRCONSO, S* semantic type),
then for each LLM phenotype text, do exact/substring match against
UMLS string table → direct CUI assignment.

학술적 정당성: UMLS는 standard ontology, all-CUI string table은 표준 자원.
SymCat CUI list 자체를 KG에 inject하는 게 아니라 UMLS 전체 phenotype CUI
strings을 사용 (벤치마크 외).

CLAUDE.md 원칙:
- 1: KG construction시 vocabulary 검증 후 진행 (이 단계가 그 자체).
- 3: 벤치마크 데이터 사용 X. UMLS는 표준 ontology.
- 5: SymCat의 specific symptom list 사용 X.
"""
from __future__ import annotations
import json, re, argparse
from pathlib import Path
from collections import defaultdict


def load_umls_phenotype_strings(mrconso_path, mrsty_path=None):
    """Load all UMLS strings for phenotype-related semantic types.
    Returns: dict[normalized_string] -> set[CUI]
    """
    # Phenotype-related TUI: Finding, Sign or Symptom, etc.
    # T033 Finding, T184 Sign or Symptom, T046 Pathologic Function, T037 Injury or Poisoning
    # T047 Disease or Syndrome (allowed but separate)
    phen_tuis = {"T033", "T184", "T046", "T037", "T048", "T049", "T191", "T190"}
    # Load CUIs with phen TUI
    if mrsty_path:
        phen_cuis = set()
        with open(mrsty_path) as f:
            for line in f:
                parts = line.split("|")
                if len(parts) >= 2 and parts[1] in phen_tuis:
                    phen_cuis.add(parts[0])
        print(f"  Phenotype CUIs (from MRSTY): {len(phen_cuis):,}", flush=True)
    else:
        phen_cuis = None  # accept all

    print(f"  Scanning MRCONSO for string → CUI...", flush=True)
    str2cuis = defaultdict(set)
    n = 0
    with open(mrconso_path) as f:
        for line in f:
            n += 1
            parts = line.split("|")
            if len(parts) < 15: continue
            c, lang = parts[0], parts[1]
            if lang != "ENG": continue
            if phen_cuis is not None and c not in phen_cuis: continue
            s = parts[14].strip().lower()
            if not s or len(s) < 3 or len(s) > 100: continue
            # Normalize
            s = re.sub(r"\s+", " ", s)
            str2cuis[s].add(c)
            if n % 2_000_000 == 0:
                print(f"    scanned {n//1_000_000}M lines, {len(str2cuis):,} strings", flush=True)
    print(f"  Done: {len(str2cuis):,} unique phenotype strings", flush=True)
    return str2cuis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mrconso", default="/windows/data/umls_subset/MRCONSO.RRF")
    ap.add_argument("--mrsty", default="/windows/data/umls_subset/MRSTY.RRF")
    ap.add_argument("--fallback_scispacy", default=None,
                    help="optional path to v85_cui_edges.jsonl for fallback when no direct match")
    args = ap.parse_args()

    print("Loading UMLS phenotype strings...", flush=True)
    mrsty = args.mrsty if Path(args.mrsty).exists() else None
    str2cuis = load_umls_phenotype_strings(args.mrconso, mrsty)

    # Load IE records
    records = []
    with open(args.ie_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"IE records: {len(records)}", flush=True)

    # disease name → cui mapping
    name_to_cui = {}
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    for dn, info in icd.items():
        if info.get("cui"): name_to_cui[dn] = info["cui"]
    sym_dis = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    for dn, info in sym_dis.items():
        if info.get("umls_cui") and dn not in name_to_cui:
            name_to_cui[dn] = info["umls_cui"]

    # Fallback (scispaCy) — load if provided
    fallback_map = {}
    if args.fallback_scispacy and Path(args.fallback_scispacy).exists():
        # Reload IE phrase → CUI mapping by inverting scispaCy edges
        # But scispaCy edges file has per-disease edges; rebuild phrase→CUI from raw IE
        # Actually we'll skip fallback for now (scispaCy already tried), use only direct match
        pass

    # Map each LLM phen → CUI(s)
    n_total = 0; n_mapped = 0
    n_records = 0; n_edges = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout:
        for r in records:
            dn = r["disease"]
            dcui = name_to_cui.get(dn)
            if not dcui: continue
            edges = {}
            for phen_text, info in r["phenotypes"].items():
                n_total += 1
                phen_low = phen_text.lower().strip()
                phen_low = re.sub(r"\s+", " ", phen_low)
                # Strategy: exact > prefix > substring match
                cands = set()
                # Exact match
                if phen_low in str2cuis:
                    cands = str2cuis[phen_low]
                else:
                    # Strip leading qualifiers
                    cleaned = re.sub(r"^(severe|mild|moderate|acute|chronic|sudden|recurrent)\s+", "", phen_low)
                    if cleaned != phen_low and cleaned in str2cuis:
                        cands = str2cuis[cleaned]
                    # Strip parenthetical
                    cleaned2 = re.sub(r"\s*\([^)]*\)\s*", "", phen_low).strip()
                    if not cands and cleaned2 in str2cuis:
                        cands = str2cuis[cleaned2]
                if not cands: continue
                # If multiple CUIs match, take alphabetically smallest (deterministic)
                cui = sorted(cands)[0]
                if cui == dcui: continue
                p = info.get("prob", 0.0)
                # If duplicate, take max
                if cui in edges:
                    edges[cui] = max(edges[cui], p)
                else:
                    edges[cui] = p
                n_mapped += 1
            if not edges: continue
            fout.write(json.dumps({
                "disease": dn, "dcui": dcui, "source": r.get("source",""),
                "edges": edges
            }) + "\n")
            n_records += 1
            n_edges += len(edges)
    print(f"\nMapped {n_mapped}/{n_total} phen texts ({100*n_mapped/n_total:.1f}%)")
    print(f"Saved {n_records} disease records, {n_edges} edges → {args.out}")


if __name__ == "__main__":
    main()
