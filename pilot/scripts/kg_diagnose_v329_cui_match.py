#!/usr/bin/env python3
"""v329: CUI-based matching evaluation.

Evaluation pipeline:
  1. Pre-extract CUIs from each unique DDXPlus evidence (223 questions) using scispaCy + UMLS linker
  2. For each test patient, build patient_cuis = ∪ CUIs(evidences) — fast lookup
  3. For each disease, score = |patient_cuis ∩ disease_cuis| (from disease_kg_cuis.json)
  4. Pick top-1 disease per patient
  5. Evaluate against gold

This bypasses LLM stage 1 entirely — purely CUI-based matching.
Establishes whether CUI matching alone is competitive with LLM+KG (v327 = 68.45%).
"""
from __future__ import annotations
import sys, json, csv, ast, time, warnings
from pathlib import Path
from collections import defaultdict, Counter
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT


def main():
    print("Loading scispaCy...")
    import spacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls", "k": 1, "threshold": 0.85
    })

    # Load DDXPlus evidence questions
    print("Loading DDXPlus evidence questions...")
    with open("data/ddxplus/release_evidences.json") as f: ev_info = json.load(f)
    print(f"  {len(ev_info):,} unique evidences")

    # Pre-extract CUIs per evidence
    print("Extracting CUIs per evidence (one-time)...")
    t0 = time.time()
    ev_to_cuis = {}
    for eid, info in ev_info.items():
        q = info.get("question_en", "")
        if not q:
            ev_to_cuis[eid] = []
            continue
        # Append value meanings (multi-value evidences)
        vm = info.get("value_meaning", {})
        full_text = q
        if isinstance(vm, dict):
            for v_dict in vm.values():
                if isinstance(v_dict, dict):
                    en_val = v_dict.get("en", "")
                    if en_val: full_text += " " + en_val
        doc = nlp(full_text)
        cuis = set()
        for ent in doc.ents:
            if ent._.kb_ents:
                cuis.add(ent._.kb_ents[0][0])
        ev_to_cuis[eid] = sorted(cuis)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Save evidence CUI map
    Path("/mnt/medkg/kg").mkdir(exist_ok=True)
    with open("/mnt/medkg/kg/ddxplus_evidence_cuis.json", "w") as f:
        json.dump(ev_to_cuis, f)

    # Load KG CUIs
    print("Loading KG CUIs...")
    kg_cuis = json.load(open("/mnt/medkg/kg/disease_kg_cuis.json"))
    kg_cuis = {k: set(v) for k, v in kg_cuis.items()}
    print(f"  {len(kg_cuis):,} diseases")

    # DDXPlus mapping
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    # Eval on test
    print("Evaluating on 30K test...")
    N = 30000
    correct_at1 = correct_at3 = correct_at5 = correct_at10 = 0
    total = 0
    fail_per_disease = Counter()
    total_per_disease = Counter()

    t1 = time.time()
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if total >= N: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            # Build patient CUIs
            patient_cuis = set()
            for ev in evs:
                base = ev.split("_@_")[0]
                if base in ev_to_cuis:
                    patient_cuis.update(ev_to_cuis[base])
            # Score each disease
            scores = []
            for dc in dcs_list:
                kg = kg_cuis.get(dc, set())
                score = len(patient_cuis & kg)
                scores.append((dc, score))
            scores.sort(key=lambda x: -x[1])
            ranked = [dc for dc, s in scores]
            true_name = next((dn for dn,info in cond.items() if dn in icd_map and icd_map[dn]['cui']==true_cui), "?")
            total += 1
            total_per_disease[true_name] += 1
            if ranked[0] == true_cui: correct_at1 += 1
            else: fail_per_disease[true_name] += 1
            if true_cui in ranked[:3]: correct_at3 += 1
            if true_cui in ranked[:5]: correct_at5 += 1
            if true_cui in ranked[:10]: correct_at10 += 1
            if total % 5000 == 0:
                print(f"  {total}/{N}: @1={100*correct_at1/total:.2f}% ({time.time()-t1:.0f}s)")

    print(f"\nv329 CUI-match GTPA@1 = {100*correct_at1/total:.2f}%")
    print(f"  @3 = {100*correct_at3/total:.2f}%")
    print(f"  @5 = {100*correct_at5/total:.2f}%")
    print(f"  @10 = {100*correct_at10/total:.2f}%")

    # Per-disease analysis
    print("\nTop failures (per-disease):")
    for d, fc_ in fail_per_disease.most_common(15):
        tot = total_per_disease[d]
        print(f"  {d:30s}  {fc_:5d}/{tot:5d} fail ({100*fc_/tot:.1f}%)")


if __name__ == "__main__":
    main()
