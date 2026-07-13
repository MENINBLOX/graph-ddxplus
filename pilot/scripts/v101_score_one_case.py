#!/usr/bin/env python3
"""v101 attribute-aware scoring on seed=42 anaphylaxis patient.

평가 방법론:
- Patient evidence → JSON 구조 (DDXPlus token → phenotype + attributes)
- Disease profile JSON (v101 IE output)
- 각 phenotype 매칭 시 attribute alignment 계산
- 단순 yes/no가 아닌 attribute-weighted similarity
"""
from __future__ import annotations
import json
from collections import defaultdict


def normalize_name(s):
    """Light normalization for phenotype/disease name matching."""
    s = s.lower().strip()
    s = s.replace("-", " ")
    return s


def phen_similarity(name1, name2):
    """Crude phenotype name similarity (no UMLS for prototype)."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if n1 == n2: return 1.0
    if n1 in n2 or n2 in n1: return 0.8
    # Check key medical synonyms
    syn_groups = [
        {"edema","swelling","angioedema"},
        {"hives","urticaria","rash"},
        {"pruritus","itching","itchy"},
        {"dyspnea","shortness of breath","difficulty breathing","breathlessness","sob"},
        {"stridor","high pitched breathing"},
        {"wheezing","wheeze","bronchospasm"},
        {"pain","ache","abdominal cramp","cramp"},
        {"nausea","feeling nauseous"},
        {"vomiting","throwing up","emesis"},
        {"throat swelling","airway swelling","laryngeal edema"},
        {"hypotension","low blood pressure","dizziness"},
        {"hypersensitivity","allergic","atopy","atopic"},
    ]
    for g in syn_groups:
        n1_in = any(p in n1 for p in g) or n1 in g
        n2_in = any(p in n2 for p in g) or n2 in g
        if n1_in and n2_in: return 0.9
    # Word overlap
    w1 = set(n1.split()); w2 = set(n2.split())
    if w1 & w2:
        return 0.5 * len(w1 & w2) / max(len(w1), len(w2))
    return 0.0


def attr_alignment(patient_attrs, profile_attrs):
    """Compare attributes between patient evidence and disease profile.
    Returns score in [0, 1] = 1.0 if all known attrs align, 0.5 if missing, 0.0 if mismatch."""
    if not profile_attrs and not patient_attrs:
        return 0.5  # neutral
    scores = []
    # Location
    pat_loc = set(patient_attrs.get("location", []))
    prof_loc = set(profile_attrs.get("location", []))
    if pat_loc and prof_loc:
        # Anatomy overlap (Jaccard) + nearby-region partial credit
        inter = pat_loc & prof_loc
        if inter:
            scores.append(1.0)
        else:
            # Partial credit for adjacent regions
            nearby_groups = [
                {"face","cheek","lip","tongue","mouth","throat","larynx","head","eye","eyelid","ear","nose"},
                {"leg","thigh","knee","ankle","foot"},
                {"arm","shoulder","elbow","wrist","hand"},
                {"chest","lung","heart","back"},
                {"abdomen","epigastric","liver","kidney","pelvis"},
                {"skin","generalized"},
            ]
            partial = 0
            for g in nearby_groups:
                if (pat_loc & g) and (prof_loc & g):
                    partial = 0.5; break
            scores.append(partial)
    elif not pat_loc and not prof_loc:
        scores.append(0.5)  # neutral
    else:
        scores.append(0.3)  # one side missing
    # Severity
    pat_sev = patient_attrs.get("severity") or patient_attrs.get("severity_scale")
    prof_sev = profile_attrs.get("severity")
    if pat_sev and prof_sev:
        # Map numeric to category
        if isinstance(pat_sev, (int, float)):
            pat_cat = ("mild" if pat_sev <= 3 else
                       "moderate" if pat_sev <= 6 else
                       "severe" if pat_sev <= 9 else "critical")
        else: pat_cat = pat_sev
        scores.append(1.0 if pat_cat == prof_sev else 0.5)
    # Onset
    pat_on = patient_attrs.get("onset")
    prof_on = profile_attrs.get("onset")
    if pat_on and prof_on:
        scores.append(1.0 if pat_on == prof_on else 0.3)
    return sum(scores)/len(scores) if scores else 0.5


def patient_seed42():
    """Encode seed=42 anaphylaxis patient as JSON-style evidence."""
    return {
        "demographics": {"age": 62, "sex": "F"},
        "evidences": [
            {"name": "Edema", "attributes": {"location": ["cheek","face"]}},
            {"name": "Pruritus", "attributes": {"severity_scale": 9}},
            {"name": "Pain", "attributes": {
                "location": ["abdomen","epigastric","groin"],
                "character": ["cramping","sharp"],
                "severity_scale": 4,
                "onset_scale": 6
            }},
            {"name": "Dyspnea", "attributes": {}},
            {"name": "Stridor", "attributes": {"location": ["throat","larynx"]}},
            {"name": "Wheezing", "attributes": {"location": ["lung","chest"]}},
            {"name": "Skin Lesion", "attributes": {
                "location": ["neck","arm","mouth"],
                "character": ["raised","large"]
            }},
            {"name": "Pruritus", "attributes": {"severity_scale": 9}},
            {"name": "Nausea", "attributes": {}},
            {"name": "Vomiting", "attributes": {}},
            {"name": "Hypersensitivity", "attributes": {"onset": "chronic"}},
        ]
    }


def score_disease(patient, disease_record):
    """Compute attribute-aware score for one disease."""
    profile_phens = disease_record.get("phenotypes", [])
    if not profile_phens: return 0.0, []
    total = 0.0
    matched = []
    for pat_ev in patient["evidences"]:
        best_match = 0.0
        best_prof_phen = None
        for prof_phen in profile_phens:
            sim = phen_similarity(pat_ev["name"], prof_phen.get("name",""))
            if sim < 0.3: continue
            attr_score = attr_alignment(pat_ev.get("attributes",{}),
                                        prof_phen.get("attributes",{}))
            freq = prof_phen.get("attributes",{}).get("frequency", 0.5)
            combined = sim * (0.5 + 0.5*attr_score) * freq
            if combined > best_match:
                best_match = combined
                best_prof_phen = prof_phen
        if best_match > 0:
            total += best_match
            matched.append((pat_ev["name"],
                            best_prof_phen.get("name") if best_prof_phen else "",
                            round(best_match, 3)))
    # Normalize by # patient evidence (so disease with more phenotypes doesn't unfairly win)
    return total / len(patient["evidences"]), matched


def main():
    data = json.load(open("pilot/data/cache/v101_recursive_ie.json"))
    patient = patient_seed42()

    # Merge all diseases from all queries
    all_diseases = {}  # disease_name → merged phenotype list
    for q_info in data["by_query"].values():
        if "diseases" not in q_info: continue
        for d in q_info["diseases"]:
            dn = d.get("disease","?")
            if dn not in all_diseases:
                all_diseases[dn] = {"disease": dn, "phenotypes": []}
            all_diseases[dn]["phenotypes"].extend(d.get("phenotypes",[]))

    print(f"=== Scoring seed=42 patient against {len(all_diseases)} v101 diseases ===\n")
    scores = []
    for dn, dr in all_diseases.items():
        s, matched = score_disease(patient, dr)
        scores.append((s, dn, matched))
    scores.sort(reverse=True)

    print(f"--- Top 10 by attribute-aware score ---")
    for i, (s, dn, matched) in enumerate(scores[:10]):
        is_truth = "naphyl" in dn.lower() or "allergic" in dn.lower()
        mark = " ⭐ ANAPHYLAXIS-like" if is_truth else ""
        print(f"  [{i+1}] {dn}: {s:.4f}{mark}")
        for pat_name, prof_name, sc in matched[:3]:
            print(f"      {pat_name} ↔ {prof_name}: {sc}")
        print()

    # Compare: did anaphylaxis (or variants) win?
    anaph_ranks = [(i+1, dn, s) for i, (s, dn, _) in enumerate(scores)
                   if "naphyl" in dn.lower() or "allergic" in dn.lower()]
    print(f"\n=== Anaphylaxis-like rankings ===")
    for r, dn, s in anaph_ranks[:5]:
        print(f"  Rank {r}: {dn} (score {s:.4f})")


if __name__ == "__main__":
    main()
