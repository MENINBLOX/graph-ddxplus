#!/usr/bin/env python3
"""v103 attribute-aware scoring on seed=42 anaphylaxis patient.

비교: v85/v95_full (CUI+IDF cosine) vs v103 (phenotype name + attribute alignment).
"""
from __future__ import annotations
import sys, csv, ast, json, random, pickle, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT


# Phenotype name synonym groups (for matching patient evidence to KG phenotype)
SYN_GROUPS = [
    {"edema","swelling","angioedema","oedema"},
    {"urticaria","hives","rash","exanthem","exanthema","skin rash","skin lesion","lesion"},
    {"pruritus","itching","itchy","pruritis"},
    {"pain","ache","cramping","cramp","abdominal cramping","abdominal pain","abdominal cramps"},
    {"nausea","feeling nauseous","nauseous"},
    {"vomiting","emesis","throwing up"},
    {"dyspnea","shortness of breath","difficulty breathing","breathlessness","sob","respiratory difficulty","respiratory distress"},
    {"wheezing","wheeze","bronchospasm","wheezing sound"},
    {"stridor","high-pitched breathing","stridorous breathing","airway obstruction"},
    {"hypotension","low blood pressure"},
    {"dizziness","lightheadedness","dizzy"},
    {"hypersensitivity","allergic","allergy","atopy","atopic"},
    {"fever","pyrexia","febrile"},
    {"cough","coughing"},
]


def name_sim(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b: return 1.0
    if a in b or b in a: return 0.85
    # Check synonyms
    for g in SYN_GROUPS:
        in_a = any(s in a or a in s for s in g)
        in_b = any(s in b or b in s for s in g)
        if in_a and in_b: return 0.9
    # Word overlap
    wa, wb = set(a.split()), set(b.split())
    if wa & wb:
        return 0.5 * len(wa & wb) / max(len(wa), len(wb))
    return 0.0


# Loose anatomy adjacency for partial credit
NEARBY_ANATOMY = [
    {"face","cheek","lip","tongue","mouth","throat","larynx","head","eye","eyelid","ear","nose"},
    {"leg","thigh","knee","ankle","foot"},
    {"arm","shoulder","elbow","wrist","hand"},
    {"chest","lung","heart","back"},
    {"abdomen","epigastric","liver","kidney","pelvis","groin"},
    {"skin","generalized","systemic"},
]


def location_align(pat_locs, prof_loc_dist):
    """Jaccard on location values, plus nearby-region partial credit."""
    if not pat_locs or not prof_loc_dist: return 0.5  # neutral
    pat_set = set(pat_locs)
    prof_set = set(prof_loc_dist.keys())
    if pat_set & prof_set:
        # Direct match — sum of profile probabilities for patient locations
        return sum(prof_loc_dist.get(loc, 0) for loc in pat_set)
    # Partial credit for nearby
    for g in NEARBY_ANATOMY:
        if (pat_set & g) and (prof_set & g):
            return 0.4
    return 0.0


def attr_align(pat_attrs, prof_attrs):
    """Combine alignment scores per attribute."""
    scores = []
    # Location
    pat_loc = pat_attrs.get("location", [])
    prof_loc = prof_attrs.get("location_dist", {})
    if pat_loc or prof_loc:
        scores.append(location_align(pat_loc, prof_loc))
    # Severity (exact match in distribution)
    pat_sev = pat_attrs.get("severity")
    prof_sev = prof_attrs.get("severity_dist", {})
    if pat_sev and prof_sev:
        scores.append(prof_sev.get(pat_sev, 0))
    # Onset
    pat_on = pat_attrs.get("onset_pace")
    prof_on = prof_attrs.get("onset_dist", {})
    if pat_on and prof_on:
        scores.append(prof_on.get(pat_on, 0))
    # Character
    pat_char = pat_attrs.get("character", [])
    prof_char = prof_attrs.get("character_dist", {})
    if pat_char and prof_char:
        scores.append(sum(prof_char.get(c, 0) for c in pat_char))
    return sum(scores)/len(scores) if scores else 0.5  # neutral if no comparable attrs


def patient_seed42_json():
    """seed=42 anaphylaxis patient → JSON-style evidence with attributes."""
    return [
        {"name": "edema", "attributes": {"location": ["cheek","face"]}},
        {"name": "pruritus", "attributes": {"severity": "severe"}},
        {"name": "pain", "attributes": {
            "location": ["abdomen","epigastric","groin"],
            "character": ["cramping","sharp"],
            "severity": "mild",
            "onset_pace": "rapid"
        }},
        {"name": "dyspnea", "attributes": {}},
        {"name": "stridor", "attributes": {"location": ["throat","larynx"]}},
        {"name": "wheezing", "attributes": {"location": ["lung","chest"]}},
        {"name": "skin lesion", "attributes": {"location": ["neck","arm","mouth"]}},
        {"name": "nausea", "attributes": {}},
        {"name": "vomiting", "attributes": {}},
        {"name": "hypersensitivity", "attributes": {}},
    ]


def score_disease(patient, disease_cui, G):
    """Sum over patient evidences: phen_name match × attribute alignment × frequency."""
    if disease_cui not in G: return 0
    total = 0
    matched = []
    for pat_ev in patient:
        best_score = 0
        best_prof_name = None
        for _, prof_phen, edge_attrs in G.out_edges(disease_cui, data=True):
            sim = name_sim(pat_ev["name"], prof_phen)
            if sim < 0.3: continue
            attr_a = attr_align(pat_ev["attributes"], edge_attrs)
            freq = edge_attrs["frequency"]
            combined = sim * (0.5 + 0.5*attr_a) * freq
            if combined > best_score:
                best_score = combined
                best_prof_name = prof_phen
        if best_score > 0:
            total += best_score
            matched.append((pat_ev["name"], best_prof_name, round(best_score,3)))
    return total / len(patient), matched


def main():
    G = pickle.load(open("pilot/data/onlykg_graph_v103.pkl", "rb"))
    cui2name = {cui: G.nodes[cui].get("name","?")
                for cui in G.nodes if G.nodes[cui].get("ntype")=="disease"}

    patient = patient_seed42_json()
    truth_cui = "C0685898"  # Anaphylaxis

    print(f"=== seed=42 patient — v103 attribute-aware scoring ===\n")
    print(f"Truth: {cui2name.get(truth_cui,'?')} ({truth_cui})\n")

    # Score all diseases
    scores = []
    for d_cui in cui2name:
        s, matched = score_disease(patient, d_cui, G)
        scores.append((s, d_cui, matched))
    scores.sort(reverse=True)

    print("--- Top 10 by v103 attribute-aware score ---")
    rank_truth = None
    for i, (s, d_cui, matched) in enumerate(scores[:10]):
        mark = " ⭐ TRUTH" if d_cui == truth_cui else ""
        print(f"  [{i+1}] {cui2name[d_cui]:<40} ({d_cui}): {s:.4f}{mark}")
        for pn, prof, sc in matched[:3]:
            print(f"      {pn} ↔ {prof}: {sc}")
        print()

    # Truth rank
    for i, (s, d_cui, _) in enumerate(scores):
        if d_cui == truth_cui:
            rank_truth = i+1; break
    print(f"\n=== Truth (Anaphylaxis) rank: {rank_truth} / {len(scores)} ===")
    print(f"\nBaseline 비교: v95_full=1, v101=5, v103=?")


if __name__ == "__main__":
    main()
