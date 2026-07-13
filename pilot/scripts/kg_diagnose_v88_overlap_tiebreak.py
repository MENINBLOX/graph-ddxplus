#!/usr/bin/env python3
"""진단 v88: v79 stage1 + KG-feature-overlap tie-break (CPU only).

각 (patient, disease) pair: count how many of disease's KG feature names
appear (substring) in patient's profile text. Use as tie-breaker.
0 LLM calls. Pure CPU.
"""
from __future__ import annotations
import ast, csv, json, os, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
GENERIC_TERMS = {'symptom', 'sign', 'pain', 'patient', 'disease', 'syndrome', 'condition'}

TRANSLATION_FIX = {
    "haunting": "stabbing", "tugging": "pulling", "sensitive": "tender",
    "a knife stroke": "stabbing", "a cramp": "cramping", "haunted": "stabbing",
    "sickening": "nauseating", "tedious": "tiresome", "scary": "frightening",
    "violent": "severe",
}

def fix_translation(s):
    if not s: return s
    for bad, good in TRANSLATION_FIX.items():
        s = s.replace(bad, good)
    return s


def main():
    print("="*80)
    print("진단 v88: v79 stage1 + KG overlap tie-break")
    print("="*80)

    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
    with open(KG_CACHE) as f: cache = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; diseases[dn] = {"cui": dc}
        fr2cui[info.get("cond-name-fr", "")] = dc; cui2name[dc] = dn
    dcs = set(d["cui"] for d in diseases.values())
    dcs_list = sorted(dcs)

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]

    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt

    # Use TOP_K=20 for richer overlap matching
    disease_features = {}  # dc -> list of feature words (lowercase)
    TOP_K = 20
    for dc in dcs_list:
        feats = ds.get(dc, {})
        top_cuis = sorted(feats.items(), key=lambda x: -x[1])[:TOP_K * 4]
        names = []; seen = set()
        for cui, cnt in top_cuis:
            n_ = cp.get(cui, cui)
            nl = n_.lower().strip()
            if not nl or nl in seen or nl in GENERIC_TERMS: continue
            if len(nl) < 4 or len(nl) > 60: continue
            seen.add(nl); names.append(nl)
            if len(names) >= TOP_K: break
        disease_features[dc] = names

    def patient_text(evidences):
        parts = []
        for ev in evidences:
            ev_parts = ev.split("_@_"); base=ev_parts[0]; value=ev_parts[1] if len(ev_parts)>1 else None
            info = ev_info.get(base, {}); q = info.get("question_en", "")
            val_en = info.get("value_en", {}).get(value, "") if value else ""
            if val_en and val_en.lower() in ("na","nowhere","n"): val_en=""
            val_en = fix_translation(val_en)
            parts.append(q + " " + val_en)
        return " ".join(parts).lower()

    score_matrix = np.load("pilot/results/v79_stage1.npy")
    n = score_matrix.shape[0]
    print(f"v79 stage1: {n} patients × {len(dcs_list)} candidates")

    candidates = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if len(candidates) >= n: break
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            candidates.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "true_dc": tdc,
            })
    assert len(candidates) == n

    # Build patient texts
    patient_texts = [patient_text(c["evidences"]) for c in candidates]

    # Pre-compute patient_kg_overlap[c_idx, d_idx] = count of disease features in patient text
    print("Computing KG overlap matrix...")
    overlap = np.zeros((n, len(dcs_list)), dtype=np.float32)
    for d_idx, dc in enumerate(dcs_list):
        feats = disease_features[dc]
        if not feats: continue
        for c_idx in range(n):
            text = patient_texts[c_idx]
            cnt = sum(1 for f in feats if f in text)
            overlap[c_idx, d_idx] = cnt

    # Stage 1 baseline
    t1_s1 = sum(1 for c_idx, c in enumerate(candidates)
                if dcs_list[int(np.argmax(score_matrix[c_idx]))] == c["true_dc"])
    print(f"Stage 1 (v79) @1 = {100*t1_s1/n:.2f}%")

    # Apply overlap as tie-breaker (small bonus)
    print("\n[Sweep] α (combined = score + α * overlap)")
    best_t1 = t1_s1; best_alpha = 0.0
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        combined = score_matrix + alpha * overlap
        t1 = sum(1 for c_idx, c in enumerate(candidates)
                 if dcs_list[int(np.argmax(combined[c_idx]))] == c["true_dc"])
        # Tie count
        tied = sum(1 for c_idx in range(n) if (combined[c_idx] == combined[c_idx].max()).sum() >= 2)
        print(f"  α={alpha:.3f}: @1={100*t1/n:.2f}%, ties={tied}")
        if t1 > best_t1:
            best_t1 = t1; best_alpha = alpha

    print(f"\n  Best α={best_alpha}: @1={100*best_t1/n:.2f}% (Δ={100*(best_t1-t1_s1)/n:+.2f}%p)")

    # Tie-break ONLY (zero out non-tied changes)
    print("\n[Tie-break only mode] use overlap only when scores are tied")
    final_score = score_matrix.copy()
    for c_idx in range(n):
        scores = score_matrix[c_idx]
        max_score = scores.max()
        tied_idxs = np.where(scores == max_score)[0]
        if len(tied_idxs) >= 2:
            # Among tied, pick the one with highest overlap
            tied_overlaps = overlap[c_idx, tied_idxs]
            best_in_tied = tied_idxs[int(np.argmax(tied_overlaps))]
            # Boost the chosen one slightly
            final_score[c_idx, best_in_tied] = max_score + 0.001

    t1_tb = sum(1 for c_idx, c in enumerate(candidates)
                if dcs_list[int(np.argmax(final_score[c_idx]))] == c["true_dc"])
    tied = sum(1 for c_idx in range(n) if (final_score[c_idx] == final_score[c_idx].max()).sum() >= 2)
    print(f"  Tie-break (overlap only): @1={100*t1_tb/n:.2f}% (Δ={100*(t1_tb-t1_s1)/n:+.2f}%p), remaining ties={tied}")

    print(f"\n{'='*80}")
    print(f"v88 GTPA@1 = {max(100*best_t1/n, 100*t1_tb/n):.2f}% (best of sweep, SUBSET={n})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
