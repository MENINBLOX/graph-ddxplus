#!/usr/bin/env python3
"""Build full SymCat parsed JSON for all 801 diseases."""
import json, sys
sys.path.insert(0, "data/symcat")
from parse import parse_symcat_conditions, parse_symcat_symptoms

conds = parse_symcat_conditions("data/symcat/symcat-801-diseases.csv")
syms = parse_symcat_symptoms("data/symcat/symcat-474-symptoms.csv")
print(f"Conditions: {len(conds)}, Symptoms: {len(syms)}", flush=True)

# Build disease_symptom_pairs using NAMES (compatible with existing eval code)
# Existing symptom_umls_mapping.json keys are SYMPTOM NAMES (e.g. 'Sharp abdominal pain')
# Disease keys are CONDITION NAMES (e.g. 'Abdominal aortic aneurysm')
# So we convert slug → name using the parse outputs

sym_slug_to_name = {slug: info["name"] for slug, info in syms.items()}

disease_symptom_pairs = {}
for slug, info in conds.items():
    name = info["condition_name"]
    sym_list = []
    for sym_slug, sym_data in info["symptoms"].items():
        sym_name = sym_slug_to_name.get(sym_slug)
        if not sym_name: continue
        prob = sym_data["probability"]
        sym_list.append([sym_name, prob])
    sym_list.sort(key=lambda x: -x[1])
    if sym_list:
        disease_symptom_pairs[name] = sym_list

# Stats
n_pairs = sum(len(v) for v in disease_symptom_pairs.values())
print(f"Disease in pairs: {len(disease_symptom_pairs)}", flush=True)
print(f"Total pairs: {n_pairs}", flush=True)
print(f"Avg symptoms/disease: {n_pairs/len(disease_symptom_pairs):.1f}", flush=True)

out_path = "data/symcat/symcat_parsed_full.json"
with open(out_path, "w") as f:
    json.dump({
        "disease_symptom_pairs": disease_symptom_pairs,
        "diseases": conds,
        "symptoms": syms,
        "statistics": {
            "n_diseases": len(disease_symptom_pairs),
            "n_symptoms": len(syms),
            "n_pairs": n_pairs,
        }
    }, f, ensure_ascii=False)
print(f"Saved → {out_path}", flush=True)
