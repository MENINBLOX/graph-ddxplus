#!/usr/bin/env python3
"""Fix 39 empty evidence mappings via:
1. Re-run scispaCy with lower threshold (0.65) on cleaned text
2. Apply hand-curated fallback for clinically critical evidences

Output: /mnt/medkg/kg/ddxplus_evidence_value_cuis_v2.json (overwrites v2)
"""
from __future__ import annotations
import sys, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import scispacy, spacy
from scispacy.linking import EntityLinker

IN = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
OUT = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis_v2.json"

# Hand-curated fallback: evidence name → CUI list (universal medical knowledge)
# Applied only when scispaCy fails to extract anything meaningful
HAND_FALLBACK = {
    "pale":          ["C0030232"],  # Pallor
    "dysp_effort":   ["C0035288"],  # Exertional dyspnea
    "gain_poids":    ["C0043094"],  # Weight Gain
    "rhino_pur":     ["C0235998"],  # Purulent nasal discharge
    "f17.210":       ["C0037369"],  # Smoking
    "Z99.2":         ["C0011946"],  # Dialysis
    "ains":          ["C0003211"],  # Anti-Inflammatory Agents, Non-Steroidal
    "drogues_stimul":["C0038280"],  # Stimulant
    "drogues_IV":    ["C0086135"],  # Intravenous drug abuser
    "drogues_anal":  ["C0002772"],  # Analgesics
    "trav1":         ["C0040802"],  # Travel
    "horm1":         ["C0033308"],  # Hormone therapy
    "ménorr":        ["C0025323"],  # Menorrhagia
    "naco":          ["C0003280"],  # Anticoagulants
    "impression_mort":["C0233492"], # Sense of impending doom
    "rhinite_aller": ["C0035455"],  # Rhinitis, Allergic
    "rhinite_inf":   ["C0035430"],  # Infectious rhinitis
    "rhinite_vaso":  ["C0035490"],  # Vasomotor rhinitis
    "hernie_hiatale":["C0019291"],  # Hiatal Hernia
    "laryngospasme": ["C0023068"],  # Laryngismus
    "surg1":         ["C0038894"],  # Surgical procedures
    "diab":          ["C0011849"],  # Diabetes
    "hypert":        ["C0020538"],  # Hypertension
    "obese":         ["C0028754"],  # Obesity
    "fum_pass":      ["C0040329"],  # Tobacco Smoke Pollution
    "crowd":         ["C0026057"],  # Multifamily Housing
    "dayc":          ["C0205321"],  # Daycare
    "itss_risque":   ["C0036916"],  # Sexually Transmitted Diseases
    # Additional 13:
    "move":          ["C0015259"],  # Exercise
    "c00-d48":       ["C0006826"],  # Neoplasm (active cancer)
    "vaccination":   ["C0042196"],  # Vaccination
    "malf_cardiaque":["C0018798"],  # Heart Defects, Congenital
    "drink_energie": ["C2745672"],  # Energy drink
    "cafe":          ["C0010111"],  # Coffee
    "immob1":        ["C0021046"],  # Immobilization
    "ap_asian":      ["C0078988"],  # Asians
    "urban1":        ["C0042029"],  # Urban
    "tagri":         ["C0001482"],  # Agriculture
    "tmine":         ["C0026000"],  # Mining
    "tconst":        ["C0010477"],  # Construction
    "suburb":        ["C0205341"],  # Suburban
}


def main():
    print("Loading existing value-aware mapping...")
    existing = json.load(open(IN))
    print(f"  {len(existing)} evidences")

    # Find empty ones
    empty_keys = []
    for k, m in existing.items():
        has_content = m.get('_question') or any(m.get(k2) for k2 in m if k2 != '_question')
        if not has_content:
            empty_keys.append(k)
    print(f"  Empty: {len(empty_keys)}")

    # Load scispaCy at lower threshold
    print("Loading scispaCy (threshold=0.65)...")
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": 0.65,
        "max_entities_per_mention": 3,
    })

    KEEP_TUIS = {"T184","T033","T046","T047","T048","T191","T037","T039","T067",
                 "T023","T024","T029","T030","T031","T017",
                 "T121","T058","T109"}  # also drugs T121, healthcare T058 for risk factor evidences
    cui2tuis = {}
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.split("|")
            cui2tuis.setdefault(p[0], set()).add(p[1])

    QUESTION_PATTERNS = [
        r"\bdo you (have|feel|experience|smoke|attend|work|live|undergo|currently)\b",
        r"\bhave you (had|been|ever|recently|gained|traveled)\b",
        r"\bare you (currently|taking|using)?\b",
        r"\bdid you\b", r"\bcan you\b",
        r"\bcurrently\b", r"\brecently\b", r"\bever\b",
        r"\bin the (last|past) \d+\s*\w*\b",
        r"\bof\b", r"\bwith\b", r"\bor\b", r"\band\b",
        r"\?",
    ]

    with open("data/ddxplus/release_evidences.json") as f:
        ev_info = json.load(f)

    fixed = dict(existing)
    n_hand_fallback = 0; n_re_extracted = 0; n_still_empty = 0
    for k in empty_keys:
        info = ev_info.get(k, {})
        q_en = info.get("question_en", "")
        cleaned = q_en
        for pat in QUESTION_PATTERNS:
            cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[?,;:!\(\)\"]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        cuis = set()
        if cleaned:
            doc = nlp(cleaned)
            for ent in doc.ents:
                if not ent._.kb_ents: continue
                for cui, score in ent._.kb_ents[:2]:
                    if score < 0.70: continue
                    tuis = cui2tuis.get(cui, set())
                    if tuis and not (tuis & KEEP_TUIS): continue
                    cuis.add(cui)
                    break

        if k in HAND_FALLBACK:
            # Hand-curated takes precedence over potentially noisy re-extraction
            cuis = set(HAND_FALLBACK[k])
            n_hand_fallback += 1
            tag = "FALLBACK"
        elif cuis:
            n_re_extracted += 1
            tag = "RE-EXTRACT"
        else:
            n_still_empty += 1
            tag = "STILL EMPTY"
        print(f"  {tag:13s} {k:25s} → {sorted(cuis)}")

        # Update fixed
        if cuis:
            fixed[k] = dict(existing[k])
            fixed[k]['_question'] = sorted(cuis)

    print(f"\nSummary: re-extracted={n_re_extracted}, hand-fallback={n_hand_fallback}, still empty={n_still_empty}")
    print(f"Saving to {OUT}...")
    with OUT.open("w") as f:
        json.dump(fixed, f, indent=2)
    print("Done")


if __name__ == "__main__":
    main()
