#!/usr/bin/env python3
"""Re-extract DDXPlus patient evidence CUIs from cleaned question text.

Original extraction polluted by questionnaire framing words:
  fievre → [Fever, Thermometer, Measured] (noise)
  pale   → [Skin Specimen, Usual] (no actual Pallor CUI)

Clean approach:
  1. Strip questionnaire framing ("Do you have", "Have you", etc.)
  2. scispaCy linker on cleaned text
  3. Filter to SYMPTOM/FINDING/DISEASE/ANATOMY TUIs only

Output: /mnt/medkg/kg/ddxplus_evidence_cuis_clean.json
"""
from __future__ import annotations
import sys, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import scispacy, spacy
from scispacy.linking import EntityLinker

OUT = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis_clean.json"

# Semantic types to KEEP
KEEP_TUIS = {
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T191",  # Neoplastic Process
    "T037",  # Injury or Poisoning
    "T039",  # Physiologic Function
    "T067",  # Phenomenon or Process
    "T023",  # Body Part, Organ, or Organ Component
    "T024",  # Tissue
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction
    "T031",  # Body Substance
    "T017",  # Anatomical Structure
}

# Questionnaire framing patterns to strip
QUESTION_PATTERNS = [
    r"\bdo you (have|feel|experience)\b",
    r"\bhave you (had|been|ever|recently)\b",
    r"\bis (your|the|it)\b",
    r"\bare you\b",
    r"\bdid you\b",
    r"\bcan you\b",
    r"\bwhere (is|do you|does it)\b",
    r"\bwhich (part|area)\b",
    r"\bhow (much|long|severe|intense)\b",
    r"\bin the past \d+\s*(days?|weeks?|months?|years?)\b",
    r"\b(either felt or )?measured with a thermometer\b",
    r"\bmore than usual\b",
    r"\bthan (usual|normal|before)\b",
    r"\brelated to your (reason for )?(consulting|consultation)\b",
    r"\bof the consultation\b",
    r"\bsomewhere\b",
    r"\bany\b",
    r"\bobjectiv(é|ee?d?)\b",
    r"\bavez[\-\s]vous\b",
    r"\b[Aa]vez[\-\s]vous\b",
    r"\?",
    r"^a\s+", r"^an\s+", r"^the\s+",
]
NOISE_CUIS = {
    "C0009818",  # Consultation
    "C0039818",  # Thermometer
    "C0444706",  # Measured
    "C0444099",  # Skin Specimen (false from "skin paler")
    "C3538928",  # Usual
    "C0205390",  # Normal
    "C1457868",  # Symptoms (too generic)
    "C0221423",  # Illness (too generic)
    "C0150312",  # Present
    "C0442743",  # Patient
    "C0205245",  # Severe
}


def load_ev_tuis():
    """Load CUI → TUI from MRSTY."""
    cui2tuis = {}
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.split("|")
            cui2tuis.setdefault(p[0], set()).add(p[1])
    return cui2tuis


def clean(text: str) -> str:
    t = text
    for pat in QUESTION_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[?,;:!\(\)]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    print("Loading evidences...")
    with open("data/ddxplus/release_evidences.json") as f:
        ev_info = json.load(f)
    print(f"  {len(ev_info)} evidences")

    print("Loading scispaCy + linker...")
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": 0.75,
    })
    print("  loaded")

    cui2tuis = load_ev_tuis()

    new_ev_cuis = {}
    total_kept = 0; total_dropped = 0
    samples_shown = 0
    for ev_name, info in ev_info.items():
        q_en = info.get("question_en", "")
        # For value_meaning entries (anatomical locations), iterate values too
        value_meanings = info.get("value_meaning", {})

        # Collect text candidates: question + each value
        texts_to_process = [q_en]
        for v_key, v_info in value_meanings.items():
            if isinstance(v_info, dict):
                v_text = v_info.get("en", v_info.get("fr", ""))
                if v_text: texts_to_process.append(v_text)

        all_cuis = set()
        for text in texts_to_process:
            if not text: continue
            cleaned = clean(text)
            if not cleaned: continue
            doc = nlp(cleaned)
            for ent in doc.ents:
                if not ent._.kb_ents: continue
                cui, score = ent._.kb_ents[0]
                if score < 0.85: continue
                # Filter by semantic type
                tuis = cui2tuis.get(cui, set())
                if tuis and not (tuis & KEEP_TUIS): continue
                if cui in NOISE_CUIS: continue
                all_cuis.add(cui)
        new_ev_cuis[ev_name] = sorted(all_cuis)

        if samples_shown < 15 and q_en:
            print(f"  {ev_name:40s} '{q_en[:60]}' → {len(all_cuis)} CUIs")
            samples_shown += 1
        total_kept += len(all_cuis)

    print(f"\nTotal CUI mappings: {total_kept}")
    print(f"Saving to {OUT}...")
    with OUT.open("w") as f:
        json.dump(new_ev_cuis, f, indent=2)
    print("Done")


if __name__ == "__main__":
    main()
