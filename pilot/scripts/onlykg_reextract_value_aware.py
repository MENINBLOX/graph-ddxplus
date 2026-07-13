#!/usr/bin/env python3
"""Value-aware evidence CUI extraction.

For each (evidence_name, value) pair, extract CUIs specific to that value.
Binary evidences (data_type=B): single mapping.
Categorical (C) / Multi-choice (M): per-value mapping.

Patient row encoding "douleurxx_endroitducorps_@_tete" → look up
{"douleurxx_endroitducorps": {"tete": [Head_CUI]}}

Output: /mnt/medkg/kg/ddxplus_evidence_value_cuis.json
"""
from __future__ import annotations
import sys, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import scispacy, spacy
from scispacy.linking import EntityLinker

OUT = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"

KEEP_TUIS = {
    "T184", "T033", "T046", "T047", "T048", "T191", "T037", "T039", "T067",
    "T023", "T024", "T029", "T030", "T031", "T017",
}

QUESTION_PATTERNS = [
    r"\bdo you (have|feel|experience)\b",
    r"\bhave you (had|been|ever|recently)\b",
    r"\bis (your|the|it)\b",
    r"\bare you\b", r"\bdid you\b", r"\bcan you\b",
    r"\bwhere (is|do you|does it)\b",
    r"\bwhich (part|area)\b",
    r"\bhow (much|long|severe|intense|fast|precisely)\b",
    r"\bin the past \d+\s*(days?|weeks?|months?|years?)\b",
    r"\b(either felt or )?measured with a thermometer\b",
    r"\bmore than usual\b",
    r"\bthan (usual|normal|before)\b",
    r"\brelated to your (reason for )?(consulting|consultation)\b",
    r"\bof the consultation\b",
    r"\bsomewhere\b",
    r"\bcharacterize your\b",
    r"\?",
]
NOISE_CUIS = {
    "C0009818", "C0039818", "C0444706", "C0444099", "C3538928",
    "C0205390", "C1457868", "C0221423", "C0150312", "C0442743",
    "C0205245",  # Severe
}


def load_cui2tuis():
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
    # Remove (R)/(L)/(D)/(G) markers
    t = re.sub(r"\([RLDG]\)", "", t)
    t = re.sub(r"[?,;:!\(\)\"]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_cuis(nlp, text, cui2tuis, thresh=0.85):
    if not text: return set()
    cleaned = clean(text)
    if not cleaned: return set()
    doc = nlp(cleaned)
    out = set()
    for ent in doc.ents:
        if not ent._.kb_ents: continue
        for cui, score in ent._.kb_ents[:3]:
            if score < thresh: continue
            tuis = cui2tuis.get(cui, set())
            if tuis and not (tuis & KEEP_TUIS): continue
            if cui in NOISE_CUIS: continue
            out.add(cui)
            break  # top accepted per entity
    return out


def main():
    print("Loading evidences + scispaCy...")
    with open("data/ddxplus/release_evidences.json") as f:
        ev_info = json.load(f)
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": 0.75,
    })
    cui2tuis = load_cui2tuis()

    out = {}
    n_with_value = 0
    sample_shown = 0
    for ev_name, info in ev_info.items():
        q_en = info.get("question_en", "")
        vm = info.get("value_meaning", {})
        data_type = info.get("data_type", "B")
        ev_out = {}
        # Always include the question's own CUIs as "_question"
        q_cuis = extract_cuis(nlp, q_en, cui2tuis)
        ev_out["_question"] = sorted(q_cuis)
        # Per-value CUIs
        if vm:
            n_with_value += 1
            for val_key, val_info in vm.items():
                if not isinstance(val_info, dict): continue
                val_text = val_info.get("en") or val_info.get("fr", "")
                if not val_text: continue
                if val_text.lower() in {"nowhere", "n", "y", "yes", "no"}:
                    ev_out[val_key] = []
                    continue
                v_cuis = extract_cuis(nlp, val_text, cui2tuis)
                ev_out[val_key] = sorted(v_cuis)
        out[ev_name] = ev_out
        if sample_shown < 5 and vm:
            print(f"  {ev_name} ({data_type}, {len(vm)} values): question={q_cuis}")
            for vk, vc in list(ev_out.items())[1:5]:
                print(f"    {vk}: {vc[:3]}")
            sample_shown += 1

    print(f"\nEvidences with value_meaning: {n_with_value}")
    print(f"Total evidences: {len(out)}")
    print(f"\nSaving to {OUT}...")
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)
    print("Done")


if __name__ == "__main__":
    main()
