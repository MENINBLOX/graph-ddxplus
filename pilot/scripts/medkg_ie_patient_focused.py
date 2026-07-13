#!/usr/bin/env python3
"""Patient-focused IE: force extraction in patient-reportable vocabulary.

Hypothesis: DDXPlus/SymCat questionnaires ask patient-reportable + anatomically-grounded
phens. Our previous IE extracts academic clinical signs (rhonchi, troponin)
that patients can't answer.

This prompt forces gemma-4-E4B to extract ONLY phenotypes that:
1. A patient could observe and answer in a questionnaire
2. Match a standard medical questionnaire vocabulary (universal — used by
   DDXPlus, SymCat, MedlinePlus symptom checkers, NHS NHS Direct, WebMD, etc.)
3. Include anatomical location where pain/lesion occurs

Universal: no DDXPlus 49 disease names; vocab list is universal patient-care.
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_pubmed import log

IE_PROMPT = """# Task: Patient-Reportable Symptom Extraction

Extract symptoms patients with "{disease}" would report when answering a standard medical questionnaire.

# Symptom Categories (a patient questionnaire covers these)

A. **Sensory symptoms** (what the patient feels):
   - pain (location, intensity 1-10, type: sharp/dull/burning/stabbing/throbbing/cramping)
   - itching, burning, numbness, tingling
   - fever, chills, fatigue, weakness
   - dizziness, lightheadedness, confusion

B. **Functional changes** (what is different):
   - cough (productive vs dry), sputum
   - shortness of breath, wheezing
   - nausea, vomiting, diarrhea, constipation, blood in stool
   - loss of appetite, weight loss/gain
   - palpitations, fast heart rate
   - urinary frequency, urgency, blood in urine
   - sweating, hot flashes, night sweats

C. **Visible / palpable changes**:
   - rash, skin lesion, redness, swelling (with LOCATION on body)
   - bruising, bleeding, discoloration
   - lumps, bumps, hernia
   - hair loss, jaundice

D. **Anatomical location** (where symptoms occur — answer "where on the body"):
   - chest, abdomen (upper/lower, left/right), back, head, face
   - throat, neck, shoulder, arm (upper/lower), elbow, wrist, hand
   - hip, thigh, knee, ankle, foot, toe
   - eye, ear, nose, mouth, tongue
   - perineum, groin, genitalia

E. **Triggers & context** (what makes it worse/better):
   - exertion, rest, eating, lying down, breathing
   - exposure to allergens, cold, heat
   - travel, contact with sick person

# DO NOT extract

- Clinical examination signs that only physicians find (rhonchi, murmur, Murphy's sign, Babinski)
- Laboratory values (CBC, troponin, BUN, electrolytes)
- Imaging findings (ground-glass opacity, consolidation on CT)
- Mechanisms, pathways, gene names, antibodies
- Treatments, drugs, procedures
- The disease name itself

# Output (one symptom per line, plain text, no bullets)

# Abstract
{text}

# Patient-reportable symptoms (one per line):
"""


def parse(text: str):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        # Remove leading bullets / numbers
        line = re.sub(r'^[-•\d.)]+\s*', '', line).strip()
        if not line or len(line) > 80: continue
        # Reject if contains "the patient", "physician", etc. that suggest meta-text
        low = line.lower()
        if any(b in low for b in ['the patient', 'physician', 'doctor', 'clinician', 'should', 'must', 'might']): continue
        out.append(line)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(Path('pilot/data/pubmed_ddx_extra/ddxplus_anchored_v2.jsonl')))
    ap.add_argument("--output", default=str(MEDKG_ROOT / 'processed' / 'edges_patient_focused_ie.jsonl'))
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            try: records.append(json.loads(line))
            except: pass
    log(f"Loaded {len(records):,} abstracts")
    if not records: return

    with open('data/ddxplus/disease_icd10_cui_mapping.json') as f: icd = json.load(f)
    cui2name = {info['cui']: dn for dn, info in icd.items() if 'cui' in info}

    log("Loading vLLM...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=400)

    BATCH = 256
    n_edges = 0; n_done = 0
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_path.open('w') as out:
        for chunk_start in range(0, len(records), BATCH):
            chunk = records[chunk_start:chunk_start + BATCH]
            convs = []
            for r in chunk:
                dname = cui2name.get(r['cui'], r['cui'])
                convs.append([{"role": "user", "content": IE_PROMPT.format(
                    disease=dname, text=r['text'][:2500])}])
            outputs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outputs):
                try: text = o.outputs[0].text
                except: text = ""
                for p in parse(text):
                    edge = {
                        "disease": cui2name.get(r['cui'], r['cui']),
                        "umls_cui": r['cui'],
                        "phenotype": p,
                        "source": "pubmed_patient_focused",
                        "source_id": r.get('pmid'),
                        "extracted_by": "gemma-4-E4B",
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            log(f"  {n_done}/{len(records)}  edges={n_edges:,}  rate={rate:.1f}/s")
    log(f"Done. {n_edges:,} patient-focused edges → {out_path}")


if __name__ == "__main__":
    main()
