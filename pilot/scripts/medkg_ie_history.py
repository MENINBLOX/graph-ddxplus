#!/usr/bin/env python3
"""History/Risk Factor IE: extract patient history & risk factors.

DDXPlus questionnaire 분석 결과 missing CUIs:
- Patient history (smoking, drinking, prior diseases, immunization)
- Risk factors (pregnancy, diet, occupation, travel, contact)
- Family history (death of relative, hereditary conditions)
- Lifestyle (sedentary, sleep, stress)

PubMed academic abstracts focus on symptoms.
MedlinePlus/Wikipedia have richer history sections.
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_pubmed import log

IE_PROMPT = """# Task: Patient History & Risk Factor Extraction

Extract risk factors, patient history, and lifestyle elements relevant to "{disease}" that would be ASKED in a medical questionnaire BEFORE physical examination.

# Categories

A. **Personal History**:
   - smoking, alcohol use, drug use, addictive behavior
   - prior diseases (HIV, COPD, asthma, diabetes, hypertension)
   - prior surgery, hospitalization
   - immunization status, vaccinations
   - pregnancy status, recent childbirth

B. **Family History**:
   - family death from similar condition
   - hereditary conditions (cardiac, cancer, autoimmune)

C. **Environmental & Occupational**:
   - recent travel (specific regions)
   - exposure (sick person contact, occupational hazard, allergen)
   - chemical exposure (toxins, pollutants)

D. **Lifestyle Factors**:
   - sedentary lifestyle, recent immobility
   - diet (poor diet, malnutrition)
   - sleep disorder
   - stress, anxiety

E. **Demographics**:
   - age group (elderly, child)
   - sex-specific factors

# DO NOT extract
- Acute symptoms (cough, pain, fever) — those go in symptom IE
- Lab/imaging findings
- Treatments/drugs
- Mechanisms, genes
- The disease name itself

# Output (one risk factor per line, plain text)

# Abstract Title
{title}

# Abstract
{text}

# Risk factors / Patient history (one per line):
"""


def parse(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        line = re.sub(r'^[-•\d.)]+\s*', '', line).strip()
        if not line or len(line) > 80: continue
        low = line.lower()
        if any(b in low for b in ['the patient', 'physician', 'doctor', 'clinician', 'should ', 'must ']): continue
        out.append(line)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            try: records.append(json.loads(line))
            except: pass
    log(f"Loaded {len(records)} abstracts")
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
                    disease=dname, title=r.get('title',''), text=r['text'][:2500])}])
            outputs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outputs):
                try: text = o.outputs[0].text
                except: text = ""
                for p in parse(text):
                    edge = {
                        "disease": cui2name.get(r['cui'], r['cui']),
                        "umls_cui": r['cui'],
                        "phenotype": p,
                        "source": "history_ie",
                        "source_id": r.get('pmid'),
                        "extracted_by": "gemma-4-E4B",
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            log(f"  {n_done}/{len(records)} edges={n_edges:,} rate={rate:.1f}/s")
    log(f"Done. {n_edges} history edges → {out_path}")


if __name__ == "__main__":
    main()
