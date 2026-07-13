#!/usr/bin/env python3
"""IE prompt v4 — 2-tier SOAP categorization, benchmark-blind few-shot.

Categories (2-tier):
  subjective: patient-perceivable (symptom + history + demographic)
  objective:  clinician-measured (physical_sign + lab_finding + imaging_finding)

Sub-tags preserved for KG inspection / optional ablation.

Few-shot examples chosen to be benchmark-blind (Iron Deficiency Anemia,
Sarcoidosis, Pulmonary Embolism — diverse evidence pattern, no overlap with
DDXPlus 49 / SymCat 50 / RareBench / ER-Reason dominant diseases).
"""
from __future__ import annotations
import os, sys, json, re, argparse, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

ALT_DIR = MEDKG_ROOT / "pubmed_alt"


IE_PROMPT = """# Task: Disease-Evidence Extraction with SOAP Categorization

Extract clinically meaningful evidence features of "{disease}" from the text, classified at TWO levels:

## Level 1 (top-tier, required): subjective | objective

- **subjective**: anything a patient could perceive or report themselves
- **objective**: anything requiring clinical examination, measurement, or test

## Level 2 (sub-tier, required): pick one matching the top-tier

If subjective:
  - **symptom**: current self-perceived sensation (chest pain, fatigue, nausea, blurred vision)
  - **history**: past event, exposure, or risk factor (smoking history, prior MI, recent travel)
  - **demographic**: age, sex, ethnicity, pregnancy status

If objective:
  - **physical_sign**: examiner-observable on physical exam (jaundice, edema, murmur, rash, lymphadenopathy)
  - **lab_finding**: laboratory test abnormality with directionality (elevated troponin, leukocytosis, hypokalemia)
  - **imaging_finding**: finding on imaging study (consolidation on chest X-ray, ST elevation on ECG, brain atrophy on MRI)

## Rules

- Each evidence must be **mechanistically or epidemiologically associated** with the disease.
- Skip negated evidences ("denies fever" → DO NOT extract).
- One evidence per line.
- Use lay vocabulary for symptoms (what a patient would say); technical for signs/labs/imaging.

## Output Format

CAT: <level1>.<level2>: <evidence text>

## Examples (diverse pattern, all 6 sub-categories represented)

### Example 1 — Iron deficiency anemia
CAT: subjective.symptom: fatigue
CAT: subjective.symptom: shortness of breath on exertion
CAT: subjective.demographic: female of reproductive age
CAT: subjective.history: chronic blood loss
CAT: objective.physical_sign: pale conjunctiva
CAT: objective.physical_sign: koilonychia
CAT: objective.lab_finding: low hemoglobin
CAT: objective.lab_finding: low ferritin
CAT: objective.lab_finding: microcytic red blood cells

### Example 2 — Sarcoidosis
CAT: subjective.symptom: persistent dry cough
CAT: subjective.symptom: fatigue
CAT: subjective.history: African American ethnicity
CAT: objective.physical_sign: bilateral hilar lymphadenopathy on exam
CAT: objective.physical_sign: erythema nodosum
CAT: objective.lab_finding: elevated serum ACE level
CAT: objective.lab_finding: hypercalcemia
CAT: objective.imaging_finding: bilateral hilar adenopathy on chest X-ray
CAT: objective.imaging_finding: pulmonary nodular infiltrates on CT

### Example 3 — Pulmonary embolism
CAT: subjective.symptom: sudden chest pain
CAT: subjective.symptom: shortness of breath
CAT: subjective.history: recent immobilization
CAT: subjective.history: oral contraceptive use
CAT: objective.physical_sign: tachycardia
CAT: objective.physical_sign: unilateral leg swelling
CAT: objective.lab_finding: elevated D-dimer
CAT: objective.imaging_finding: filling defect on CT pulmonary angiogram
CAT: objective.imaging_finding: S1Q3T3 pattern on ECG

Only output CAT lines. No preamble, no commentary, no explanation. Each line must follow `CAT: <level1>.<level2>: <text>` exactly.

# Section
{section}

# Text
{text}

# Categorized Evidences for {disease}
"""


VALID_PAIRS = {
    ("subjective", "symptom"), ("subjective", "history"), ("subjective", "demographic"),
    ("objective", "physical_sign"), ("objective", "lab_finding"), ("objective", "imaging_finding"),
}


def parse_categorized_v4(text):
    """Parse CAT lines into (level1, level2, evidence) tuples."""
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(r"CAT\s*:\s*([a-z_]+)\s*\.\s*([a-z_]+)\s*:\s*(.+)", line, re.IGNORECASE)
        if not m: continue
        l1 = m.group(1).strip().lower()
        l2 = m.group(2).strip().lower()
        feat = m.group(3).strip().rstrip(".,;:")
        feat = re.sub(r"^[\*\-•\d\.\s]+", "", feat).strip()
        if (l1, l2) not in VALID_PAIRS: continue
        if not (4 < len(feat) < 200): continue
        out.append((l1, l2, feat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuis_file", default="pilot/data/seeds/v4_test_pilot.jsonl")
    ap.add_argument("--out", default=str(MEDKG_ROOT / "processed" / "edges_universal_v4.jsonl"))
    ap.add_argument("--max_abstracts_per_cui", type=int, default=20)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()

    target_cuis = []
    cui_to_name = {}
    with open(args.cuis_file) as f:
        for line in f:
            e = json.loads(line)
            target_cuis.append(e["cui"])
            cui_to_name[e["cui"]] = e.get("primary_name", e.get("name", ""))
    # Partition
    target_cuis = [c for i, c in enumerate(target_cuis) if i % args.of == args.part]
    print(f"[part {args.part}/{args.of}] Target CUIs: {len(target_cuis)}", flush=True)

    records = []
    for cui in target_cuis:
        fp = ALT_DIR / f"{cui}.jsonl"
        if not fp.exists() or fp.stat().st_size == 0: continue
        with fp.open() as f:
            cnt = 0
            for line in f:
                line = line.strip()
                if not line: continue
                try: e = json.loads(line)
                except: continue
                ab = (e.get("abstract") or "").strip()
                if len(ab) < 50: continue
                records.append({
                    "disease": e.get("disease_name", cui_to_name.get(cui, cui)),
                    "cui": cui,
                    "pmid": e.get("pmid", ""),
                    "title": e.get("title", ""),
                    "abstract": ab[:2500],
                })
                cnt += 1
                if cnt >= args.max_abstracts_per_cui: break
    print(f"  loaded {len(records)} abstracts", flush=True)
    if not records:
        print("  no records, exit", flush=True)
        return

    print("Loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              tensor_parallel_size=args.tp, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=600)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    BATCH = 512
    t0 = time.time()
    with open(args.out, "w") as out:
        for cs in range(0, len(records), BATCH):
            chunk = records[cs:cs+BATCH]
            convs = [[{"role": "user", "content": IE_PROMPT.format(
                disease=r["disease"], section="abstract", text=r["abstract"]
            )}] for r in chunk]
            outs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outs):
                try: text = o.outputs[0].text
                except: text = ""
                for l1, l2, feat in parse_categorized_v4(text):
                    edge = {"disease": r["disease"], "umls_cui": r["cui"],
                            "phenotype": feat, "level1": l1, "level2": l2,
                            "category": l2,  # for backward compat with v3 readers
                            "source": "pubmed_alt_v4", "source_id": r["pmid"],
                            "pmid": r["pmid"], "title": r["title"],
                            "extracted_by": "gemma-4-E4B-v4"}
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
            elapsed = time.time() - t0
            rate = (cs + len(chunk)) / max(elapsed, 1)
            eta = (len(records) - cs - len(chunk)) / max(rate, 1)
            print(f"  [part {args.part}] {cs+len(chunk)}/{len(records)} "
                  f"edges={n_edges:,} ETA={eta/60:.0f}min", flush=True)
    print(f"Done. v4 edges: {n_edges:,} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
