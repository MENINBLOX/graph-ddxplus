#!/usr/bin/env python3
"""Universal 6-way categorized IE on incremental PubMed abstracts.

Reads new CUI seed file (priority list), processes their pubmed_alt/{cui}.jsonl
abstracts, and outputs categorized edges to processed/edges_universal_v3.jsonl.

Categories (6):
  patient_reportable | clinical_sign | lab_finding | imaging_finding | history | demographic

Each output edge:
  {disease, umls_cui, phenotype, category, source, source_id, pmid, ...}

Designed for cross-benchmark KG: filter by `category` at eval time.
"""
from __future__ import annotations
import os, sys, json, re, argparse
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

ALT_DIR = MEDKG_ROOT / "pubmed_alt"


IE_PROMPT = """# Task: Universal Disease-Evidence Extraction with Category Labels

Extract clinically meaningful evidence features of "{disease}" from the text below, classified into 6 categories.

# Categories

1. **patient_reportable** — symptoms a patient describes in their own words (chest pain, fatigue, nausea, fever, blurred vision)
2. **clinical_sign** — physical exam findings observed by a clinician (jaundice, peripheral edema, hepatomegaly, heart murmur, rash)
3. **lab_finding** — laboratory test result abnormalities with directionality (elevated troponin, leukocytosis, hyperkalemia, ANA positive, low CD4)
4. **imaging_finding** — findings on imaging studies (consolidation on chest X-ray, mass on CT, ST elevation on ECG, white matter hyperintensity on MRI)
5. **history** — past events, exposures, or risk factors (smoking history, prior MI, family history of cancer, recent travel to malaria zone)
6. **demographic** — patient demographics relevant to the differential (elderly, female, African American, pregnant, age <5)

# Rules

- Each evidence must be **mechanistically or epidemiologically associated** with the disease (not just co-mentioned).
- Skip negated evidences ("denies fever" → do NOT extract).
- Use lay vocabulary when the patient would describe it; use technical when only a clinician would observe it.
- Each line must be ONE evidence with its category. Do not list "and" or comma-separated groups.

# Output Format

CAT: <category>: <evidence text>

Example for "Acute appendicitis":
CAT: patient_reportable: right lower quadrant abdominal pain
CAT: patient_reportable: anorexia
CAT: patient_reportable: nausea
CAT: clinical_sign: McBurney point tenderness
CAT: clinical_sign: rebound tenderness
CAT: clinical_sign: low-grade fever
CAT: lab_finding: leukocytosis
CAT: imaging_finding: appendiceal wall thickening on ultrasound
CAT: history: recent gastroenteritis
CAT: demographic: young adult

Only output CAT lines. No preamble, no commentary.

# Section
{section}

# Text
{text}

# Categorized Evidences for {disease}
"""


VALID_CATS = {"patient_reportable", "clinical_sign", "lab_finding",
              "imaging_finding", "history", "demographic"}


def parse_categorized_v3(text):
    """Parse CAT lines into (category, evidence) pairs."""
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(r"CAT\s*:\s*([a-z_]+)\s*:\s*(.+)", line, re.IGNORECASE)
        if not m: continue
        cat = m.group(1).strip().lower()
        feat = m.group(2).strip().rstrip(".,;:")
        feat = re.sub(r"^[\*\-•\d\.\s]+", "", feat).strip()
        if cat not in VALID_CATS: continue
        if not (4 < len(feat) < 200): continue
        out.append((cat, feat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuis_file", default="pilot/data/seeds/multibench_priority_seeds.jsonl",
                    help="JSONL with cui field, restrict IE to these CUIs")
    ap.add_argument("--out", default=str(MEDKG_ROOT / "processed" / "edges_universal_v3.jsonl"))
    ap.add_argument("--max_abstracts_per_cui", type=int, default=30)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    args = ap.parse_args()

    # Load target CUIs
    target_cuis = []
    cui_to_name = {}
    with open(args.cuis_file) as f:
        for line in f:
            e = json.loads(line)
            target_cuis.append(e["cui"])
            cui_to_name[e["cui"]] = e.get("primary_name", "")
    print(f"Target CUIs: {len(target_cuis)}", flush=True)

    # Gather abstracts
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
    print(f"Loaded {len(records)} abstracts from {len(target_cuis)} CUIs", flush=True)
    if not records:
        print("No abstracts to process. Exiting.", flush=True)
        return

    # vLLM batch
    print("Loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              tensor_parallel_size=args.tp,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=500)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    BATCH = 512
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
                for cat, feat in parse_categorized_v3(text):
                    edge = {
                        "disease": r["disease"],
                        "umls_cui": r["cui"],
                        "phenotype": feat,
                        "category": cat,
                        "source": "pubmed_alt_v3",
                        "source_id": r["pmid"],
                        "pmid": r["pmid"],
                        "title": r["title"],
                        "extracted_by": "gemma-4-E4B-v3",
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
            print(f"  {cs+len(chunk)}/{len(records)} edges={n_edges:,}", flush=True)
    print(f"Done. Universal v3 IE edges: {n_edges:,} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
