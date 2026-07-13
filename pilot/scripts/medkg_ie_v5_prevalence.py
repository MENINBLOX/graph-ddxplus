#!/usr/bin/env python3
"""IE v5 — Prevalence-aware extraction.

Extracts disease-evidence with frequency qualifier:
  always (0.95) / common (0.75) / frequent (0.55) /
  occasional (0.30) / uncommon (0.15) / rare (0.05)

Output edges include `prevalence` (numeric 0-1) replacing `weight`.
"""
from __future__ import annotations
import os, sys, json, re, argparse, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

ALT_DIR = MEDKG_ROOT / "pubmed_alt"


IE_PROMPT = """# Task: Disease-Evidence Extraction with Clinical Prevalence

Extract evidence features of "{disease}" from the text, with clinical PREVALENCE indicator.

## Level 1 (top-tier): subjective | objective
## Level 2 (sub): symptom | history | demographic | physical_sign | lab_finding | imaging_finding
## Level 3 (NEW — prevalence qualifier):

| Qualifier  | Mapped P(E|D) | Use when text says... |
|------------|---------------|------------------------|
| always     | 0.95 | "all patients", "universal", "pathognomonic", "100%", "diagnostic feature" |
| common     | 0.75 | "most patients", "typically", "characteristic", "majority", "≥75%" |
| frequent   | 0.55 | "often", "frequently", ">50%", "in the majority" |
| occasional | 0.30 | "sometimes", "may have", "in some cases", "30-50%" |
| uncommon   | 0.15 | "less frequently", "in a minority", "15-30%" |
| rare       | 0.05 | "rare", "occasionally seen", "atypical", "<10%" |

If text doesn't specify, default to **frequent** (0.55) for clearly emphasized features, **occasional** (0.30) for casually mentioned.

## Output Format

CAT: <level1>.<level2>: <evidence> | freq: <qualifier>

## Examples (benchmark-blind diverse pattern)

### Example 1 — Iron deficiency anemia
CAT: subjective.symptom: fatigue | freq: common
CAT: subjective.symptom: shortness of breath on exertion | freq: frequent
CAT: subjective.demographic: female of reproductive age | freq: common
CAT: subjective.history: chronic blood loss | freq: frequent
CAT: objective.physical_sign: pale conjunctiva | freq: common
CAT: objective.physical_sign: koilonychia | freq: rare
CAT: objective.lab_finding: low hemoglobin | freq: always
CAT: objective.lab_finding: low ferritin | freq: common
CAT: objective.lab_finding: microcytic red blood cells | freq: common

### Example 2 — Sarcoidosis
CAT: subjective.symptom: persistent dry cough | freq: frequent
CAT: subjective.symptom: fatigue | freq: common
CAT: subjective.history: African American ethnicity | freq: occasional
CAT: objective.physical_sign: bilateral hilar lymphadenopathy on exam | freq: common
CAT: objective.physical_sign: erythema nodosum | freq: occasional
CAT: objective.lab_finding: elevated serum ACE level | freq: frequent
CAT: objective.lab_finding: hypercalcemia | freq: uncommon
CAT: objective.imaging_finding: bilateral hilar adenopathy on chest X-ray | freq: common
CAT: objective.imaging_finding: pulmonary nodular infiltrates on CT | freq: frequent

### Example 3 — Pulmonary embolism
CAT: subjective.symptom: sudden chest pain | freq: frequent
CAT: subjective.symptom: shortness of breath | freq: common
CAT: subjective.history: recent immobilization | freq: occasional
CAT: subjective.history: oral contraceptive use | freq: occasional
CAT: objective.physical_sign: tachycardia | freq: common
CAT: objective.physical_sign: unilateral leg swelling | freq: occasional
CAT: objective.lab_finding: elevated D-dimer | freq: common
CAT: objective.imaging_finding: filling defect on CT pulmonary angiogram | freq: always
CAT: objective.imaging_finding: S1Q3T3 pattern on ECG | freq: uncommon

Only output CAT lines. Each line: `CAT: <level1>.<level2>: <text> | freq: <qualifier>`.

# Section
{section}

# Text
{text}

# Categorized Evidences for {disease}
"""


PREV_MAP = {
    "always": 0.95, "common": 0.75, "frequent": 0.55,
    "occasional": 0.30, "uncommon": 0.15, "rare": 0.05,
}
VALID_PAIRS = {
    ("subjective", "symptom"), ("subjective", "history"), ("subjective", "demographic"),
    ("objective", "physical_sign"), ("objective", "lab_finding"), ("objective", "imaging_finding"),
}


def parse_prevalence_v5(text):
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(
            r"CAT\s*:\s*([a-z_]+)\s*\.\s*([a-z_]+)\s*:\s*(.+?)\s*\|\s*freq\s*:\s*([a-z_]+)",
            line, re.IGNORECASE
        )
        if not m: continue
        l1, l2 = m.group(1).strip().lower(), m.group(2).strip().lower()
        feat = m.group(3).strip().rstrip(".,;:")
        freq = m.group(4).strip().lower()
        feat = re.sub(r"^[\*\-•\d\.\s]+", "", feat).strip()
        if (l1, l2) not in VALID_PAIRS: continue
        if freq not in PREV_MAP: continue
        if not (4 < len(feat) < 200): continue
        out.append((l1, l2, feat, freq, PREV_MAP[freq]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuis_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_abstracts_per_cui", type=int, default=15)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    target_cuis = []
    cui_to_name = {}
    with open(args.cuis_file) as f:
        for line in f:
            e = json.loads(line)
            target_cuis.append(e["cui"])
            cui_to_name[e["cui"]] = e.get("primary_name", e.get("name", ""))
    print(f"Target CUIs: {len(target_cuis)}", flush=True)

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
    print(f"Loaded {len(records)} abstracts", flush=True)
    if not records:
        print("No records, exit", flush=True); return

    import os as _os
    _os.environ["CUDA_HOME"] = "/usr/local/cuda"
    _os.environ["PATH"] = "/usr/local/cuda/bin:" + _os.environ.get("PATH", "")

    print("Loading vLLM 0.21...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=600)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    BATCH = 256
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
                for l1, l2, feat, freq_q, prev in parse_prevalence_v5(text):
                    edge = {"disease": r["disease"], "umls_cui": r["cui"],
                            "phenotype": feat, "level1": l1, "level2": l2,
                            "category": l2, "freq_qualifier": freq_q,
                            "prevalence": prev,  # NEW: numeric P(E|D) estimate
                            "source": "pubmed_v5_prevalence",
                            "pmid": r["pmid"], "title": r["title"],
                            "extracted_by": "gemma-4-E4B-v5"}
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
            elapsed = time.time() - t0
            print(f"  {cs+len(chunk)}/{len(records)} edges={n_edges} ({elapsed:.0f}s)", flush=True)
    print(f"Done. v5 edges: {n_edges:,} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
