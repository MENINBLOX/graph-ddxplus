#!/usr/bin/env python3
"""Categorized IE: extract disease features in 5 clinical categories.

Inspired by NB analysis showing these categories are discriminative across
DDXPlus 49. Generalizable to other benchmarks (real clinics also question
along these axes).

Categories:
  1. EXPOSURE — epidemiologic exposure (contagious context, settings, contacts)
  2. SYSTEMIC — systemic toxicity (severe fatigue, chills, anorexia, weight loss)
  3. RESPIRATORY — respiratory details (productive cough, sputum quality, dyspnea)
  4. PAIN_PROFILE — pain characteristics (location, quality, radiation, severity)
  5. COMORBIDITY — typical associated chronic conditions or risk diseases

Source sections targeted: epidemiology, history and physical, signs and symptoms,
differential diagnosis, etiology, causes, history.

Output: $MEDKG_ROOT/processed/edges_categorized.jsonl
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

SEC_PATH = MEDKG_ROOT / "processed" / "sections.jsonl"
OUT_PATH = MEDKG_ROOT / "processed" / "edges_categorized.jsonl"
LOG_PATH = MEDKG_ROOT / "logs" / "ie_categorized.log"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_SECTIONS = {"epidemiology","etiology","causes","history","history and physical",
                   "signs and symptoms","differential diagnosis","differential diagnoses",
                   "evaluation","presentation","introduction"}

IE_PROMPT = """# Task: Categorized Clinical Feature Extraction

Extract characteristic features of "{disease}" from the text, organized into 5 clinical categories.

# Categories

1. **EXPOSURE** — epidemiologic exposure context (contagion via close contact, crowded settings, daycare, sick contacts, secondhand smoke, occupational, travel, season, age groups affected)
2. **SYSTEMIC** — systemic toxicity / constitutional symptoms (severe fatigue, malaise, chills/rigors, anorexia, weight loss, diaphoresis, severe vs mild)
3. **RESPIRATORY** — respiratory details if relevant (productive cough with purulent sputum vs dry cough; dyspnea characterization; wheezing; runny nose color)
4. **PAIN_PROFILE** — pain characteristics if relevant (anatomic location, radiation, quality (sharp/dull/burning/pleuritic), severity, exacerbating factors)
5. **COMORBIDITY** — typical associated chronic conditions or context (COPD, heart failure, immunocompromised, diabetes, prior infections)

# Output Format

For each category that has clear content for this disease in the text, output one or more lines:
CAT: <category>: <feature>

Example:
CAT: EXPOSURE: spreads via close contact in crowded settings
CAT: SYSTEMIC: high fever with rigors
CAT: RESPIRATORY: productive cough with purulent sputum

Only output CAT lines. No other text. Skip categories with no clear content. Each feature should be a discriminative descriptor (with modifier when present).

# Section
{section_name}

# Text
{text}

# Categorized Features for {disease}"""


def parse_categorized(text):
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(r"CAT\s*:\s*([A-Z_]+)\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            cat = m.group(1).strip().upper()
            feat = m.group(2).strip().rstrip(".,;:")
            feat = re.sub(r"^[\*\-•\d\.\s]+", "", feat).strip()
            if cat in {"EXPOSURE","SYSTEMIC","RESPIRATORY","PAIN_PROFILE","COMORBIDITY"} \
               and 5 < len(feat) < 200:
                out.append((cat, feat))
    return out


def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(s + "\n")


def main():
    t0 = time.time()
    sections = []
    with SEC_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: s = json.loads(line)
            except: continue
            sn = (s.get("section_name") or "").lower().strip()
            if sn not in TARGET_SECTIONS: continue
            t = (s.get("text") or "").strip()
            if len(t) < 100: continue
            s["text"] = t[:3000]
            sections.append(s)
    log(f"Loaded {len(sections):,} target sections")

    log("Loading vLLM...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image":0,"audio":0})
    sampling = SamplingParams(temperature=0, max_tokens=500)

    BATCH = 1024
    n_edges = 0; n_done = 0
    with OUT_PATH.open("w") as out:
        for cs in range(0, len(sections), BATCH):
            chunk = sections[cs:cs+BATCH]
            convs = [[{"role":"user","content":IE_PROMPT.format(
                disease=s["disease"], section_name=s.get("section_name",""),
                text=s["text"]
            )}] for s in chunk]
            outs = llm.chat(convs, sampling)
            for s, o in zip(chunk, outs):
                try: text = o.outputs[0].text
                except: text = ""
                for cat, feat in parse_categorized(text):
                    edge = {"disease": s["disease"], "umls_cui": s.get("umls_cui"),
                            "phenotype": feat, "category": cat,
                            "source": s.get("source",""),
                            "source_id": s.get("source_id"),
                            "section_name": s.get("section_name"),
                            "extracted_by": "gemma-4-E4B",
                            "edge_type": "categorized"}
                    out.write(json.dumps(edge, ensure_ascii=False)+"\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            log(f"  {cs+len(chunk)}/{len(sections):,}  edges={n_edges:,}")
    log(f"Done. Categorized edges: {n_edges:,}")


if __name__ == "__main__":
    main()
