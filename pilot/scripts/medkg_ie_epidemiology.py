#!/usr/bin/env python3
"""Epidemiology IE: extract risk-factor / transmission / setting / age context
from 'epidemiology', 'etiology', 'causes', 'history' sections.

Different from phenotype IE: this captures the WHO gets the disease, not WHAT
symptoms. Critical for distinguishing diseases with overlapping symptoms but
different epidemiologic profiles (e.g., URTI common in crowded/daycare settings
vs. Pneumonia in elderly with comorbidities).

Output: $MEDKG_ROOT/processed/edges_epidemiology.jsonl with edge_type=epidemiologic
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

SEC_PATH = MEDKG_ROOT / "processed" / "sections.jsonl"
OUT_PATH = MEDKG_ROOT / "processed" / "edges_epidemiology.jsonl"
LOG_PATH = MEDKG_ROOT / "logs" / "ie_epidemiology.log"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_SECTIONS = {"epidemiology", "etiology", "causes", "etiology and epidemiology",
                   "history", "introduction"}

IE_PROMPT = """# Task: Epidemiologic Context Extraction

Extract EPIDEMIOLOGIC and CONTEXTUAL factors that characterize who gets the disease "{disease}", from the following text.

# What to Extract

EPIDEMIOLOGIC FACTORS in any of these categories:
1. **Population/age** (e.g., "young adults 15-40 years", "elderly", "children under 5")
2. **Setting/environment** (e.g., "daycare", "crowded living", "nursing home", "winter season")
3. **Risk factors** (e.g., "smoking", "immunocompromised state", "recent travel", "sick contacts")
4. **Transmission mode** (e.g., "person-to-person", "airborne droplets", "fecal-oral", "sexually transmitted")
5. **Comorbidity context** (e.g., "patients with COPD", "in HIV-positive individuals")

# What NOT to Extract

- Symptoms or clinical findings (those are phenotypes, not epidemiology)
- Diagnostic test results
- Treatment information
- Specific incidence/prevalence numbers (e.g., "10 per 100,000") — these are statistics, not categorical features

# Output

Output one factor per line:
EPID: <category>: <description>

Where <category> is one of: POPULATION, SETTING, RISK_FACTOR, TRANSMISSION, COMORBIDITY.

Output ONLY EPID lines. No other text.

# Section
{section_name}

# Text
{text}

# Epidemiologic Factors for {disease}"""


def parse_factors(text):
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(r"EPID\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            f = m.group(1).strip().rstrip(".,;:")
            f = re.sub(r"^[\*\-•\d\.\s]+", "", f).strip()
            if f and 5 < len(f) < 200:
                out.append(f)
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
            text = (s.get("text") or "").strip()
            if len(text) < 100: continue
            s["text"] = text[:3000]
            sections.append(s)
    log(f"Loaded {len(sections):,} epidemiology/context sections")

    log(f"Loading vLLM (gemma-4-E4B-it)...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=400)

    BATCH = 1024
    n_edges = 0; n_done = 0
    with OUT_PATH.open("w") as out:
        for chunk_start in range(0, len(sections), BATCH):
            chunk = sections[chunk_start:chunk_start + BATCH]
            convs = [[{"role": "user", "content": IE_PROMPT.format(
                disease=s["disease"], section_name=s.get("section_name", ""),
                text=s["text"]
            )}] for s in chunk]
            outputs = llm.chat(convs, sampling)
            for s, o in zip(chunk, outputs):
                try: text = o.outputs[0].text
                except: text = ""
                feats = parse_factors(text)
                for p in feats:
                    edge = {"disease": s["disease"], "umls_cui": s.get("umls_cui"),
                            "phenotype": p, "source": s.get("source", ""),
                            "source_id": s.get("source_id"),
                            "section_name": s.get("section_name"),
                            "extracted_by": "gemma-4-E4B",
                            "edge_type": "epidemiologic"}
                    for k in ("pmid","revid","chapter_title","topic_title","title"):
                        if s.get(k): edge[k] = s[k]
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta_min = (len(sections) - n_done) / max(rate, 0.001) / 60
            log(f"  {chunk_start+len(chunk)}/{len(sections):,}  edges={n_edges:,}  rate={rate:.1f}/s ETA={eta_min:.0f}min")
    log(f"Done. Epidemiologic edges: {n_edges:,}")


if __name__ == "__main__":
    main()
