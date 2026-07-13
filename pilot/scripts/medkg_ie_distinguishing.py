#!/usr/bin/env python3
"""Distinguishing-features IE: extract DISCRIMINATIVE clinical features from
'differential diagnosis', 'diagnosis', 'history and physical', 'signs and symptoms'
sections.

Different from medkg_ie_multi_source.py:
  - Targets only diagnostic-reasoning sections (not full chapter)
  - Prompt asks for features that DISTINGUISH the target disease from similar ones
  - Output edge type: "distinguishing" (separate from "phenotype")

Output: $MEDKG_ROOT/processed/edges_distinguishing.jsonl
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

SEC_PATH = MEDKG_ROOT / "processed" / "sections.jsonl"
OUT_PATH = MEDKG_ROOT / "processed" / "edges_distinguishing.jsonl"
LOG_PATH = MEDKG_ROOT / "logs" / "ie_distinguishing.log"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_SECTIONS = {
    "differential diagnosis", "differential diagnoses", "differential",
    "diagnosis", "evaluation",
    "history and physical", "signs and symptoms",
    "presentation", "physical examination", "physical exam",
    "history", "symptoms",
}

IE_PROMPT = """# Task: Distinguishing Clinical Feature Extraction

Extract DISTINGUISHING clinical features for the disease "{disease}" from the following diagnosis-related text. A distinguishing feature is one that helps separate this disease from similar conditions during clinical evaluation.

# What to Extract

A distinguishing feature MUST be:
- A specific, observable clinical feature (symptom, sign, lab finding, imaging finding, or characteristic pattern)
- Stated in the text as relevant to confirming this disease OR ruling it in/out
- Specific enough to be checked in an individual patient (severity descriptor, onset pattern, anatomic location, quality, or associated context)

Express the feature with its DEFINING modifier when present. Examples:
- Not just "cough" but "non-productive dry cough"
- Not just "fever" but "high fever with rigors"
- Not just "headache" but "unilateral severe periorbital headache"
- Not just "chest pain" but "pleuritic chest pain worse with inspiration"

# What NOT to Extract

- Disease names of differentials (these are alternative diagnoses, not features)
- Treatments, medications, procedures
- Generic terms without modifier ("pain", "discomfort", "abnormality")
- Statistical/epidemiologic data ("60% of patients", "rare condition")
- Diagnostic test names without specific findings ("ECG" alone; OK: "ST-elevation on ECG")

# Output

Output one feature per line:
DISTINGUISH: <feature with modifier>

Output ONLY DISTINGUISH lines. No other text.

# Section Type
{section_name}

# Text
{text}

# Distinguishing Features for {disease}"""


def parse_features(text):
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(r"DISTINGUISH\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            f = m.group(1).strip().rstrip(".,;:")
            f = re.sub(r"^[\*\-•\d\.\s]+", "", f).strip()
            if f and 3 < len(f) < 120:
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
    log(f"Loaded {len(sections):,} target sections")

    log(f"Loading vLLM (gemma-4-E4B-it)...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=400)

    BATCH = 1024
    n_edges = 0
    n_done = 0
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
                feats = parse_features(text)
                for p in feats:
                    edge = {"disease": s["disease"], "umls_cui": s.get("umls_cui"),
                            "phenotype": p, "source": s.get("source", ""),
                            "source_id": s.get("source_id"),
                            "section_name": s.get("section_name"),
                            "section_id": s.get("section_id"),
                            "extracted_by": "gemma-4-E4B",
                            "edge_type": "distinguishing"}
                    for k in ("pmid", "revid", "chapter_title", "topic_title", "title"):
                        if s.get(k): edge[k] = s[k]
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta_min = (len(sections) - n_done) / max(rate, 0.001) / 60
            log(f"  {chunk_start+len(chunk)}/{len(sections):,}  edges={n_edges:,}  rate={rate:.1f}/s ETA={eta_min:.0f}min")

    log(f"Done. Distinguishing edges: {n_edges:,}")


if __name__ == "__main__":
    main()
