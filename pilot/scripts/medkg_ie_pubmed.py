#!/usr/bin/env python3
"""PubMed IE: extract disease–phenotype edges from crawled PubMed abstracts.

Input:
  - $MEDKG_ROOT/pubmed/{cui}.jsonl   one file per CUI, each line is {pmid, title, abstract, cui, disease_name}

Each abstract is a single LLM IE prompt anchored to the disease name (per-CUI).
The disease anchor is the same as the seed list (UMLS preferred name + benchmark
disease name when applicable; we use the file's `disease_name` field).

Same v3 principle-only IE prompt (clinical phenotype extraction) used for textbooks.
gemma-4-E4B-it via vLLM, batched.

Output:
  - $MEDKG_ROOT/processed/edges_pubmed_ie.jsonl
"""
from __future__ import annotations
import os, sys, json, re, time, argparse
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

PUBMED_DIR = MEDKG_ROOT / "pubmed"
OUT_PATH = MEDKG_ROOT / "processed" / "edges_pubmed_ie.jsonl"
LOG_PATH = MEDKG_ROOT / "logs" / "pubmed_ie.log"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

IE_PROMPT = """# Task: Clinical Phenotype Extraction

Extract diagnostic phenotypes (clinical features) for the disease "{disease}" from the following PubMed abstract.

# Definition

A diagnostic phenotype is a clinically observable abnormality of an individual patient that a clinician evaluates when establishing a diagnosis. It belongs to exactly one of the following observation modalities:
- A subjective sensation or complaint reported directly by the patient
- A physical finding objectively observed by a clinician during examination
- A quantitative biochemical, hematological, or immunological abnormality measured in body fluids or tissues
- An abnormal structural or functional finding detected by medical imaging
- A microscopic or gross abnormality identified in tissue specimens

# Inclusion Criteria

A candidate qualifies only if ALL of the following hold:
- It is explicitly stated in the abstract as occurring in patients with the target disease
- It denotes a specific, discrete clinical observation rather than a category, summary, or qualitative descriptor
- It is an attribute of the patient's body, not of an external entity, instrument, study, intervention, or biological mechanism
- It is identifiable as a single phenotypic concept rather than a compound clause

# Exclusion Criteria

A candidate MUST NOT be extracted if any of the following applies:
- It names the target disease itself, any of its subtypes, alternative designations, or closely synonymous terms
- It refers to a gene, gene product, protein, receptor, antigen, antibody, or any molecular entity
- It refers to a pathophysiological or molecular mechanism, pathway, or biological process
- It refers to any therapeutic, surgical, pharmacological, or supportive intervention
- It refers to a diagnostic instrument, scoring system, questionnaire, scale, or composite index
- It refers to study design, sample characteristics, prevalence, incidence, mortality, or epidemiological measurement
- It refers to a different disease mentioned only for differential diagnosis or comparison
- It is a generic narrative term denoting an outcome, course, severity grade, or clinical state without naming a specific abnormality
- It refers to a temporal pattern, demographic attribute, comorbidity status, or risk factor that is not itself a clinical observation

# Precision Standard

If a candidate's classification is ambiguous, it must be excluded. False positives degrade knowledge graph quality more than false negatives.

# Output Specification

Express each phenotype using canonical medical terminology consistent with controlled biomedical vocabularies. Normalize non-canonical phrasing to its canonical form when the canonical form is unambiguous; otherwise reuse the abstract's wording.

Output exactly one phenotype per line, each line in the form:
PHENOTYPE: <term>

The output must consist solely of such PHENOTYPE lines. Do not produce any analytical commentary, reasoning, headers, bullet markers, numbering, or surrounding text.

# Source Type

This text comes from: PubMed abstract
Title: {title}

# Text

{text}

# Extracted Phenotypes for {disease}"""


def parse_phenotypes(text):
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(r"PHENOTYPE\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            f = m.group(1).strip().rstrip(".,;:")
            f = re.sub(r"^[\*\-•\d\.\s]+", "", f).strip()
            if f and 2 < len(f) < 100:
                out.append(f)
    return out


def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(s + "\n")


def load_abstracts(limit_cuis=0, limit_per_cui=20, min_chars=50):
    """Yield records {disease, cui, pmid, title, abstract}."""
    records = []
    files = sorted(PUBMED_DIR.glob("*.jsonl"))
    if limit_cuis > 0:
        files = files[:limit_cuis]
    for fp in files:
        size = fp.stat().st_size
        if size == 0:
            continue
        with fp.open() as f:
            n_kept = 0
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                ab = (e.get("abstract") or "").strip()
                if len(ab) < min_chars: continue
                records.append({
                    "disease": e.get("disease_name", ""),
                    "cui": e.get("cui", fp.stem),
                    "pmid": e.get("pmid", ""),
                    "title": e.get("title", ""),
                    "abstract": ab,
                })
                n_kept += 1
                if n_kept >= limit_per_cui: break
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_cuis", type=int, default=0, help="0 = all")
    ap.add_argument("--limit_per_cui", type=int, default=20)
    ap.add_argument("--max_chars", type=int, default=2500, help="abstract truncation")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--resume_from", type=int, default=0, help="skip first N records (for restart)")
    args = ap.parse_args()

    t0 = time.time()
    log(f"PubMed IE start. limit_cuis={args.limit_cuis}, limit_per_cui={args.limit_per_cui}")

    records = load_abstracts(args.limit_cuis, args.limit_per_cui)
    log(f"Loaded {len(records):,} abstracts from {PUBMED_DIR}")
    if args.resume_from > 0:
        records = records[args.resume_from:]
        log(f"Resuming from offset {args.resume_from}, remaining = {len(records):,}")

    # Truncate
    for r in records:
        r["abstract"] = r["abstract"][:args.max_chars]

    log(f"Loading vLLM (gemma-4-E4B-it)...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="google/gemma-4-E4B-it",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    write_mode = "a" if args.resume_from > 0 else "w"
    with OUT_PATH.open(write_mode) as out:
        n_edges = 0
        n_done = 0
        BATCH = args.batch_size
        for chunk_start in range(0, len(records), BATCH):
            chunk = records[chunk_start:chunk_start + BATCH]
            convs = [[{"role": "user", "content": IE_PROMPT.format(
                disease=r["disease"], title=r["title"], text=r["abstract"]
            )}] for r in chunk]
            outputs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outputs):
                try:
                    text = o.outputs[0].text
                except Exception:
                    text = ""
                phens = parse_phenotypes(text)
                for p in phens:
                    edge = {
                        "disease": r["disease"],
                        "umls_cui": r["cui"],
                        "phenotype": p,
                        "source": "pubmed",
                        "source_id": r["pmid"],
                        "section_name": "abstract",
                        "section_id": r["pmid"],
                        "title": r["title"],
                        "extracted_by": "gemma-4-E4B",
                        "pmid": r["pmid"],
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta_min = (len(records) - n_done) / max(rate, 0.001) / 60
            log(f"  Chunk {chunk_start+len(chunk)}/{len(records):,}  edges={n_edges:,}  rate={rate:.1f}/s  ETA={eta_min:.0f}min")

    log(f"\nPubMed IE complete. Total edges: {n_edges:,} → {OUT_PATH}")


if __name__ == "__main__":
    main()
