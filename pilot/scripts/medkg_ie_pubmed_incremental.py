#!/usr/bin/env python3
"""Incremental PubMed IE: process only CUIs that are crawled but not yet IE'd.

Reads $MEDKG_ROOT/processed/edges_pubmed_ie.jsonl to determine processed CUIs,
then runs IE on the remaining crawled CUIs and APPENDS to the same file.

Same prompt and model as medkg_ie_pubmed.py.
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_pubmed import IE_PROMPT, parse_phenotypes, log

PUBMED_DIR = MEDKG_ROOT / "pubmed"
OUT_PATH = MEDKG_ROOT / "processed" / "edges_pubmed_ie.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_processed_cuis():
    if not OUT_PATH.exists(): return set()
    out = set()
    with OUT_PATH.open() as f:
        for line in f:
            try: out.add(json.loads(line).get("umls_cui", ""))
            except Exception: pass
    return out


def load_abstracts_for(cuis, limit_per_cui=20, min_chars=50):
    records = []
    for cui in sorted(cuis):
        fp = PUBMED_DIR / f"{cui}.jsonl"
        if not fp.exists() or fp.stat().st_size == 0: continue
        with fp.open() as f:
            n_kept = 0
            for line in f:
                line = line.strip()
                if not line: continue
                try: e = json.loads(line)
                except: continue
                ab = (e.get("abstract") or "").strip()
                if len(ab) < min_chars: continue
                records.append({
                    "disease": e.get("disease_name", ""),
                    "cui": e.get("cui", cui),
                    "pmid": e.get("pmid", ""),
                    "title": e.get("title", ""),
                    "abstract": ab,
                })
                n_kept += 1
                if n_kept >= limit_per_cui: break
    return records


def main():
    t0 = time.time()
    processed = load_processed_cuis()
    log(f"Processed CUIs: {len(processed):,}")
    crawled = {p.stem for p in PUBMED_DIR.glob("*.jsonl") if p.stat().st_size > 0}
    log(f"Crawled non-empty CUIs: {len(crawled):,}")
    new_cuis = crawled - processed
    log(f"NEW CUIs to IE: {len(new_cuis):,}")
    if not new_cuis:
        log("Nothing to do."); return

    records = load_abstracts_for(new_cuis)
    log(f"Loaded {len(records):,} abstracts")
    for r in records: r["abstract"] = r["abstract"][:2500]

    log(f"Loading vLLM (gemma-4-E4B-it)...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=350)

    BATCH = 2048
    n_edges = 0
    n_done = 0
    with OUT_PATH.open("a") as out:
        for chunk_start in range(0, len(records), BATCH):
            chunk = records[chunk_start:chunk_start + BATCH]
            convs = [[{"role": "user", "content": IE_PROMPT.format(
                disease=r["disease"], title=r["title"], text=r["abstract"]
            )}] for r in chunk]
            outputs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outputs):
                try: text = o.outputs[0].text
                except: text = ""
                phens = parse_phenotypes(text)
                for p in phens:
                    edge = {"disease": r["disease"], "umls_cui": r["cui"],
                            "phenotype": p, "source": "pubmed",
                            "source_id": r["pmid"], "section_name": "abstract",
                            "section_id": r["pmid"], "title": r["title"],
                            "extracted_by": "gemma-4-E4B", "pmid": r["pmid"]}
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta_min = (len(records) - n_done) / max(rate, 0.001) / 60
            log(f"  Chunk {chunk_start+len(chunk)}/{len(records):,}  edges={n_edges:,}  rate={rate:.1f}/s  ETA={eta_min:.0f}min")
    log(f"Done. New edges: {n_edges:,}")


if __name__ == "__main__":
    main()
