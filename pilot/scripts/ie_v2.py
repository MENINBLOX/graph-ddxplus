"""Corpus-v2 information extraction: one pass, extractive, article-level provenance.

For every article of every disease the model lists the clinical items the text associates with the
disease, COPYING each item exactly as written. Extractive output is what makes provenance exact: an
item is kept only if its phrase is found in the article, and the sentence containing it becomes the
edge's evidence (research design RQ3). Items the model paraphrased are dropped by the same rule, for
every disease alike.

Benchmark-blind: the prompt holds the disease name and the article text, nothing else; no examples
(zero-shot), no cap on the number of items.

    bash pilot/scripts/run_vllm.sh pilot/scripts/ie_v2.py --sources DIR --out OUT.jsonl \
         [--model google/gemma-4-E4B-it] [--limit-diseases N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

PROMPT = """You extract clinical information about the disease "{disease}" from the text below.

List every item that the text associates with {disease}:
- clinical features of patients with {disease}: symptoms, physical signs, laboratory, imaging or other test findings, and complications;
- risk factors, exposures, past or coexisting conditions, family history, medications, and age or sex patterns.

Rules:
- Each item is ONE finding or factor, named by a short phrase of at most six words copied EXACTLY from the text. Never output a sentence or clause. Do not paraphrase, translate, abbreviate or merge items.
- Leave out anything the text states is absent or not associated (for example "no fever").
- Leave out treatments, therapeutic procedures and doses, but keep findings that appear inside such sentences.
- Leave out items the text attributes to a different disease.
- There is no limit on the number of items.

Output one JSON object per line and nothing else:
{{"t": "<exact phrase from the text>", "k": "<symptom|sign|test|complication|risk|history|exposure|medication|demographic>"}}

Text:
{src}"""

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def windows(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    out, i = [], 0
    while i < len(text):
        out.append((i, text[i:i + size]))
        if i + size >= len(text):
            break
        i += size - overlap
    return out


def parse(raw: str) -> list[dict]:
    items = []
    for line in raw.splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = (o.get("t") or "").strip()
        if 1 < len(t) < 200:
            items.append({"t": t, "k": (o.get("k") or "").strip().lower()})
    return items


def locate(text: str, phrase: str) -> tuple[int, int] | None:
    i = text.lower().find(phrase.lower())
    if i < 0:
        return None
    s = max(text.rfind(". ", 0, i), text.rfind("? ", 0, i), text.rfind("! ", 0, i))
    s = 0 if s < 0 else s + 2
    e = min([x for x in (text.find(". ", i), text.find("? ", i), text.find("! ", i)) if x >= 0] or [len(text)])
    return s, min(len(text), e + 1)


def build_chat(model: str, max_new: int, util: float, eager: bool = True):
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
    from vllm import LLM, SamplingParams
    kw = {"limit_mm_per_prompt": {"image": 0, "audio": 0}} if "gemma" in model.lower() else {}
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=util,
              enforce_eager=eager, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)

    def chat(prompts):
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs], sum(len(o.outputs[0].token_ids) for o in outs)
    return chat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--win", type=int, default=6000)
    ap.add_argument("--overlap", type=int, default=400)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64, help="windows per generate call")
    ap.add_argument("--util", type=float, default=float(os.environ.get("VLLM_GPU_UTIL", "0.85")))
    ap.add_argument("--limit-diseases", type=int, default=0)
    ap.add_argument("--files", default="", help="optional JSON list of source files (pilot)")
    ap.add_argument("--no-eager", action="store_true", help="allow CUDA graphs (faster decoding)")
    a = ap.parse_args()

    files = json.load(open(a.files)) if a.files else sorted(glob.glob(f"{a.sources}/*.json"))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = []
    for f in files:
        rec = json.load(open(f))
        if rec["id"] in done:
            continue
        todo.append(rec)
        if a.limit_diseases and len(todo) >= a.limit_diseases:
            break
    print(f"ie_v2: model={a.model} diseases={len(todo)} (done {len(done)})", flush=True)
    chat = build_chat(a.model, a.max_new, a.util, eager=not a.no_eager)

    # flatten to windows, process in batches, write per disease once all its windows finished
    jobs = []
    for rec in todo:
        for d in rec["docs"]:
            for off, w in windows(d["text"], a.win, a.overlap):
                jobs.append((rec["id"], rec["name"], d["doc_id"], off, w))
    pending = {rec["id"]: {"id": rec["id"], "name": rec["name"], "items": [], "n_windows": 0,
                           "n_raw": 0, "n_located": 0} for rec in todo}
    remaining = {}
    for j in jobs:
        remaining[j[0]] = remaining.get(j[0], 0) + 1
    for rid, r in pending.items():         # diseases without any text are finished immediately
        if rid not in remaining:
            with open(a.out, "a") as fo:
                fo.write(json.dumps(r) + "\n")
    t0, tok = time.time(), 0
    with open(a.out, "a") as fo:
        for b in range(0, len(jobs), a.batch):
            batch = jobs[b:b + a.batch]
            raws, nt = chat([PROMPT.format(disease=nm, src=w) for _, nm, _, _, w in batch])
            tok += nt
            for (rid, nm, doc, off, w), raw in zip(batch, raws):
                r = pending[rid]
                r["n_windows"] += 1
                for it in parse(raw):
                    r["n_raw"] += 1
                    loc = locate(w, it["t"])
                    if loc is None:
                        continue
                    r["n_located"] += 1
                    r["items"].append({"t": it["t"], "k": it["k"], "doc": doc,
                                       "sent": w[loc[0]:loc[1]].strip()[:600]})
                remaining[rid] -= 1
                if remaining[rid] == 0:
                    fo.write(json.dumps(pending.pop(rid)) + "\n")
                    fo.flush()
            el = time.time() - t0
            print(f"  windows {min(b + a.batch, len(jobs))}/{len(jobs)}  {el:.0f}s  "
                  f"out-tok/s {tok / max(el, 1e-9):.0f}", flush=True)


if __name__ == "__main__":
    main()
