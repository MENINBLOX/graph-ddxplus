"""LLM axis of research design T: zero-shot open-set diagnosis of dated phenopacket cases.

The model sees the HPO labels of the observed features and lists 10 diagnoses. Each name is mapped
to universe keys by exact normalized string match only (universe names and synonyms, UMLS disease
strings, OMIM titles from HPO annotations), then scored with the same lineage rule as the graphs.
Unmapped names count as misses; the mapping rate is reported.

    python llm_temporal.py generate --model Qwen/Qwen2.5-7B-Instruct --out pilot/cache/v2/temporal/llm_qwen25_7b_raw.json
    python llm_temporal.py score --raw ...raw.json --out ...json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disease_link  # noqa: E402
import v2_ids  # noqa: E402
from temporal_eval import load_cases  # noqa: E402

PROMPT = ("A patient has the following phenotypic features (Human Phenotype Ontology terms):\n{feats}\n\n"
          "List the 10 most likely diagnoses, most likely first. Give specific disease names "
          "(for genetic disorders, the specific syndrome or OMIM disease name). "
          "Output exactly 10 lines, one disease name per line, with no numbering and no explanation.")
HPOA_LATEST = "data/external_kg/hpoa_history/v2026-09-01.hpoa"


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def hpo_labels() -> dict[str, str]:
    g = json.load(open("data/medkg/hp.json"))["graphs"][0]
    return {n["id"].rsplit("/", 1)[-1].replace("_", ":"): n["lbl"] for n in g["nodes"] if n.get("lbl")}


def generate(a) -> None:
    from vllm import LLM, SamplingParams
    lab = hpo_labels()
    import zipfile
    z = zipfile.ZipFile("data/external_benchmarks/phenopacket_store/all_phenopackets.zip")
    names = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}
    cases = load_cases()
    if a.limit:
        cases = cases[:: max(1, len(cases) // a.limit)][: a.limit]
    prompts = []
    for c in cases:
        raw = z.read(names[c["file"]]) if c["release"] == "0.1.27" else open(f"data/confirm/phenopacket_new/{c['file']}", "rb").read()
        p = json.loads(raw)
        feats = [lab.get(f["type"]["id"], f["type"].get("label", "")) for f in p.get("phenotypicFeatures", [])
                 if not f.get("excluded")]
        prompts.append([{"role": "user", "content": PROMPT.format(feats="\n".join(f"- {x}" for x in feats if x))}])
    extra = {"limit_mm_per_prompt": {"image": 0, "audio": 0}} if "gemma" in a.model.lower() else {}
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.88, quantization=a.quant or None,
              enable_prefix_caching=True, enforce_eager=a.eager, **extra)
    outs = llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens),
                    chat_template_kwargs={"enable_thinking": False})
    json.dump({"model": a.model, "prompt": PROMPT,
               "rows": [{"file": c["file"], "text": o.outputs[0].text} for c, o in zip(cases, outs)]},
              open(a.out, "w"))
    print("generated", len(cases), flush=True)


def name_index() -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for k, v in v2_ids.universe().items():
        forms = [v["name"]]
        syn = v.get("syn")
        if isinstance(syn, str) and syn.startswith("["):
            syn = ast.literal_eval(syn)
        forms += list(syn or [])
        for f in forms:
            idx[norm(re.sub(r"\((disorder|disease|finding)\)$", "", f))].add(k)
    for line in open(HPOA_LATEST):
        f = line.split("\t")
        if len(f) > 1 and f[0].startswith("OMIM:"):
            keys = v2_ids.gold_keys({f[0]})
            for nm in {f[1], re.sub(r";.*$", "", f[1])}:
                idx[norm(nm)] |= keys
    return idx


def to_keys(name: str, idx) -> set[str]:
    n = norm(re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", name))
    keys = set(idx.get(n, ()))
    if not keys:
        cui, _ = disease_link.link_disease(name)
        if cui:
            keys = v2_ids.gold_keys({cui})
    return keys


def score(a) -> None:
    from graph_dx import GraphDx
    raw = json.load(open(a.raw))
    idx = name_index()
    cases = {c["file"]: c for c in load_cases()}
    lin = GraphDx(a.uri, a.auth, 0.5, 2.0, 0.0, 0.0)
    rows, n_names, n_mapped = [], 0, 0
    for r in raw["rows"]:
        c = cases[r["file"]]
        preds = [x for x in (ln.strip() for ln in r["text"].splitlines()) if x][:10]
        keys = [to_keys(p, idx) for p in preds]
        n_names += len(preds)
        n_mapped += sum(bool(k) for k in keys)
        ok = lin.lineage(v2_ids.gold_keys(c["gold"]))
        pos = next((i + 1 for i, k in enumerate(keys) if k & ok), None)
        rows.append({"file": c["file"], "release": c["release"], "pmid": c["pmid"], "omim": c["omim"],
                     "date": c["date"].isoformat(), "rank": pos})
    lin.drv.close()
    n = len(rows)
    json.dump({"model": raw["model"], "map_rate": n_mapped / max(1, n_names), "rows": rows}, open(a.out, "w"))
    print(f"{raw['model']}: n={n} @1 {100 * sum(r['rank'] == 1 for r in rows) / n:.1f} "
          f"@10 {100 * sum(r['rank'] is not None for r in rows) / n:.1f} name map rate {n_mapped / max(1, n_names):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generate", "score"])
    ap.add_argument("--model", default="")
    ap.add_argument("--quant", default="")
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--raw", default="")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (generate if a.step == "generate" else score)(a)


if __name__ == "__main__":
    main()
