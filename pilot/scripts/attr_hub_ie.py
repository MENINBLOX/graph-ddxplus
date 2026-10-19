"""Qualify the findings that actually need qualifying: the HUB symptoms.

Measured on the production KG, the findings a patient reports have a median document frequency of
2,249 (shared with thousands of diseases -- pain, fever, weakness), while the edges our IE had
qualified have a median df of 87. The extraction was spending its attributes on rare,
disease-specific findings and leaving the hub symptoms bare, which is why 21,361 Clinical-course
values in the KG and 940 on the patient side never once landed on the same (disease, finding).

Targeting hubs is also what the scoring says. A high-df finding contributes almost no IDF on its
own; whether it discriminates depends entirely on HOW it presents. "Pain" is worthless, "pain,
substernal, pressure-like, worse on exertion" is nearly diagnostic. So the attributes are worth
most exactly where we were not extracting them.

The rule is benchmark-blind: the target set is chosen from the KG's own df statistic, never from
any benchmark's question list (CLAUDE.md §5).

Usage: python attr_hub_ie.py --kg KG.json --preds IN.json --out OUT.json [--topk 12]
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v146_decomp import build_backend
from v153_production_ie import windows
from v145_strategy_scope import in_src
from attr_open_ie import is_qualifier

PROMPT = '''You are a meticulous clinical information extractor. The SOURCE below describes "{disease}".

SOURCE:
"""{src}"""

These findings are COMMON to many diseases, so the fact that they occur says little; what distinguishes this disease is HOW they present in it:

{names}

For each of those findings, and ONLY those, report everything the SOURCE says about the way it presents in this disease -- its site, its quality, how bad it is, its time course or pattern, its distribution or laterality, what brings it on or makes it worse, what makes it better, or anything else the SOURCE states about its form.

Rules:
- "finding": copy it exactly from the list above. Ignore findings not on the list.
- "quote": the qualifying words, VERBATIM from the SOURCE. Never paraphrase, never infer.
- "relation": your own short label for what the quote says (site, quality, intensity, time course, distribution, what makes it worse, what makes it better, ...).
- If the SOURCE says nothing about how a finding presents, return nothing for it. Do not invent.
- A statement that the finding worsened over time is a time course, not a factor that worsens it.
- Do not report the disease's treatment, the response to treatment, or a post-treatment state.

Output ONLY JSON: {{"qualifiers":[{{"finding":"","quote":"","relation":""}}]}}'''

_OBJ = re.compile(r"\{[^{}]*\}")


def parse_objs(txt: str) -> list[dict]:
    out = []
    for m in _OBJ.finditer(txt):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and o.get("quote") and o.get("finding"):
            out.append(o)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, help="supplies the df statistic used to pick hubs")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=12, help="hub findings qualified per disease")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--win", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    kg = json.load(open(a.kg, encoding="utf-8"))
    df = {p: i.get("df", 0) for p, i in kg["phenotypes"].items()}
    name2cui = {}
    for p, i in kg["phenotypes"].items():
        n = (i.get("name") or "").strip().lower()
        if n:
            name2cui.setdefault(n, p)

    preds = json.load(open(a.preds, encoding="utf-8"))
    if a.resume and os.path.exists(a.out):
        done = json.load(open(a.out, encoding="utf-8"))
        for c, v in done.items():
            if c in preds and v.get("_hub"):
                preds[c] = v
        print(f"resume: {sum(1 for v in preds.values() if v.get('_hub'))} done", flush=True)

    def hubs(v):
        """The disease's findings that are shared with the most other diseases."""
        scored = []
        for f in v.get("findings", []):
            cui = name2cui.get(f["name"].strip().lower())
            if cui:
                scored.append((df.get(cui, 0), f["name"]))
        scored.sort(reverse=True)
        return [n for _, n in scored[: a.topk]]

    todo = [c for c, v in preds.items()
            if not v.get("_hub") and v.get("source") and hubs(v)]
    if a.n:
        todo = todo[: a.n]
    print(f"hub-targeted attribute IE: {len(todo)} diseases, top-{a.topk} hub findings each",
          flush=True)

    def save():
        tmp = a.out + ".tmp"
        json.dump(preds, open(tmp, "w"))
        os.replace(tmp, a.out)

    chat = build_backend(a.backend, a.bs, a.max_new)
    n_q = 0
    for ci in range(0, len(todo), a.chunk):
        batch = todo[ci:ci + a.chunk]
        jobs, meta = [], []
        for cui in batch:
            v = preds[cui]
            hs = hubs(v)
            for w in windows(v["source"], a.win):
                jobs.append(PROMPT.format(disease=v.get("disease", cui), src=w,
                                          names="\n".join("- " + h for h in hs)))
                meta.append((cui, w))
        outs = chat(jobs)
        for (cui, w), txt in zip(meta, outs):
            wl = w.lower()
            by_name = {f["name"]: f for f in preds[cui]["findings"]}
            for o in parse_objs(txt):
                tgt = by_name.get(str(o.get("finding", "")).strip().lower())
                if tgt is None:
                    continue
                q = str(o.get("quote", "")).strip().lower()
                if len(q) < 3 or not in_src(q, wl) or not is_qualifier(q, tgt["name"]):
                    continue
                bag = tgt.setdefault("qualifiers", [])
                if not any(x["quote"] == q for x in bag):
                    bag.append({"quote": q, "relation": str(o.get("relation", "")).strip().lower()})
                    n_q += 1
        for cui in batch:
            preds[cui]["_hub"] = True
        save()
        print(f"  chunk {ci//a.chunk + 1}/{(len(todo)+a.chunk-1)//a.chunk}: {len(jobs)} prompts, "
              f"qualifiers so far {n_q}", flush=True)

    # did the attributes land where patients actually look? report the df of what we qualified
    qdf = []
    for c, v in preds.items():
        for f in v.get("findings", []):
            if f.get("qualifiers"):
                cui = name2cui.get(f["name"].strip().lower())
                if cui:
                    qdf.append(df.get(cui, 0))
    qdf.sort()
    print(f"\nqualifiers added: {n_q}")
    if qdf:
        print(f"median df of qualified findings: {qdf[len(qdf)//2]} "
              f"(patients report findings with median df 2249; the previous pass qualified df 87)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
