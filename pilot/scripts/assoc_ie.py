"""Extract the NON-SYMPTOM evidence a diagnosis rests on: history, exposure, risk, comorbidity.

CLAUDE.md §11 says patient evidence is one pool -- a history of myocardial infarction counts exactly
as a report of pain -- but the production IE prompt says the opposite: "EXCLUDE causes/risk factors,
epidemiology, treatments/drugs, diagnostic procedures". The KG therefore cannot hold what a large
share of DDXPlus asks about, and the numbers show it exactly:

    evidence type                      instances   recall
    T184 sign/symptom                      9,230    78.4%   <- the symptom IE works
    T033 finding (travel history, ...)     5,232    16.1%
    T047 disease (past history)            5,811    26.8%
    T061/T121/T054/T037/T031/... (9 types) 2,916     0.0%   <- excluded by prompt design

"Travel History" (C0489542) is the single most-missed evidence CUI in almost every DDXPlus disease.
This pass extracts that class from the same sources, as free text, and is therefore no different in
kind from the symptom IE -- it is not a disease-symptom mapping table (§3), and the query is the
disease name only (§5).

Findings are never touched: this writes a separate `associations` list, so the symptom graph is
unchanged and the two can be evaluated separately.

Usage: python assoc_ie.py --preds IN.json --out OUT.json [--n 0]
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v146_decomp import build_backend
from v153_production_ie import windows
from v145_strategy_scope import in_src

PROMPT = '''You are a meticulous clinical information extractor. Read the SOURCE about "{disease}".

SOURCE:
"""{src}"""

Extract everything the SOURCE states about WHO GETS this disease and WHAT PRECEDES it -- the facts a clinician would elicit from the history rather than observe as a symptom. Use ONLY what the SOURCE explicitly states.

Report each as one item with:
- "item": the thing itself, named as briefly as the SOURCE allows ("recent travel", "smoking", "immunosuppression", "prior myocardial infarction", "anticoagulant use", "crowded living conditions", "raw milk consumption").
- "kind": one of exposure | risk_factor | past_disease | medication | procedure | family_history | occupation | habit | demographic.

Rules:
- Include exposures and contacts, habits, occupations, prior or coexisting diseases, medications and procedures the patient had BEFORE this illness, and inherited or family predisposition.
- EXCLUDE the symptoms and signs of this disease -- those are handled elsewhere. Also exclude its treatment, its complications, and its diagnostic tests.
- Do not infer. If the SOURCE does not state it, leave it out.
- Name the factor, not a sentence: "recent travel to an endemic area", not "patients who have recently travelled to an endemic area may present with".

Output ONLY JSON: {{"associations":[{{"item":"","kind":""}}]}}'''

_OBJ = re.compile(r"\{[^{}]*\}")
KINDS = {"exposure", "risk_factor", "past_disease", "medication", "procedure",
         "family_history", "occupation", "habit", "demographic"}
_SENTENCE = re.compile(r"\b(patients?|may|can|should|often|usually|is|are|were|has|have)\b", re.I)


def parse_objs(txt: str) -> list[dict]:
    out = []
    for m in _OBJ.finditer(txt):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and o.get("item"):
            out.append(o)
    return out


def keep(item: str) -> bool:
    """An association is a named factor, not a sentence about one."""
    it = item.strip().lower()
    return 2 <= len(it.split()) <= 7 or (len(it.split()) == 1 and len(it) > 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--win", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    preds = json.load(open(a.preds, encoding="utf-8"))
    if a.resume and os.path.exists(a.out):
        done = json.load(open(a.out, encoding="utf-8"))
        for c, v in done.items():
            if c in preds and v.get("_assoc"):
                preds[c] = v
        print(f"resume: {sum(1 for v in preds.values() if v.get('_assoc'))} done", flush=True)
    todo = [c for c, v in preds.items() if not v.get("_assoc") and v.get("source")]
    if a.n:
        todo = todo[: a.n]
    print(f"association IE: {len(todo)} diseases", flush=True)

    def save():
        tmp = a.out + ".tmp"
        json.dump(preds, open(tmp, "w"))
        os.replace(tmp, a.out)

    chat = build_backend(a.backend, a.bs, a.max_new)
    n_a = 0
    for ci in range(0, len(todo), a.chunk):
        batch = todo[ci:ci + a.chunk]
        jobs, meta = [], []
        for cui in batch:
            v = preds[cui]
            for w in windows(v["source"], a.win):
                jobs.append(PROMPT.format(disease=v.get("disease", cui), src=w))
                meta.append((cui, w))
        outs = chat(jobs)
        for (cui, w), txt in zip(meta, outs):
            wl = w.lower()
            bag = preds[cui].setdefault("associations", [])
            seen = {x["item"] for x in bag}
            for o in parse_objs(txt):
                it = str(o.get("item", "")).strip().lower()
                kind = str(o.get("kind", "")).strip().lower()
                if not it or it in seen or kind not in KINDS:
                    continue
                if not keep(it) or _SENTENCE.search(it) or not in_src(it, wl):
                    continue
                bag.append({"item": it, "kind": kind})
                seen.add(it)
                n_a += 1
        for cui in batch:
            preds[cui]["_assoc"] = True
        save()
        print(f"  chunk {ci//a.chunk + 1}/{(len(todo)+a.chunk-1)//a.chunk}: {len(jobs)} prompts, "
              f"associations so far {n_a}", flush=True)

    import collections
    kinds = collections.Counter(x["kind"] for v in preds.values() for x in v.get("associations", []))
    n_d = sum(1 for v in preds.values() if v.get("associations"))
    print(f"\nassociations {n_a} over {n_d} diseases ({n_a/max(1,n_d):.1f}/disease)")
    for k, c in kinds.most_common():
        print(f"  {k:16} {c}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
