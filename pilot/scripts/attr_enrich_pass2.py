"""Pass 2: attribute enrichment over an ALREADY-extracted finding list.

A single prompt cannot do both jobs well. Measured on the same 28 case-report sources, moving the
neglected attributes into their own prompt rules raised aggravating 3 -> 16 extractions and
location 34% -> 42%, but cost 12% of the findings (1274 -> 1125) and halved timing (7% -> 4%):
attention spent on attributes is taken from exhaustiveness.

So the two jobs are split. Pass 1 (v153) stays exactly as validated -- exhaustive finding list.
Pass 2 re-reads the SAME source with pass 1's finding names supplied, and only asks: for each of
these findings, what does the source say about how it presents? Nothing is extracted here that
pass 1 did not already find, so pass 2 cannot inflate the finding set; it can only fill attribute
cells. Source-grounded (values must appear in the source) and benchmark-blind -- the finding list
comes from our own pass 1, never from a benchmark.

Attribute precision is NOT assumed: clinical prose states an aggravating factor as "dyspnea on
exertion" far more often than "aggravated by exertion", and a naive keyword audit both over-counts
(course words: "worsening dyspnea") and under-counts (no explicit trigger verb). So the prompt
names the distinction explicitly and the output is audited qualitatively, not by regex.

Usage: python attr_enrich_pass2.py --preds ie_pred_X.json --out ie_pred_X_enriched.json
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v146_decomp import build_backend
from v153_production_ie import ATTRS, windows
from v145_strategy_scope import in_src

PROMPT = '''You are a meticulous clinical information extractor. Below is a SOURCE about "{disease}" and a LIST OF FINDINGS already extracted from it. Do NOT add or remove findings. For each finding, report ONLY what the SOURCE explicitly says about HOW that finding presents.

SOURCE:
"""{src}"""

FINDINGS: {names}

For each finding, fill these fields, copying the SOURCE's own words. Leave "" when the SOURCE does not state it for THAT finding.
- "location": the anatomical site.
- "character": the quality (sharp, dull, burning, crushing, throbbing, colicky, cramping, pressure-like, tearing...).
- "severity": mild/moderate/severe, or a 0-10 rating.
- "onset": sudden or gradual.
- "timing": the temporal pattern (nocturnal, postprandial, episodic, intermittent, exertional, morning, periodic...).
- "aggravating": what makes the finding worse, brings it on, triggers or precipitates it -- e.g. exertion, lying down, a meal, coughing, deep breathing, movement, standing, heat, stress, menstruation, fasting. IMPORTANT: this means a FACTOR OR CIRCUMSTANCE, not the mere fact that the finding got worse over time. "progressively worsening dyspnea" states no aggravating factor and must give "". "dyspnea on exertion" states one: "exertion".
- "relieving": what the patient can do or take that eases or aborts the finding -- e.g. rest, sitting up, leaning forward, a position change, antacids, nitroglycerin, an analgesic, food, sleep. This must be a FACTOR: "the swelling improved slightly" states none; "pain relieved by sitting forward" states "sitting forward". CRITICAL: definitive treatment of the DISEASE is not a relieving factor. If the source says the finding improved after antibiotics, chemotherapy, surgery, transfusion, dialysis, oxygen, or "after starting treatment", give "" -- that is the treatment course, not how the symptom behaves.

Output ONLY JSON: {{"findings":[{{"name":"<copy from FINDINGS>","location":"","character":"","severity":"","onset":"","timing":"","aggravating":"","relieving":""}}]}}'''

_OBJ = re.compile(r"\{[^{}]*\}")


def parse_objs(txt: str) -> list[dict]:
    """Collect every complete finding object, so a response cut at max_new still contributes."""
    out = []
    for m in _OBJ.finditer(txt):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and str(o.get("name", "")).strip():
            out.append(o)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pass-1 v153 predictions to enrich")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=5120)
    ap.add_argument("--win", type=int, default=8000)
    ap.add_argument("--max-names", type=int, default=60,
                    help="findings per pass-2 prompt (a long list crowds out the answer)")
    ap.add_argument("--chunk", type=int, default=200, help="diseases per checkpoint")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: diseases already marked enriched in --out "
                         "are skipped (a full-corpus pass is long and must survive a dead shell)")
    a = ap.parse_args()

    preds = json.load(open(a.preds, encoding="utf-8"))
    if a.resume and os.path.exists(a.out):
        done = json.load(open(a.out, encoding="utf-8"))
        for cui, v in done.items():
            if cui in preds and v.get("_pass2"):
                preds[cui] = v
        print(f"resume: {sum(1 for v in preds.values() if v.get('_pass2'))} diseases already enriched",
              flush=True)
    todo = [c for c, v in preds.items()
            if not v.get("_pass2") and v.get("source") and v.get("findings")]
    print(f"pass 2: {len(todo)} diseases to enrich (of {len(preds)})", flush=True)

    def save():
        tmp = a.out + ".tmp"
        json.dump(preds, open(tmp, "w"))
        os.replace(tmp, a.out)

    chat = build_backend(a.backend, a.bs, a.max_new)
    filled = {k: 0 for k in ATTRS}
    for ci in range(0, len(todo), a.chunk):
        batch_cuis = todo[ci:ci + a.chunk]
        jobs, meta = [], []
        for cui in batch_cuis:
            v = preds[cui]
            names = [f["name"] for f in v["findings"]]
            for w in windows(v["source"], a.win):
                for i in range(0, len(names), a.max_names):
                    jobs.append(PROMPT.format(disease=v.get("disease", cui), src=w,
                                              names="; ".join(names[i:i + a.max_names])))
                    meta.append((cui, w))
        outs = chat(jobs)
        # merge: fill EMPTY attribute cells of pass-1 findings only; never add a finding
        for (cui, w), txt in zip(meta, outs):
            by_name = {f["name"]: f for f in preds[cui]["findings"]}
            wl = w.lower()
            for o in parse_objs(txt):
                tgt = by_name.get(str(o.get("name", "")).strip().lower())
                if tgt is None:
                    continue
                for k in ATTRS:
                    val = str(o.get(k, "") or "").strip().lower()
                    if not val or str(tgt.get(k, "") or "").strip():
                        continue
                    if k == "onset":
                        if val not in ("sudden", "gradual"):
                            continue
                    elif not in_src(val, wl):
                        continue                  # keep only wording present in the source
                    tgt[k] = val
                    filled[k] += 1
        for cui in batch_cuis:
            preds[cui]["_pass2"] = True
        save()
        print(f"  chunk {ci//a.chunk + 1}/{(len(todo)+a.chunk-1)//a.chunk}: {len(jobs)} prompts, "
              f"cells added so far {sum(filled.values())}", flush=True)

    n = sum(len(v.get("findings", [])) for v in preds.values())
    have = {k: sum(1 for v in preds.values() for f in v["findings"]
                   if str(f.get(k, "") or "").strip()) for k in ATTRS}
    print(f"\nfindings unchanged: {n}")
    print("attribute fill after pass 2 (cells added by pass 2 in parentheses):")
    for k in ATTRS:
        print(f"  {k:12} {have[k]:6}/{n} ({100*have[k]/max(1,n):3.0f}%)   +{filled[k]}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
