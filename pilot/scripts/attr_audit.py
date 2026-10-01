"""Audit attribute RECALL against the source itself: what the source states vs what the IE kept.

The production IE has two jobs at once -- enumerate every finding, and qualify each finding -- and
the finding list wins the competition for attention. To separate "the source never said it" from
"the IE did not extract it", this pass does ONLY the second job, with no exhaustiveness pressure
and no finding list to produce: it walks the source and quotes, verbatim, every clause that says
HOW something presents.

Verbatim quoting is what makes the audit checkable. Each claim is verified against the source
string before it is counted, so the audit cannot inflate recall with invented statements, and the
quotes can be read by hand. No external gold and no parametric knowledge is used -- the source is
the ground truth, which is the only reference available under the gold-free regime (CLAUDE.md).

Output: {cui: {disease, statements:[{quote, finding, attr}]}} plus a comparison against the
production predictions:
  * present-in-source AND extracted   -> the IE is working
  * present-in-source NOT extracted   -> PROMPT problem (fixable)
  * few statements present at all     -> SOURCE problem (needs a denser corpus)

Usage: python attr_audit.py --preds ie_pred_X.json --out audit.json [--n 30]
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v146_decomp import build_backend
from v153_production_ie import windows

ATTRS = ("location", "character", "severity", "onset", "timing", "aggravating", "relieving")

PROMPT = '''You are auditing a clinical text. Below is a SOURCE about "{disease}".

SOURCE:
"""{src}"""

Find EVERY place where the SOURCE says something about HOW a symptom, sign or finding PRESENTS -- not merely that it occurs. Quote each one VERBATIM from the SOURCE. Do not paraphrase, do not infer, do not add clinical knowledge of your own: if the SOURCE does not say it, it does not exist for this task.

The kinds of statement to find, and what each means:
- "location": the body site of a finding ("substernal", "left lower quadrant").
- "character": its quality ("burning", "sharp", "colicky", "pressure-like").
- "severity": how intense it is ("mild", "excruciating", "8/10").
- "onset": whether it starts suddenly or gradually.
- "timing": its temporal pattern ("nocturnal", "postprandial", "episodic", "on waking").
- "aggravating": a FACTOR that makes it worse or brings it on ("worse on exertion", "provoked by cold"). A statement that it merely got worse over time is NOT this.
- "relieving": a FACTOR that eases it ("relieved by sitting forward", "improves with antacids"). Definitive treatment of the disease is NOT this.

For each statement give: "quote" (verbatim substring of the SOURCE), "finding" (the symptom/sign it qualifies, as the SOURCE names it), "attr" (one of: location, character, severity, onset, timing, aggravating, relieving).

Output ONLY JSON: {{"statements":[{{"quote":"","finding":"","attr":""}}]}}'''

_OBJ = re.compile(r"\{[^{}]*\}")


def parse_objs(txt: str) -> list[dict]:
    out = []
    for m in _OBJ.finditer(txt):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and o.get("quote"):
            out.append(o)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30, help="diseases to audit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--win", type=int, default=8000)
    a = ap.parse_args()

    import random
    preds = json.load(open(a.preds, encoding="utf-8"))
    pool = [c for c, v in preds.items() if len(v.get("source", "") or "") > 3000 and v.get("findings")]
    random.Random(a.seed).shuffle(pool)
    sel = pool[: a.n]

    jobs, meta = [], []
    for cui in sel:
        v = preds[cui]
        for w in windows(v["source"], a.win):
            jobs.append(PROMPT.format(disease=v.get("disease", cui), src=w))
            meta.append((cui, w))
    print(f"auditing {len(sel)} diseases -> {len(jobs)} prompts", flush=True)

    chat = build_backend(a.backend, a.bs, a.max_new)
    outs = chat(jobs)

    audit: dict[str, dict] = {}
    n_claim = n_grounded = 0
    for (cui, w), txt in zip(meta, outs):
        wl = w.lower()
        rec = audit.setdefault(cui, {"disease": preds[cui].get("disease", cui), "statements": []})
        for o in parse_objs(txt):
            n_claim += 1
            q = str(o.get("quote", "")).strip()
            at = str(o.get("attr", "")).strip().lower()
            if at not in ATTRS or len(q) < 4:
                continue
            if q.lower() not in wl:          # verbatim check -> the audit cannot invent statements
                continue
            n_grounded += 1
            rec["statements"].append({"quote": q, "finding": str(o.get("finding", "")).strip(),
                                      "attr": at})
    json.dump(audit, open(a.out, "w"), indent=1)

    # ---- compare source-truth against what production actually kept --------------------------
    print(f"\nclaims {n_claim} -> verbatim-verified {n_grounded} "
          f"({100*n_grounded/max(1,n_claim):.0f}%; the rest were paraphrases and were dropped)")
    print(f"\n{'attr':14}{'in SOURCE':>11}{'IE extracted':>14}{'recall':>9}")
    tot_src = tot_ie = 0
    for at in ATTRS:
        in_src = sum(1 for r in audit.values() for s in r["statements"] if s["attr"] == at)
        got = sum(1 for c in audit for f in preds[c]["findings"]
                  if str(f.get(at, "") or "").strip())
        tot_src += in_src; tot_ie += got
        print(f"  {at:12}{in_src:11}{got:14}{(100*got/in_src if in_src else float('nan')):8.0f}%")
    print(f"  {'TOTAL':12}{tot_src:11}{tot_ie:14}{100*tot_ie/max(1,tot_src):8.0f}%")
    per = [len(r["statements"]) for r in audit.values()]
    per.sort()
    print(f"\nattribute statements the SOURCE actually contains, per disease: "
          f"median {per[len(per)//2] if per else 0}, max {per[-1] if per else 0}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
