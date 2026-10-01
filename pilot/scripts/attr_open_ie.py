"""Open-schema attribute IE: extract every qualifier the source attaches to a finding.

The previous pass filled a fixed table of seven slots. Two things went wrong with that. A reviewer
can ask why those seven, and -- measured -- a qualifier the source states but the table has no
column for is simply lost: the audit found 519 verbatim attribute statements across 30 diseases of
which the slot schema could consume 144.

Here the prompt names NO vocabulary and NO slot list. It asks only for the qualifier text, verbatim,
plus the finding it qualifies and a free-form relation label the model may write in its own words.
The controlled vocabulary lives downstream in `hpo_modifier` (HPO Clinical modifier, 347 terms /
16 families), so:
  * the prompt stays short -- adding attribute RULES alone already cost 12% of the findings, and a
    347-term list would cost far more;
  * the vocabulary can change without re-running a 20+ hour extraction (re-normalizing is minutes);
  * "no pre-specified scope" is real rather than nominal.

Whatever fails to normalize is KEPT as free text with its quote, and the normalization rate is
reported -- an attribute we cannot standardize is a limitation to state, not something to hide.

Usage: python attr_open_ie.py --preds IN.json --out OUT.json [--n 0 = all]
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v146_decomp import build_backend
from v153_production_ie import windows
from v145_strategy_scope import in_src

PROMPT = '''You are a meticulous clinical information extractor. Below is a SOURCE about "{disease}" and a list of FINDINGS already extracted from it.

SOURCE:
"""{src}"""

FINDINGS: {names}

For each finding, report every QUALIFIER the SOURCE attaches to it -- anything the SOURCE says about how that finding presents, rather than merely that it occurs. Do not limit yourself to any fixed set of qualifier types: report whatever the SOURCE actually states.

Rules:
- "finding": copy it from the FINDINGS list. Do not invent findings.
- "quote": the qualifying words, copied VERBATIM from the SOURCE. Never paraphrase.
- "relation": your own short label for what the quote says about the finding (for example: site, quality, intensity, time course, what makes it worse, what makes it better, distribution, position, trigger, duration -- or any other label that fits).
- Report a qualifier only if the SOURCE states it for THAT finding. If the SOURCE says a finding simply got worse over time, that is a time course, not a factor that worsens it.
- Do not report the treatment of the disease, the patient's response to treatment, or a post-treatment state.

Output ONLY JSON: {{"qualifiers":[{{"finding":"","quote":"","relation":""}}]}}'''

_OBJ = re.compile(r"\{[^{}]*\}")

# Removing the slot list removes the model's guard rail too: it starts returning findings and whole
# clauses ("adults present with melena", "acts as a lead point for intussusception") instead of
# qualifiers. These are rejected structurally rather than by re-introducing a vocabulary -- a
# qualifier is a short phrase that modifies the finding, so it is not a clause, not narrative, and
# not the finding itself.
_CLAUSE = re.compile(r"\b(present\w*|occur\w*|cause[sd]?|causing|result\w*|lead\w*|include[sd]?|"
                     r"associate\w*|report\w*|describe[sd]?|note[sd]?|suggest\w*|indicat\w*|"
                     r"may be|can be|is a|are a|acts? as)\b", re.I)
_VAGUE = re.compile(r"^(a |an |the )?(variety|number|range|spectrum|form|type|kind|example)s?\b", re.I)
MAX_WORDS = 8


# The model labels each extraction with the relation it thinks it found, and that label is a
# reliable type signal: audited on 197,885 extractions, the ones labelled `type` (33,421, the
# single largest group) are subtypes or synonym glosses -- "solid tumors of the skin = melanoma" --
# and `prevalence` / `description` / `presence` / `manifestation` are epidemiology or narrative.
# Keeping only relations that name a way of presenting raises precision without adding any value
# vocabulary back into the prompt.
_QUALIFIER_RELATION = re.compile(
    r"(quality|character|nature|intensity|severity|degree|distribution|location|site|position|"
    r"laterality|radiat|time ?course|timing|tempo|frequency|duration|onset|course|pattern|"
    r"progress|trigger|aggravat|precipitat|provok|reliev|ameliorat|worse|better|exacerbat|"
    r"appearance|morpholog|consistency|colou?r|size|shape|extent)", re.I)


def is_qualifier(quote: str, finding: str, relation: str = "") -> bool:
    """A qualifier modifies the finding; it is not the finding, a clause, or narrative."""
    q = quote.strip().lower()
    f = finding.strip().lower()
    if not q or len(q.split()) > MAX_WORDS:
        return False
    if q == f or f in q or q in f:
        return False                      # restating the finding carries no extra information
    if _CLAUSE.search(q) or _VAGUE.match(q):
        return False
    if relation and not _QUALIFIER_RELATION.search(relation):
        return False                      # the model itself called this something other than a qualifier
    return True


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
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all diseases")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=5120)
    ap.add_argument("--win", type=int, default=8000)
    ap.add_argument("--max-names", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=300, help="diseases per checkpoint")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    preds = json.load(open(a.preds, encoding="utf-8"))
    if a.resume and os.path.exists(a.out):
        done = json.load(open(a.out, encoding="utf-8"))
        for cui, v in done.items():
            if cui in preds and v.get("_open"):
                preds[cui] = v
        print(f"resume: {sum(1 for v in preds.values() if v.get('_open'))} done", flush=True)
    todo = [c for c, v in preds.items()
            if not v.get("_open") and v.get("source") and v.get("findings")]
    if a.n:
        todo = todo[: a.n]
    print(f"open-schema attribute IE: {len(todo)} diseases", flush=True)

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
            names = [f["name"] for f in v["findings"]]
            for w in windows(v["source"], a.win):
                for i in range(0, len(names), a.max_names):
                    jobs.append(PROMPT.format(disease=v.get("disease", cui), src=w,
                                              names="; ".join(names[i:i + a.max_names])))
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
                if len(q) < 3 or not in_src(q, wl):
                    continue                      # keep only wording present in the source
                if not is_qualifier(q, tgt["name"]):
                    continue                      # a finding or a clause, not a qualifier
                rel = str(o.get("relation", "")).strip().lower()
                bag = tgt.setdefault("qualifiers", [])
                if not any(x["quote"] == q for x in bag):
                    bag.append({"quote": q, "relation": rel})
                    n_q += 1
        for cui in batch:
            preds[cui]["_open"] = True
        save()
        print(f"  chunk {ci//a.chunk + 1}/{(len(todo)+a.chunk-1)//a.chunk}: "
              f"{len(jobs)} prompts, qualifiers so far {n_q}", flush=True)

    # normalization report: how much of the open extraction reaches a standard concept
    import hpo_modifier
    voc = hpo_modifier.load()
    import collections
    fam = collections.Counter()
    nq = norm = 0
    for v in preds.values():
        for f in v.get("findings", []):
            for x in f.get("qualifiers", []):
                nq += 1
                hs = voc.classes(x["quote"], x.get("relation", ""))
                if hs:
                    norm += 1
                    for h in hs:
                        fam[voc.label.get(voc.family(h), "?")] += 1
    print(f"\nqualifiers extracted: {nq}   normalized to an HPO clinical modifier: "
          f"{norm} ({100*norm/max(1,nq):.0f}%)")
    print("families found (this IS the attribute schema, discovered rather than pre-declared):")
    for f, n in fam.most_common(20):
        print(f"  {f:34} {n}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
