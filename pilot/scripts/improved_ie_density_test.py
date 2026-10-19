"""Prove IE density is raisable: re-run IE on a sample with a stronger exhaustive +
mandatory-attribute prompt and compare findings/disease and location fill vs the current KG.
(User: density is low because IE extracted too little -- verified 98% of location-less pain
findings have a location word in the source.)
"""
import sys, json, re, argparse
from collections import Counter
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend

HEAD = '''You are a meticulous clinical information extractor. Read the SOURCE about the disease "{disease}" and extract EVERY symptom/sign/phenotype a patient with this disease can have that the SOURCE mentions. Be EXHAUSTIVE -- do not stop early; list all of them, atomically.

SOURCE:
"""{src}"""

Rules:
- Strip anatomical SITE and laterality off the finding name and put them in "location" ("substernal chest pain" -> name "pain", location "substernal chest"). WHENEVER the source gives a body site for a finding, you MUST fill location -- do not leave it empty.
- Keep intensity/quality words for "character" (sharp, dull, burning, crushing, throbbing...) and fill "character" whenever stated.
- Fill "severity" (mild/moderate/severe or a 0-10 number), "onset" (sudden/gradual), "timing", "aggravating", "relieving" ONLY when the source states them for THAT finding, else "".
- EXCLUDE the disease name, causes/risk factors, epidemiology, treatments, tests, normal findings.

Output ONLY JSON: {{"findings":[{{"name":"","location":"","character":"","severity":"","onset":"","timing":"","aggravating":"","relieving":""}}]}}'''


def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return []
    try:
        return [f for f in json.loads(m.group(0)).get("findings", []) if isinstance(f, dict)]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    old = json.load(open("pilot/cache/prod/ie_pred_full.json"))
    dmap = json.load(open("data/ddxplus/disease_umls_mapping.json"))["mapping"]
    ddx = [dmap[k]["umls_cui"] for k in dmap if dmap[k].get("umls_cui")]
    # take DDXPlus disease CUIs that have a source + old prediction, plus fill up to n
    picks = [c for c in ddx if c in old and old[c].get("source")][:a.n]
    print(f"re-IE on {len(picks)} DDXPlus diseases", flush=True)

    chat = build_backend(a.backend, bs=len(picks), max_new=2000)
    prompts = [HEAD.replace("{disease}", old[c]["disease"]).replace("{src}", old[c]["source"][:4000])
               for c in picks]
    outs = chat(prompts)

    o_find = o_loc = n_find = n_loc = n_char = n_sev = 0
    for c, out in zip(picks, outs):
        of = old[c].get("findings", [])
        o_find += len(of)
        o_loc += sum(1 for f in of if str(f.get("location", "")).strip())
        nf = parse(out)
        n_find += len(nf)
        n_loc += sum(1 for f in nf if str(f.get("location", "")).strip())
        n_char += sum(1 for f in nf if str(f.get("character", "")).strip())
        n_sev += sum(1 for f in nf if str(f.get("severity", "")).strip())
    k = len(picks)
    print(f"\n=== density: current KG vs improved IE (same model, {k} diseases) ===")
    print(f"  findings/disease : {o_find/k:.1f}  ->  {n_find/k:.1f}")
    print(f"  location fill     : {100*o_loc/max(1,o_find):.0f}%  ->  {100*n_loc/max(1,n_find):.0f}%")
    print(f"  (improved) character fill {100*n_char/max(1,n_find):.0f}%  severity fill {100*n_sev/max(1,n_find):.0f}%")


if __name__ == "__main__":
    main()
