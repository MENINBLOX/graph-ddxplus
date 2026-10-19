"""Validate the biochemical-IE lever: re-extract RAMEDIS/LIRICAL disease phenotypes with a
schema that explicitly admits laboratory/biochemical diagnostic findings, and measure whether
exact-match coverage of patient evidence rises (the sole predictor of Recall@1 per the forensic).
Content is already in the ingested source text -> strict-zero-shot safe.
"""
import sys, json, re, argparse
from collections import defaultdict, Counter
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend
from umls_link import link as umls_link

FINDING_TUI = {"T184", "T047", "T048", "T046", "T191", "T019", "T190", "T037",
               "T020", "T049", "T050", "T033", "T034"}

PROMPT = '''You are a meticulous clinical information extractor. Read the SOURCE about the disease "{disease}" and extract EVERY finding a patient with this disease can have that the SOURCE states, atomically and EXHAUSTIVELY.

SOURCE:
"""{src}"""

Extract BOTH:
(1) clinical symptoms and signs (e.g., seizures, intellectual disability, hepatomegaly), AND
(2) LABORATORY and BIOCHEMICAL abnormalities that are diagnostic of the disease — elevated or decreased metabolites, amino acids, organic acids, hormones, or enzymes; enzyme deficiencies; and "-emia"/"-uria" findings (e.g., "hyperphenylalaninemia", "elevated 17-hydroxyprogesterone", "N-acetylaspartic acid accumulation", "elevated urinary organic acids"). These biochemical markers are often the PATHOGNOMONIC finding; do NOT omit them in favor of clinical symptoms only.

Use only facts stated in the SOURCE. Strip anatomical site/laterality/intensity modifiers off the finding name. Exclude the disease name itself, causes, epidemiology, treatments, and normal findings.
Output ONLY JSON: {{"findings":["...","..."]}}'''


def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return []
    try:
        fs = json.loads(m.group(0)).get("findings", [])
    except Exception:
        return []
    return [f if isinstance(f, str) else f.get("name", "") for f in fs if f]


def link(nm):
    nm = re.sub(r"\b(severe|mild|moderate|acute|chronic|recurrent|elevated|increased|decreased|high|low)\b",
               "", nm, flags=re.I).strip()
    return umls_link(nm, FINDING_TUI)[0] if nm else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backend", default="vllm")
    a = ap.parse_args()

    old = json.load(open("pilot/cache/prod/ie_pred_full.json"))
    kg = json.load(open("pilot/cache/prod/kg.json"))
    prof = defaultdict(set)
    for e in kg["edges"]:
        prof[e["d"]].add(e["p"])
    dmap = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]
    hpo = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]

    def hcui(h):
        i = hpo.get(h)
        return i.get("umls_cui") if isinstance(i, dict) else i if isinstance(i, str) else None

    # RAMEDIS patients grouped by true disease CUI
    dz_pat = defaultdict(list)
    for line in open("data/rarebench/data/RAMEDIS.jsonl"):
        r = json.loads(line)
        ev = {hcui(h) for h in r["Phenotype"] if hcui(h)}
        for t in r["RareDisease"]:
            info = dmap.get(t)
            if info and info.get("umls_cui") and ev:
                dz_pat[info["umls_cui"]].append(ev)

    picks = [cu for cu in dz_pat if cu in old and old[cu].get("source")][:a.n]
    print(f"biochem re-IE on {len(picks)} RAMEDIS diseases", flush=True)
    chat = build_backend(a.backend, bs=len(picks), max_new=1500)
    outs = chat([PROMPT.replace("{disease}", old[c]["disease"]).replace("{src}", old[c]["source"][:4000])
                 for c in picks])

    old_cov = new_cov = union_cov = tot = 0
    for c, out in zip(picks, outs):
        new_cuis = {x for x in (link(n) for n in parse(out)) if x}
        old_cuis = prof.get(c, set())
        for ev in dz_pat[c]:
            tot += len(ev)
            old_cov += len(ev & old_cuis)
            new_cov += len(ev & new_cuis)
            union_cov += len(ev & (old_cuis | new_cuis))
    print(f"\n=== RAMEDIS exact-match coverage of patient evidence ({len(picks)} diseases) ===")
    print(f"  old profile        : {100*old_cov/max(1,tot):.1f}%")
    print(f"  biochem re-IE alone: {100*new_cov/max(1,tot):.1f}%")
    print(f"  union (old ∪ new)  : {100*union_cov/max(1,tot):.1f}%")


if __name__ == "__main__":
    main()
