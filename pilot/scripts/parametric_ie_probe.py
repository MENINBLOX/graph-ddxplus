"""VERIFY-FIRST probe: does disease-name-only EXHAUSTIVE parametric IE (LLM knowledge, no
source-grounding) recover the phenotypes that pubmed-grounded IE genuinely misses?

Background: unified eval forensic showed 94% of RareBench patient evidence is genuinely
ABSENT from the true disease's KG profile (pubmed-abstract IE, degree~6). CLAUDE.md §4-6
allow disease-name-only exhaustive LLM IE (no benchmark/mapping data). This tests whether
relaxing the self-imposed source-grounding lifts true-disease recall.

Compares, on a sample of RareBench diseases: current pubmed-KG profile recall vs
parametric-IE profile recall of the SAME patients' phenotypes. Prints delta.
"""
from __future__ import annotations
import sys, json, re, argparse
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend
from umls_link import link as umls_link

FINDING_TUI = {"T184", "T047", "T048", "T046", "T191", "T019", "T190", "T037",
               "T020", "T049", "T050", "T033", "T034"}
STRIP = {"severe", "mild", "moderate", "acute", "chronic", "recurrent", "progressive"}

PROMPT = (
    "You are a clinical knowledge expert. For the disease named below, list EXHAUSTIVELY "
    "every clinical finding (symptom, sign, phenotype) that occurs in patients with this "
    "disease, drawing on established medical knowledge. Be comprehensive: include common "
    "AND rare manifestations across ALL organ systems (neurologic, cardiac, renal, "
    "hepatic, skeletal, dermatologic, ophthalmologic, metabolic, hematologic, etc.). "
    "Strip anatomical-site and intensity modifiers off the finding name (\"abdominal "
    "pain\"->\"pain\", but keep the site in the location field). Do NOT include tests, "
    "procedures, medications, or normal findings.\n"
    "Output ONLY JSON: {\"findings\":[{\"name\":\"\",\"location\":\"\",\"severity\":\"\","
    "\"onset\":\"\",\"character\":\"\"}]}\n"
    "Disease: {disease}"
)


def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return []
    try:
        return [f for f in json.loads(m.group(0)).get("findings", []) if isinstance(f, dict)]
    except Exception:
        return []


def link_findings(findings):
    cuis = set()
    for f in findings:
        nm = f.get("name", "").strip()
        if not nm:
            continue
        nm = " ".join(w for w in nm.split() if w.lower() not in STRIP) or nm
        cid, _ = umls_link(nm, FINDING_TUI)
        if cid:
            cuis.add(cid)
    return cuis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--backend", default="vllm")
    a = ap.parse_args()

    kg = json.load(open("pilot/cache/prod/kg.json"))
    prof = defaultdict(set)
    for e in kg["edges"]:
        prof[e["d"]].add(e["p"])

    dis_map = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]
    hpo_map = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]

    def hcui(hp):
        i = hpo_map.get(hp)
        return i.get("umls_cui") if isinstance(i, dict) else i if isinstance(i, str) else None

    # gather patients per true-disease CUI (RAMEDIS)
    dz_pat = defaultdict(list)
    dz_name = {}
    for line in open("data/rarebench/data/RAMEDIS.jsonl"):
        r = json.loads(line)
        ev = {hcui(h) for h in r["Phenotype"] if hcui(h)}
        if not ev:
            continue
        for t in r["RareDisease"]:
            info = dis_map.get(t)
            if info and info.get("umls_cui"):
                cu = info["umls_cui"]
                dz_pat[cu].append(ev)
                dz_name[cu] = info.get("disease_name", t)

    # pick diseases that HAVE a KG profile (so pubmed-recall is measurable) with patients
    picks = [cu for cu in dz_pat if cu in prof][:a.n]
    print(f"probing {len(picks)} RAMEDIS diseases (disease-name-only parametric IE)", flush=True)

    chat = build_backend(a.backend, bs=len(picks), max_new=1400)
    prompts = [PROMPT.replace("{disease}", dz_name[cu]) for cu in picks]
    outs = chat(prompts)

    tot_pub = tot_par = tot_union = n_ev = 0
    for cu, out in zip(picks, outs):
        par_cuis = link_findings(parse(out))
        pub_cuis = prof.get(cu, set())
        evs = dz_pat[cu]
        for ev in evs:
            n_ev += len(ev)
            tot_pub += len(ev & pub_cuis)
            tot_par += len(ev & par_cuis)
            tot_union += len(ev & (pub_cuis | par_cuis))
        print(f"  {dz_name[cu][:38]:38} pub_prof={len(pub_cuis):3} par_prof={len(par_cuis):3} "
              f"patients={len(evs)}")

    print(f"\n=== true-disease recall on {len(picks)} diseases ===")
    print(f"  pubmed-grounded profile : {100*tot_pub/max(1,n_ev):5.1f}%")
    print(f"  parametric profile      : {100*tot_par/max(1,n_ev):5.1f}%")
    print(f"  union (pubmed ∪ param)  : {100*tot_union/max(1,n_ev):5.1f}%")


if __name__ == "__main__":
    main()
