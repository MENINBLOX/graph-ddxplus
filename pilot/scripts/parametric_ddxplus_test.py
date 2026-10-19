"""Decisive test: does disease-name-only EXHAUSTIVE parametric IE (LLM knowledge, §4-6 safe)
build COMMON-disease profiles good enough to approach the DDXPlus method-ceiling (@1 95%)?

Ceiling analysis showed our overlap scorer hits @1=95% when the KG has DDXPlus's true
disease->symptom knowledge. Our pubmed KG gets @1=31% => KG-content bottleneck. The LLM knows
common-disease symptoms well (unlike rare metabolic), so parametric IE on the 49 DDXPlus
disease names may close the gap -- WITHOUT any benchmark/mapping data (strict zero-shot).
"""
from __future__ import annotations
import sys, json, csv, ast, math
from collections import defaultdict, Counter
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend
from umls_link import link as umls_link

FINDING_TUI = {"T184", "T047", "T048", "T046", "T191", "T019", "T190", "T037",
               "T020", "T049", "T050", "T033", "T034"}
STRIP = {"severe", "mild", "moderate", "acute", "chronic", "recurrent", "progressive"}
PROMPT = (
    "You are a clinical knowledge expert. For the disease named below, list EXHAUSTIVELY every "
    "clinical finding (symptom, sign) a patient with this disease may present with, from "
    "established medical knowledge. Include the common presenting complaints AND associated "
    "findings across organ systems. Strip anatomical-site and intensity modifiers off the name. "
    "No tests, procedures, medications, or normal findings.\n"
    'Output ONLY JSON: {"findings":[{"name":""}]}\n'
    "Disease: {disease}"
)


def parse(t):
    import re
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return []
    try:
        fs = json.loads(m.group(0)).get("findings", [])
    except Exception:
        return []
    out = []
    for f in fs:
        if isinstance(f, dict):
            out.append(f.get("name", "") or f.get("finding", ""))
        elif isinstance(f, str):
            out.append(f)
    return [x for x in out if x]


def link(nm):
    nm = " ".join(w for w in nm.split() if w.lower() not in STRIP) or nm
    cid, _ = umls_link(nm, FINDING_TUI)
    return cid


def main():
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    dmap = json.load(open("data/ddxplus/disease_umls_mapping.json"))["mapping"]
    umap = json.load(open("data/ddxplus/umls_mapping.json"))["mapping"]
    fr2en = {info.get("cond-name-fr") or info.get("condition_name"): en for en, info in cond.items()}

    diseases = [(en, dmap[en]["umls_cui"]) for en in dmap if dmap[en].get("umls_cui")]
    names = [en for en, _ in diseases]
    print(f"parametric IE on {len(names)} DDXPlus diseases", flush=True)
    chat = build_backend("vllm", bs=len(names), max_new=1200)
    outs = chat([PROMPT.replace("{disease}", en) for en in names])

    profile = {}
    for (en, cu), out in zip(diseases, outs):
        cuis = {c for c in (link(n) for n in parse(out)) if c}
        profile[en] = cuis
    df = Counter()
    for en in profile:
        for c in profile[en]:
            df[c] += 1
    idf = {c: math.log(len(profile) / v) for c, v in df.items()}
    avg = sum(len(v) for v in profile.values()) / len(profile)
    print(f"avg parametric profile size: {avg:.1f}", flush=True)

    def ev_cui(e):
        for c in (e, e.split("_@_")[0], e.split("_")[0]):
            if c in umap:
                return umap[c]["cui"]
        return None

    cands = list(profile)
    n = c1 = c3 = c5 = c10 = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if n >= 3000:
                break
            en = fr2en.get(row["PATHOLOGY"])
            if en not in profile:
                continue
            ev = {c for c in (ev_cui(e) for e in ast.literal_eval(row["EVIDENCES"])) if c}
            if not ev:
                continue
            sc = {}
            for d in cands:
                p = profile[d]
                sc[d] = sum(idf.get(c, 0) for c in ev if c in p) / max(len(p) ** 0.5, 1)
            ranked = sorted(cands, key=lambda d: -sc[d])
            r = ranked.index(en) + 1
            n += 1; c1 += r == 1; c3 += r <= 3; c5 += r <= 5; c10 += r <= 10
    print(f"\n=== parametric-KG DDXPlus ({n} patients, 49-way) ===")
    print(f"  @1={100*c1/n:.1f}  @3={100*c3/n:.1f}  @5={100*c5/n:.1f}  @10={100*c10/n:.1f}")


if __name__ == "__main__":
    main()
