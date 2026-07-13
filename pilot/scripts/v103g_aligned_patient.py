"""Vocab-alignment test: map the PATIENT evidence with the SAME scispaCy linker
used for the KG, so patient and KG phenotype CUIs come from one pipeline (instead
of patient=value_cuis hand map vs KG=scispaCy → CUI mismatch like
iliac-fossa C0278463 vs groin C0018834).

Render each patient's evidences to English (question_en + value English), run
scispaCy NER+UMLS linker → patient CUIs. Match against the scispaCy KG. Compare
@1 to the value_cuis baseline (33.62).
"""
import sys, json, csv, ast, math, pickle
from collections import defaultdict
import spacy
from scispacy.linking import EntityLinker


def render(evs, evmeta):
    parts = []
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            m = evmeta.get(base, {})
            vm = m.get("value_meaning", {}).get(val, {})
            en = vm.get("en", "") if isinstance(vm, dict) else ""
            q = m.get("question_en", "")
            if en and en != "nowhere":
                parts.append(f"{en}")
        else:
            m = evmeta.get(ev, {})
            # strip "Do you have ..." to keep the finding-ish text
            q = m.get("question_en", ev)
            parts.append(q)
    return ". ".join(parts)


def main():
    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    evmeta = json.load(open("data/ddxplus/release_evidences.json"))
    fr2cui = {info.get("cond-name-fr", ""): icd[dn]["cui"] for dn, info in cond.items() if dn in icd}
    dcs = sorted(set(fr2cui.values()))

    def L(f): return pickle.load(open(f"pilot/data/cache/{f}", "rb"))
    Gs = [L("v103ddx49_sci_kg.pkl"), L("v103pres_ddx49_sci_kg.pkl"), L("v103sci_ddx49_kg.pkl")]
    P = {}
    for d in dcs:
        w = defaultdict(float)
        for G in Gs:
            if d in G:
                for _, p, e in G.out_edges(d, data=True):
                    if e.get("etype") == "HAS_PHENOTYPE": w[p] += e.get("n_mentions", 0.0)
        P[d] = {p: x/(x+2.0) for p, x in w.items() if x > 0}
    all_evs = set().union(*[set(p) for p in P.values()])
    N = len(P); df = defaultdict(int)
    for pr in P.values():
        for e in pr: df[e] += 1
    idf = {e: math.log((N+1)/(df[e]+1))+1.0 for e in df}
    Pw = {d: {e: p*(idf.get(e, 1.0)**0.75) for e, p in pr.items()} for d, pr in P.items()}
    dl = [d for d in Pw if Pw[d]]

    print("Loading scispaCy...", flush=True)
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True,
                 "linker_name": "umls", "k": 3, "threshold": 0.80, "max_entities_per_mention": 1})

    rows = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if len(rows) >= 3000: break
            tc = fr2cui.get(row["PATHOLOGY"])
            if tc not in P or not P[tc]: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            rows.append((tc, render(evs, evmeta)))
    print(f"{len(rows)} patients, scispaCy linking...", flush=True)

    n = c1 = c3 = c10 = 0; covsum = 0.0
    texts = [t for _, t in rows]
    for (tc, _), doc in zip(rows, nlp.pipe(texts, batch_size=64)):
        pos = set()
        for entt in doc.ents:
            for cui, sc in entt._.kb_ents[:1]:
                pos.add(cui)
        pos &= all_evs
        if not pos: continue
        patv = {e: idf.get(e, 1.0)**0.75 for e in pos}
        pn = math.sqrt(sum(v*v for v in patv.values())) or 1e-9
        sc = {}
        for d in dl:
            pr = Pw[d]; dn = math.sqrt(sum(v*v for v in pr.values())) or 1e-9
            sc[d] = sum(patv[e]*pr[e] for e in pos if e in pr)/(pn*dn)
        ranked = sorted(dl, key=lambda d: -sc[d]); rk = ranked.index(tc)+1
        n += 1; c1 += rk == 1; c3 += rk <= 3; c10 += rk <= 10
        covsum += len([e for e in pos if e in P[tc]])/max(len(pos), 1)
    print(f"\nN={n}  (scispaCy-aligned patient)  @1={100*c1/n:.2f} @3={100*c3/n:.2f} @10={100*c10/n:.2f}", flush=True)
    print(f"GT 프로필 coverage 평균={100*covsum/n:.1f}%  (value_cuis baseline @1=33.62, cov=46.6%)", flush=True)


if __name__ == "__main__":
    main()
