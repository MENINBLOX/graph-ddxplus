"""Where does the literature graph lose recall? Stage-wise decomposition on phenopacket patients.

For every (case, observed HPO feature) whose gold disease HPOA annotates with that feature, the gold
disease's literature evidence for the feature is classified by the first pipeline stage that lost it:

  no_docs        the disease has no usable source article
  not_in_text    no HPO label/synonym of the feature occurs in the disease's article text
  not_extracted  the text names the feature but no IE item for the disease contains that name
  not_linked     an IE item contains the name but its phrase maps to no CUI / a different CUI
  filtered       mapped to the feature's CUI but dropped (generic article, excluded semantic type)
  offtopic_only  in the graph only from articles whose title does not name the disease (weight 0)
  near           absent, but a parent or child HPO term's CUI is a topic edge
  edge           a topic edge exists

String matching is a lower bound on presence (paraphrases are missed), so not_in_text is an upper
bound on collection loss and not_extracted a lower bound on extraction loss.

    python recall_decomp.py --out pilot/cache/v2/recall/decomp.json
    python recall_decomp.py --sources DIR1,DIR2 --ie A.jsonl,B.jsonl --link full --out ...   # merged corpus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curated_agreement as CA  # noqa: E402
import v2_ids  # noqa: E402
from build_graph_v2 import _norm, hpo_name_index, iter_sources, linked_forms, name_forms  # noqa: E402
from temporal_eval import load_cases  # noqa: E402

V2 = "pilot/cache/v2"
STAGES = ["no_docs", "not_in_text", "not_extracted", "not_linked", "filtered", "offtopic_only", "near", "edge"]


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def hpo_terms():
    g = json.load(open(CA.HP_JSON))["graphs"][0]
    short = lambda u: u.rsplit("/", 1)[-1].replace("_", ":")  # noqa: E731
    names = {}
    for n in g["nodes"]:
        if not n.get("lbl"):
            continue
        f = {norm(n["lbl"])} | {norm(s["val"]) for s in n.get("meta", {}).get("synonyms", [])
                                 if s.get("pred") in ("hasExactSynonym", "hasRelatedSynonym")}
        names[short(n["id"])] = {x for x in f if len(x) >= 4}
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie", default=f"{V2}/ie_gemma_full.jsonl", help="comma-separated extraction files")
    ap.add_argument("--table", default=f"{V2}/phrase2cui_gemma.json")
    ap.add_argument("--sources", default=f"{V2}/sources", help="as build_graph_v2.py --sources")
    ap.add_argument("--link", choices=["table", "full"], default="table",
                    help="full: link like build_graph_v2.py --split-coord --hpo-names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h2c, par = CA.hpo_to_cui()
    kids = defaultdict(set)
    for c, ps in par.items():
        for p in ps:
            kids[p].add(c)
    names = hpo_terms()
    uni = dict(v2_ids.universe())
    for k in json.load(open(f"{V2}/exclude_nonhuman_diseases.json")):
        uni.pop(k, None)

    # curated (disease key, HPO) pairs, so only features the curators tie to the gold disease count
    cur = defaultdict(set)
    for line in open(CA.HPOA):
        f = line.rstrip("\n").split("\t")
        if len(f) > 10 and f[0].startswith("OMIM:") and f[2] != "NOT" and f[10] == "P":
            for key in v2_ids.gold_keys({f[0]}):
                cur[key].add(f[3])

    cases = load_cases()
    import zipfile
    z = zipfile.ZipFile("data/external_benchmarks/phenopacket_store/all_phenopackets.zip")
    zn = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}
    need = defaultdict(set)                        # disease key -> HPO ids observed in its patients
    tasks = []
    for c in cases:
        raw = z.read(zn[c["file"]]) if c["release"] == "0.1.27" else open(f"data/confirm/phenopacket_new/{c['file']}", "rb").read()
        p = json.loads(raw)
        hpos = {f["type"]["id"] for f in p.get("phenotypicFeatures", []) if not f.get("excluded")}
        keys = v2_ids.gold_keys(c["gold"]) & set(uni)
        for k in keys:
            for h in hpos & cur[k]:
                need[k].add(h)
                tasks.append((c["file"], k, h))
    print(f"cases {len(cases)}, (case, gold, curated feature) triples {len(tasks)}, diseases {len(need)}", flush=True)

    use = defaultdict(int)
    need_docs = {}
    for dz, dz_docs in iter_sources(a.sources):
        for x in dz_docs:
            use[x["doc_id"]] += 1
        if dz in need:
            need_docs[dz] = dz_docs
    generic = {d for d, n in use.items() if n >= 20}
    text, title = {}, {}
    for k in need:
        docs = [x for x in need_docs.get(k, []) if x["doc_id"] not in generic]
        text[k] = " " + norm(" ".join(x["text"] for x in docs)) + " "
        title[k] = {x["doc_id"]: _norm(x.get("title") or "") for x in docs}
    table = json.load(open(a.table))
    hpo = hpo_name_index() if a.link == "full" else None
    items = defaultdict(list)                      # disease -> (normalized phrase, cui, doc, tui)
    for line in (ln for path in a.ie.split(",") for ln in open(path)):
        r = json.loads(line)
        if r["id"] in need:
            for it in r["items"]:
                ph = " " + norm(it["t"]) + " "
                if it.get("cui"):
                    items[r["id"]].append((ph, it["cui"], it["doc"], it.get("tui")))
                elif hpo is not None:
                    links = linked_forms(it["t"], table, True, hpo) or [(it["t"], {})]
                    items[r["id"]] += [(ph, t.get("cui"), it["doc"], t.get("tui")) for _, t in links]
                else:
                    t = table.get(it["t"].lower().strip(), {})
                    items[r["id"]].append((ph, t.get("cui"), it["doc"], t.get("tui")))
    excluded_tui = {"T058", "T059", "T060", "T061"}

    verdict = {}
    for k, hs in need.items():
        forms = name_forms(uni[k])
        topic_cuis, any_cuis = set(), set()
        for ph, cui, doc, tui in items[k]:
            if cui and doc not in generic and tui not in excluded_tui:
                any_cuis.add(cui)
                if any(f in title[k].get(doc, "") for f in forms):
                    topic_cuis.add(cui)
        for h in hs:
            cs = h2c.get(h, set())
            nm = names.get(h, set())
            near_cs = set().union(*(h2c.get(x, set()) for x in par.get(h, set()) | kids.get(h, set())))
            if cs & topic_cuis:
                v = "edge"
            elif near_cs & topic_cuis:
                v = "near"
            elif cs & any_cuis:
                v = "offtopic_only"
            elif any(cui in cs for _, cui, _, _ in items[k]):
                v = "filtered"
            elif not title[k]:
                v = "no_docs"
            elif not any(f" {n} " in text[k] for n in nm):
                v = "not_in_text"
            elif any(f" {n} " in ph for ph, *_ in items[k] for n in nm):
                v = "not_linked"
            else:
                v = "not_extracted"
            verdict[(k, h)] = v

    trip = Counter(verdict[(k, h)] for _, k, h in tasks)
    pairs = Counter(verdict.values())
    out = {"triples": len(tasks), "pairs": len(verdict), "diseases": len(need),
           "by_triple": {s: [trip[s], round(100 * trip[s] / len(tasks), 1)] for s in STAGES},
           "by_pair": {s: [pairs[s], round(100 * pairs[s] / len(verdict), 1)] for s in STAGES}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump([{"d": k, "hpo": h, "v": v} for (k, h), v in verdict.items()],
              open(a.out.replace(".json", "_pairs.json"), "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
