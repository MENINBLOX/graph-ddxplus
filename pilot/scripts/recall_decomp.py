"""Where does the literature graph lose recall? Stage-wise decomposition on phenopacket patients.

For every (case, observed HPO feature) whose gold disease HPOA annotates with that feature, the gold
disease's literature evidence for the feature is classified by the first pipeline stage that lost it:

  no_docs        the disease has no usable source article
  not_in_text    no HPO label/synonym of the feature occurs in the disease's article text
  not_extracted  the text names the feature but no IE item for the disease contains that name
  not_linked     an IE item contains the name but its phrase maps to no CUI / a different CUI
  filtered       mapped to the feature's CUI but dropped (generic article, excluded semantic type,
                 document not held for the disease, with --no-self a concept naming the disease)
  offtopic_only  in the graph only from articles whose title does not name the disease (scored at
                 offtopic_w: 0 in design D, .5 or 1 in the v3 graphs)
  near           absent, but a parent or child HPO term's CUI is a topic edge
  edge           a topic edge exists

String matching is a lower bound on presence (paraphrases are missed), so not_in_text is an upper
bound on collection loss and not_extracted a lower bound on extraction loss.

By default (--feature-cui xref) a feature is matched by its hp.json UMLS xrefs and near looks one
HPO is_a step up or down. The ranker instead queries one CUI per feature (hpo_umls_mapping.json, as
temporal_eval.load_cases), which is also the CUI --hpo-names adds to the graph, and expands it only
upward through the graph's Concept IS_A (phenotype_parents.json; outgoing only from a concept that
has an edge). --feature-cui ranker matches by that CUI, both by it or an xref; in either mode near
means a parent through Concept IS_A, and three stages are added:

  no_cui         the feature has no CUI, so the ranker never sees it
  near_child     only a child concept has a topic edge (not scored by the ranker)
  near_offtopic  only a parent's off-topic edge (scored at decay * offtopic_w)

These modes read every disease's items, since IS_A and idf depend on the whole graph, and report
the triples the ranker scores at all (`scored`, at --offtopic-w) whatever their stage.

Documents, the generic-document rule, linking and topic titles follow build_graph_v2.py; pass the
flags the graph was built with (--link full = --split-coord --hpo-names).

    python recall_decomp.py --out pilot/cache/v2/recall/decomp.json            # design D (base_nm)
    python recall_decomp.py --sources DIR1,DIR2 --ie A.jsonl,B.jsonl.gz --table T.json --link full \\
        --topic-genes --no-self [--phrases LLM1.jsonl,LLM2.jsonl] --out ...                # v3 variants
    python recall_decomp.py ... --feature-cui ranker --offtopic-w 0.5 --out ...       # ranker's CUIs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curated_agreement as CA
import opendx_eval as O
import v2_ids
from build_graph_v2 import (
    _norm,
    article,
    causal_gene_forms,
    hpo_name_index,
    iter_sources,
    linked_forms,
    name_forms,
)
from jsonl_io import open_text
from temporal_eval import NEW, load_cases

V2 = "pilot/cache/v2"
STAGES = ["no_docs", "not_in_text", "not_extracted", "not_linked", "filtered", "offtopic_only", "near", "edge"]
RANKER_STAGES = ["no_cui", "no_docs", "not_in_text", "not_extracted", "not_linked", "filtered", "near_child",
                 "near_offtopic", "offtopic_only", "near", "edge"]
PHENOTYPE_PARENTS = "pilot/cache/umls_maps/phenotype_parents.json"
EXCLUDED_TUI = {"T058", "T059", "T060", "T061"}
MAX_DOC_DISEASES = 20
LEADING_ID = re.compile(r'\{"id": "([^"]+)"')


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def hpo_terms() -> dict[str, set[str]]:
    with open(CA.HP_JSON) as f:
        g = json.load(f)["graphs"][0]
    names = {}
    for n in g["nodes"]:
        if not n.get("lbl"):
            continue
        f = {norm(n["lbl"])} | {norm(s["val"]) for s in n.get("meta", {}).get("synonyms", [])
                                 if s.get("pred") in ("hasExactSynonym", "hasRelatedSynonym")}
        names[n["id"].rsplit("/", 1)[-1].replace("_", ":")] = {x for x in f if len(x) >= 4}
    return names


def records(paths: str, keep: set[str] | None) -> Iterator[dict]:
    """Extraction records of the diseases in `keep` (None: all); other lines are skipped before parsing."""
    for path in paths.split(","):
        with open_text(path) as f:
            for ln in f:
                if keep is not None and (m := LEADING_ID.match(ln)) and m.group(1) not in keep:
                    continue
                r = json.loads(ln)
                if keep is None or r["id"] in keep:
                    yield r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie", default=f"{V2}/ie_gemma_full.jsonl", help="comma-separated extraction files (.gz read too)")
    ap.add_argument("--table", default=f"{V2}/phrase2cui_gemma.json")
    ap.add_argument("--sources", default=f"{V2}/sources", help="as build_graph_v2.py --sources")
    ap.add_argument("--link", choices=["table", "full"], default="table",
                    help="full: link like build_graph_v2.py --split-coord --hpo-names")
    ap.add_argument("--topic-genes", action="store_true", help="as build_graph_v2.py --topic-genes")
    ap.add_argument("--no-self", action="store_true", help="as build_graph_v2.py --no-self")
    ap.add_argument("--phrases", default="",
                    help="extraction files whose item phrases also count as extracted (not_extracted vs "
                         "not_linked); for llm_dict_relink.py output, the LLM files it re-linked, because it "
                         "drops the LLM items in which the dictionary finds nothing")
    ap.add_argument("--feature-cui", choices=["xref", "ranker", "both"], default="xref",
                    help="CUIs a feature is matched by: hp.json UMLS xrefs, the ranker's hpo_umls_mapping CUI, "
                         "or either (module docstring)")
    ap.add_argument("--offtopic-w", type=float, default=0.0,
                    help="the graph's frozen offtopic_w, for `scored` (design E: S-mix .5, S-llm 1, S-dict .5, D 0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    whole = a.feature_cui != "xref"
    h2c, par = CA.hpo_to_cui()
    kids = defaultdict(set)
    for c, ps in par.items():
        for p in ps:
            kids[p].add(c)
    names = hpo_terms()
    uni = dict(v2_ids.universe())
    with open(f"{V2}/exclude_nonhuman_diseases.json") as f:
        for k in json.load(f):
            uni.pop(k, None)

    # curated (disease key, HPO) pairs, so only features the curators tie to the gold disease count
    cur = defaultdict(set)
    with open(CA.HPOA) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > 10 and f[0].startswith("OMIM:") and f[2] != "NOT" and f[10] == "P":
                for key in v2_ids.gold_keys({f[0]}):
                    cur[key].add(f[3])

    cases = load_cases()
    z = zipfile.ZipFile("data/external_benchmarks/phenopacket_store/all_phenopackets.zip")
    zn = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}
    need = defaultdict(set)                        # disease key -> HPO ids observed in its patients
    tasks = []
    for c in cases:
        if c["release"] == "0.1.27":
            raw = z.read(zn[c["file"]])
        else:
            with open(f"{NEW}/{c['file']}", "rb") as fb:
                raw = fb.read()
        p = json.loads(raw)
        hpos = {f["type"]["id"] for f in p.get("phenotypicFeatures", []) if not f.get("excluded")}
        keys = v2_ids.gold_keys(c["gold"]) & set(uni)
        for k in keys:
            for h in hpos & cur[k]:
                need[k].add(h)
                tasks.append((c["file"], k, h))
    print(f"cases {len(cases)}, (case, gold, curated feature) triples {len(tasks)}, diseases {len(need)}", flush=True)

    # generic-document rule of build_graph_v2.py: counted per article, a table document (TBL<PMCID>)
    # adding nothing for a disease that also holds the article
    use = defaultdict(int)
    need_docs = {}
    held_all = {}
    for dz, dz_docs in iter_sources(a.sources):
        ids = {x["doc_id"] for x in dz_docs}
        for x in dz_docs:
            if not (x["doc_id"].startswith("TBL") and article(x["doc_id"]) in ids):
                use[article(x["doc_id"])] += 1
        if dz in need:
            need_docs[dz] = dz_docs
        elif whole and dz in uni:
            held_all[dz] = {sys.intern(i) for i in ids}
    generic = {d for d, n in use.items() if n >= MAX_DOC_DISEASES}
    del use
    gene_forms = causal_gene_forms() if a.topic_genes else {}
    text, held, topic_doc = {}, {}, {}
    for k in need:
        all_docs = need_docs.pop(k, [])
        docs = [x for x in all_docs if article(x["doc_id"]) not in generic]
        held[k] = {x["doc_id"] for x in all_docs}
        text[k] = " " + norm(" ".join(x["text"] for x in docs)) + " "
        forms = name_forms(uni[k]) | gene_forms.get(k, set())
        topic_doc[k] = {x["doc_id"]: any(f in _norm(x.get("title") or "") for f in forms) for x in docs}
    print(f"generic documents {len(generic)}, diseases with usable documents {sum(map(bool, topic_doc.values()))}, "
          f"max RSS {rss_gb():.1f} GB", flush=True)

    with open(a.table) as f:
        table = json.load(f)
    hpo = hpo_name_index() if a.link == "full" else None
    self_cuis = defaultdict(set)
    if a.no_self:
        for c, p in v2_ids.disease_isa():
            self_cuis[c].add(p)
        for k in uni:
            self_cuis[k].add(k)

    def links(it: dict) -> list[tuple[str | None, str | None]]:
        if it.get("cui"):
            out = [(it["cui"], it.get("tui"))]
            hc = hpo.get(_norm(it["t"]).strip()) if hpo is not None else None
            if hc and hc != it["cui"]:
                out.append((hc, ""))
            return out
        if hpo is not None:
            return [(t.get("cui"), t.get("tui")) for _, t in linked_forms(it["t"], table, True, hpo)] or [(None, None)]
        t = table.get(it["t"].lower().strip(), {})
        return [(t.get("cui"), t.get("tui"))]

    # per disease: item phrases, every linked CUI, CUIs that reach an edge, CUIs that reach a topic edge
    phrases = {k: set() for k in need}
    linked = {k: set() for k in need}
    any_cuis = {k: set() for k in need}
    topic_cuis = {k: set() for k in need}
    edge_cuis = defaultdict(set)                   # whole graph: other diseases -> concepts with an edge
    n_items = 0
    for r in records(a.ie, None if whole else set(need)):
        k = r["id"]
        if k not in need:
            if k in held_all:
                hk, ek, sk = held_all[k], edge_cuis[k], self_cuis.get(k, ())
                for it in r.get("items", []):
                    if it["doc"] in hk and article(it["doc"]) not in generic:
                        ek.update(c for c, t in links(it) if c and t not in EXCLUDED_TUI and c not in sk)
            continue
        for it in r.get("items", []):
            n_items += 1
            phrases[k].add(" " + norm(it["t"]) + " ")
            if it.get("src_t"):                    # llm_dict_relink.py: the LLM phrase the string was found in
                phrases[k].add(" " + norm(it["src_t"]) + " ")
            doc = it["doc"]
            kept = article(doc) not in generic and doc in held[k]
            for cui, tui in links(it):
                if not cui:
                    continue
                linked[k].add(cui)
                if kept and tui not in EXCLUDED_TUI and cui not in self_cuis.get(k, ()):
                    any_cuis[k].add(cui)
                    if topic_doc[k].get(doc, False):
                        topic_cuis[k].add(cui)
    print(f"items of these diseases {n_items}, max RSS {rss_gb():.1f} GB", flush=True)
    if a.phrases:
        n_extra = 0
        for r in records(a.phrases, set(need)):
            for it in r.get("items", []):
                ph = " " + norm(it["t"]) + " "
                if ph not in phrases[r["id"]]:
                    phrases[r["id"]].add(ph)
                    n_extra += 1
        print(f"phrases added from --phrases {n_extra}, max RSS {rss_gb():.1f} GB", flush=True)
    # one string per disease; "\0" never occurs in a normalized phrase, so no name spans two phrases
    phrase_text = {k: "\0".join(v) for k, v in phrases.items()}
    del phrases

    def lost(k: str, h: str) -> str:
        """Stage of a feature none of whose CUIs the disease's items reach, decided on its HPO names."""
        nm = names.get(h, set())
        if not topic_doc[k]:
            return "no_docs"
        if not any(f" {n} " in text[k] for n in nm):
            return "not_in_text"
        if any(f" {n} " in phrase_text[k] for n in nm):
            return "not_linked"
        return "not_extracted"

    verdict = {}
    if not whole:
        for k, hs in need.items():
            for h in hs:
                cs = h2c.get(h, set())
                near_cs = set().union(*(h2c.get(x, set()) for x in par.get(h, set()) | kids.get(h, set())))
                if cs & topic_cuis[k]:
                    v = "edge"
                elif near_cs & topic_cuis[k]:
                    v = "near"
                elif cs & any_cuis[k]:
                    v = "offtopic_only"
                elif cs & linked[k]:
                    v = "filtered"
                else:
                    v = lost(k, h)
                verdict[(k, h)] = v
        stages, extra = STAGES, {}
    else:
        for k in need:
            edge_cuis[k] = any_cuis[k]
        del held_all
        df = Counter(c for cs in edge_cuis.values() for c in cs)
        n_dz = sum(map(bool, edge_cuis.values()))
        with open(PHENOTYPE_PARENTS) as f:
            pp = json.load(f)
        # counts build_graph_v2.py logs and writes to its summary, to check this pass against the graph
        graph = {"diseases_with_findings": n_dz, "edges": sum(map(len, edge_cuis.values())),
                 "concepts": len(set(df) | {p for c in df for p in pp.get(c, ())}),
                 "concept_is_a": sum(len(pp.get(c, ())) for c in df)}
        del edge_cuis
        print(f"whole graph {graph}, max RSS {rss_gb():.1f} GB", flush=True)
        pp_kids = defaultdict(set)
        for c, ps in pp.items():
            for p in ps:
                pp_kids[p].add(c)
        _, patient_cui = O._rb_maps()

        def mode_cuis(h: str) -> set[str]:
            pc = patient_cui(h)
            cs = {pc} if pc else set()
            return cs if a.feature_cui == "ranker" else cs | h2c.get(h, set())

        def up(cs: set[str]) -> set[str]:
            return {p for c in cs if c in df for p in pp.get(c, ())}

        scored, hpo_near = {}, {}
        for k, hs in need.items():
            topic, anyc = topic_cuis[k], any_cuis[k]
            for h in hs:
                cs = mode_cuis(h)
                ups = up(cs)
                if not cs:
                    v = "no_cui"
                elif cs & topic:
                    v = "edge"
                elif ups & topic:
                    v = "near"
                elif cs & anyc:
                    v = "offtopic_only"
                elif ups & anyc:
                    v = "near_offtopic"
                elif set().union(*(pp_kids.get(c, set()) for c in cs)) & topic:
                    v = "near_child"
                elif cs & linked[k]:
                    v = "filtered"
                else:
                    v = lost(k, h)
                verdict[(k, h)] = v
                # what graph_dx.RANK adds for this feature: idf is log(n_dz / df), 0 for a concept on every disease
                pc = patient_cui(h)
                reach = {pc} | up({pc}) if pc else set()
                scored[(k, h)] = any(df[c] < n_dz and (c in topic or (a.offtopic_w > 0 and c in anyc)) for c in reach)
                if set().union(*(mode_cuis(x) for x in par.get(h, set()))) & topic:
                    hpo_near[(k, h)] = "parent"
                elif set().union(*(mode_cuis(x) for x in kids.get(h, set()))) & topic:
                    hpo_near[(k, h)] = "child"
        stages = RANKER_STAGES
        n_sc = sum(scored[(k, h)] for _, k, h in tasks)
        sc = Counter(verdict[(k, h)] for _, k, h in tasks if scored[(k, h)])
        hn = Counter((verdict[(k, h)], hpo_near[(k, h)]) for _, k, h in tasks if (k, h) in hpo_near)
        extra = {"graph": graph, "scored": [n_sc, round(100 * n_sc / len(tasks), 1)],
                 "scored_by_stage": {s: sc[s] for s in stages},
                 "hpo_near_by_stage": {s: {"parent": hn[(s, "parent")], "child_only": hn[(s, "child")]}
                                       for s in stages}}

    trip = Counter(verdict[(k, h)] for _, k, h in tasks)
    pairs = Counter(verdict.values())
    out = {"args": vars(a), "triples": len(tasks), "pairs": len(verdict), "diseases": len(need),
           "by_triple": {s: [trip[s], round(100 * trip[s] / len(tasks), 1)] for s in stages},
           "by_pair": {s: [pairs[s], round(100 * pairs[s] / len(verdict), 1)] for s in stages}, **extra}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    with open(a.out.replace(".json", "_pairs.json"), "w") as f:
        n_cases = Counter((k, h) for _, k, h in tasks)
        rows = [{"d": k, "hpo": h, "v": v, "n": n_cases[(k, h)]} for (k, h), v in verdict.items()]
        if whole:
            for r in rows:
                r["s"] = int(scored[(r["d"], r["hpo"])])
                r["hn"] = hpo_near.get((r["d"], r["hpo"]), "")
        json.dump(rows, f)
    print(json.dumps(out, indent=1))
    print(f"max RSS {rss_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
