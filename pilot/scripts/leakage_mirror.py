"""Data-driven leakage check on graphs built without Mention nodes (research design §9.24, §9.32 P0-1).

The §9.24 rule (leakage_check.py) flags a patient when one document supports, for a gold disease, at
least half of the patient's concepts and at least five of them. leakage_check.py reads the supporting
documents from Mention nodes, which only the S-mix build on 7689 has. Here the documents behind each
(gold disease, patient concept) edge come from a mirror of build_graph_v2.py edge construction over the
extraction files: per-disease sources and the generic-document rule, held documents, linking
(--split-coord --hpo-names, the flags of every variant used here), excluded semantic types and
--no-self. Every graph is checked against its build summary (edges, mentions, documents, diseases with
findings, concepts) and S-mix also against the Mention graph, edge by edge.

A component is one extraction run over one corpus; a graph is the union of its components, with or
without self edges (a self edge links a disease to its own or a parent disease's concept, a property
of the pair, so one pass serves both builds).

    python leakage_mirror.py component U3g --work DIR     # one pass per component (2-5 GB each)
    python leakage_mirror.py mention --work DIR           # S-mix documents from the 7689 Mention graph
    python leakage_mirror.py flags --work DIR             # check graphs, flag the nine development sets
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval
import v2_ids
from build_graph_v2 import _norm, article, hpo_name_index, iter_sources, linked_forms
from jsonl_io import open_text
from tune_dev import dev_sets

V2, V3 = "pilot/cache/v2", "pilot/cache/v3"
VARIANTS = Path(f"{V2}/devD/variants")
OUT = Path(f"{V3}/p0/leakage")
SRC2, SRC3 = f"{V2}/sources", f"{V2}/sources,{V3}/sources_new"
T2, T3 = f"{V2}/phrase2cui_gemma.json", f"{V3}/phrase2cui_v3.json"
LLM2, LLM3 = f"{V2}/ie_gemma_full.jsonl", f"{V2}/ie_gemma_full.jsonl,{V3}/ie_v3.jsonl"
HPO2, HPO3 = f"{V2}/dict_v2.jsonl.gz", f"{V2}/dict_v2.jsonl.gz,{V3}/dict_v3.jsonl.gz"
UMLS2, UMLS3 = f"{V2}/dict_umls_v2.jsonl.gz", f"{V2}/dict_umls_v2.jsonl.gz,{V3}/dict_umls_v3.jsonl.gz"
LLMD3 = f"{V3}/llmd_v2.jsonl.gz,{V3}/llmd_v3.jsonl.gz"
# (extraction files, phrase table, sources) as run_graph_variants.sh passes them
COMPONENTS = {
    "L2": (LLM2, T2, SRC2), "H2": (HPO2, T2, SRC2), "U2": (UMLS2, T2, SRC2),
    "L3": (LLM3, T3, SRC3), "H3g": (HPO3, T2, SRC3), "H3v": (HPO3, T3, SRC3), "U3g": (UMLS3, T2, SRC3),
    "U3v": (UMLS3, T3, SRC3), "M3": (LLMD3, T3, SRC3),
}
# graph -> (components, --no-self)
GRAPHS = {
    "v2dict": (["H2"], False), "v2dict_ns": (["H2"], True), "v2umls": (["U2"], False), "v2umls_ns": (["U2"], True),
    "base_split_tg_ns": (["L2"], True), "v2union_ns": (["L2", "H2"], True), "v2unionu_ns": (["L2", "U2"], True),
    "v3dict": (["H3g"], False), "v3dict_ns": (["H3g"], True), "v3umls": (["U3g"], False), "v3umls_ns": (["U3g"], True),
    "v3llm_ns": (["L3"], True), "v3union_ns": (["L3", "H3v"], True), "v3unionu_ns": (["L3", "U3v"], True),
    "v3llmd_ns": (["M3"], True),
}
SMIX, SMIX_URI = "v3llmd_ns", "bolt://localhost:7689"
EXCLUDED_TUI = {"T058", "T059", "T060", "T061"}
MAX_DOC_DISEASES = 20
COVER, MIN_CONCEPTS = 0.5, 5
CUI = re.compile(r"C\d{7}")
MENTION_Q = """
MATCH (:Disease {id: $d})<-[:OF]-(m:Mention)-[:MENTIONS]->(c:Concept)
WHERE c.cui IN $cs
MATCH (m)-[:FROM]->(x:Doc)
RETURN c.cui AS c, collect(DISTINCT x.id) AS docs
"""


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def universe() -> dict:
    uni = dict(v2_ids.universe())
    with open(f"{V2}/exclude_nonhuman_diseases.json") as f:
        for k in json.load(f):
            uni.pop(k, None)
    return uni


def self_cuis(uni: dict) -> dict[str, set[str]]:
    out = defaultdict(set)
    for c, p in v2_ids.disease_isa():
        out[c].add(p)
    for k in uni:
        out[k].add(k)
    return out


def dev_patients() -> dict[str, list[tuple[tuple[str, ...], frozenset[str]]]]:
    """Per set, the patients graph_dx.py ranks (gold keys and concepts both non-empty), in its order."""
    opendx_eval.RAW_GOLD = True
    out = {}
    for b, pats in dev_sets().items():
        rows = [(tuple(sorted(v2_ids.gold_keys(g))), frozenset(ev)) for g, ev in pats]
        out[b] = [(g, ev) for g, ev in rows if g and ev]
    return out


def needed(pats: dict) -> dict[str, set[str]]:
    need = defaultdict(set)
    for rows in pats.values():
        for gold, ev in rows:
            for g in gold:
                need[g] |= ev
    return need


def run_component(name: str, work: Path) -> None:
    ie, table_path, sources = COMPONENTS[name]
    t0 = time.time()
    uni = universe()
    didx = {k: i for i, k in enumerate(sorted(uni))}
    selfc = self_cuis(uni)
    need = needed(dev_patients())

    held, use = {}, defaultdict(int)
    for dz, dz_docs in iter_sources(sources):
        held[dz] = {sys.intern(x["doc_id"]) for x in dz_docs}
        for x in dz_docs:
            if not (x["doc_id"].startswith("TBL") and article(x["doc_id"]) in held[dz]):
                use[article(x["doc_id"])] += 1
    generic = {d for d, n in use.items() if n >= MAX_DOC_DISEASES}
    del use
    print(f"{name}: generic documents {len(generic)}, max RSS {rss_gb():.1f} GB", flush=True)
    with open(table_path) as f:
        table = json.load(f)
    hpo = hpo_name_index()

    @lru_cache(maxsize=1 << 21)
    def table_links(phrase: str) -> tuple[tuple[str, str | None], ...]:
        return tuple((t["cui"], t.get("tui")) for _, t in linked_forms(phrase, table, True, hpo))

    def links(it: dict) -> tuple[tuple[str, str | None], ...]:
        if it.get("cui"):
            hc = hpo.get(_norm(it["t"]).strip())
            return ((it["cui"], it["tui"]), (hc, "")) if hc and hc != it["cui"] else ((it["cui"], it["tui"]),)
        return table_links(it["t"])

    edges, self_edges = set(), set()
    mentions = Counter()                       # "ns": links kept by --no-self, "self": links it drops
    docs_ns, docs_self = set(), set()
    support = defaultdict(set)                 # (gold disease, patient concept) -> documents
    n_table_items = 0
    for path in ie.split(","):
        with open_text(path) as f:
            for ln in f:
                r = json.loads(ln)
                d = r["id"]
                if d not in uni:
                    continue
                hd, sk, nd, base = held.get(d, ()), selfc.get(d, ()), need.get(d, ()), didx[d] << 32
                for it in r.get("items", []):
                    doc = it["doc"]
                    if doc not in hd or article(doc) in generic:
                        continue
                    n_table_items += not it.get("cui")
                    for c, tui in links(it):
                        if tui in EXCLUDED_TUI:
                            continue
                        if not CUI.fullmatch(c):
                            raise SystemExit(f"{name}: concept {c!r} is not a CUI; the edge code needs one")
                        code = base | int(c[1:])
                        edges.add(code)
                        if c in sk:
                            self_edges.add(code)
                            mentions["self"] += 1
                            docs_self.add(doc)
                        else:
                            mentions["ns"] += 1
                            docs_ns.add(doc)
                        if c in nd:
                            support[(d, c)].add(doc)
        print(f"{name}: read {path}, edges {len(edges)}, max RSS {rss_gb():.1f} GB, {time.time() - t0:.0f}s", flush=True)
    res = {"name": name, "ie": ie, "table": table_path, "sources": sources, "table_items": n_table_items,
           "edges": np.sort(np.fromiter(edges, dtype=np.int64, count=len(edges))),
           "self_edges": np.sort(np.fromiter(self_edges, dtype=np.int64, count=len(self_edges))),
           "mentions": dict(mentions), "docs_ns": docs_ns, "docs_self": docs_self,
           "support": {k: sorted(v) for k, v in support.items()}}
    work.mkdir(parents=True, exist_ok=True)
    with open(work / f"{name}.pkl", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{name}: done, edges {len(edges)}, self {len(self_edges)}, mentions {dict(mentions)}, "
          f"support pairs {len(support)}, max RSS {rss_gb():.1f} GB, {time.time() - t0:.0f}s", flush=True)


def run_mention(work: Path, auth: str) -> None:
    """(gold disease, patient concept) -> documents of its Mentions in the S-mix build on 7689 (read only)."""
    need = needed(dev_patients())
    out = {}
    drv = GraphDatabase.driver(SMIX_URI, auth=tuple(auth.split("/", 1)))
    with drv.session(default_access_mode="READ") as s:
        for i, (d, cs) in enumerate(sorted(need.items())):
            for r in s.run(MENTION_Q, d=d, cs=sorted(cs)):
                out[(d, r["c"])] = sorted(r["docs"])
            if i % 500 == 0:
                print(f"mention: {i}/{len(need)} diseases, pairs {len(out)}", flush=True)
    drv.close()
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "mention_smix.pkl", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"mention: {len(out)} supported pairs", flush=True)


def graph_counts(comps: list[dict], no_self: bool, uni: dict) -> dict:
    """The counts build_graph_v2.py writes to its summary, from the components' edge codes."""
    edges = np.unique(np.concatenate([c["edges"] for c in comps]))
    if no_self:
        edges = np.setdiff1d(edges, np.unique(np.concatenate([c["self_edges"] for c in comps])), assume_unique=True)
    docs = set().union(*(c["docs_ns"] for c in comps))
    if not no_self:
        docs |= set().union(*(c["docs_self"] for c in comps))
    with open("pilot/cache/umls_maps/phenotype_parents.json") as f:
        parents = json.load(f)
    cuis = {f"C{x:07d}" for x in np.unique(edges & 0xFFFFFFFF).tolist()}
    return {"diseases": len(uni), "diseases_with_findings": int(np.unique(edges >> 32).size),
            "concepts": len(cuis | {p for c in cuis for p in parents.get(c, ())}), "edges": int(edges.size),
            "mentions": sum(c["mentions"].get("ns", 0) + (0 if no_self else c["mentions"].get("self", 0)) for c in comps),
            "docs": len(docs)}


def flag_patients(pats: dict, support: dict, unit=lambda doc: doc) -> dict:
    """Per set and patient: the largest number of the patient's concepts one document supports for a gold
    disease, that document, and the §9.24 flag."""
    out = {}
    for b, rows in pats.items():
        best, docs, flags = [], [], []
        for gold, ev in rows:
            per_doc = defaultdict(set)
            for c in ev:
                for g in gold:
                    for x in support.get((g, c), ()):
                        per_doc[unit(x)].add(c)
            doc, cs = max(per_doc.items(), key=lambda kv: (len(kv[1]), kv[0]), default=("", set()))
            n = len(cs)
            best.append(n)
            flag = n >= MIN_CONCEPTS and n / len(ev) >= COVER
            flags.append(int(flag))
            docs.append(doc if flag else "")
        out[b] = {"n_max": best, "flag": flags, "doc": docs}
    return out


def run_flags(work: Path) -> None:
    uni = universe()
    selfc = self_cuis(uni)
    pats = dev_patients()
    with open(work / "mention_smix.pkl", "rb") as f:
        mention = pickle.load(f)
    res = {"rule": {"cover": COVER, "min_concepts": MIN_CONCEPTS, "unit": "document id as in the Mention graph "
                    "(an article's table document TBL<PMCID> is a document of its own); article_flag merges them"},
           "patients": {b: len(rows) for b, rows in pats.items()}, "graphs": {}}
    cache: dict[str, dict] = {}
    for g, (names, no_self) in GRAPHS.items():
        for n in names:
            if n not in cache:
                with open(work / f"{n}.pkl", "rb") as f:
                    cache[n] = pickle.load(f)
        comps = [cache[n] for n in names]
        got = graph_counts(comps, no_self, uni)
        with open(VARIANTS / f"graph_{g}.json") as f:
            want = json.load(f)
        bad = {k: (v, want.get(k)) for k, v in got.items() if v != want.get(k)}
        support = defaultdict(set)
        for c in comps:
            for (d, cui), docs in c["support"].items():
                if not (no_self and cui in selfc.get(d, ())):
                    support[(d, cui)].update(docs)
        entry = {"components": names, "no_self": no_self, "mirror": got, "summary_match": not bad, "mismatch": bad}
        if g == SMIX:
            diff = [k for k in set(support) | set(mention) if set(support.get(k, ())) != set(mention.get(k, ()))]
            entry["mention_check"] = {"pairs_mirror": len(support), "pairs_mention": len(mention),
                                      "pairs_differing": len(diff), "examples": [list(k) for k in sorted(diff)[:5]]}
            support = {k: set(v) for k, v in mention.items()}      # S-mix: the Mention graph itself
        fl = flag_patients(pats, support)
        art = flag_patients(pats, support, article)
        for b in fl:
            fl[b]["article_flag"] = art[b]["flag"]
        entry["sets"] = fl
        entry["flagged"] = {b: sum(v["flag"]) for b, v in fl.items()}
        entry["flagged_article"] = {b: sum(v["article_flag"]) for b, v in fl.items()}
        res["graphs"][g] = entry
        print(f"{g:17s} summary {'ok' if not bad else bad} flagged {entry['flagged']}"
              + (f" mention_check {entry['mention_check']}" if g == SMIX else ""), flush=True)
        for n in list(cache):
            if all(n not in GRAPHS[h][0] for h in list(GRAPHS)[list(GRAPHS).index(g) + 1:]):
                del cache[n]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "flags.json", "w") as f:
        json.dump(res, f)
    print(f"saved {OUT / 'flags.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["component", "mention", "flags"])
    ap.add_argument("name", nargs="?", choices=sorted(COMPONENTS))
    ap.add_argument("--work", required=True, type=Path, help="directory for component and Mention pickles")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    a = ap.parse_args()
    if a.stage == "component":
        if not a.name:
            ap.error("component needs a name")
        run_component(a.name, a.work)
    elif a.stage == "mention":
        run_mention(a.work, a.auth)
    else:
        run_flags(a.work)


if __name__ == "__main__":
    main()
