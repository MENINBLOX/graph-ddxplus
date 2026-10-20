"""What the mixed graph (S-mix) holds that the dictionary graph (S-dict) does not, and whether rank wins rest on it.

(1) HAS_FINDING edge sets of S-mix (v3llmd_ns, 7689, built with Mentions) and S-dict (v3umls_ns, 7688) at the
(disease id, concept CUI) level: shared, S-mix-only and S-dict-only edges, each with its n_docs distribution
(1, 2-4, >= 5) and the share with an on-topic document (n_topic_docs >= 1). A shared edge has an n_docs in each
graph; both are given.
(2) Evidence origin of S-mix edges from their Mentions: dict_relink when the Mention carries a span (the LLM phrase
inside which llm_dict_relink.py found a UMLS string), llm_table otherwise (the phrase table linked the LLM phrase
or a coordinated part of it), and whether the dictionary can name the edge's concept at all. A seeded sample of
S-mix-only edges goes to smix_only_examples.csv with one Mention and what the S-dict pipeline does with the edge's
documents: dict_ie.py --vocab umls over every sentence, linked as build_graph_v2.py links dictionary items (the
string's CUI, its HPO name, the v2 phrase table for a string without a CUI), and for each occurrence of a UMLS
string of the concept whether S-dict extracts it, drops it for a negation cue, or loses it inside another match.
(3) Development rank wins at lineage Top-10 of the frozen runs (one graph ranks the gold in Top-10, the other
does not): the winner's edges from the gold disease, and from the disease that hit, to the patient's concepts as
the RANK query of graph_dx.py reaches them (a patient CUI that is a Concept node, at weight 1, and its IS_A
parents, at weight decay; only concepts with an idf). An edge is winner-only when the other graph lacks that
(disease, CUI) edge; its score share is its part of the disease's 1-hop finding sum. The disease that hit comes
from re-ranking the patient on both graphs with the runs' configuration, which must reproduce the stored ranks.
The same gold-edge counts over all evaluated patients of a set are the base rate; the loser's gold edges and
the ten diseases it ranked above the gold are counted the same way from the loser's side.

The reading of the sample (class new_form / dict_should_match / likely_wrong per row) is written by hand and
merged afterwards with --reading.

    python pilot/scripts/dict_mix_overlap.py [--n-examples 30 --seed 0]
    python pilot/scripts/dict_mix_overlap.py --reading pilot/cache/v3/review/smix_only_reading.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import dict_ie
import numpy as np
import opendx_eval
import v2_ids
from build_graph_v2 import _norm, hpo_name_index, linked_forms
from graph_dx import RANK, UNIFIED, GraphDx
from neo4j import GraphDatabase
from tune_dev import dev_sets

MIX, DICT = "S-mix", "S-dict"
URI = {MIX: "bolt://localhost:7689", DICT: "bolt://localhost:7688"}
AUTH = tuple(os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only").split("/", 1))
RUN = {MIX: "pilot/cache/v2/devD/variants/dev_v3llmd_ns_ot0.5.json",
       DICT: "pilot/cache/v2/devD/variants/dev_v3umls_ns_ot0.5.json"}
FROZEN = "pilot/cache/v2/devD/frozen_config_E.json"
SOURCES = ("pilot/cache/v2/sources", "pilot/cache/v3/sources_new")    # SRC3 of run_graph_variants.sh, in order
TABLE_V2 = "pilot/cache/v2/phrase2cui_gemma.json"                     # the phrase table S-dict was built with
SRDEF = "data/2025AB-full/config/2025AB/SRDEF"
EXCLUDED_TUI = {"T058", "T059", "T060", "T061"}                        # build_graph_v2.py --exclude-tuis default
OUT = Path("pilot/cache/v3/review")
# set -> the graph whose wins the review asks about; the other direction is computed as a contrast
WINS = {"RareBench-RAMEDIS": MIX, "RareBench-LIRICAL": MIX, "Phenopacket-store": MIX,
        "DDXPlus-test": DICT, "SymCat": DICT}
READING = {"new_form": "paraphrase or implicit wording no dictionary string of the concept matches",
           "dict_should_match": "a dictionary string of the concept is in the text; S-dict did not keep it",
           "likely_wrong": "the sentence does not support the finding for the disease"}
CUI_BITS = 24           # CUI number (C + 7 digits) < 2**24; edge key = disease index << 24 | CUI number
MASK = (1 << CUI_BITS) - 1

EDGES = """
MATCH (d:Disease)
CALL (d) {
  MATCH (d)-[e:HAS_FINDING]->(c:Concept)
  RETURN collect(c.cui) AS cs, collect(e.n_docs) AS nd, collect(e.n_topic_docs) AS nt, collect(e.n_mentions) AS nm
}
RETURN d.id AS id, cs, nd, nt, nm
"""
ORIGINS = """
MATCH (d:Disease)
CALL (d) {
  MATCH (d)<-[:OF]-(m:Mention)-[:MENTIONS]->(c:Concept)
  WITH c.cui AS cui, count(*) AS n, count(m.span) AS rl
  RETURN collect(cui) AS cs, collect(n) AS ns, collect(rl) AS rls
}
RETURN d.id AS id, cs, ns, rls
"""
EXAMPLE = """
MATCH (d:Disease {id: $d})<-[:OF]-(m:Mention)-[:MENTIONS]->(c:Concept {cui: $c})
MATCH (m)-[:FROM]->(x:Doc)
RETURN d.name AS disease, collect({phrase: m.phrase, span: m.span, sentence: m.sentence, topic: m.topic,
       doc_id: x.id, pmid: x.pmid, title: x.title}) AS mentions
"""


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def pct(k: float, n: float) -> float | None:
    return round(100 * k / n, 2) if n else None


class Edges:
    """The HAS_FINDING edges of one graph as arrays sorted by edge key."""

    def __init__(self, s, dz_index: dict[str, int], n: int):
        key = np.empty(n, np.int64)
        cols = {f: np.empty(n, np.int32) for f in ("nd", "nt", "nm")}
        i = 0
        for r in s.run(EDGES):
            k = len(r["cs"])
            if not k:
                continue
            key[i:i + k] = (dz_index[r["id"]] << CUI_BITS) | np.fromiter((int(c[1:]) for c in r["cs"]), np.int64, k)
            for f, col in cols.items():
                col[i:i + k] = r[f]
            i += k
        if i != n:
            sys.exit(f"streamed {i} edges, the graph holds {n}")
        o = np.argsort(key)
        self.key = key[o]
        self.nd, self.nt, self.nm = (cols[f][o] for f in ("nd", "nt", "nm"))
        self.concepts = np.unique(self.key & MASK)

    def has(self, keys: np.ndarray) -> np.ndarray:
        pos = np.minimum(np.searchsorted(self.key, keys), len(self.key) - 1)
        return self.key[pos] == keys

    def of(self, dz: int) -> slice:
        lo, hi = np.searchsorted(self.key, [dz << CUI_BITS, (dz + 1) << CUI_BITS])
        return slice(int(lo), int(hi))


def ndocs_profile(nd: np.ndarray, nt: np.ndarray) -> dict:
    n = len(nd)
    bins = {"1": int((nd == 1).sum()), "2-4": int(((nd >= 2) & (nd <= 4)).sum()), ">=5": int((nd >= 5).sum())}
    topic = int((nt >= 1).sum())
    return {"edges": n, "n_docs": bins, "n_docs_pct": {k: pct(v, n) for k, v in bins.items()},
            "n_topic_docs>=1": topic, "n_topic_docs>=1_pct": pct(topic, n)}


def origin_profile(mn: np.ndarray, rl: np.ndarray, nd: np.ndarray, nt: np.ndarray) -> dict:
    groups = {"all_dict_relink": rl == mn, "all_llm_table": rl == 0}
    groups["both"] = ~groups["all_dict_relink"] & ~groups["all_llm_table"]
    return {g: {"pct": pct(int(m.sum()), len(mn)), **ndocs_profile(nd[m], nt[m])} for g, m in groups.items()}


def overlap_part(mix: Edges, dic: Edges, mn: np.ndarray, rl: np.ndarray) -> dict:
    in_dict = dic.has(mix.key)
    in_mix = mix.has(dic.key)
    shared = int(in_dict.sum())
    mix_only_cuis = mix.key[~in_dict] & MASK
    dict_only_cuis = dic.key[~in_mix] & MASK
    no_dict_concept = ~np.isin(mix_only_cuis, dic.concepts)
    no_mix_concept = ~np.isin(dict_only_cuis, mix.concepts)
    nd_m, nd_d = mix.nd[in_dict], dic.nd[in_mix]          # same edge order: both arrays sorted by key
    return {
        "edges": {MIX: len(mix.key), DICT: len(dic.key), "shared": shared,
                  "S-mix-only": len(mix.key) - shared, "S-dict-only": len(dic.key) - shared,
                  "shared_pct_of_S-mix": pct(shared, len(mix.key)), "shared_pct_of_S-dict": pct(shared, len(dic.key)),
                  "jaccard_pct": pct(shared, len(mix.key) + len(dic.key) - shared)},
        "n_docs": {"shared (S-mix n_docs)": ndocs_profile(nd_m, mix.nt[in_dict]),
                   "shared (S-dict n_docs)": ndocs_profile(nd_d, dic.nt[in_mix]),
                   "S-mix-only": ndocs_profile(mix.nd[~in_dict], mix.nt[~in_dict]),
                   "S-dict-only": ndocs_profile(dic.nd[~in_mix], dic.nt[~in_mix])},
        "shared_n_docs_compared": {"S-dict more docs": int((nd_d > nd_m).sum()), "equal": int((nd_d == nd_m).sum()),
                                   "S-mix more docs": int((nd_d < nd_m).sum()),
                                   "median_ratio_S-dict_to_S-mix": float(np.median(nd_d / nd_m))},
        "concepts": {MIX: len(mix.concepts), DICT: len(dic.concepts),
                     "shared": int(np.isin(mix.concepts, dic.concepts).sum()),
                     "S-mix-only edges on a concept with no S-dict edge": int(no_dict_concept.sum()),
                     "distinct concepts of those S-mix-only edges": len(np.unique(mix_only_cuis[no_dict_concept])),
                     "S-dict-only edges on a concept with no S-mix edge": int(no_mix_concept.sum()),
                     "distinct concepts of those S-dict-only edges": len(np.unique(dict_only_cuis[no_mix_concept]))},
        "origin": {"S-mix-only": origin_profile(mn[~in_dict], rl[~in_dict], mix.nd[~in_dict], mix.nt[~in_dict]),
                   "shared": origin_profile(mn[in_dict], rl[in_dict], mix.nd[in_dict], mix.nt[in_dict])},
    }


# ---- (2) sampled S-mix-only edges against the S-dict pipeline


class TextTable:
    """The v2 phrase table looked up in its JSON text: S-dict sends only strings without a CUI through it, and the
    parsed table would take several GB."""

    def __init__(self, path: str):
        self.text = Path(path).read_text(encoding="utf-8")
        self.cache: dict[str, dict] = {}

    def get(self, key: str, default=None):
        if key not in self.cache:
            head = json.dumps(key) + ": {"
            i = self.text.find(head)
            while i > 0 and self.text[i - 2:i] != ", " and i != 1:
                i = self.text.find(head, i + 1)
            if i < 0:
                self.cache[key] = None
            else:
                j = self.text.index("}", i)
                self.cache[key] = json.loads(self.text[i + len(head) - 1:j + 1])
        v = self.cache[key]
        return default if v is None else v


def match_spans(toks: list[str], names) -> list[tuple[int, int, str, bool]]:
    """dict_ie.matches with positions and the negated matches it drops: (start, length, string, negated)."""
    out, i = [], 0
    while i < len(toks):
        for n in range(min(dict_ie.MAX_N, len(toks) - i), 0, -1):
            cand = " ".join(toks[i:i + n])
            if cand in names:
                out.append((i, n, cand, bool(dict_ie.NEG & set(toks[max(0, i - 5):i]))))
                i += n - 1
                break
        i += 1
    return out


def string_links(m: str, names, hpo, table) -> set[str]:
    """CUIs build_graph_v2.py gives a dictionary item spelling m."""
    if names[m]:
        out = {names[m][0]}
        if hc := hpo.get(_norm(m).strip()):
            out.add(hc)
        return out
    return {t["cui"] for _, t in linked_forms(m, table, True, hpo) if t.get("tui") not in EXCLUDED_TUI}


def contains(toks: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return bool(n) and any(toks[i:i + n] == sub for i in range(len(toks) - n + 1))


def doc_texts(d: str, doc_ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for src in SOURCES:
        p = f"{src}/{d.replace(':', '_')}.json"
        if os.path.exists(p):
            for x in json.loads(Path(p).read_text())["docs"]:
                if x["doc_id"] in doc_ids and x["doc_id"] not in out:
                    out[x["doc_id"]] = x["text"]
    return out


def semantic_types(cuis: set[str]) -> dict[str, list[tuple[str, str]]]:
    """CUI -> (TUI, semantic type name); the extracted MRSTY leaves the name empty, SRDEF has it."""
    with open(SRDEF) as fi:
        sty = {f[1]: f[2] for ln in fi if (f := ln.split("|"))[0] == "STY"}
    out: dict[str, list[tuple[str, str]]] = {}
    with open("data/umls_extracted/MRSTY.RRF") as fi:
        for ln in fi:
            f = ln.split("|", 2)
            if f[0] in cuis:
                out.setdefault(f[0], []).append((f[1], sty.get(f[1], "")))
    return out


def umls_info(cuis: set[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Preferred English name and semantic types of each CUI. Names come from the dictionary's MRCONSO, then from
    the larger subset for concepts that have no English string there."""
    best: dict[str, tuple[int, str]] = {}
    for path in ("data/umls_extracted/MRCONSO.RRF", "data/umls_subset/MRCONSO.RRF"):
        want = cuis - set(best)
        with open(path, encoding="utf-8") as fi:
            for ln in fi:
                f = ln.split("|", 15)
                if f[0] in want and f[1] == "ENG":
                    rank = (f[2] == "P") + (f[4] == "PF") + (f[6] == "Y")
                    if rank > best.get(f[0], (-1, ""))[0]:
                        best[f[0]] = (rank, f[14])
    tuis = {c: [t for t, _ in v] for c, v in semantic_types(cuis).items()}
    return {c: v[1] for c, v in best.items()}, tuis


def reachability_part(mix: Edges, in_dict: np.ndarray, mn: np.ndarray, rl: np.ndarray, names, hpo) -> dict:
    """S-mix edges by whether the S-dict extraction can name their concept at all: a UMLS finding string that
    names it, or the HPO name of such a string. Strings without a CUI also reach concepts through the v2 phrase
    table; those concepts are not counted as named here."""
    named = {v[0] for v in names.values() if v} | {hpo[k] for k in names if k in hpo}
    named_arr = np.fromiter((int(c[1:]) for c in named), np.int64, len(named))
    out = {"named_concepts": len(named)}
    only = ~in_dict
    for grp, m in (("S-mix-only", only), ("shared", in_dict)):
        ok = np.isin(mix.key[m] & MASK, named_arr)
        nd, nt, gmn, grl = mix.nd[m], mix.nt[m], mn[m], rl[m]
        out[grp] = {"edges": int(m.sum()),
                    "concept named by the dictionary": {"pct": pct(int(ok.sum()), len(ok)), **ndocs_profile(nd[ok], nt[ok]),
                                                        "origin": origin_profile(gmn[ok], grl[ok], nd[ok], nt[ok])},
                    "concept not named by the dictionary": {"pct": pct(int((~ok).sum()), len(ok)),
                                                            **ndocs_profile(nd[~ok], nt[~ok])}}
        if grp == "S-mix-only":
            cuis = mix.key[m][~ok] & MASK
            sty = semantic_types({f"C{int(c):07d}" for c in np.unique(cuis)})
            by_tui: Counter = Counter()
            finding = 0
            for c, k in zip(*np.unique(cuis, return_counts=True)):
                ts = sty.get(f"C{int(c):07d}", [])
                finding += int(k) * any(t in dict_ie.FINDING_TUI for t, _ in ts)
                for t, name in ts:
                    by_tui[f"{t} {name}"] += int(k)
            out[grp]["concept not named by the dictionary"].update({
                "distinct concepts": len(np.unique(cuis)),
                "edges whose concept has a finding semantic type": finding,
                "edges by semantic type (a concept may have several)": dict(by_tui.most_common(20))})
    return out


def diagnose(d: str, c: str, ms: list[dict], names, hpo, table, concept_strings: set[str]) -> dict:
    """The S-dict pipeline on the documents of one S-mix edge."""
    texts = doc_texts(d, {m["doc_id"] for m in ms})
    status: Counter = Counter()
    linked_docs = 0
    for text in texts.values():
        links: set[str] = set()
        for s in dict_ie.SENT.split(text):
            toks = dict_ie.TOK.findall(s.lower())
            spans = match_spans(toks, names)
            kept = [x[2] for x in spans if not x[3]]
            if kept != dict_ie.matches(s, names):
                sys.exit(f"match_spans differs from dict_ie.matches on {s!r}")
            for m in kept:
                links |= string_links(m, names, hpo, table)
            for k in concept_strings:
                st = k.split()
                for j in range(len(toks) - len(st) + 1):
                    if toks[j:j + len(st)] != st:
                        continue
                    same = [x for x in spans if x[0] == j and x[2] == k]
                    cover = [x for x in spans if x[0] <= j < x[0] + x[1] or j <= x[0] < j + len(st)]
                    if same:
                        status["negated" if same[0][3] else "extracted"] += 1
                    else:
                        status["inside other match: " + "/".join(sorted({x[2] for x in cover}))] += 1
        linked_docs += c in links
    verbatim = []
    for m in ms:
        llm = m["span"] or m["phrase"]
        text = texts.get(m["doc_id"])
        verbatim.append(None if text is None else contains(dict_ie.TOK.findall(text.lower()),
                                                            dict_ie.TOK.findall(llm.lower())))
    return {"docs_found": len(texts), "docs": len({m["doc_id"] for m in ms}),
            "sdict_links_concept_in_docs": linked_docs, "concept_string_occurrences": dict(status),
            "llm_phrase_verbatim_in_doc": verbatim}


def examples_part(mix: Edges, in_dict: np.ndarray, ids: list[str], n: int, seed: int, names, hpo,
                  table) -> list[dict]:
    only = np.flatnonzero(~in_dict)
    picks = sorted(random.Random(seed).sample(range(len(only)), n))
    edges = [(ids[int(mix.key[only[i]] >> CUI_BITS)], f"C{int(mix.key[only[i]] & MASK):07d}", int(only[i]))
             for i in picks]
    drv = GraphDatabase.driver(URI[MIX], auth=AUTH)
    with drv.session() as s:
        got = {(d, c): s.run(EXAMPLE, d=d, c=c).single() for d, c, _ in edges}
    drv.close()
    names_c, tuis = umls_info({c for _, c, _ in edges})
    want = {c for _, c, _ in edges}
    strings: dict[str, set[str]] = {c: set() for c in want}
    for k, v in names.items():
        if v and v[0] in want:
            strings[v[0]].add(k)
        if (h := hpo.get(k)) in want:
            strings[h].add(k)
    rows = []
    for d, c, i in edges:
        r = got[(d, c)]
        ms = sorted(r["mentions"], key=lambda m: (m["doc_id"], m["phrase"], m["sentence"], m["span"] or ""))
        if len(ms) != mix.nm[i]:
            sys.exit(f"{d} -> {c}: {len(ms)} Mentions, edge says {mix.nm[i]}")
        m = random.Random(f"{seed}:{d}:{c}").choice(ms)
        per = Counter("dict_relink" if x["span"] is not None else "llm_table" for x in ms)
        diag = diagnose(d, c, ms, names, hpo, table, strings[c])
        toks = dict_ie.TOK.findall(m["sentence"].lower())
        in_sent = [f"{x[2]}{' (negated)' if x[3] else ''} -> {'/'.join(sorted(string_links(x[2], names, hpo, table))) or '-'}"
                   for x in match_spans(toks, names)]
        rows.append({"disease_id": d, "disease": r["disease"], "cui": c, "concept": names_c.get(c, "(no English UMLS string here)"),
                     "concept_tuis": "/".join(tuis.get(c, [])), "n_docs": int(mix.nd[i]),
                     "n_topic_docs": int(mix.nt[i]), "n_mentions": int(mix.nm[i]),
                     "mentions_dict_relink": per["dict_relink"], "mentions_llm_table": per["llm_table"],
                     "edge_origin": "both" if len(per) == 2 else next(iter(per)),
                     "phrase": m["phrase"], "span": m["span"] or "", "sentence": m["sentence"], "doc_id": m["doc_id"],
                     "pmid": m["pmid"] or "", "title": m["title"] or "",
                     "concept_dict_strings": len(strings[c]),
                     "concept_dict_strings_examples": " | ".join(sorted(strings[c], key=lambda x: (len(x), x))[:5]),
                     "concept_strings_in_docs": "; ".join(f"{k}: {v}" for k, v in sorted(
                         diag["concept_string_occurrences"].items())) or "none",
                     "sdict_links_concept_in_docs": f"{diag['sdict_links_concept_in_docs']}/{diag['docs_found']}",
                     "llm_phrase_verbatim_in_doc": "/".join("?" if v is None else "y" if v else "n"
                                                            for v in diag["llm_phrase_verbatim_in_doc"]),
                     "dict_in_sentence": "; ".join(in_sent),
                     "all_mentions": [{k: x[k] for k in ("phrase", "span", "sentence", "doc_id", "topic")} for x in ms]})
    return rows


CSV_COLS = ["row", "disease_id", "disease", "cui", "concept", "concept_tuis", "n_docs", "n_topic_docs", "n_mentions",
            "mentions_dict_relink", "mentions_llm_table", "edge_origin", "phrase", "span", "sentence", "doc_id",
            "pmid", "title", "llm_phrase_verbatim_in_doc", "concept_dict_strings", "concept_dict_strings_examples",
            "concept_strings_in_docs", "sdict_links_concept_in_docs", "dict_in_sentence"]


# ---- (3) rank wins


def concept_info(s) -> tuple[dict[int, float | None], dict[int, list[int]]]:
    idf = {int(r["cui"][1:]): r["idf"] for r in s.run("MATCH (c:Concept) RETURN c.cui AS cui, c.idf AS idf")}
    parents: dict[int, list[int]] = {}
    for r in s.run("MATCH (c:Concept)-[:IS_A]->(p:Concept) RETURN c.cui AS c, p.cui AS p"):
        parents.setdefault(int(r["c"][1:]), []).append(int(r["p"][1:]))
    return idf, parents


def reach(ev: set[str], idf: dict, parents: dict, decay: float) -> dict[int, float]:
    """Concept -> weight, as the first two clauses of the RANK query."""
    w: dict[int, float] = {}
    for cui in ev:
        if not (cui[:1] == "C" and cui[1:].isdigit()) or (c0 := int(cui[1:])) not in idf:
            continue
        w[c0] = 1.0
        for p in parents.get(c0, ()):
            w[p] = max(w.get(p, 0.0), decay)
    return {c: x for c, x in w.items() if idf[c] is not None}


def disease_edges(dzs: list[int], w: dict[int, float], g: Edges, other: Edges, idf: dict, beta: float,
                  gamma: float) -> dict:
    out = {"edges": 0, "only": 0, "only_concept_absent_in_other": 0, "score": 0.0, "score_only": 0.0}
    wc = np.fromiter(w, np.int64, len(w))
    for dz in dzs:
        sl = g.of(dz)
        hit = np.flatnonzero(np.isin(g.key[sl] & MASK, wc))
        if not len(hit):
            continue
        keys = g.key[sl][hit]
        only = ~other.has(keys)
        absent = ~np.isin(keys & MASK, other.concepts)
        for j, k, o, a in zip(hit, keys, only, absent):
            c = int(k & MASK)
            nd, nt = g.nd[sl][j], g.nt[sl][j]
            term = w[c] * idf[c] * float(nd) ** beta * (1.0 if nt > 0 else gamma)
            out["edges"] += 1
            out["score"] += term
            if o:
                out["only"] += 1
                out["score_only"] += term
                out["only_concept_absent_in_other"] += bool(a)
    return out


def share(x: dict) -> float | None:
    return x["score_only"] / x["score"] if x["score"] else None


def summary(rows: list[dict], who: str) -> dict:
    n = len(rows)
    sh = [v for r in rows if (v := share(r[who])) is not None]
    return {"patients": n,
            "with_edge": sum(r[who]["edges"] > 0 for r in rows),
            "with_graph_only_edge": sum(r[who]["only"] > 0 for r in rows),
            "with_graph_only_edge_pct": pct(sum(r[who]["only"] > 0 for r in rows), n),
            "all_edges_graph_only": sum(r[who]["edges"] > 0 and r[who]["only"] == r[who]["edges"] for r in rows),
            "with_graph_only_edge_on_concept_absent_in_other":
                sum(r[who]["only_concept_absent_in_other"] > 0 for r in rows),
            "edges_median": statistics.median(r[who]["edges"] for r in rows) if n else None,
            "graph_only_edges_median": statistics.median(r[who]["only"] for r in rows) if n else None,
            "finding_sum_median": round(statistics.median(r[who]["score"] for r in rows), 3) if n else None,
            "graph_only_score_share_median": round(statistics.median(sh), 3) if sh else None,
            "graph_only_score_share_mean": round(statistics.fmean(sh), 3) if sh else None}


def scored(dx: GraphDx, ev: set[str], k: int = 10) -> tuple[dict[str, float], dict[str, int]]:
    """GraphDx.rank's two terms: finding scores of its k*4 window and unified-evidence counts."""
    with dx.drv.session() as s:
        fs = {r["id"]: r["score"] for r in s.run(
            RANK, ev=sorted(ev), decay=dx.decay, mu=dx.mu, smed=dx.smed, lbar=dx.lbar, alpha=dx.alpha, beta=dx.beta,
            gamma=dx.gamma, cands=None, k=k * 4)}
        uni = {r["id"]: r["n"] for r in s.run(UNIFIED, ev=sorted(ev))} if dx.unified_w else {}
    return fs, uni


def top(fs: dict[str, float], uni: dict[str, int], unified_w: float, k: int = 10) -> list[str]:
    sc = dict(fs)
    for d, n in uni.items():
        sc[d] = sc.get(d, 0.0) + unified_w * n
    return sorted(sc, key=lambda d: (-sc[d], d))[:k]


def ranking_part(mix: Edges, dic: Edges, dz_index: dict[str, int], args: dict) -> tuple[dict, list[dict]]:
    opendx_eval.RAW_GOLD = True
    sets = dev_sets(args["ddx_n"])
    runs = {g: {r["bench"]: r["ranks_lineage"] for r in read_json(RUN[g])["results"]} for g in RUN}
    graphs = {MIX: mix, DICT: dic}
    info, dx, norm = {}, {}, {}
    for g in (MIX, DICT):
        dx[g] = GraphDx(URI[g], "/".join(AUTH), args["alpha"], args["mu"], args["decay"], args["unified_w"],
                        args["support_beta"], args["offtopic_w"], args["full_unified"])
        with dx[g].drv.session() as s:
            info[g] = concept_info(s)
            # the length normalization of the RANK query, per disease
            norm[g] = {r["id"]: (r["n"] + dx[g].mu * (1.0 - r["sc"] / (r["sc"] + dx[g].smed)) * dx[g].lbar)
                       ** dx[g].alpha for r in s.run("MATCH (d:Disease) RETURN d.id AS id, d.n_findings AS n, "
                                                     "d.src_chars AS sc")}
    beta, gamma, decay = args["support_beta"], args["offtopic_w"], args["decay"]
    out, details = {}, []
    for b, focus in WINS.items():
        pats = [(gk, ev) for g, ev in sets[b] if (gk := v2_ids.gold_keys(g)) and ev]
        if any(len(runs[g][b]) != len(pats) for g in runs):
            sys.exit(f"{b}: {len(pats)} evaluated patients, runs hold {[len(runs[g][b]) for g in runs]}")
        res = {"focus_winner": focus}
        for win in (focus, DICT if focus == MIX else MIX):
            lose = DICT if win == MIX else MIX
            idf, parents = info[win]
            lidf, lpar = info[lose]
            rows = []
            for i, (gk, ev) in enumerate(pats):
                w = reach(ev, idf, parents, decay)
                dzs = [dz_index[k] for k in sorted(gk) if k in dz_index]
                rows.append({"i": i, "gold": disease_edges(dzs, w, graphs[win], graphs[lose], idf, beta, gamma)})
            rw, rl = runs[win][b], runs[lose][b]
            wins, mismatch = [], 0
            for r in rows:
                i = r["i"]
                if rw[i] is None or rl[i] is not None:
                    continue
                gk, ev = pats[i]
                lin = dx[win].lineage(gk)
                fs, uni = {}, {}
                for g in (win, lose):
                    fs[g], uni[g] = scored(dx[g], ev)
                ranked = {g: top(fs[g], uni[g], dx[g].unified_w) for g in (win, lose)}
                pos = {g: next((j + 1 for j, d in enumerate(ranked[g]) if d in lin), None) for g in ranked}
                if pos[win] != rw[i] or pos[lose] is not None:
                    mismatch += 1
                    continue
                hit = ranked[win][pos[win] - 1]
                w = reach(ev, idf, parents, decay)
                wl = reach(ev, lidf, lpar, decay)
                # Partial counterfactual: the winner's finding score of every lineage disease in its window without
                # its winner-only edges, all other diseases and every normalization left as they are.
                fs2 = dict(fs[win])
                for d in fs[win]:
                    if d in lin:
                        e = disease_edges([dz_index[d]], w, graphs[win], graphs[lose], idf, beta, gamma)
                        if abs(fs[win][d] * norm[win][d] - e["score"]) > 1e-6 * max(1.0, e["score"]):
                            sys.exit(f"{b} patient {i}: {d} finding sum {e['score']} vs RANK {fs[win][d] * norm[win][d]}")
                        fs2[d] = (e["score"] - e["score_only"]) / norm[win][d]
                pos2 = next((j + 1 for j, d in enumerate(top(fs2, uni[win], dx[win].unified_w)) if d in lin), None)
                comp = [disease_edges([dz_index[d]], wl, graphs[lose], graphs[win], lidf, beta, gamma)
                        for d in ranked[lose]]
                r.update({"hit": disease_edges([dz_index[hit]], w, graphs[win], graphs[lose], idf, beta, gamma),
                          "hit_id": hit, "hit_is_gold": hit in gk,
                          "unified": {g: uni[g].get(hit, 0) for g in (win, lose)},
                          "rank_without_winner_only": pos2,
                          "gold_in_loser": disease_edges([dz_index[k] for k in sorted(gk) if k in dz_index], wl,
                                                         graphs[lose], graphs[win], lidf, beta, gamma),
                          "loser_top10": {"diseases": len(comp), "with_graph_only_edge": sum(x["only"] > 0 for x in comp),
                                          "score": sum(x["score"] for x in comp),
                                          "score_only": sum(x["score_only"] for x in comp)}})
                wins.append(r)
                details.append({"set": b, "winner": win, "patient": i, "rank_winner": rw[i], "gold": sorted(gk),
                                "hit": hit, "hit_is_gold": hit in gk, "gold_edges": r["gold"], "hit_edges": r["hit"],
                                "unified_credit_of_hit": r["unified"],
                                "rank_without_winner_only_edges": pos2, "gold_edges_in_loser": r["gold_in_loser"],
                                "loser_top10": r["loser_top10"], "loser_top10_ids": ranked[lose]})
            top10 = [r for r in rows if rw[r["i"]] is not None]
            both = [r for r in top10 if rl[r["i"]] is not None]
            kept = sum(r["rank_without_winner_only"] is not None for r in wins)
            res[f"{win} wins"] = {
                "winner": win, "evaluated": len(rows), "wins": len(wins), "rank_check_mismatches": mismatch,
                "gold_diseases_of_wins": len({tuple(sorted(pats[r["i"]][0])) for r in wins}),
                "gold": summary(wins, "gold"), "hit": summary(wins, "hit"),
                "hit_is_gold": sum(r["hit_is_gold"] for r in wins),
                "hit_unified_credit_in_winner": sum(r["unified"][win] > 0 for r in wins),
                "hit_unified_credit_in_loser": sum(r["unified"][lose] > 0 for r in wins),
                "still_top10_without_winner_only_edges": kept,
                "still_top10_without_winner_only_edges_pct": pct(kept, len(wins)),
                "base_gold_all_evaluated": summary(rows, "gold"),
                "base_gold_winner_top10": summary(top10, "gold"),
                "base_gold_both_top10": summary(both, "gold"),
                "loser_gold": summary(wins, "gold_in_loser"),
                "loser_top10": {
                    "diseases_with_loser_only_edge_median":
                        statistics.median(r["loser_top10"]["with_graph_only_edge"] for r in wins) if wins else None,
                    "loser_only_score_share_median": round(statistics.median(
                        share(r["loser_top10"]) or 0.0 for r in wins), 3) if wins else None}}
            g_ = res[f"{win} wins"]
            print(f"{b:18s} {win} wins {g_['wins']:4d} (rank mismatches {mismatch}): gold has {win}-only edge "
                  f"{g_['gold']['with_graph_only_edge']}/{g_['wins']} ({g_['gold']['with_graph_only_edge_pct']}%), "
                  f"hit {g_['hit']['with_graph_only_edge']}; base all "
                  f"{g_['base_gold_all_evaluated']['with_graph_only_edge_pct']}%, both top10 "
                  f"{g_['base_gold_both_top10']['with_graph_only_edge_pct']}%; still top10 without them {kept}; "
                  f"loser top10 loser-only share {g_['loser_top10']['loser_only_score_share_median']}", flush=True)
        out[b] = res
    for d in dx.values():
        d.drv.close()
    return out, details


def merge_reading(path: str) -> None:
    res_path = OUT / "dict_mix_overlap.json"
    res = json.loads(res_path.read_text())
    with open(path, newline="") as f:
        reading = {int(r["row"]): r for r in csv.DictReader(f)}
    rows = res["examples"]["rows"]
    if sorted(reading) != list(range(1, len(rows) + 1)) or any(r["class"] not in READING for r in reading.values()):
        sys.exit(f"{path}: need rows 1..{len(rows)} with class in {sorted(READING)}")
    res["sample_reading"] = {
        "caveat": f"Claude's own reading of the {len(rows)} sampled S-mix-only edges against their sentences and the "
                  "S-dict diagnostics, not a clinical judgment. One class per edge; likely_wrong is chosen whenever "
                  "the sentence does not support the edge, whatever the wording.",
        "classes": READING, "counts": dict(Counter(reading[k]["class"] for k in sorted(reading))),
        "subtype_counts": dict(Counter(f"{reading[k]['class']}/{reading[k].get('subtype') or '-'}"
                                       for k in sorted(reading))),
        "rows": [{"row": k, "disease": rows[k - 1]["disease"], "concept": rows[k - 1]["concept"],
                  "class": reading[k]["class"], "subtype": reading[k].get("subtype") or "",
                  "note": reading[k]["note"]} for k in sorted(reading)]}
    res_path.write_text(json.dumps(res, indent=1))
    ex = OUT / "smix_only_examples.csv"
    with open(ex, newline="") as f:
        table = list(csv.DictReader(f))
    with open(ex, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS + ["my_reading", "my_reading_subtype", "my_note"])
        w.writerows([r[c] for c in CSV_COLS] + [reading[int(r["row"])][k] for k in ("class", "subtype", "note")]
                    for r in table)
    print(json.dumps(res["sample_reading"]["counts"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reading", default="",
                    help="CSV row,class,subtype,note: merge this reading of the sample and stop")
    a = ap.parse_args()
    if a.reading:
        merge_reading(a.reading)
        return
    frozen = read_json(FROZEN)["graph_systems"]
    args = read_json(RUN[MIX])["args"]
    if {k: v for k, v in args.items() if k not in ("out", "uri")} != \
            {k: v for k, v in read_json(RUN[DICT])["args"].items() if k not in ("out", "uri")}:
        sys.exit("the two development runs differ in configuration")

    graphs, ids = {}, None
    for g in (MIX, DICT):
        drv = GraphDatabase.driver(URI[g], auth=AUTH)
        with drv.session() as s:
            gid = sorted(r["id"] for r in s.run("MATCH (d:Disease) RETURN d.id AS id"))
            n = s.run("MATCH ()-[e:HAS_FINDING]->() RETURN count(e) AS n").single()["n"]
            if n != frozen[g]["edges"] or len(gid) != frozen[g]["diseases"]:
                sys.exit(f"{URI[g]}: {n} edges / {len(gid)} diseases, frozen {g} has "
                         f"{frozen[g]['edges']} / {frozen[g]['diseases']}")
            if ids is None:
                ids = gid
            elif gid != ids:
                sys.exit("the two graphs hold different Disease sets")
            dz_index = {d: i for i, d in enumerate(ids)}
            print(f"{g}: streaming {n} edges", flush=True)
            graphs[g] = Edges(s, dz_index, n)
        drv.close()
    mix, dic = graphs[MIX], graphs[DICT]

    print("S-mix: Mention origin per edge", flush=True)
    mn = np.zeros(len(mix.key), np.int32)
    rl = np.zeros(len(mix.key), np.int32)
    drv = GraphDatabase.driver(URI[MIX], auth=AUTH)
    with drv.session() as s:
        for r in s.run(ORIGINS):
            if not r["cs"]:
                continue
            keys = (dz_index[r["id"]] << CUI_BITS) | np.fromiter((int(c[1:]) for c in r["cs"]), np.int64, len(r["cs"]))
            pos = np.searchsorted(mix.key, keys)
            if not (mix.has(keys)).all():
                sys.exit(f"{r['id']}: Mentions on a concept without a HAS_FINDING edge")
            mn[pos], rl[pos] = r["ns"], r["rls"]
    drv.close()
    if (mn != mix.nm).any():
        sys.exit(f"{int((mn != mix.nm).sum())} S-mix edges whose Mention count differs from n_mentions")

    res = {"note": "edge = (disease id, concept CUI); S-mix = v3llmd_ns on 7689 (with Mentions), S-dict = v3umls_ns "
                   "on 7688; both at the frozen_config_E edge and disease counts",
           "graphs": {g: {"uri": URI[g], "variant": frozen[g]["variant"], "edges": len(graphs[g].key),
                          "diseases": len(ids), "run": RUN[g]} for g in graphs},
           "overlap": overlap_part(mix, dic, mn, rl)}
    o = res["overlap"]
    print(json.dumps(o["edges"]), flush=True)
    for k, v in o["n_docs"].items():
        print(f"  {k:24s} {v['edges']:9d}  n_docs 1/2-4/>=5 % {v['n_docs_pct']}  topic>=1 {v['n_topic_docs>=1_pct']}%")
    for k, v in o["origin"].items():
        print(f"  origin {k}: " + ", ".join(f"{g} {x['pct']}%" for g, x in v.items()))
    print(f"  concepts {o['concepts']}", flush=True)

    in_dict = dic.has(mix.key)
    print("loading the UMLS finding-string index and the v2 phrase table", flush=True)
    names = dict_ie.umls_index()
    hpo = hpo_name_index()
    res["dictionary_reach"] = reachability_part(mix, in_dict, mn, rl, names, hpo)
    for g in ("S-mix-only", "shared"):
        x = res["dictionary_reach"][g]
        print(f"  {g}: concept named by the dictionary {x['concept named by the dictionary']['pct']}%", flush=True)
    table = TextTable(TABLE_V2)
    rows = examples_part(mix, in_dict, ids, a.n_examples, a.seed, names, hpo, table)
    del names, hpo, table
    res["examples"] = {"csv": str(OUT / "smix_only_examples.csv"), "seed": a.seed, "n": len(rows),
                       "sampling": "uniform over S-mix-only edges sorted by (disease id, CUI); one Mention per edge "
                                   "shown, chosen with Random(f'{seed}:{disease}:{cui}')",
                       "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "smix_only_examples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        w.writerows([k, *(r[c] for c in CSV_COLS[1:])] for k, r in enumerate(rows, 1))
    del in_dict, mn, rl
    gc.collect()

    res["ranking"], res["ranking_wins"] = ranking_part(mix, dic, dz_index, args)
    res["ranking_note"] = ("wins = lineage Top-10 in the winner's stored run and not in the loser's; gold = edges of the "
                           "gold universe keys to the patient's reached concepts, hit = the same for the disease the "
                           "winner ranked at the stored position; graph-only = an edge of the graph the counts are "
                           "taken from (the winner for gold, hit and base_*, the loser for loser_gold and loser_top10) "
                           "that the other graph lacks; score share = its part of the 1-hop finding sum "
                           "(w * idf * n_docs^beta * topic factor, before length normalization and unified evidence); "
                           "base_* = the same gold counts over all evaluated patients, over those the winner ranks "
                           "in Top-10, and over those both rank in Top-10; loser_top10 = the ten diseases the loser "
                           "ranked (all above the gold)")
    (OUT / "dict_mix_overlap.json").write_text(json.dumps(res, indent=1))
    print(f"saved {OUT / 'dict_mix_overlap.json'}, {OUT / 'smix_only_examples.csv'}")


if __name__ == "__main__":
    main()
