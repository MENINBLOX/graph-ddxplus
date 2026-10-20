"""What the LLM phrase table adds to the S-dict graph (research design §9.32 P0-2).

build_graph_v2.py links a UMLS dictionary item that carries no CUI (dict_ie.py: a string naming several finding
concepts, none by its unique preferred name) through linked_forms(): the phrase table for the whole string and
its coordinated parts, then the HPO name index. S-dict (v3umls_ns) used the gemma phrase table; the pure
dictionary graph v3umlsnt_ns (run_pure_dict.sh) passes an empty one, so only the HPO name index links such an
item. This script replays the build's item filters with the v3umls_ns flags and counts, per item, the concepts
each table yields, then the (disease, concept) edges whose document or topic-document count differs. The counts
must reconcile exactly with the two build summaries (edges and mentions), or the script stops.

    python pilot/scripts/pure_dict_diff.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

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

V = Path("pilot/cache/v2")
V3 = Path("pilot/cache/v3")
OUT = V3 / "p0/pure_dict"
IE = [V / "dict_umls_v2.jsonl.gz", V3 / "dict_umls_v3.jsonl.gz"]
SOURCES = f"{V}/sources,{V3}/sources_new"
TABLE = V / "phrase2cui_gemma.json"
S_DICT = V / "devD/variants/graph_v3umls_ns.json"
PURE = OUT / "graph_v3umlsnt_ns.json"
# build_graph_v2.py defaults, which run_graph_variants.sh leaves unchanged
EXCLUDED_DZ = V / "exclude_nonhuman_diseases.json"
EXCLUDED_TUI = {"T058", "T059", "T060", "T061"}
MAX_DOC_DISEASES = 20
ID = re.compile(r'\{"id": "([^"]+)"')
TOP = 25


class Corpus:
    """The build's item filters and topic rule for v3umls_ns (no year cut, no gene cut, --topic-genes, --no-self)."""

    def __init__(self) -> None:
        self.uni = dict(v2_ids.universe())
        for k in json.loads(EXCLUDED_DZ.read_text()):
            self.uni.pop(k, None)
        self.self_cuis: dict[str, set[str]] = defaultdict(set)
        for c, p in v2_ids.disease_isa():
            self.self_cuis[c].add(p)
        for k in self.uni:
            self.self_cuis[k].add(k)
        self.gene_forms = causal_gene_forms()
        use: Counter = Counter()
        self.title: dict[str, str] = {}
        self.held: dict[str, set[str]] = {}
        for dz, docs in iter_sources(SOURCES):
            self.held[dz] = {sys.intern(x["doc_id"]) for x in docs}
            for x in docs:
                if not (x["doc_id"].startswith("TBL") and article(x["doc_id"]) in self.held[dz]):
                    use[article(x["doc_id"])] += 1
                self.title[x["doc_id"]] = _norm(x.get("title") or "")
        self.generic = {d for d, n in use.items() if n >= MAX_DOC_DISEASES}
        self.forms: dict[str, set[str]] = {}

    def kept(self, d: str, doc: str) -> bool:
        return d in self.uni and doc in self.held.get(d, ()) and article(doc) not in self.generic

    def concepts(self, d: str, links: list[tuple[str, dict]]) -> list[str]:
        return [t["cui"] for _, t in links if t.get("tui") not in EXCLUDED_TUI and t["cui"] not in self.self_cuis[d]]

    def topic(self, d: str, doc: str) -> bool:
        if d not in self.forms:
            self.forms[d] = name_forms(self.uni[d]) | self.gene_forms.get(d, set())
        return any(f in self.title.get(doc, "") for f in self.forms[d])


def items(only: set[str] | None = None):
    """(disease, item) over both dictionary files; with `only`, other diseases' lines are not parsed."""
    for path in IE:
        with open_text(str(path)) as f:
            for ln in f:
                if only is not None and ((m := ID.match(ln)) is None or m.group(1) not in only):
                    continue
                r = json.loads(ln)
                for it in r.get("items", []):
                    yield r["id"], it


def cui_item_concepts(c: Corpus, d: str, it: dict, hpo: dict[str, str]) -> list[str]:
    links = [(it["t"], {"cui": it["cui"], "tui": it["tui"]})]
    hc = hpo.get(_norm(it["t"]).strip())
    if hc and hc != it["cui"]:
        links.append((it["t"], {"cui": hc, "tui": ""}))
    return c.concepts(d, links)


def main() -> None:
    table = json.loads(TABLE.read_text())
    hpo = hpo_name_index()
    c = Corpus()
    print(f"corpus: {len(c.uni)} diseases, {len(c.title)} documents, {len(c.generic)} generic", flush=True)

    # pass 1: the items without a CUI, linked with the phrase table (T) and with the empty table (E)
    n: Counter = Counter()
    outcome: Counter = Counter()
    strings: Counter = Counter()
    lost_strings: Counter = Counter()
    lost_link: dict[str, list[str]] = {}
    diff: dict[tuple[str, str], list[set[str]]] = {}      # (d, c) -> [T-only docs, T-only topic, E-only docs, E-only topic]
    for d, it in items():
        n["items"] += 1
        if it.get("cui"):
            continue
        n["cuiless"] += 1
        if not c.kept(d, it["doc"]):
            continue
        n["cuiless_kept"] += 1
        ct = set(c.concepts(d, linked_forms(it["t"], table, True, hpo)))
        ce = set(c.concepts(d, linked_forms(it["t"], {}, True, hpo)))
        n["mentions_T"] += len(ct)
        n["mentions_E"] += len(ce)
        strings[it["t"]] += 1
        if ct == ce:
            outcome["same links" if ct else "no link either way"] += 1
            continue
        outcome["table only (no link without it)" if not ce else
                "table adds concepts" if ce < ct else "differs both ways"] += 1
        lost_strings[it["t"]] += 1
        lost_link.setdefault(it["t"], sorted(ct - ce))
        topic = c.topic(d, it["doc"])
        for side, cs in ((0, ct - ce), (2, ce - ct)):
            for cui in cs:
                e = diff.setdefault((d, cui), [set(), set(), set(), set()])
                e[side].add(it["doc"])
                if topic:
                    e[side + 1].add(it["doc"])
    print(f"pass 1: {dict(n)}; {len(diff)} (disease, concept) keys differ", flush=True)

    # pass 2: what the shared links give those keys (CUI items, and the links both tables make)
    common: dict[tuple[str, str], list[set[str]]] = {k: [set(), set()] for k in diff}
    for d, it in items({d for d, _ in diff}):
        if not c.kept(d, it["doc"]):
            continue
        if it.get("cui"):
            cs = set(cui_item_concepts(c, d, it, hpo))
        else:
            cs = (set(c.concepts(d, linked_forms(it["t"], table, True, hpo)))
                  & set(c.concepts(d, linked_forms(it["t"], {}, True, hpo))))
        for cui in cs:
            if (e := common.get((d, cui))) is not None:
                e[0].add(it["doc"])
                if c.topic(d, it["doc"]):
                    e[1].add(it["doc"])

    edges: Counter = Counter()
    for k, (td, tt, ed, et) in diff.items():
        cd, cdt = common[k]
        dt, de = cd | td, cd | ed
        nt_t, nt_e = len(cdt | tt), len(cdt | et)
        if dt and not de:
            edges["removed"] += 1
        elif de and not dt:
            edges["added"] += 1
        elif len(dt) != len(de) or nt_t != nt_e:
            edges["n_docs or n_topic_docs changed"] += 1
        else:
            edges["mention count only"] += 1

    s_dict, pure = json.loads(S_DICT.read_text()), json.loads(PURE.read_text())
    want_edges = s_dict["edges"] - edges["removed"] + edges["added"]
    want_mentions = s_dict["mentions"] - n["mentions_T"] + n["mentions_E"]
    if (want_edges, want_mentions) != (pure["edges"], pure["mentions"]):
        raise SystemExit(f"does not reconcile: edges {want_edges} vs {pure['edges']}, "
                         f"mentions {want_mentions} vs {pure['mentions']}")

    res = {"note": "items = UMLS dictionary items (dict_umls_v2 + dict_umls_v3); cuiless_kept = items without a CUI "
                   "that pass the build's document filters; mentions_T/E = their concept links with the gemma phrase "
                   "table / the empty table (HPO name index only), after the excluded-TUI and no-self filters. Edges "
                   "counted over the (disease, concept) keys where the two tables differ; the reconciliation with the "
                   "build summaries passed.",
           "counts": dict(n), "item_outcome": dict(outcome), "edges": dict(edges),
           "summaries": {"S-dict v3umls_ns": s_dict, "pure v3umlsnt_ns": pure},
           "top_cuiless_strings": strings.most_common(TOP),
           "top_strings_losing_links": [[s, k, lost_link[s]] for s, k in lost_strings.most_common(TOP)]}
    (OUT / "diff.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("counts", "item_outcome", "edges")}, indent=1))
    print(f"summaries: S-dict edges {s_dict['edges']} mentions {s_dict['mentions']} concepts {s_dict['concepts']}; "
          f"pure edges {pure['edges']} mentions {pure['mentions']} concepts {pure['concepts']}")
    for s, k, cs in res["top_strings_losing_links"]:
        print(f"  {k:9d}  {s!r:40s} table -> {cs}")
    print(f"saved {OUT / 'diff.json'}")


if __name__ == "__main__":
    main()
