"""What the generic-document rule of build_graph_v2.py removes (an article listed >= N times across disease files,
twice when one file lists it twice, is dropped for all).

Per corpus: generic articles, diseases left with no usable document, diseases that lose an on-topic document (title names
the disease), and the development-set gold diseases among them. On-topic is the builder's title test (name_forms in
the normalised title, without causal-gene titles; "exact" uses only the full name and synonyms, not the family stem), so a fix that keeps on-topic generic articles can be sized before any rebuild.

    python generic_doc_audit.py --sources pilot/cache/v2/sources,pilot/cache/v3/sources_new --out X.json [--max-doc-diseases 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval  # noqa: E402
import v2_ids  # noqa: E402
from build_graph_v2 import _norm, article, iter_sources, name_forms  # noqa: E402
from tune_dev import dev_sets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--max-doc-diseases", type=int, default=20)
    ap.add_argument("--exclude-diseases", default="pilot/cache/v2/exclude_nonhuman_diseases.json",
                    help="same default as build_graph_v2.py: these diseases are not graph diseases")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    uni = dict(v2_ids.universe())
    for k in json.load(open(a.exclude_diseases)) if a.exclude_diseases else ():
        uni.pop(k, None)
    use: Counter = Counter()
    docs: dict[str, set[str]] = {}
    topic: dict[str, set[str]] = defaultdict(set)      # disease -> its on-topic articles
    exact: dict[str, set[str]] = defaultdict(set)      # same, by the full name or a synonym only (no family stem)
    for dz, dz_docs in iter_sources(a.sources):
        held = {x["doc_id"] for x in dz_docs}
        for x in dz_docs:                              # counted as build_graph_v2.py does, excluded diseases included
            if not (x["doc_id"].startswith("TBL") and article(x["doc_id"]) in held):
                use[article(x["doc_id"])] += 1
        if dz not in uni:
            continue
        arts = {sys.intern(article(x["doc_id"])) for x in dz_docs}
        forms = name_forms(uni[dz])
        full = {f for n in [uni[dz]["name"], *uni[dz].get("syn", [])] if n and len(f := _norm(n)) >= 5}
        docs[dz] = arts
        for x in dz_docs:
            t = _norm(x.get("title") or "")
            if any(f in t for f in forms):
                topic[dz].add(article(x["doc_id"]))
            if any(f in t for f in full):
                exact[dz].add(article(x["doc_id"]))
    generic = {d for d, n in use.items() if n >= a.max_doc_diseases}
    with_docs = [dz for dz, s in docs.items() if s]
    emptied = [dz for dz in with_docs if not docs[dz] - generic]
    topic_lost = {dz: topic[dz] & generic for dz in with_docs if topic[dz] & generic}
    rescued = [dz for dz in emptied if topic[dz] & generic]
    exact_lost = {dz: exact[dz] & generic for dz in with_docs if exact[dz] & generic}
    exact_kept = Counter(a for s_ in exact_lost.values() for a in s_)   # article -> diseases that would keep it
    opendx_eval.RAW_GOLD = True
    emptied_set = set(emptied)
    pat_empty = pat_topic = pat_exact = pat_total = 0
    for pats in dev_sets().values():
        for g, ev in pats:
            keys = {k for k in v2_ids.gold_keys(g) if k in docs and docs[k]} if ev else set()
            if not keys:
                continue
            pat_total += 1
            pat_empty += keys <= emptied_set          # every gold key with documents loses all of them
            pat_topic += bool(keys & topic_lost.keys())
            pat_exact += bool(keys & exact_lost.keys())
    top = Counter()
    for d in generic:
        top[d] = use[d]
    out = {
        "sources": a.sources, "max_doc_diseases": a.max_doc_diseases,
        "articles": len(use), "generic_articles": len(generic),
        "diseases_with_documents": len(with_docs),
        "diseases_emptied": len(emptied),
        "diseases_emptied_with_on_topic_generic": len(rescued),
        "diseases_losing_on_topic_articles": len(topic_lost),
        "on_topic_generic_articles": len(set().union(*topic_lost.values())) if topic_lost else 0,
        "diseases_emptied_with_exact_topic_generic": sum(1 for dz in emptied if exact[dz] & generic),
        "diseases_losing_exact_topic_articles": len(exact_lost),
        "exact_topic_generic_articles": len(exact_kept),
        "exact_topic_generic_max_diseases_per_article": max(exact_kept.values(), default=0),
        "exact_topic_generic_articles_kept_by_20plus": sum(1 for n in exact_kept.values() if n >= 20),
        "dev_patients_with_documented_gold": pat_total,
        "dev_patients_gold_emptied": pat_empty,             # strict keys; lineage descendants may still score
        "dev_patients_gold_losing_on_topic": pat_topic,
        "dev_patients_gold_losing_exact_topic": pat_exact,
        "examples_emptied": [(dz, uni[dz]) for dz in emptied[:40]],
        "most_shared_generic": [(d, n) for d, n in top.most_common(30)],
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith(("examples", "most"))}, indent=1))


if __name__ == "__main__":
    main()
