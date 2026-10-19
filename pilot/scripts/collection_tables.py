"""Table pretest: do PMC tables carry the curated findings the paragraph text misses?

fetch_v2.py keeps paragraphs only. Genetic case series put the phenotype in a patient-by-feature
table, and a section check on the collection pretest found 59% of the newly found curated features
in tables. For the same 300 sampled diseases (collection_pretest.py), this fetches the tables of
(a) the articles already in the corpus and (b) the extra PMC articles, and scores text recall and
the random-disease control as in collection_pretest.py.

    set -a; . ./.env; set +a
    python collection_tables.py --pretest pilot/cache/v2/collect_pretest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curated_agreement as CA  # noqa: E402
from collection_pretest import efetch  # noqa: E402
from fetch_v2 import FetchError  # noqa: E402
from recall_decomp import hpo_terms, norm  # noqa: E402

V2 = "pilot/cache/v2"


def tables(pmcids: list[str]) -> dict[str, list[dict]]:
    """PMCID -> [{caption, rows: [[cell, ...], ...]}]"""
    out = {}
    ids = [p[3:] if p.startswith("PMC") else p for p in pmcids if p]
    for i in range(0, len(ids), 20):
        for art in efetch("pmc", ids[i:i + 20]).iter("article"):
            aid = {e.get("pub-id-type"): (e.text or "").strip() for e in art.iter("article-id")}
            pid = "PMC" + aid.get("pmc", aid.get("pmcid", "")).lstrip("PMC")
            ts = []
            for tw in art.iter("table-wrap"):
                cap = " ".join(" ".join(x.itertext()) for x in tw.iter("caption")).split()
                rows = [[" ".join("".join(c.itertext()).split()) for c in tr if c.tag in ("td", "th")]
                        for tr in tw.iter("tr")]
                ts.append({"caption": " ".join(cap), "rows": [r for r in rows if any(r)]})
            out[pid] = ts
    return out


def table_text(ts: list[dict]) -> str:
    return " ".join(t["caption"] + " " + " ".join(" | ".join(r) for r in t["rows"]) for t in ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretest", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(f"{a.pretest}/tables", exist_ok=True)
    res = json.load(open(f"{a.pretest}/result.json"))
    omims = [r["omim"] for r in res["per_disease"]]
    use = Counter()
    for f in glob.glob(f"{V2}/sources/*.json"):
        for x in json.load(open(f))["docs"]:
            use[x["doc_id"]] += 1

    def job(o):
        p = f"{a.pretest}/tables/{o.replace(':', '_')}.json"
        if os.path.exists(p):
            return
        rec = json.load(open(f"{a.pretest}/docs/{o.replace(':', '_')}.json"))
        bp = f"{V2}/sources/{rec['key'].replace(':', '_')}.json"
        base = [x for x in json.load(open(bp))["docs"] if use[x["doc_id"]] < 20] if os.path.exists(bp) else []
        bids = [x.get("pmcid") or "" for x in base if (x.get("pmcid") or "").startswith("PMC")]
        nids = [d["id"] for d in rec["pmc"]]
        try:
            t = tables(list(dict.fromkeys(bids + nids)))
        except FetchError:
            return
        json.dump({"base_ids": bids, "new_ids": nids, "tables": t}, open(p, "w"))
        print(o, len(bids), len(nids), sum(len(v) for v in t.values()), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, omims))

    cur = defaultdict(set)
    for line in open(CA.HPOA):
        f = line.rstrip("\n").split("\t")
        if len(f) > 10 and f[0].startswith("OMIM:") and f[2] != "NOT" and f[10] == "P":
            cur[f[0]].add(f[3])
    names = hpo_terms()
    done = [o for o in omims if os.path.exists(f"{a.pretest}/tables/{o.replace(':', '_')}.json")]
    rng = random.Random(8)
    ctrl = {o: rng.choice([x for x in done if x != o]) for o in done}
    V = ["base", "base+tables", "pmc_more", "pmc_more+tables", "pmc_full", "pmc_full+tables"]
    hit, chit = {v: [0, 0] for v in V}, {v: [0, 0] for v in V}
    for o in done:
        rec = json.load(open(f"{a.pretest}/docs/{o.replace(':', '_')}.json"))
        tb = json.load(open(f"{a.pretest}/tables/{o.replace(':', '_')}.json"))
        bp = f"{V2}/sources/{rec['key'].replace(':', '_')}.json"
        base = " ".join(x["text"] for x in (json.load(open(bp))["docs"] if os.path.exists(bp) else [])
                        if use[x["doc_id"]] < 20)
        bt = " ".join(table_text(tb["tables"].get(i, [])) for i in tb["base_ids"])
        nt = " ".join(table_text(tb["tables"].get(i, [])) for i in tb["new_ids"])
        clin = " ".join(d["clinical"] for d in rec["pmc"])
        full = " ".join(d["full"] for d in rec["pmc"])
        tx = {"base": base, "base+tables": base + " " + bt, "pmc_more": base + " " + clin,
              "pmc_more+tables": base + " " + bt + " " + clin + " " + nt, "pmc_full": base + " " + full,
              "pmc_full+tables": base + " " + bt + " " + full + " " + nt}
        tx = {k: " " + norm(v) + " " for k, v in tx.items()}
        own, other = cur[o], cur[ctrl[o]] - cur[o]
        for v in V:
            hit[v][0] += sum(any(f" {n} " in tx[v] for n in names.get(x, ())) for x in own)
            hit[v][1] += len(own)
            chit[v][0] += sum(any(f" {n} " in tx[v] for n in names.get(x, ())) for x in other)
            chit[v][1] += len(other)
    out = {"diseases": len(done),
           "recall": {v: round(100 * hit[v][0] / max(1, hit[v][1]), 1) for v in V},
           "control_recall": {v: round(100 * chit[v][0] / max(1, chit[v][1]), 1) for v in V}}
    json.dump(out, open(f"{a.pretest}/tables_result.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
