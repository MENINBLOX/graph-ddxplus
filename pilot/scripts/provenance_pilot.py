"""Pilot: can each KG finding be attributed to the specific article it came from?

The deep sources concatenated many articles into one blob per disease and kept no article IDs, so an
edge can be traced to its disease's text but not to a paper. This pilot re-issues the same
disease-name queries (deep_fetch.pmc_ids), fetches each article's text SEPARATELY with its PMCID,
and asks for every extracted finding of the disease: which articles state it?

Reported per disease and overall:
  in_blob   finding surface string found in the stored blob (upper bound for string attribution)
  attrib    finding found in at least one separately fetched article (or the StatPearls chapter)
Query results can drift since the original fetch date, so attrib <= in_blob is expected.

    python provenance_pilot.py --n 20 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_fetch import DROP_SEC, KEEP_SEC, pmc_ids  # noqa: E402
from statpearls_fetch import _get, eutil, search as sp_search, semiology_text, to_text  # noqa: E402

P = "pilot/cache/prod/"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def pmc_articles(ids: list[str]) -> dict[str, str]:
    """{PMCID: kept-section text} for each article, same section filter as deep_fetch.pmc_text."""
    if not ids:
        return {}
    xml = eutil("efetch.fcgi", {"db": "pmc", "id": ",".join(ids), "retmode": "xml"})
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}
    out = {}
    for art in root.iter("article"):
        pid = next((("PMC" + e.text.lstrip("PMC")) for e in art.iter("article-id")
                    if e.get("pub-id-type") in ("pmc", "pmcid") and e.text), None)
        body = art.find(".//body")
        if not pid or body is None:
            continue
        parts = []
        for sec in body.iter("sec"):
            te = sec.find("title")
            title = "".join(te.itertext()).strip() if te is not None else ""
            if title and (DROP_SEC.search(title) or not KEEP_SEC.search(title)):
                continue
            parts += [" ".join(" ".join(p.itertext()).split()) for p in sec.iter("p")]
        out[pid] = " ".join(parts)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=P + "provenance_pilot.json")
    a = ap.parse_args()

    src_dir = P + "sources_full"
    cuis = sorted(f[:-5] for f in os.listdir(src_dir) if f.startswith("C"))
    pick = set(random.Random(a.seed).sample(cuis, a.n))

    import ijson
    findings: dict[str, list[str]] = {}
    with open(P + "ie_pred_merged29_promoted.json", "rb") as fh:
        for cui, v in ijson.kvitems(fh, ""):
            if cui in pick:
                findings[cui] = sorted({f["name"] for f in v.get("findings") or [] if f.get("name")})
                if len(findings) == len(pick):
                    break
    alt = json.load(open("pilot/cache/umls_maps/disease_cui2name.json", encoding="utf-8"))

    rows, tot = [], {"findings": 0, "in_blob": 0, "attrib": 0}
    for cui in sorted(pick):
        src = json.load(open(f"{src_dir}/{cui}.json", encoding="utf-8"))
        name, blob = src["disease"], norm(src["text"])
        docs = {}
        docs.update(pmc_articles(pmc_ids(name, 10, "review[Publication Type]", alt.get(cui, ""))))
        docs.update(pmc_articles(pmc_ids(name, 6, '"case reports"[Publication Type]', alt.get(cui, ""))))
        acc = src.get("provenance", {}).get("statpearls")
        if acc:
            docs[acc] = semiology_text(to_text(_get(f"https://www.ncbi.nlm.nih.gov/books/{acc}/")))[0]
        docs = {k: norm(v) for k, v in docs.items() if v}
        fs = findings.get(cui, [])
        per = []
        for f in fs:
            k = norm(f)
            hits = [d for d, t in docs.items() if k in t]
            per.append({"finding": f, "in_blob": k in blob, "docs": hits})
        n_blob = sum(p["in_blob"] for p in per)
        n_att = sum(bool(p["docs"]) for p in per)
        tot["findings"] += len(per)
        tot["in_blob"] += n_blob
        tot["attrib"] += n_att
        rows.append({"cui": cui, "disease": name, "n_docs": len(docs), "n_findings": len(per),
                     "in_blob": n_blob, "attrib": n_att, "findings": per})
        print(f"{name[:40]:40} docs {len(docs):3}  findings {len(per):4}  "
              f"in_blob {n_blob:4}  attrib {n_att:4}", flush=True)
    t = tot["findings"] or 1
    print(f"\nTOTAL findings {tot['findings']}  in_blob {100 * tot['in_blob'] / t:.1f}%  "
          f"attributed to an article {100 * tot['attrib'] / t:.1f}%")
    json.dump({"totals": tot, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
