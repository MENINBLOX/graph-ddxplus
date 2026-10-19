"""Corpus v2: per-article literature collection with provenance (research design §3, RQ3).

Differences from the v1 deep fetch, each closing a reviewer attack point:
  * every disease in the universe gets the SAME caps (R3: no deeper collection for benchmark golds);
  * articles are stored separately with PMID / PMCID / NBK accession, title, year, license, so every
    extracted finding can be traced to the article and sentence it came from (RQ3);
  * articles whose PMID is a benchmark case's source publication are dropped (R2 leakage).
Queries use the disease NAME only (and its ontology synonyms as fallback), never benchmark content.

    python fetch_v2.py --universe U.json --out-dir DIR --exclude-pmids X.json [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_fetch import DROP_SEC, KEEP_SEC, name_variants  # noqa: E402
from statpearls_fetch import eutil  # noqa: E402

class FetchError(RuntimeError):
    """A request failed (empty body, rate-limit JSON, unparsable XML). Distinguished from a valid
    empty result: v2's first full run treated failures as "no literature", which silently left
    ~20% of zero-document diseases (e.g. Croup) without sources."""


def _retry(fn, tries: int = 6):
    import time
    for k in range(tries):
        try:
            return fn()
        except FetchError:
            time.sleep(2 * (k + 1))
    raise FetchError("gave up after retries")


def _ids(term: str, retmax: int) -> list[str]:
    def once():
        r = eutil("esearch.fcgi", {"db": "pmc", "term": term, "retmax": retmax, "retmode": "json",
                                   "sort": "relevance"})
        try:
            j = json.loads(r)
        except Exception:
            raise FetchError("esearch: unparsable or empty response")
        if "esearchresult" not in j or "ERROR" in j.get("esearchresult", {}) or j.get("error"):
            raise FetchError(f"esearch error: {str(j)[:120]}")
        return j["esearchresult"].get("idlist", [])
    return _retry(once)


def pmc_ids(names: list[str], retmax: int, ptype: str) -> list[str]:
    for field in ("[Title]", ""):
        for v in names:
            ids = _ids(f'"{v}"{field} AND open access[filter] AND {ptype}', retmax)
            if ids:
                return ids
    return []


def pmc_articles(ids: list[str], kind: str, exclude: set[str], per_cap: int) -> list[dict]:
    if not ids:
        return []
    def once():
        xml = eutil("efetch.fcgi", {"db": "pmc", "id": ",".join(ids), "retmode": "xml"})
        try:
            return ET.fromstring(xml)
        except ET.ParseError:
            raise FetchError("efetch: unparsable or empty XML")
    root = _retry(once)
    out = []
    for art in root.iter("article"):
        aid = {e.get("pub-id-type"): (e.text or "").strip() for e in art.iter("article-id")}
        pmcid = aid.get("pmc") or aid.get("pmcid") or ""
        pmcid = ("PMC" + pmcid.lstrip("PMC")) if pmcid else ""
        pmid = aid.get("pmid", "")
        if pmid and pmid in exclude:
            out.append({"excluded": True, "pmid": pmid, "pmcid": pmcid})
            continue
        body = art.find(".//body")
        if body is None or not pmcid:
            continue
        parts = []
        for sec in body.iter("sec"):
            te = sec.find("title")
            title = "".join(te.itertext()).strip() if te is not None else ""
            if title and (DROP_SEC.search(title) or not KEEP_SEC.search(title)):
                continue
            parts += [" ".join(" ".join(p.itertext()).split()) for p in sec.iter("p")]
        text = " ".join(parts)[:per_cap]
        if not text:
            continue
        tt = art.find(".//title-group/article-title")
        yr = art.find(".//pub-date/year")
        lic = art.find(".//permissions/license")
        lic_s = ""
        if lic is not None:
            ref = lic.find("{http://www.niso.org/schemas/ali/1.0/}license_ref")
            link = lic.find(".//ext-link")
            lic_s = ((ref.text or "").strip() if ref is not None else "") or \
                lic.get("{http://www.w3.org/1999/xlink}href") or \
                (link.get("{http://www.w3.org/1999/xlink}href") if link is not None else "") or \
                lic.get("license-type") or " ".join("".join(lic.itertext()).split())[:120]
        out.append({"doc_id": pmcid, "pmid": pmid, "pmcid": pmcid, "kind": kind,
                    "title": " ".join("".join(tt.itertext()).split()) if tt is not None else "",
                    "year": yr.text if yr is not None else "", "license": lic_s, "text": text})
    return out


BOOKS: dict[str, list[dict]] = {}
BOOK_LICENSE = {"statpearls": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                "genereviews": "GeneReviews (University of Washington); noncommercial use permitted"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude-pmids", required=True)
    ap.add_argument("--reviews", type=int, default=6)
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--per-article", type=int, default=12000, help="chars kept per article")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extend-delta", default="",
                    help="extend mode: re-query with the (larger) caps, add only articles not yet stored "
                         "to the existing source files, and write the new articles to this directory")
    ap.add_argument("--book-index", default="pilot/cache/v2/book_index.json")
    ap.add_argument("--seed-order", type=int, default=0, help="shuffle seed for a representative pilot")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    uni = json.load(open(a.universe))
    exclude = set(json.load(open(a.exclude_pmids)))
    for d in json.load(open(a.book_index)).values():
        BOOKS.setdefault(_norm(d["title"]), []).append(d)
    keys = sorted(uni)
    if a.seed_order:
        import random
        random.Random(a.seed_order).shuffle(keys)
    if a.extend_delta:
        os.makedirs(a.extend_delta, exist_ok=True)
        todo = [k for k in keys if not os.path.exists(f"{a.extend_delta}/{k.replace(':', '_')}.json")]
    else:
        todo = [k for k in keys if not os.path.exists(f"{a.out_dir}/{k.replace(':', '_')}.json")]
    if a.limit:
        todo = todo[:a.limit]
    print(f"fetch v2: {len(todo)} diseases, caps reviews={a.reviews} cases={a.cases} "
          f"per-article={a.per_article}", flush=True)

    def one(k: str):
        try:
            return _one(k)
        except FetchError as e:          # not written: the next run retries this disease
            return k, -1, 0, 0

    def _one(k: str):
        info = uni[k]
        # name_variants() drops forms shorter than 6 characters, so "Croup", "Gout" or "Mumps" were
        # never queried at all; the disease's own name is always tried first.
        base = [info["name"]] if len(info["name"].strip()) >= 3 else []
        names = list(dict.fromkeys(base + name_variants(info["name"]) +
                                   [s for s in info.get("syn", []) if s]))
        docs = []
        docs += pmc_articles(pmc_ids(names, a.reviews, "review[Publication Type]"), "review", exclude, a.per_article)
        docs += pmc_articles(pmc_ids(names, a.cases, '"case reports"[Publication Type]'), "case_report", exclude, a.per_article)
        # Bookshelf chapters from the versioned LitArch bundles (book_index_v2.py), matched by
        # exact normalized title against the same disease-name forms used for PMC.
        seen = set()
        for nm in names:
            for d in BOOKS.get(_norm(nm), []):
                if d["doc_id"] in seen:
                    continue
                seen.add(d["doc_id"])
                docs.append({"doc_id": d["doc_id"], "pmid": "", "pmcid": "", "kind": d["book"],
                             "title": d["title"], "year": "", "license": BOOK_LICENSE[d["book"]],
                             "text": d["text"][:a.per_article * 2]})
        rec = {"id": k, "name": info["name"], "source_vocab": info["src"],
               "docs": [d for d in docs if not d.get("excluded")],
               "excluded_pmids": [d["pmid"] for d in docs if d.get("excluded")]}
        path = f"{a.out_dir}/{k.replace(':', '_')}.json"
        if a.extend_delta:
            old = json.load(open(path)) if os.path.exists(path) else {**rec, "docs": [], "excluded_pmids": []}
            have = {d["doc_id"] for d in old["docs"]}
            new = [d for d in rec["docs"] if d["doc_id"] not in have and d["kind"] in ("review", "case_report")]
            old["docs"] += new
            old["excluded_pmids"] = sorted(set(old.get("excluded_pmids", [])) | set(rec["excluded_pmids"]))
            tmp = path + ".tmp"
            json.dump(old, open(tmp, "w"))
            os.replace(tmp, path)
            json.dump({**rec, "docs": new}, open(f"{a.extend_delta}/{k.replace(':', '_')}.json", "w"))
            rec = {**rec, "docs": new}
        else:
            json.dump(rec, open(path, "w"))
        return k, len(rec["docs"]), sum(len(d["text"]) for d in rec["docs"]), len(rec["excluded_pmids"])

    n = errors = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k, nd, nc, nx in ex.map(one, todo):
            n += 1
            errors += nd < 0
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} last {k} docs={nd} chars={nc} excluded={nx} errors={errors}", flush=True)
    if errors:
        raise SystemExit(f"{errors} diseases failed; re-run to retry them")


if __name__ == "__main__":
    main()
