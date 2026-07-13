#!/usr/bin/env python3
"""Deep PubMed crawl — fix the 20-abstract/disease cap.

Diagnosis (2026-05-31): prior crawl capped abstracts_per_cui=20 (disk median 13),
producing artificially low symptom coverage (62% DDXPlus). PubMed itself contains
thousands of abstracts/disease including classic + case-report papers that DO
describe lay symptoms. This crawler retrieves up to --depth abstracts per disease
with reliable efetch (batched + retries), keeping only records that have an
abstract, so the effective abstract count is high.

Strict zero-shot: query is the disease NAME only (benchmark-blind). Output
schema matches the existing PubMed cache: {pmid, title, abstract} per line.
"""
from __future__ import annotations
import sys, os, json, time, argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
UA = "medkg-research/1.0 (academic)"
KEY = os.environ.get("NCBI_API_KEY", "")


def _get(url, params, timeout, tries=4):
    if KEY:
        params = {**params, "api_key": KEY}
    for t in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": UA})
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(0.5 * (t + 1))
    return None


def esearch(query, retmax):
    r = _get(ESEARCH, {"db": "pubmed", "term": query, "retmax": str(retmax),
                       "sort": "relevance", "retmode": "json"}, 30)
    if r is None:
        return []
    try:
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def efetch(pmids):
    r = _get(EFETCH, {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}, 90)
    if r is None:
        return []
    try:
        root = ET.fromstring(r.text)
    except Exception:
        return []
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid_e = art.find(".//PMID")
        title_e = art.find(".//ArticleTitle")
        parts = []
        for a in art.findall(".//AbstractText"):
            label = a.get("Label", "")
            txt = "".join(a.itertext())
            parts.append(f"{label}: {txt}" if label else txt)
        abstract = " ".join(parts).strip()
        if not abstract:
            continue
        out.append({"pmid": pmid_e.text if pmid_e is not None else "",
                    "title": (("".join(title_e.itertext())).strip()
                              if title_e is not None else ""),
                    "abstract": abstract})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="TSV: cui<TAB>name per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--depth", type=int, default=300,
                    help="esearch retmax (PMIDs requested per disease)")
    ap.add_argument("--min_abstracts", type=int, default=0,
                    help="skip re-crawl if existing file already has >= this many")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    delay = 0.11 if KEY else 0.4
    rows = [l.rstrip("\n").split("\t") for l in open(args.manifest) if l.strip()]
    print(f"deep crawl: {len(rows)} diseases, depth={args.depth}, "
          f"api_key={'set' if KEY else 'NONE'}", flush=True)

    for i, row in enumerate(rows):
        cui, name = row[0], row[1]
        fp = out / f"{cui}.jsonl"
        if fp.exists() and args.min_abstracts:
            n = sum(1 for _ in open(fp))
            if n >= args.min_abstracts:
                print(f"  [{i+1}/{len(rows)}] {name}: skip ({n} cached)", flush=True)
                continue
        pmids = esearch(name, args.depth)
        time.sleep(delay)
        recs = []
        for j in range(0, len(pmids), 100):
            recs += efetch(pmids[j:j + 100])
            time.sleep(delay)
        with open(fp, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [{i+1}/{len(rows)}] {name} ({cui}): {len(pmids)} pmids → "
              f"{len(recs)} abstracts", flush=True)


if __name__ == "__main__":
    main()
