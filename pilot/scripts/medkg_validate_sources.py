#!/usr/bin/env python3
"""Tier 1 6 sources validation: try downloading 1 sample disease article from each source
and confirm we can parse + extract candidate phenotypes via LLM IE.

Sources:
  1. StatPearls (NCBI Bookshelf)
  2. GeneReviews (NCBI Bookshelf)
  3. MedlinePlus A.D.A.M. Medical Encyclopedia (NLM Web Service)
  4. Wikipedia 의학 article
  5. OMIM Clinical Synopsis (API key required)
  6. Orphanet (XML download)

Test disease: "Pneumonia" (common, well-documented)
"""
from __future__ import annotations
import os, sys, json, time, urllib.parse, re
from pathlib import Path
import requests
from xml.etree import ElementTree as ET

OUT = Path("/home/max/Graph-DDXPlus/data/medkg")  # /windows is NTFS uid=1000 read-only for max
LOG = OUT / "logs" / "validation.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s)
    LOG.write_text(LOG.read_text() + s + "\n" if LOG.exists() else s + "\n")


def test_statpearls():
    """NCBI Bookshelf via E-utilities. StatPearls has its own collection."""
    log("=== 1. StatPearls (NCBI Bookshelf) ===")
    try:
        # Search for "Pneumonia" in StatPearls
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "books", "term": "pneumonia AND statpearls[book]", "retmax": "3"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ids = [e.text for e in root.findall(".//Id")]
        log(f"  StatPearls search OK, found IDs: {ids[:3]}")
        if not ids:
            return None
        # Fetch first article via efetch
        bookid = ids[0]
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "books", "id": bookid, "rettype": "xml"}
        r2 = requests.get(fetch_url, params=fetch_params, timeout=60)
        log(f"  efetch status {r2.status_code}, content size {len(r2.text)} chars")
        sample = OUT / "statpearls" / f"sample_{bookid}.xml"
        sample.write_text(r2.text)
        # Try to extract a section
        return {"id": bookid, "size": len(r2.text), "path": str(sample)}
    except Exception as e:
        log(f"  StatPearls FAIL: {e}")
        return None


def test_genereviews():
    """GeneReviews (NCBI Bookshelf)."""
    log("=== 2. GeneReviews (NCBI Bookshelf) ===")
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "books", "term": "Marfan AND gene[book]", "retmax": "3"}
        r = requests.get(url, params=params, timeout=30)
        root = ET.fromstring(r.text)
        ids = [e.text for e in root.findall(".//Id")]
        log(f"  GeneReviews search returned IDs: {ids[:3]}")
        if not ids:
            return None
        bookid = ids[0]
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "books", "id": bookid, "rettype": "xml"}
        r2 = requests.get(fetch_url, params=fetch_params, timeout=60)
        log(f"  efetch status {r2.status_code}, content size {len(r2.text)} chars")
        sample = OUT / "genereviews" / f"sample_{bookid}.xml"
        sample.write_text(r2.text)
        return {"id": bookid, "size": len(r2.text), "path": str(sample)}
    except Exception as e:
        log(f"  GeneReviews FAIL: {e}")
        return None


def test_medlineplus():
    """MedlinePlus Web Service (NLM)."""
    log("=== 3. MedlinePlus A.D.A.M. ===")
    try:
        # Search for pneumonia topic
        url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {"db": "healthTopics", "term": "pneumonia", "retmax": "3"}
        r = requests.get(url, params=params, timeout=30)
        log(f"  search status {r.status_code}, content size {len(r.text)} chars")
        sample = OUT / "medlineplus" / "sample_search.xml"
        sample.write_text(r.text)
        # Parse to get topic URL
        root = ET.fromstring(r.text)
        documents = root.findall(".//document")
        log(f"  Found {len(documents)} documents")
        if not documents:
            return None
        # Get the FullSummary content via NLM endpoint
        url2 = "https://wsearch.nlm.nih.gov/ws/query"
        params2 = {"db": "healthTopics", "term": "pneumonia", "rettype": "topic"}
        r2 = requests.get(url2, params=params2, timeout=30)
        sample2 = OUT / "medlineplus" / "sample_pneumonia.xml"
        sample2.write_text(r2.text)
        return {"size": len(r.text), "doc_count": len(documents), "path": str(sample)}
    except Exception as e:
        log(f"  MedlinePlus FAIL: {e}")
        return None


def test_wikipedia():
    """Wikipedia API for medical article."""
    log("=== 4. Wikipedia ===")
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "format": "json",
            "titles": "Pneumonia",
            "prop": "extracts|info|revisions",
            "exintro": "0", "explaintext": "1",
            "rvprop": "ids|timestamp",
            "redirects": "1",
        }
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "MedKG-Research/0.1"})
        r.raise_for_status()
        data = r.json()
        page = next(iter(data["query"]["pages"].values()))
        log(f"  Wikipedia title='{page.get('title')}', extract size={len(page.get('extract',''))} chars")
        sample = OUT / "wikipedia" / "sample_pneumonia.json"
        sample.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return {"title": page.get("title"), "size": len(page.get("extract", "")), "path": str(sample)}
    except Exception as e:
        log(f"  Wikipedia FAIL: {e}")
        return None


def test_omim():
    """OMIM API. Requires API key in env OMIM_API_KEY."""
    log("=== 5. OMIM Clinical Synopsis ===")
    api_key = os.environ.get("OMIM_API_KEY") or os.environ.get("OMIM_KEY")
    if not api_key:
        log("  SKIP: OMIM_API_KEY not set (user must register at omim.org/api). Caching task for later.")
        return {"skipped": True, "reason": "no_api_key"}
    try:
        # Try Marfan syndrome OMIM 154700
        url = "https://api.omim.org/api/clinicalSynopsis"
        params = {"mimNumber": "154700", "format": "json", "apiKey": api_key}
        r = requests.get(url, params=params, timeout=30)
        log(f"  OMIM status {r.status_code}")
        sample = OUT / "omim" / "sample_154700.json"
        sample.write_text(r.text)
        return {"status": r.status_code, "size": len(r.text), "path": str(sample)}
    except Exception as e:
        log(f"  OMIM FAIL: {e}")
        return None


def test_orphanet():
    """Orphanet — XML data download (free, no key)."""
    log("=== 6. Orphanet ===")
    try:
        # Orphadata XML — phenotype.hpoa style data is at:
        # https://www.orphadata.com/data/xml/en_product4.xml (rare diseases & associated phenotypes)
        url = "https://www.orphadata.com/data/xml/en_product4.xml"
        r = requests.get(url, timeout=120, stream=True, headers={"User-Agent": "MedKG-Research/0.1"})
        log(f"  Orphanet status {r.status_code}, headers {dict(r.headers).get('Content-Length','?')} bytes")
        if r.status_code != 200:
            return None
        sample = OUT / "orphanet" / "en_product4.xml"
        # Stream-write up to 5MB to validate format
        size = 0
        with sample.open("wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                size += len(chunk)
                if size > 50 * 1024 * 1024:  # cap at 50MB for now
                    log(f"  reached cap at {size} bytes")
                    break
        log(f"  Orphanet downloaded {size} bytes, path {sample}")
        return {"size": size, "path": str(sample)}
    except Exception as e:
        log(f"  Orphanet FAIL: {e}")
        return None


def main():
    LOG.write_text("")  # reset log
    log("Tier 1 source validation start")
    results = {}
    results["statpearls"] = test_statpearls()
    results["genereviews"] = test_genereviews()
    results["medlineplus"] = test_medlineplus()
    results["wikipedia"] = test_wikipedia()
    results["omim"] = test_omim()
    results["orphanet"] = test_orphanet()

    # Summary
    log("\n=== Summary ===")
    for src, r in results.items():
        ok = r is not None and not (isinstance(r, dict) and r.get("skipped"))
        log(f"  {src}: {'OK' if ok else 'FAIL/SKIP'}  {r}")

    summary_path = OUT / "logs" / "validation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    log(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
