#!/usr/bin/env python3
"""PubMed crawl for combined seed list.

For each seed CUI:
  - Search PubMed using preferred disease name (or list of synonyms)
  - Fetch top-N abstracts (default N=20)
  - Save per-CUI JSONL with PMID + title + abstract

Output: $MEDKG_ROOT/pubmed/{cui}.jsonl

Resumable: skip CUIs that already have a saved file.

Usage:
  python medkg_pubmed_crawl.py [--limit=N] [--abstracts_per_cui=K]
"""
from __future__ import annotations
import sys, os, json, time, argparse, requests
from pathlib import Path
from xml.etree import ElementTree as ET
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

PUBMED_DIR = MEDKG_ROOT / "pubmed"
PUBMED_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = MEDKG_ROOT / "logs" / "pubmed_crawl.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

UA = "MedKG-Research/0.1 (academic; max@meninblox.com)"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(s + "\n")


def esearch(query, retmax):
    params = {"db": "pubmed", "term": query, "retmax": str(retmax),
              "sort": "relevance", "retmode": "json"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(ESEARCH, params=params, timeout=30, headers={"User-Agent": UA})
    if r.status_code != 200:
        return []
    try:
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def efetch(pmids):
    if not pmids: return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(EFETCH, params=params, timeout=60, headers={"User-Agent": UA})
    if r.status_code != 200:
        return []
    try:
        root = ET.fromstring(r.text)
    except Exception:
        return []
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid_elem = art.find(".//PMID")
        title_elem = art.find(".//ArticleTitle")
        abst_elems = art.findall(".//AbstractText")
        if pmid_elem is None or title_elem is None: continue
        pmid = pmid_elem.text
        title = (title_elem.text or "").strip()
        abstract_parts = []
        for a in abst_elems:
            label = a.get("Label", "")
            text = a.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts).strip()
        out.append({"pmid": pmid, "title": title, "abstract": abstract})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--abstracts_per_cui", type=int, default=20)
    ap.add_argument("--seed", default=str(MEDKG_ROOT / "seeds" / "combined_seed.jsonl"))
    args = ap.parse_args()

    log(f"PubMed crawl start. seed={args.seed}, abstracts_per_cui={args.abstracts_per_cui}, NCBI_API_KEY={'set' if NCBI_API_KEY else 'NOT set (3 req/s limit)'}")

    seeds = []
    with open(args.seed) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            seeds.append(json.loads(line))
    if args.limit > 0:
        seeds = seeds[:args.limit]
    log(f"Loaded {len(seeds):,} seeds")

    # Resume: skip already-done CUIs
    done = {p.stem for p in PUBMED_DIR.glob("*.jsonl")}
    log(f"Already crawled: {len(done):,} CUIs")
    pending = [s for s in seeds if s["cui"] not in done]
    log(f"Pending: {len(pending):,} CUIs")

    rate_delay = 0.11 if NCBI_API_KEY else 0.4   # 9/s with key, 2.5/s without
    n_ok = 0
    n_empty = 0
    n_err = 0
    t0 = time.time()
    for i, s in enumerate(pending):
        cui = s["cui"]
        name = s["name"]
        # Build query: prefer phrase match
        query = f'"{name}"[Title/Abstract] OR "{name}"[MeSH Terms]'
        try:
            pmids = esearch(query, args.abstracts_per_cui)
            time.sleep(rate_delay)
            if not pmids:
                # Try a more lenient query
                pmids = esearch(name, args.abstracts_per_cui)
                time.sleep(rate_delay)
            if pmids:
                articles = efetch(pmids)
                time.sleep(rate_delay)
                if articles:
                    out_path = PUBMED_DIR / f"{cui}.jsonl"
                    with out_path.open("w") as out:
                        for a in articles:
                            a["cui"] = cui
                            a["disease_name"] = name
                            out.write(json.dumps(a, ensure_ascii=False) + "\n")
                    n_ok += 1
                else:
                    n_empty += 1
                    # Mark as done with empty file to skip on resume
                    (PUBMED_DIR / f"{cui}.jsonl").write_text("")
            else:
                n_empty += 1
                (PUBMED_DIR / f"{cui}.jsonl").write_text("")
        except Exception as e:
            n_err += 1
            log(f"  ERR {cui} ({name}): {e}")
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / max(elapsed, 1)
            eta_min = (len(pending) - i - 1) / max(rate, 0.001) / 60
            log(f"  Progress: {i+1}/{len(pending):,}  ok={n_ok}, empty={n_empty}, err={n_err}, ETA={eta_min:.0f}min")

    log(f"\nFinal: ok={n_ok}, empty={n_empty}, err={n_err}, total={len(pending)}")


if __name__ == "__main__":
    main()
