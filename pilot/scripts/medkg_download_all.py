#!/usr/bin/env python3
"""Tier 1 raw text bulk download.

Strategy:
  - Disease seeds: DDXPlus 49 + SymCat 50 (used) + RareBench top-100 most frequent + ~200 SNOMED CORE common.
  - Each disease → query multiple sources → store raw text + provenance metadata.

Sources (parallel where possible):
  1. StatPearls (NCBI Bookshelf): esearch+esummary→NBK ID→HTML fetch
  2. GeneReviews (NCBI Bookshelf): same approach with gene[book] filter
  3. MedlinePlus A.D.A.M. Encyclopedia: NLM Web Service + topic crawl
  4. Wikipedia: API extract per disease
  5. Orphanet: Already downloaded en_product4.xml (rare disease phenotypes)

Output:
  /home/max/Graph-DDXPlus/data/medkg/{source}/{slug}.{ext}
  /home/max/Graph-DDXPlus/data/medkg/raw/manifest.jsonl  ← all (disease, source, file, status)
"""
from __future__ import annotations
import os, sys, json, time, re, urllib.parse
from pathlib import Path
import requests
from xml.etree import ElementTree as ET

OUT = Path("/home/max/Graph-DDXPlus/data/medkg")
MANIFEST = OUT / "raw" / "manifest.jsonl"
LOG = OUT / "logs" / "download.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

UA = "MedKG-Research/0.1 (academic research; contact: max@meninblox.com)"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")  # optional


def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    with LOG.open("a") as f:
        f.write(s + "\n")


def append_manifest(entry):
    with MANIFEST.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:80]


# ===== Disease seeds =====

def load_disease_seeds():
    """Load DDXPlus 49 + SymCat (used) + RareBench (top frequencies) disease names."""
    seeds = {}

    # DDXPlus
    p = Path("/home/max/Graph-DDXPlus/data/ddxplus/disease_umls_mapping.json")
    if p.exists():
        d = json.load(p.open())
        for k, v in d.get("mapping", {}).items():
            name = (v.get("name_en") or v.get("disease_key") or k).strip()
            seeds[name] = {"sources": ["ddxplus"], "umls_cui": v.get("umls_cui")}

    # SymCat
    p = Path("/home/max/Graph-DDXPlus/data/symcat/disease_umls_mapping.json")
    if p.exists():
        d = json.load(p.open())
        for k, v in d.get("mapping", {}).items():
            name = (v.get("umls_name") or k).strip()
            entry = seeds.get(name, {"sources": []})
            entry["sources"].append("symcat")
            entry["umls_cui"] = entry.get("umls_cui") or v.get("umls_cui")
            seeds[name] = entry

    # RareBench — sample 200 by frequency (we don't have freq, so first 200)
    p = Path("/home/max/Graph-DDXPlus/data/rarebench/disease_umls_mapping.json")
    if p.exists():
        d = json.load(p.open())
        for i, (k, v) in enumerate(d.get("mapping", {}).items()):
            if i >= 200: break
            name = (v.get("disease_name") or v.get("umls_name") or k).strip()
            entry = seeds.get(name, {"sources": []})
            entry["sources"].append("rarebench")
            entry["umls_cui"] = entry.get("umls_cui") or v.get("umls_cui")
            seeds[name] = entry

    return seeds


# ===== StatPearls / GeneReviews via NCBI Bookshelf =====

def ncbi_search(term, book_filter):
    """esearch in books db with book filter — TITLE-only to avoid spurious matches.
    Returns list of section UIDs (already filtered to chapters whose title matches term).
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    # Quote multi-word terms so NCBI treats it as phrase
    quoted_term = f'"{term}"' if " " in term else term
    params = {"db": "books", "term": f"{quoted_term}[title] AND {book_filter}[book]", "retmax": "5"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": UA})
    if r.status_code != 200:
        return []
    root = ET.fromstring(r.text)
    return [e.text for e in root.findall(".//Id")]


def ncbi_summary_to_nbk(uids):
    """esummary on books returns chapter accession IDs (NBK...)"""
    if not uids: return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "books", "id": ",".join(uids), "retmode": "json"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": UA})
    if r.status_code != 200:
        return []
    try:
        data = r.json()
        results = []
        for uid in data.get("result", {}).get("uids", []):
            entry = data["result"][uid]
            chapter_nbk = entry.get("chapteraccessionid")
            chapter_id = entry.get("chapterid")
            book = entry.get("book")
            if chapter_nbk and chapter_id:
                results.append({"nbk": chapter_nbk, "chapter_uid": chapter_id, "book": book})
        # dedupe by chapter_uid
        seen = set()
        out = []
        for r in results:
            if r["chapter_uid"] in seen: continue
            seen.add(r["chapter_uid"])
            out.append(r)
        return out
    except Exception:
        return []


def fetch_book_chapter_html(nbk):
    """Fetch full HTML for an NBK chapter."""
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk}/"
    r = requests.get(url, timeout=60, headers={"User-Agent": UA})
    return r.text if r.status_code == 200 else None


def crawl_ncbi_book(disease_name, book_filter, out_dir, max_chapters=3):
    """For one disease, do esearch+esummary+HTML fetch in given book collection.
    Title-restricted, fetches up to max_chapters relevant articles.
    Returns dict with list of fetched chapter info, or None if no match.
    """
    uids = ncbi_search(disease_name, book_filter)
    if not uids: return None
    chapters = ncbi_summary_to_nbk(uids)
    if not chapters: return None

    # Get titles to filter actually relevant chapters
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    chapter_uids = [str(c["chapter_uid"]) for c in chapters]
    params = {"db": "books", "id": ",".join(chapter_uids), "retmode": "json"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    try:
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": UA})
        title_data = r.json().get("result", {})
    except Exception:
        title_data = {}

    fetched = []
    seen_nbk = set()
    for c in chapters:
        if len(fetched) >= max_chapters: break
        nbk = c["nbk"]
        if nbk in seen_nbk: continue
        seen_nbk.add(nbk)
        title = title_data.get(str(c["chapter_uid"]), {}).get("title", "")
        # Verify disease keyword in chapter title
        d_lower = disease_name.lower()
        t_lower = title.lower()
        if not any(w in t_lower for w in d_lower.split() if len(w) > 3):
            continue  # skip irrelevant
        html = fetch_book_chapter_html(nbk)
        if not html: continue
        out_path = out_dir / f"{nbk}_{slug(disease_name)}.html"
        out_path.write_text(html, encoding="utf-8")
        fetched.append({"nbk": nbk, "title": title, "size": len(html), "path": str(out_path)})

    if not fetched: return None
    return {"chapters": fetched, "n_chapters": len(fetched)}


# ===== MedlinePlus =====

def crawl_medlineplus(disease_name, out_dir):
    """Search MedlinePlus, save the matched topic content."""
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {"db": "healthTopics", "term": disease_name, "retmax": "3"}
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": UA})
    if r.status_code != 200: return None
    out_path = out_dir / f"{slug(disease_name)}.xml"
    out_path.write_text(r.text, encoding="utf-8")
    # Check if any document found
    try:
        root = ET.fromstring(r.text)
        docs = root.findall(".//document")
        return {"size": len(r.text), "n_docs": len(docs), "path": str(out_path)} if docs else None
    except Exception:
        return None


# ===== Wikipedia =====

def crawl_wikipedia(disease_name, out_dir):
    """Fetch Wikipedia article text + infobox via API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json",
        "titles": disease_name,
        "prop": "extracts|info|revisions",
        "explaintext": "1", "rvprop": "ids|timestamp",
        "redirects": "1",
    }
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": UA})
    if r.status_code != 200: return None
    try:
        data = r.json()
        page = next(iter(data["query"]["pages"].values()))
        if "missing" in page or page.get("pageid", 0) <= 0:
            return None
        out_path = out_dir / f"{slug(disease_name)}.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return {"title": page.get("title"), "size": len(page.get("extract", "")), "path": str(out_path), "pageid": page.get("pageid")}
    except Exception as e:
        return None


# ===== Main loop =====

def main():
    seeds = load_disease_seeds()
    log(f"Loaded {len(seeds)} disease seeds")

    sp_dir = OUT / "statpearls"
    gr_dir = OUT / "genereviews"
    mp_dir = OUT / "medlineplus"
    wp_dir = OUT / "wikipedia"
    for d in [sp_dir, gr_dir, mp_dir, wp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Reset manifest
    if MANIFEST.exists(): MANIFEST.unlink()

    counts = {"statpearls": 0, "genereviews": 0, "medlineplus": 0, "wikipedia": 0}
    fails = {"statpearls": 0, "genereviews": 0, "medlineplus": 0, "wikipedia": 0}

    diseases_list = list(seeds.items())
    for idx, (name, meta) in enumerate(diseases_list):
        if idx % 10 == 0:
            log(f"Progress: {idx}/{len(diseases_list)}  counts={counts} fails={fails}")
        for source, fn, dirpath in [
            ("statpearls", lambda n: crawl_ncbi_book(n, "statpearls", sp_dir), sp_dir),
            ("genereviews", lambda n: crawl_ncbi_book(n, "gene", gr_dir), gr_dir),
            ("medlineplus", lambda n: crawl_medlineplus(n, mp_dir), mp_dir),
            ("wikipedia", lambda n: crawl_wikipedia(n, wp_dir), wp_dir),
        ]:
            try:
                result = fn(name)
                if result:
                    counts[source] += 1
                    append_manifest({"disease": name, "umls_cui": meta.get("umls_cui"), "source": source, "result": result, "status": "ok"})
                else:
                    fails[source] += 1
                    append_manifest({"disease": name, "umls_cui": meta.get("umls_cui"), "source": source, "status": "miss"})
            except Exception as e:
                fails[source] += 1
                append_manifest({"disease": name, "umls_cui": meta.get("umls_cui"), "source": source, "status": "error", "error": str(e)})
            # Rate limit (NCBI: 3/s without key, 10/s with)
            time.sleep(0.4 if NCBI_API_KEY else 0.5)

    log(f"\nFinal counts: {counts}")
    log(f"Final fails: {fails}")
    log(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
