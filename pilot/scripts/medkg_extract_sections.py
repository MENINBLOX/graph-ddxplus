#!/usr/bin/env python3
"""Extract disease-relevant text sections from each source type.

For each downloaded raw file → produce {disease, source, source_id, section, text} entries.
Output: /home/max/Graph-DDXPlus/data/medkg/processed/sections.jsonl
"""
from __future__ import annotations
import json, re
from pathlib import Path
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

ROOT = Path("/home/max/Graph-DDXPlus/data/medkg")
MANIFEST = ROOT / "raw" / "manifest.jsonl"
OUT = ROOT / "processed" / "sections.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Section names we care about (clinical phenotype-relevant)
RELEVANT_STATPEARLS_SECTIONS = {
    "history and physical", "evaluation", "introduction",
    "etiology", "epidemiology", "pathophysiology",
    "histopathology", "treatment", "clinical features",
    "differential diagnosis", "prognosis", "complications",
}


def extract_statpearls(html_path):
    """StatPearls/GeneReviews HTML: NCBI Bookshelf. Extract h2 sections."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    # Extract metadata
    meta_title = soup.find("meta", {"name": "citation_title"})
    pmid = soup.find("meta", {"name": "citation_pmid"})
    title = meta_title["content"] if meta_title else None
    pmid_val = pmid["content"] if pmid else None

    sections = []
    for h2 in soup.find_all("h2"):
        sec_name = h2.get_text(strip=True).lower()
        sec_id = h2.get("id", "")
        # Only keep article body sections (have section_id starting with _article or _NBK)
        if not sec_id or not (sec_id.startswith("_article") or sec_id.startswith("_NBK")):
            continue
        # Skip irrelevant sections
        skip_keywords = ("review questions", "references", "media", "clinical pearls",
                          "deterrence and patient education", "pearls and other issues",
                          "enhancing healthcare team outcomes", "education", "interprofessional",
                          "continuing education")
        if any(kw in sec_name for kw in skip_keywords):
            continue
        # Keep only clinically relevant sections
        relevant_kw = ("history", "physical", "presentation", "evaluation", "introduction",
                        "etiology", "pathophysiology", "histopathology", "epidemiology",
                        "clinical features", "diagnosis", "differential", "complications", "treatment",
                        "prognosis", "signs", "symptoms")
        if not any(kw in sec_name for kw in relevant_kw):
            continue
        text_parts = []
        for sib in h2.find_next_siblings():
            if sib.name == "h2": break
            text_parts.append(sib.get_text(" ", strip=True))
        text = " ".join(text_parts).strip()
        if len(text) < 100: continue
        sections.append({"section_id": sec_id, "section_name": sec_name, "text": text})
    return {"title": title, "pmid": pmid_val, "sections": sections}


def extract_medlineplus(xml_path):
    """MedlinePlus search XML: extract topic FullSummary."""
    try:
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    docs = root.findall(".//document")
    out = []
    for doc in docs[:3]:
        url = doc.get("url", "")
        title = ""
        full_summary = ""
        for c in doc.findall(".//content"):
            n = c.get("name", "")
            if n == "title": title = c.text or ""
            elif n == "FullSummary":
                # FullSummary is HTML
                soup = BeautifulSoup(c.text or "", "html.parser")
                full_summary = soup.get_text(" ", strip=True)
            elif n == "snippet" and not full_summary:
                full_summary = c.text or ""
        if title and full_summary and len(full_summary) > 100:
            out.append({"title": title, "url": url, "text": full_summary})
    return {"docs": out} if out else None


def extract_wikipedia(json_path):
    """Wikipedia: extract intro + Signs and Symptoms section if present."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page or page.get("pageid", 0) <= 0: return None
    extract = page.get("extract", "")
    revid = page.get("revisions", [{}])[0].get("revid")
    title = page.get("title", "")
    # Split by section headers (=== or ==)
    sections = []
    cur_name, cur_text = "Introduction", []
    skip_kw = ("external links", "references", "see also", "further reading", "notes",
                "external", "bibliography", "footnotes", "sources", "cited works")
    relevant_kw = ("introduction", "signs and symptoms", "symptoms", "presentation",
                    "diagnosis", "history", "causes", "etiology", "pathophysiology",
                    "complications", "prognosis", "differential", "physical", "clinical")
    for line in extract.split("\n"):
        m = re.match(r"^\s*=+\s*(.+?)\s*=+\s*$", line)
        if m:
            if cur_text:
                low = cur_name.lower()
                if any(kw in low for kw in relevant_kw) and not any(kw in low for kw in skip_kw):
                    text = " ".join(cur_text).strip()
                    if len(text) > 100:
                        sections.append({"section_name": low, "text": text})
            cur_name = m.group(1)
            cur_text = []
        else:
            cur_text.append(line)
    if cur_text:
        low = cur_name.lower()
        if any(kw in low for kw in relevant_kw) and not any(kw in low for kw in skip_kw):
            text = " ".join(cur_text).strip()
            if len(text) > 100:
                sections.append({"section_name": low, "text": text})
    return {"title": title, "pageid": page.get("pageid"), "revid": revid, "sections": sections}


def main():
    n_total = 0
    n_proc = 0
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        return
    with OUT.open("w") as out_f:
        for line in MANIFEST.read_text().split("\n"):
            if not line.strip(): continue
            entry = json.loads(line)
            if entry.get("status") != "ok": continue
            n_total += 1
            disease = entry["disease"]
            cui = entry.get("umls_cui")
            source = entry["source"]
            result = entry["result"]
            extracted = None

            try:
                if source in ("statpearls", "genereviews"):
                    chapters = result.get("chapters", [])
                    for ch in chapters:
                        path = Path(ch["path"])
                        if not path.exists(): continue
                        d = extract_statpearls(path)
                        if not d: continue
                        for sec in d["sections"]:
                            out_f.write(json.dumps({
                                "disease": disease, "umls_cui": cui,
                                "source": source, "source_id": ch["nbk"],
                                "chapter_title": d["title"], "pmid": d["pmid"],
                                "section_name": sec["section_name"],
                                "section_id": sec["section_id"],
                                "text": sec["text"],
                            }, ensure_ascii=False) + "\n")
                            n_proc += 1
                elif source == "medlineplus":
                    path = Path(result["path"])
                    if not path.exists(): continue
                    d = extract_medlineplus(path)
                    if not d: continue
                    for doc in d["docs"]:
                        out_f.write(json.dumps({
                            "disease": disease, "umls_cui": cui,
                            "source": source, "source_id": doc["url"],
                            "section_name": "fullsummary",
                            "text": doc["text"],
                            "topic_title": doc["title"],
                        }, ensure_ascii=False) + "\n")
                        n_proc += 1
                elif source == "wikipedia":
                    path = Path(result["path"])
                    if not path.exists(): continue
                    d = extract_wikipedia(path)
                    if not d: continue
                    for sec in d["sections"]:
                        out_f.write(json.dumps({
                            "disease": disease, "umls_cui": cui,
                            "source": source, "source_id": f"pageid:{d['pageid']}",
                            "revid": d["revid"], "title": d["title"],
                            "section_name": sec["section_name"],
                            "text": sec["text"],
                        }, ensure_ascii=False) + "\n")
                        n_proc += 1
            except Exception as e:
                print(f"  ERROR {source}/{disease}: {e}")
    print(f"\nProcessed {n_total} manifest entries → {n_proc} sections")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
