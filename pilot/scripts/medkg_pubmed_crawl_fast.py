#!/usr/bin/env python3
"""Concurrent PubMed crawl (drop-in faster replacement for medkg_pubmed_crawl.py).

Same contract: for each seed CUI, esearch the disease name -> efetch top-N abstracts ->
write per-CUI JSONL at $MEDKG_ROOT/pubmed/{cui}.jsonl (empty file marks a genuine no-hit so
resume skips it). Same query cascade (phrase -> lenient) and output schema.

Difference: a thread pool issues requests concurrently under ONE global token-bucket rate
limiter (NCBI allows ~9 req/s with an API key), instead of sleeping serially. Per-CUI file
writes are independent (distinct filenames) so no write races. Resumable: skips existing files.

Usage: MEDKG_ROOT=... NCBI_API_KEY=... python medkg_pubmed_crawl_fast.py [--abstracts_per_cui 20] [--workers 5]
"""
from __future__ import annotations
import sys, os, json, time, argparse, threading, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

PUBMED_DIR = MEDKG_ROOT / "pubmed"
PUBMED_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = MEDKG_ROOT / "logs" / "pubmed_crawl_fast.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "MedKG-Research/0.1 (academic; max@meninblox.com)"}
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_log_lock = threading.Lock()
def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
        print(s, flush=True)
        with LOG_PATH.open("a") as f:
            f.write(s + "\n")


class RateLimiter:
    """Global min-interval limiter shared across worker threads (NCBI ~9 req/s with key)."""
    def __init__(self, per_sec):
        self.interval = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_t = time.monotonic()
    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                sleep = self.next_t - now
            else:
                sleep = 0.0
                self.next_t = now
            self.next_t += self.interval
        if sleep > 0:
            time.sleep(sleep)


RL = RateLimiter(9 if NCBI_API_KEY else 2.5)
_session = requests.Session()
_session.headers.update(UA)


def _get(url, params):
    RL.wait()
    return _session.get(url, params=params, timeout=45)


def esearch(query, retmax):
    p = {"db": "pubmed", "term": query, "retmax": str(retmax), "sort": "relevance", "retmode": "json"}
    if NCBI_API_KEY: p["api_key"] = NCBI_API_KEY
    try:
        r = _get(ESEARCH, p)
        if r.status_code != 200: return []
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def efetch(pmids):
    if not pmids: return []
    p = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if NCBI_API_KEY: p["api_key"] = NCBI_API_KEY
    try:
        r = _get(EFETCH, p)
        if r.status_code != 200: return []
        root = ET.fromstring(r.text)
    except Exception:
        return []
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID"); title_el = art.find(".//ArticleTitle")
        if pmid_el is None or title_el is None: continue
        parts = []
        for a in art.findall(".//AbstractText"):
            lab = a.get("Label", ""); txt = a.text or ""
            parts.append(f"{lab}: {txt}" if lab else txt)
        out.append({"pmid": pmid_el.text, "title": (title_el.text or "").strip(),
                    "abstract": " ".join(parts).strip()})
    return out


def handle(cui, name, k):
    """Full pipeline for one CUI. Returns 'ok'|'empty'|'err'. Writes file atomically."""
    try:
        pmids = esearch(f'"{name}"[Title/Abstract] OR "{name}"[MeSH Terms]', k)
        if not pmids:
            pmids = esearch(name, k)
        arts = efetch(pmids) if pmids else []
        arts = [a for a in arts if a.get("abstract", "").strip()]
        out_path = PUBMED_DIR / f"{cui}.jsonl"
        tmp = out_path.with_suffix(".jsonl.tmp")
        if arts:
            with tmp.open("w") as out:
                for a in arts:
                    a["cui"] = cui; a["disease_name"] = name
                    out.write(json.dumps(a, ensure_ascii=False) + "\n")
            tmp.replace(out_path)
            return "ok"
        else:
            out_path.write_text("")  # genuine no-hit marker for resume
            return "empty"
    except Exception as e:
        log(f"  ERR {cui} ({name}): {e}")
        return "err"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--abstracts_per_cui", type=int, default=20)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", default=str(MEDKG_ROOT / "seeds" / "combined_seed.jsonl"))
    args = ap.parse_args()

    log(f"FAST crawl start. workers={args.workers}, k={args.abstracts_per_cui}, "
        f"key={'set' if NCBI_API_KEY else 'NOT set'}")
    seeds = [json.loads(l) for l in open(args.seed) if l.strip()]
    if args.limit: seeds = seeds[:args.limit]
    done = {p.stem for p in PUBMED_DIR.glob("*.jsonl")}
    pending = [s for s in seeds if s["cui"] not in done]
    log(f"seeds={len(seeds):,} done={len(done):,} pending={len(pending):,}")

    n = {"ok": 0, "empty": 0, "err": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(handle, s["cui"], s["name"], args.abstracts_per_cui): s for s in pending}
        for i, fut in enumerate(as_completed(futs)):
            n[fut.result()] += 1
            if (i + 1) % 200 == 0:
                el = time.time() - t0; rate = (i + 1) / max(el, 1)
                eta = (len(pending) - i - 1) / max(rate, 1e-3) / 60
                log(f"  Progress: {i+1}/{len(pending):,}  ok={n['ok']} empty={n['empty']} "
                    f"err={n['err']}  {rate:.1f}/s ETA={eta:.0f}min")
    log(f"\nFinal: ok={n['ok']} empty={n['empty']} err={n['err']} total={len(pending)}")


if __name__ == "__main__":
    main()
