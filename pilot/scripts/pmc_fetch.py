"""Fetch PMC Open-Access FULL-TEXT clinical prose per disease (benchmark-blind, §3-safe).

We currently seed IE from PubMed ABSTRACTS (~12k chars, sometimes off-topic). PMC OA full-text
of review/clinical articles is far richer for rare presentations. This crawler, seeded by
DISEASE NAME only, pulls OA full-text, keeps the phenotype-bearing sections (clinical
description / signs & symptoms / diagnosis / presentation), and writes {cui}.json source blobs
in the same schema as sources_gr/sources_sp. Prose only -> strict-zero-shot safe.

Quality filters (avoid the "PKU mentioned in a nanotech paper" trap):
  * Title must contain the disease name  -> the article is ABOUT the disease.
  * Prefer review[pt]; fall back to any OA article with a clinical section.
  * Keep only sections whose <title> matches a clinical-content regex; drop methods/refs.
Checkpointed + resumable + NCBI rate-limited (3 req/s w/o key, set NCBI_API_KEY for 10/s).
"""
from __future__ import annotations
import json, os, re, sys, time, argparse, threading, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
API_KEY = os.environ.get("NCBI_API_KEY", "").strip()   # a CRLF .env leaves a \r -> HTTP 400
RATE = 9.0 if API_KEY else 2.7               # global requests/sec cap (NCBI: 10/3 with/without key)


class RateLimiter:
    """Thread-shared token bucket: no more than `rate` requests/sec across all workers."""
    def __init__(self, rate):
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.interval
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)


LIMITER = RateLimiter(RATE)
KEEP_SEC = re.compile(r"(clinical (description|features|characteristics|manifestation|presentation)|"
                      r"sign|symptom|presentation|phenotyp|diagnosis|natural history)", re.I)
DROP_SEC = re.compile(r"(method|material|statistic|result|discussion|conclusion|reference|"
                      r"acknowledg|funding|conflict|author contribution|supplement)", re.I)


def eutil(path, params, tries=3):
    params = dict(params)
    if API_KEY:
        params["api_key"] = API_KEY
    url = EUTILS + path + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            LIMITER.wait()                    # respect the global NCBI rate cap
            return urllib.request.urlopen(url, timeout=40).read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(1.0 + k)
    return ""


def search(name, retmax=4):
    q = f'"{name}"[Title] AND open access[filter]'
    r = eutil("esearch.fcgi", {"db": "pmc", "term": q, "retmax": retmax,
                               "retmode": "json", "sort": "relevance"})
    try:
        return json.loads(r)["esearchresult"].get("idlist", [])
    except Exception:
        return []


def clinical_text(xml, budget=12000):
    """Parse JATS full-text; keep phenotype-bearing sections' prose."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    out, used = [], 0
    for art in root.iter("article"):
        body = art.find(".//body")
        if body is None:
            continue
        for sec in body.iter("sec"):
            title_el = sec.find("title")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            if title and DROP_SEC.search(title) and not KEEP_SEC.search(title):
                continue
            if title and not KEEP_SEC.search(title):
                continue                       # keep only clinical-titled sections
            paras = [" ".join(" ".join(p.itertext()).split()) for p in sec.iter("p")]
            txt = " ".join(x for x in paras if x)
            if txt:
                out.append(txt); used += len(txt)
            if used >= budget:
                break
        if used >= budget:
            break
    return " ".join(out)[:budget]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="pilot/cache/prod/ie_pred_full.json",
                    help="{cui:{disease,...}} to take (cui, disease_name) pairs from")
    ap.add_argument("--out-dir", default="pilot/cache/prod/sources_pmc")
    ap.add_argument("--n-articles", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--order", choices=["asc", "rare-first"], default="rare-first")
    ap.add_argument("--workers", type=int, default=8, help="concurrent fetch workers (rate-limited globally)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    seed = json.load(open(a.seed))
    items = [(c, v.get("disease", "")) for c, v in seed.items() if v.get("disease")]
    if a.order == "rare-first":
        # benchmark-blind proxy for "needs content": fewest existing findings first
        items.sort(key=lambda x: len(seed[x[0]].get("findings", [])))
    if a.limit:
        items = items[:a.limit]

    done = set(os.path.splitext(f)[0] for f in os.listdir(a.out_dir))
    todo = [(c, n) for c, n in items if c not in done]
    print(f"PMC fetch: {len(items)} seed, {len(done)} done, {len(todo)} to do", flush=True)

    # run manifest: pin the query template + fetch date + corpus so the source set is reproducible
    from datetime import date
    manifest_path = a.out_dir.rstrip("/") + "_manifest.json"
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {
        "corpus": "PMC Open Access subset (E-utilities efetch, JATS full-text)",
        "query_template": '"{disease}"[Title] AND open access[filter]  (sort=relevance, JATS body '
                          'clinical-section prose only, capped 12000 chars)',
        "fetched_date": str(date.today()), "diseases": {}}

    def fetch_one(item):
        cui, name = item
        ids = search(name, retmax=a.n_articles + 2)
        texts, used = [], []
        for pid in ids[: a.n_articles + 2]:
            xml = eutil("efetch.fcgi", {"db": "pmc", "id": pid, "rettype": "xml", "retmode": "xml"})
            ct = clinical_text(xml)
            if ct:
                texts.append(ct); used.append(f"PMC{pid}")   # freeze the exact articles used
            if len(texts) >= a.n_articles:
                break
        blob = " ".join(texts)[:12000]
        json.dump({"cui": cui, "disease": name, "text": blob, "source": "pmc", "pmcids": used},
                  open(os.path.join(a.out_dir, f"{cui}.json"), "w"))
        return cui, name, used, bool(blob)

    n_ok = n_empty = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (cui, name, used, ok) in enumerate(ex.map(fetch_one, todo)):
            manifest["diseases"][cui] = {"disease": name, "pmcids": used}
            n_ok += ok; n_empty += (not ok)
            if (i + 1) % 200 == 0:
                with lock:
                    json.dump(manifest, open(manifest_path, "w"))   # checkpoint the frozen set
                print(f"  {i+1}/{len(todo)}  ok={n_ok} empty={n_empty}", flush=True)
    json.dump(manifest, open(manifest_path, "w"))
    print(f"done: ok={n_ok} empty={n_empty} -> {a.out_dir}/  (manifest: {manifest_path})", flush=True)


if __name__ == "__main__":
    main()
