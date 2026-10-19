"""Fetch the Clinical Methods symptom chapters (identified by bookshelf_semiology_scope.py) to a
local corpus for attribute IE, with a frozen provenance manifest (reproducibility lockfile).

Clinical Methods (Walker/Hall/Hurst, 3rd ed) is open-access on NCBI Bookshelf and symptom-organized:
each chapter describes a presenting symptom's character/location/timing/aggravating/relieving AND
how the causes are DISTINGUISHED by those attributes (the differential-grade signal StatPearls/PMC
omit). §3/§4-safe: this is textbook IE input (no disease-symptom mapping table, no benchmark labels).

Writes pilot/cache/prod/sources_semiology/NBK<id>.json {nbk,title,text} + a manifest
(source, edition, fetched_date, chapters:[{nbk,title,chars}]) so the used chapter set is frozen.
"""
from __future__ import annotations
import urllib.request, re, html, json, os, argparse, time
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ.get("NCBI_API_KEY", "").strip()   # a CRLF .env leaves a \r -> HTTP 400


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 thesis-semiology"})
    for _ in range(3):
        try:
            return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(1.0)
    return ""


def clinical_text(raw):
    """Strip nav/refs; keep the chapter prose (Definition/Technique/Basic Science/Clinical
    Significance sections carry the attribute descriptions)."""
    t = re.sub(r"(?is)<(script|style|head|nav|footer|aside).*?</\1>", " ", raw)
    # drop reference list / citation blocks
    t = re.sub(r"(?is)<div[^>]*ref-list.*?</div>", " ", t)
    t = re.sub(r"(?s)</(p|div|li|h[1-6]|tr|section)>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="pilot/cache/prod/bookshelf_semiology_scope.json")
    ap.add_argument("--out-dir", default="pilot/cache/prod/sources_semiology")
    ap.add_argument("--fetched-date", required=True, help="YYYY-MM-DD (passed in; no Date.now)")
    ap.add_argument("--min-hits", type=int, default=5, help="skip chapters below this attr density")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    scope = json.load(open(a.scope))
    chapters = [r for r in scope["symptom_chapters"] if r["attr_hits"] >= a.min_hits]
    print(f"fetching {len(chapters)} symptom chapters (attr_hits>={a.min_hits}) ...", flush=True)

    def fetch(r):
        raw = get(f"https://www.ncbi.nlm.nih.gov/books/{r['nbk']}/")
        if not raw:
            return None
        txt = clinical_text(raw)
        return {"nbk": r["nbk"], "title": r["title"], "text": txt}

    manifest = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(fetch, chapters):
            if not res or len(res["text"]) < 500:
                continue
            json.dump(res, open(os.path.join(a.out_dir, f"{res['nbk']}.json"), "w"))
            manifest.append({"nbk": res["nbk"], "title": res["title"], "chars": len(res["text"])})
            print(f"  {res['nbk']:10} {len(res['text']):>7} chars  {res['title'][:48]}", flush=True)

    mf = {"source": "Clinical Methods: The History, Physical, and Laboratory Examinations",
          "authors": "Walker HK, Hall WD, Hurst JW (eds)", "edition": "3rd (1990)",
          "publisher": "Butterworths / NCBI Bookshelf (open access)",
          "fetched_date": a.fetched_date, "n_chapters": len(manifest), "chapters": manifest}
    json.dump(mf, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(f"\ncorpus: {len(manifest)} chapters, "
          f"{sum(c['chars'] for c in manifest)} chars -> {a.out_dir}")
    print(f"manifest frozen -> {a.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
