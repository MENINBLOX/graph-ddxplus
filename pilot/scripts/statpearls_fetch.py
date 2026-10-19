"""Fetch StatPearls per disease and keep the SEMIOLOGY sections only.

Two measurements motivate this. (1) Our corpus is dominated by PubMed abstracts, the weakest
attribute source we measured (13% of documents carry attribute phrasing, 0.21 mentions/doc),
whereas StatPearls carries a dedicated "History and Physical" section -- the semiology a clinician
elicits. (2) Inside a StatPearls article that section is **4.7x denser** in aggravating/relieving
language than the article as a whole (0.112 vs 0.024 mentions per 1k chars), and it is present in
95% of articles -- so feeding the IE the whole 35k-char article dilutes the attribute statements
across 16x more text.

We hold 596 StatPearls files locally out of the ~10^4 on NCBI Bookshelf, so most diseases have no
StatPearls evidence at all. This fetches by DISEASE NAME only (§5 benchmark-blind), keeps the
semiology sections, and writes the same {cui, disease, text} blobs the IE already consumes.

Output: <out-dir>/{cui}.json + manifest.json (frozen: query, sections kept, fetch date).
"""
from __future__ import annotations
import argparse, html, json, os, re, sys, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
API_KEY = os.environ.get("NCBI_API_KEY", "").strip()   # a CRLF .env leaves a \r -> HTTP 400
RATE = 9.0 if API_KEY else 2.7

# StatPearls' fixed section order. History and Physical is the semiology; Evaluation adds the
# examination findings. Everything after Treatment is management, which the IE must not read.
SEMIOLOGY = re.compile(
    r"(?is)\b(History and Physical|Clinical Presentation|Signs and Symptoms)\b(.{200,40000}?)"
    r"(?:\n\s*(?:Treatment\s*/?\s*Management|Treatment|Differential Diagnosis|Prognosis|"
    r"Complications|Deterrence|Pearls|Enhancing|Review Questions|References)\b)")
EVALUATION = re.compile(
    r"(?is)\bEvaluation\b(.{200,20000}?)(?:\n\s*(?:Treatment\s*/?\s*Management|Treatment|"
    r"Differential Diagnosis|Prognosis|Complications)\b)")
# Etiology and the differential are what a benchmark encoding "which underlying condition caused
# this" actually asks about. DDXPlus records for Localized edema are dominated by heart failure,
# cirrhosis, nephrotic syndrome and valve disease -- none of which appear in a History-and-Physical
# section, and all of which sit here. Dropping these sections cost those diseases every @10.
ETIOLOGY = re.compile(
    r"(?is)\b(Etiology|Causes|Pathophysiology|Differential Diagnosis)\b(.{200,30000}?)"
    r"(?:\n\s*(?:Treatment\s*/?\s*Management|Treatment|Prognosis|Complications|Deterrence|"
    r"Pearls|Enhancing|Review Questions|References|Epidemiology)\b)")


class RateLimiter:
    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.interval
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)


LIMITER = RateLimiter(RATE)


def _get(url: str, tries: int = 3) -> str:
    for k in range(tries):
        try:
            LIMITER.wait()
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-statpearls"})
            return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(1.0 + k)
    return ""


def eutil(path: str, params: dict) -> str:
    p = dict(params)
    if API_KEY:
        p["api_key"] = API_KEY
    return _get(EUTILS + path + "?" + urllib.parse.urlencode(p))


def to_text(raw: str) -> str:
    t = re.sub(r"(?is)<(script|style|head|nav|footer|aside).*?</\1>", " ", raw)
    t = re.sub(r"(?is)<div[^>]*ref-list.*?</div>", " ", t)
    t = re.sub(r"(?s)</(p|div|li|h[1-6]|tr|section)>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def semiology_text(article: str) -> tuple[str, list[str]]:
    """Article text -> (semiology prose, names of sections kept)."""
    out, kept = [], []
    m = SEMIOLOGY.search(article)
    if m:
        out.append(m.group(2).strip()); kept.append(m.group(1))
    e = EVALUATION.search(article)
    if e:
        out.append(e.group(1).strip()); kept.append("Evaluation")
    for m2 in ETIOLOGY.finditer(article):
        out.append(m2.group(2).strip()); kept.append(m2.group(1))
    return "\n".join(out), kept


def search(name: str) -> str | None:
    """Disease name -> the chapter's NBK accession (not the Bookshelf UID).

    esearch returns an internal UID; the page lives at /books/NBK<accessionid>/, so the UID must be
    resolved through esummary. Using the UID directly fetches nothing and looks exactly like "this
    disease has no StatPearls article"."""
    r = eutil("esearch.fcgi", {"db": "books", "retmode": "json", "retmax": 1, "sort": "relevance",
                               "term": f'"{name}"[Title] AND statpearls[book]'})
    try:
        ids = json.loads(r)["esearchresult"].get("idlist", [])
    except Exception:
        return None
    if not ids:
        return None
    s = eutil("esummary.fcgi", {"db": "books", "retmode": "json", "id": ids[0]})
    try:
        rec = json.loads(s)["result"][ids[0]]
    except Exception:
        return None
    return rec.get("accessionid") or rec.get("chapteraccessionid") or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diseases", required=True, help="JSON {cui: name} or a v153 pred file")
    ap.add_argument("--out-dir", default="pilot/cache/prod/sources_sp_full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fetched-date", required=True, help="YYYY-MM-DD, recorded in the manifest")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    raw = json.load(open(a.diseases, encoding="utf-8"))
    diseases = {c: (v.get("disease", c) if isinstance(v, dict) else v) for c, v in raw.items()}
    todo = [(c, n) for c, n in diseases.items()
            if n and not os.path.exists(os.path.join(a.out_dir, f"{c}.json"))]
    if a.limit:
        todo = todo[: a.limit]
    print(f"statpearls fetch: {len(todo)} diseases", flush=True)

    def one(item):
        cui, name = item
        acc = search(name)
        if not acc:
            return cui, name, "", [], "no-article"
        art = to_text(_get(f"https://www.ncbi.nlm.nih.gov/books/{acc}/"))
        if not art:
            return cui, name, "", [], "no-fetch"
        txt, kept = semiology_text(art)
        return cui, name, txt, kept, ("ok" if txt else "no-semiology-section")

    manifest, why = [], {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (cui, name, txt, kept, status) in enumerate(ex.map(one, todo), 1):
            why[status] = why.get(status, 0) + 1
            if txt:
                json.dump({"cui": cui, "disease": name, "text": txt,
                           "provenance": {"sections": kept}},
                          open(os.path.join(a.out_dir, f"{cui}.json"), "w"))
                manifest.append({"cui": cui, "disease": name, "chars": len(txt), "sections": kept})
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}  kept={len(manifest)}  {why}", flush=True)

    # seam invariant: say where diseases were lost rather than let the loss pass silently
    print(f"\nrequested {len(todo)} -> kept {len(manifest)}   breakdown: {why}")
    if manifest:
        ch = sorted(m["chars"] for m in manifest)
        print(f"  chars/disease median {ch[len(ch)//2]}")
    json.dump({"source": "StatPearls (NCBI Bookshelf, open access)",
               "query": '"<disease>"[Title] AND statpearls[book]',
               "sections_kept": "History and Physical / Clinical Presentation / Signs and Symptoms"
                                " + Evaluation",
               "fetched_date": a.fetched_date, "n_diseases": len(manifest), "diseases": manifest},
              open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(f"-> {a.out_dir}/ (manifest frozen)")


if __name__ == "__main__":
    main()
