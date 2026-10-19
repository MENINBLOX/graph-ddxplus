"""Fetch PMC open-access CASE REPORTS per disease — the attribute-bearing source type.

Measured attribute yield by source type (fraction of extracted findings carrying the attribute):

    source type                     aggravating  relieving  character
    StatPearls / PMC reviews             0%          0%        ~0%
    Clinical Methods (semiology)        12%          5%        21%

Disease-oriented text states WHICH findings occur, not HOW they present: the semiology a clinician
would elicit ("worse lying down", "relieved by antacids") is omitted as too obvious to write down.
A CASE REPORT is the opposite -- a narrative of one patient, where the presentation IS the content.
PMC holds 438,368 open-access case reports, so this is the largest untapped attribute source.

Benchmark-blind (§5): the query is the DISEASE NAME only; no benchmark question or symptom list.
Free text only (§3): no disease-symptom mapping table. Same output schema as sources_gr /
sources_pmc so the existing IE + normalization stages consume it unchanged.

Known bias to declare in the paper: published cases skew atypical/severe, so case reports
complement (not replace) textbook semiology rather than standing alone.

Output: <out-dir>/{cui}.json = {cui, disease, text, provenance:[{pmcid,title}]} + manifest.json.
"""
from __future__ import annotations
import json, os, re, sys, time, argparse, threading, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# .strip(): a key sourced from a CRLF .env carries a trailing \r, which makes every request a
# malformed URL -> HTTP 400 -> silently zero results.
API_KEY = os.environ.get("NCBI_API_KEY", "").strip()
RATE = 9.0 if API_KEY else 2.7               # NCBI cap: 10/s with key, 3/s without

# A case report's narrative lives in these sections; everything else is framing or literature.
KEEP_SEC = re.compile(r"(case\s*(report|presentation|description|history|summary)|"
                      r"clinical\s*(presentation|course|history|findings|description)|"
                      r"presentation|history|examination|investigation|patient)", re.I)
DROP_SEC = re.compile(r"(discussion|conclusion|reference|acknowledg|funding|conflict|"
                      r"author contribution|supplement|literature review|background|introduction)", re.I)


class RateLimiter:
    """Thread-shared token bucket: no more than `rate` requests/sec across all workers."""

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


def eutil(path: str, params: dict, tries: int = 3) -> str:
    params = dict(params)
    if API_KEY:
        params["api_key"] = API_KEY
    url = EUTILS + path + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            LIMITER.wait()
            return urllib.request.urlopen(url, timeout=40).read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(1.0 + k)
    return ""


def search(name: str, retmax: int) -> list[str]:
    """Case reports whose TITLE names the disease -> the article is ABOUT it, not a passing mention."""
    q = (f'"{name}"[Title] AND "case reports"[Publication Type] AND open access[filter]')
    r = eutil("esearch.fcgi", {"db": "pmc", "term": q, "retmax": retmax,
                               "retmode": "json", "sort": "relevance"})
    try:
        return json.loads(r)["esearchresult"].get("idlist", [])
    except Exception:
        return []


def _sec_text(sec) -> str:
    paras = [" ".join(" ".join(p.itertext()).split()) for p in sec.iter("p")]
    return " ".join(x for x in paras if x)


def case_text(xml: str, budget: int) -> tuple[str, list[str]]:
    """JATS full text -> the case narrative. Returns (text, titles_of_kept_sections).

    Many case reports carry NO section titles, so an untitled body is kept whole rather than
    dropped -- dropping it silently was exactly the failure mode that cost us half the
    GeneReviews corpus."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return "", []
    out, kept, used = [], [], 0
    for art in root.iter("article"):
        body = art.find(".//body")
        if body is None:
            continue
        secs = list(body.iter("sec"))
        if not secs:                                   # untitled body: keep the prose as-is
            txt = " ".join(" ".join(" ".join(p.itertext()).split()) for p in body.iter("p"))
            if txt:
                out.append(txt); kept.append("(untitled body)"); used += len(txt)
            continue
        for sec in secs:
            te = sec.find("title")
            title = "".join(te.itertext()).strip() if te is not None else ""
            if title and DROP_SEC.search(title):
                continue
            if title and not KEEP_SEC.search(title):
                continue
            txt = _sec_text(sec)
            if txt:
                out.append(txt); kept.append(title or "(untitled sec)"); used += len(txt)
            if used >= budget:
                break
        if used >= budget:
            break
    return " ".join(out)[:budget], kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diseases", required=True,
                    help="JSON {cui: name} or a v153 pred file (keys=cui, .disease=name)")
    ap.add_argument("--out-dir", default="pilot/cache/prod/sources_case")
    ap.add_argument("--n-articles", type=int, default=5, help="case reports per disease")
    ap.add_argument("--budget", type=int, default=24000, help="max chars kept per disease")
    ap.add_argument("--limit", type=int, default=0, help="0 = all diseases in --diseases")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fetched-date", required=True, help="YYYY-MM-DD, recorded in the manifest")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    raw = json.load(open(a.diseases, encoding="utf-8"))
    diseases = {c: (v.get("disease", c) if isinstance(v, dict) else v) for c, v in raw.items()}
    todo = [(c, n) for c, n in diseases.items()
            if n and not os.path.exists(os.path.join(a.out_dir, f"{c}.json"))]
    if a.limit:
        todo = todo[: a.limit]
    print(f"case-report fetch: {len(todo)} diseases x <= {a.n_articles} articles", flush=True)

    def one(item):
        cui, name = item
        ids = search(name, a.n_articles)
        if not ids:
            return cui, name, "", [], 0
        xml = eutil("efetch.fcgi", {"db": "pmc", "id": ",".join(ids), "retmode": "xml"})
        txt, kept = case_text(xml, a.budget)
        return cui, name, txt, kept, len(ids)

    manifest, n_txt, n_noart, n_nosec = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (cui, name, txt, kept, n_ids) in enumerate(ex.map(one, todo), 1):
            if not n_ids:
                n_noart += 1
            elif not txt:
                n_nosec += 1
            if txt:
                json.dump({"cui": cui, "disease": name, "text": txt,
                           "provenance": {"n_articles": n_ids, "sections": kept[:12]}},
                          open(os.path.join(a.out_dir, f"{cui}.json"), "w"))
                manifest.append({"cui": cui, "disease": name, "chars": len(txt), "n_articles": n_ids})
                n_txt += 1
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}  with-text={n_txt} no-article={n_noart} "
                      f"no-section={n_nosec}", flush=True)

    # seam invariant: report where diseases were lost instead of letting the loss pass silently
    print(f"\ndiseases requested      : {len(todo)}")
    print(f"  with case-report text : {n_txt} ({100*n_txt/max(1,len(todo)):.0f}%)")
    print(f"  no OA case report     : {n_noart}")
    print(f"  article but no narrative section parsed: {n_nosec}")
    if manifest:
        chars = sorted(m["chars"] for m in manifest)
        print(f"  chars/disease median  : {chars[len(chars)//2]}")
    json.dump({"source": "PMC Open Access case reports", "query": '"<disease>"[Title] AND '
               '"case reports"[Publication Type] AND open access[filter]',
               "fetched_date": a.fetched_date, "n_articles_per_disease": a.n_articles,
               "budget_chars": a.budget, "n_diseases": len(manifest), "diseases": manifest},
              open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(f"-> {a.out_dir}/ (manifest frozen)")


if __name__ == "__main__":
    main()
