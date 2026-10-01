"""Fetch DEEP per-disease evidence: many open-access articles instead of one 12k-char blob.

Every disease in the production KG was built from ~12k characters, which yields a median of 31
findings and leaves DDXPlus evidence recall at 48%. That ceiling is set by how much we read, not by
the extractor: the symptom IE already recovers 84% of the symptom evidence it is given.

This pulls the open-access literature for a disease at depth -- PMC full text (reviews first, then
any OA article), StatPearls' semiology sections, and open-access case reports -- and concatenates
them into one source blob per disease, keeping the same {cui, disease, text} schema the IE consumes.

Benchmark-blind (§5): every query is the DISEASE NAME. No benchmark question, symptom list, or
label is read here or anywhere downstream.

Usage: python deep_fetch.py --diseases {cui:name}.json --out-dir DIR --fetched-date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, html, json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statpearls_fetch import search as sp_search, semiology_text, to_text, LIMITER, eutil, _get

KEEP_SEC = re.compile(r"(?i)(clinical|sign|symptom|presentation|manifest|feature|diagnos|history|"
                      r"physical|examination|finding|course|natural history|epidemiolog|risk)")
DROP_SEC = re.compile(r"(?i)(method|statistic|acknowledg|funding|conflict|author contribution|"
                      r"supplement|reference|competing interest|data availability)")


def _search(term: str, retmax: int) -> list[str]:
    r = eutil("esearch.fcgi", {"db": "pmc", "term": term, "retmax": retmax, "retmode": "json",
                               "sort": "relevance"})
    try:
        return json.loads(r)["esearchresult"].get("idlist", [])
    except Exception:
        return []


def name_variants(name: str) -> list[str]:
    """Query forms to try for a disease label, most specific first.

    RareBench labels join an Orphanet name and an OMIM name with a slash and keep OMIM's numeric
    subtype suffixes -- "Autosomal dominant hyper-IgE syndrome/Hyper-IgE recurrent infection
    syndrome", "Ehlers-Danlos syndrome, classic type, 2". Searched verbatim they return NOTHING,
    which reads exactly like "no literature exists" and cost 93 LIRICAL diseases their sources.
    The parts are well covered: the first returns 616 open-access hits and "Hyper-IgE syndrome"
    2,307. Splitting the alias join and dropping the numeric subtype recovers them."""
    out, seen = [], set()
    parts = [p.strip() for p in re.split(r"\s*/\s*", name or "") if p.strip()]
    for p in parts:
        # "Ehlers-Danlos syndrome, classic type, 2" -> also "...classic type" and "Ehlers-Danlos syndrome"
        forms = [p,
                 re.sub(r",\s*\d+\s*$", "", p).strip(),
                 re.sub(r",\s*(type\s*)?[\dIVX]+[a-z]?\s*$", "", p).strip(),
                 p.split(",")[0].strip()]
        for f in forms:
            f = re.sub(r"\s+", " ", f)
            if len(f) >= 6 and f.lower() not in seen:
                seen.add(f.lower())
                out.append(f)
    return out[:5]


def pmc_ids(name: str, retmax: int, extra: str = "", alt: str = "") -> list[str]:
    """Title search, then progressively looser fallbacks.

    A benchmark's own label is often not what the literature titles an article: SymCat writes
    "Eye alignment disorder" and "Cervical disorder", and a title-restricted search for those
    returns nothing -- 523 of 1,157 diseases came back with zero characters for exactly this
    reason, which looks identical to "no literature exists". So fall back to the concept's UMLS
    preferred name, then drop the [Title] restriction. Still benchmark-blind: every query is a
    NAME of the disease, never its findings."""
    suffix = f" AND {extra}" if extra else ""
    cands = name_variants(name) + ([alt] if alt else [])
    for v in cands:                                   # title match on each name form
        ids = _search(f'"{v}"[Title] AND open access[filter]{suffix}', retmax)
        if ids:
            return ids
    for v in cands:                                   # then anywhere in the article
        ids = _search(f'"{v}" AND open access[filter]{suffix}', retmax)
        if ids:
            return ids
    return []


def pmc_text(ids: list[str], budget: int) -> str:
    if not ids:
        return ""
    xml = eutil("efetch.fcgi", {"db": "pmc", "id": ",".join(ids), "retmode": "xml"})
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
            te = sec.find("title")
            title = "".join(te.itertext()).strip() if te is not None else ""
            if title and (DROP_SEC.search(title) or not KEEP_SEC.search(title)):
                continue
            txt = " ".join(" ".join(" ".join(p.itertext()).split()) for p in sec.iter("p"))
            if txt:
                out.append(txt); used += len(txt)
            if used >= budget:
                return " ".join(out)[:budget]
    return " ".join(out)[:budget]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diseases", required=True, help="JSON {cui: name}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fetched-date", required=True)
    ap.add_argument("--reviews", type=int, default=10, help="PMC review articles per disease")
    ap.add_argument("--cases", type=int, default=8, help="PMC case reports per disease")
    ap.add_argument("--budget", type=int, default=250000, help="max chars kept per disease")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    diseases = json.load(open(a.diseases, encoding="utf-8"))
    # UMLS preferred name per CUI, used as the fallback query term (see pmc_ids)
    try:
        alt_name = json.load(open("pilot/cache/umls_maps/disease_cui2name.json", encoding="utf-8"))
    except Exception:
        alt_name = {}
    todo = [(c, n) for c, n in diseases.items()
            if n and not os.path.exists(os.path.join(a.out_dir, f"{c}.json"))]
    print(f"deep fetch: {len(todo)} diseases", flush=True)

    per = max(1, a.budget // 3)

    def one(item):
        cui, name = item
        parts, prov = [], {}
        alt = alt_name.get(cui, "")
        rev = pmc_text(pmc_ids(name, a.reviews, 'review[Publication Type]', alt), per)
        if rev:
            parts.append(rev); prov["pmc_review_chars"] = len(rev)
        if len(" ".join(parts)) < a.budget:
            case = pmc_text(pmc_ids(name, a.cases, '"case reports"[Publication Type]', alt), per)
            if case:
                parts.append(case); prov["pmc_case_chars"] = len(case)
        acc = sp_search(name) or (sp_search(alt) if alt and alt != name else None)
        if acc:
            art = to_text(_get(f"https://www.ncbi.nlm.nih.gov/books/{acc}/"))
            sp, kept = semiology_text(art)
            if sp:
                parts.append(sp); prov["statpearls_chars"] = len(sp); prov["statpearls"] = acc
        txt = " ".join(parts)[:a.budget]
        return cui, name, txt, prov

    manifest, n_ok = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (cui, name, txt, prov) in enumerate(ex.map(one, todo), 1):
            if txt:
                json.dump({"cui": cui, "disease": name, "text": txt, "provenance": prov},
                          open(os.path.join(a.out_dir, f"{cui}.json"), "w"))
                manifest.append({"cui": cui, "disease": name, "chars": len(txt), **prov})
                n_ok += 1
            print(f"  {i}/{len(todo)} {name[:32]:34} {len(txt):7} chars  {prov}", flush=True)

    ch = sorted(m["chars"] for m in manifest) or [0]
    print(f"\nfetched {n_ok}/{len(todo)}   chars/disease median {ch[len(ch)//2]}  total {sum(ch)}")
    json.dump({"source": "PMC open access (reviews + case reports) + StatPearls semiology",
               "query": "disease NAME only (§5 benchmark-blind)",
               "fetched_date": a.fetched_date, "n_diseases": len(manifest), "diseases": manifest},
              open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(f"-> {a.out_dir}/ (manifest frozen)")


if __name__ == "__main__":
    main()
