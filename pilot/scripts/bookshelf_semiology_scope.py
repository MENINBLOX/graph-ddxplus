"""Coverage pre-check (CLAUDE.md §1) for adding an open clinical-semiology textbook as an IE
source for symptom ATTRIBUTES (character/location/timing/aggravating/relieving) — the layer
disease-oriented sources (StatPearls/PMC) omit as "obvious to clinicians".

Target: "Clinical Methods: The History, Physical, and Laboratory Examinations" (Walker, Hall,
Hurst; 3rd ed) on NCBI Bookshelf — symptom-organized, open access, LLM-free (§3-safe). Chapters
are contiguous NBK ids. This probes the range, keeps SYMPTOM chapters (Part on symptoms), and
measures per-chapter attribute-keyword density so we know BEFORE building how much attribute
signal the corpus carries. No IE, no KG — measurement only.
"""
from __future__ import annotations
import urllib.request, re, html, time, json, argparse, os
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ.get("NCBI_API_KEY", "")

ATTR_KW = ["character", "quality", "burning", "crushing", "pressure", "squeezing", "colicky",
           "cramping", "sharp", "dull", "stabbing", "throbbing", "tearing", "radiat", "aggravat",
           "relieved", "worse", "exertion", "lying", "recumben", "bending", "position",
           "postprandial", "nocturnal", "antacid", "duration", "onset", "precipitat", "provok",
           "exacerbat", "relieving", "supine"]
# symptom-y title cue (Clinical Methods Part "Symptoms" chapters); exclude pure exam/lab/technique
SYMPTOM_CUE = re.compile(r"(?i)pain|cough|dyspnea|dyspnoea|dyspepsia|heartburn|indigestion|nausea|"
                         r"vomit|diarrhea|diarrhoea|constipat|headache|dizz|vertigo|fever|fatigue|"
                         r"weakness|edema|oedema|jaundice|palpitation|dysphagia|hematemesis|melena|"
                         r"hematochezia|hemoptysis|dysuria|hematuria|pruritus|rash|weight loss|"
                         r"bleeding|syncope|seizure|tremor|numbness|dyspha|anorexia|bloating|"
                         r"wheez|sputum|chest|abdominal|back pain|joint|swelling|incontinence")
DROP_CUE = re.compile(r"(?i)examination of|inspection|palpation|auscultation|laboratory|"
                      r"electrocardiog|radiograph|the physical|percussion|blood |urine test|"
                      r"reflexes|technique of")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 thesis-coverage"})
    try:
        return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    except Exception:
        return ""


def strip(raw):
    t = re.sub(r"(?is)<(script|style|head|nav|footer).*?</\1>", " ", raw)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"[ \t]+", " ", html.unescape(t))


def probe(nbk):
    raw = get(f"https://www.ncbi.nlm.nih.gov/books/NBK{nbk}/")
    if not raw:
        return None
    if "Clinical Methods" not in raw:
        return None
    tm = re.search(r"<title>([^<]+)</title>", raw)
    title = html.unescape(tm.group(1)).split(" - Clinical Methods")[0].strip() if tm else f"NBK{nbk}"
    txt = strip(raw); low = txt.lower()
    hits = sum(low.count(k) for k in ATTR_KW)
    is_sym = bool(SYMPTOM_CUE.search(title)) and not DROP_CUE.search(title)
    return {"nbk": f"NBK{nbk}", "title": title, "chars": len(txt), "attr_hits": hits,
            "attr_density": round(hits / max(1, len(txt) / 1000), 2), "symptom_chapter": is_sym}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--hi", type=int, default=520)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="pilot/cache/prod/bookshelf_semiology_scope.json")
    a = ap.parse_args()
    ids = list(range(a.lo, a.hi + 1))
    print(f"probing NBK{a.lo}-NBK{a.hi} ({len(ids)} ids) for Clinical Methods chapters ...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(probe, ids)):
            if r:
                rows.append(r)
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(ids)} probed, {len(rows)} CM chapters so far", flush=True)
    syms = [r for r in rows if r["symptom_chapter"]]
    syms.sort(key=lambda r: -r["attr_hits"])
    print(f"\nClinical Methods chapters found: {len(rows)}  |  symptom chapters: {len(syms)}")
    print(f"{'NBK':10} {'attr_hits':>9} {'dens/1k':>8}  title")
    for r in syms:
        print(f"  {r['nbk']:10} {r['attr_hits']:>9} {r['attr_density']:>8}  {r['title'][:52]}")
    tot_hits = sum(r["attr_hits"] for r in syms)
    print(f"\ntotal attribute-keyword hits across symptom chapters: {tot_hits}")
    json.dump({"chapters": rows, "symptom_chapters": syms}, open(a.out, "w"))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
