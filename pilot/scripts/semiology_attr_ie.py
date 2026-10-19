"""Differential-grade attribute IE from the Clinical Methods symptom chapters.

Each chapter is organized by a presenting SYMPTOM and describes how its CAUSES are distinguished
by Bates attributes (chest pain: angina = pressure/substernal/worse-exertion/relieved-by-nitro vs
reflux = burning/substernal/worse-recumbency/relieved-by-antacid vs pleuritic = sharp/worse-
inspiration). This IE extracts, per cause, that cause's attributed version of the symptom -- the
disease-SPECIFIC attribute contrast that discriminates diseases sharing a symptom, which the
disease-oriented sources (StatPearls/PMC) omit as clinician-obvious.

Benchmark-blind (input = textbook prose only, no benchmark question/symptom list), source-grounded
(only facts in the SOURCE), textbook IE (§4-safe). Output feeds a deterministic CUI-linking +
KG-merge step (semiology_build_edges.py). Output: pilot/cache/prod/semiology_ie.json =
{nbk: {symptom, causes:[{cause,character,location,radiation,timing,onset,aggravating,relieving}]}}.
"""
from __future__ import annotations
import json, os, re, sys, argparse
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend


def parse_obj(txt):
    """Extract the JSON object from a model output and return it whole (not just one key)."""
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

# chapters that are imaging/exam technique, not symptom semiology -> skip
SKIP_TITLE = re.compile(r"(?i)roentgenog|examination|chest examination")

PROMPT = '''You are a meticulous clinical information extractor. The SOURCE is a medical textbook chapter about the symptom "{symptom}". It discusses several diseases/conditions that CAUSE this symptom and how each one's version of the symptom differs.

SOURCE:
"""{src}"""

For EACH distinct disease/condition the SOURCE discusses as a cause of "{symptom}", extract how the SOURCE characterizes THAT cause's version of the symptom, USING ONLY facts explicitly stated in the SOURCE. Do not invent attributes; leave a field "" when the SOURCE does not state it for that cause.
- "cause": the disease/condition name exactly as the SOURCE names it.
- "character": the quality/nature (burning, pressure, crushing, sharp, dull, colicky, cramping, stabbing, throbbing, tearing, squeezing, aching, ...).
- "location": anatomical site.  "radiation": where it spreads/extends/radiates to.
- "timing": temporal pattern (nocturnal, postprandial, episodic, exertional, morning, ...).  "onset": sudden or gradual.
- "aggravating": what the SOURCE says makes it worse (exertion, recumbency/lying down, meals/food, deep breathing, movement, ...).
- "relieving": what the SOURCE says makes it better (rest, antacids, nitroglycerin, sitting up, ...).

Output ONLY JSON: {{"symptom":"{symptom}","causes":[{{"cause":"","character":"","location":"","radiation":"","timing":"","onset":"","aggravating":"","relieving":""}}]}}'''

FIELDS = ["cause", "character", "location", "radiation", "timing", "onset", "aggravating", "relieving"]


def symptom_of(title: str) -> str:
    t = re.sub(r"\s*(or Discomfort|and Associated Symptoms|, Chills, and Night Sweats|"
               r"and Sputum Production|, Orthopnea.*|, Frequency, and Urgency|, Melena.*)", "", title, flags=re.I)
    t = t.replace(" and ", " / ")
    return t.strip().lower()


def chunks(text: str, size: int, overlap: int = 800):
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="pilot/cache/prod/sources_semiology")
    ap.add_argument("--out", default="pilot/cache/prod/semiology_ie.json")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=13000)
    a = ap.parse_args()

    files = [f for f in sorted(os.listdir(a.src_dir)) if f.endswith(".json") and f != "manifest.json"]
    jobs = []   # (nbk, symptom, chunk_text)
    meta = {}
    for f in files:
        d = json.load(open(os.path.join(a.src_dir, f)))
        if SKIP_TITLE.search(d["title"]):
            continue
        sym = symptom_of(d["title"])
        meta[d["nbk"]] = sym
        for ch in chunks(d["text"], a.chunk):
            jobs.append((d["nbk"], sym, ch))
    print(f"semiology IE: {len(meta)} chapters -> {len(jobs)} chunks", flush=True)

    chat = build_backend(a.backend, a.bs, a.max_new)
    outs = chat([PROMPT.format(symptom=sym, src=ch) for _, sym, ch in jobs])

    # merge chunks per chapter; dedup causes by name (first non-empty attr wins)
    bynbk = {}
    for (nbk, sym, _), txt in zip(jobs, outs):
        j = parse_obj(txt)
        causes = j.get("causes", []) if isinstance(j, dict) else []
        agg = bynbk.setdefault(nbk, {"symptom": sym, "causes": {}})
        for c in causes:
            if not isinstance(c, dict):
                continue
            nm = str(c.get("cause", "")).strip()
            if not nm or len(nm) < 3:
                continue
            cell = agg["causes"].setdefault(nm.lower(), {k: "" for k in FIELDS})
            cell["cause"] = nm
            for k in FIELDS[1:]:
                v = str(c.get(k, "")).strip()
                if v and not cell[k]:
                    cell[k] = v
    result = {nbk: {"symptom": v["symptom"], "causes": list(v["causes"].values())}
              for nbk, v in bynbk.items()}
    json.dump(result, open(a.out, "w"), indent=1)

    # stats
    ncause = sum(len(v["causes"]) for v in result.values())
    fill = {k: 0 for k in FIELDS[1:]}
    for v in result.values():
        for c in v["causes"]:
            for k in fill:
                if c.get(k):
                    fill[k] += 1
    print(f"\nchapters={len(result)}  causes={ncause}  ({ncause/max(1,len(result)):.1f}/chapter)")
    print("attribute fill across causes:")
    for k in FIELDS[1:]:
        print(f"  {k:12} {fill[k]:5} ({100*fill[k]/max(1,ncause):.0f}%)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
