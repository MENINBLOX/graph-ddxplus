"""PRODUCTION IE stage of the large-scale attribute pipeline (context B = disease -> phenotypes).

Input = a disease's clinical description text (v105_sources/<CUI>.txt); output = atomic
findings + 6 candidate attributes. Reuses the VALIDATED recipe from MACCROBAT tuning:
gemma-4-E4B, single prompt, STEP1 site-atomization, conservative severity, 6-attribute schema
(v147 V4) — but re-framed for DISEASE descriptions ("symptoms a patient with this disease can
have") instead of case reports ("the patient has"). Benchmark-blind, no few-shot, temp=0.

Raw severity/onset are extracted VERBATIM here (faithful); the deterministic normalization to
controlled vocab (incl. 0-10 NRS) happens in the normalization layer (v152/v154), not here.

Usage: python v153_production_ie.py --n 15 [--out preds.json]   (run in .venv)"""
import json, glob, os, re, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v143_unified_4attr import SEV_R2, CHAR, ONSET
from v146_decomp import LOC_DECOMP, build_backend
from v147_single6 import TIMING, AGGR, RELV
from v145_strategy_scope import parse_json, clean, in_src  # `clean` kept for reference; see clean_full

SRC_V105 = "pilot/cache/v105_sources"          # Wikipedia intros (symptom-poor)
SRC_SCALEUP = "pilot/cache/scaleup_sources"    # symptom-rich "Signs and symptoms" sections
SEED = "data/medkg/seeds/combined_seed.jsonl"
MODEL = "google/gemma-4-E4B-it"
ONSET_RULE = ONSET["O2"]

STEP1_PROD = ('STEP 1 — List the abnormal clinical findings (symptoms, signs, phenotypes) the SOURCE '
              'attributes to patients with this disease, ATOMICALLY. When a finding phrase contains an '
              'anatomical SITE as a modifier, strip the site OFF the name and keep ONLY the core finding '
              '("abdominal pain" -> "pain", "hepatic fibrosis" -> "fibrosis"); the site goes to location '
              'in STEP 2. Do NOT strip intensity/quality words. EXCLUDE the disease name itself, causes/'
              'risk factors, epidemiology, treatments/drugs, diagnostics, normal findings, and headings.')

HEAD_PROD = '''You are a meticulous clinical information extractor. Read the SOURCE about the disease "{disease}" and extract EVERY symptom/sign/phenotype a patient with this disease can have that the SOURCE mentions. Be EXHAUSTIVE -- do not stop early; list all of them, atomically, USING ONLY facts explicitly stated in the SOURCE.

SOURCE:
"""{src}"""

Rules:
- Extract BOTH clinical symptoms/signs AND diagnostic LABORATORY/BIOCHEMICAL abnormalities. The biochemical markers are often the PATHOGNOMONIC finding of a rare disease -- do NOT omit them: elevated or decreased metabolites, amino acids, organic acids, hormones, or enzymes; enzyme deficiencies; and "-emia"/"-uria" abnormalities (e.g., "hyperphenylalaninemia", "elevated glutaric acid", "methylmalonic aciduria", "N-acetylaspartic acid accumulation", "elevated 17-hydroxyprogesterone").
- Strip the anatomical SITE and laterality off the finding name and put them in "location" ("substernal chest pain" -> name "pain", location "substernal chest"). WHENEVER the source gives a body site for a finding, you MUST fill location -- do not leave it empty.
- Fill "character" (sharp, dull, burning, crushing, throbbing, colicky...) whenever the source states the quality of the finding.
- Fill "severity" (mild/moderate/severe or a 0-10 number), "onset" (sudden/gradual), "timing", "aggravating", "relieving" ONLY when the source states them for THAT finding, else "".
- EXCLUDE the disease name itself, causes/risk factors, epidemiology, treatments/drugs, diagnostic procedures, normal findings, and headings.
- A COMPLICATION of this disease is still something the patient has, so extract it as a finding: SLE's pericarditis and nephritis, cirrhosis' ascites, diabetes' retinopathy. What is excluded is the treatment of a complication, not the complication.
- BUT a symptom or sign does not stop being one because of the sentence it appears in. Sources name real features inside surveillance, screening, monitoring, counselling and differential-diagnosis sentences -- "screen at every visit for pallor, abnormal bleeding, petechiae, fatigue", "watch for melena or hematemesis", "examine for clubbing and cyanosis". Extract those findings (pallor, abnormal bleeding, petechiae, fatigue, melena, hematemesis, clubbing, cyanosis); it is the SCREENING ACT that is excluded, not the findings it looks for.
- EXCLUDE the patient's RESPONSE TO TREATMENT and any post-treatment state: a narrative source reports the clinical course, so phrases like "improved hemoglobin", "improved WBC", "afebrile", "resolution of the necrosis" describe what the treatment achieved, not a finding of the disease. Extract the abnormality itself (anemia, leukopenia, fever, necrosis), never its improvement.
- EXCLUDE one patient's MEASURED VALUES. "oxygen saturation 84% on room air", "temperature 38.9 °C", "white cell count of 8200/mm3", "potassium of 2.8 mmol/L" are that patient's numbers, not properties of the disease. Name the abnormality instead: hypoxemia, fever, leukocytosis, hypokalemia.
- EXCLUDE bare anatomical STRUCTURES ("epiglottis", "left ventricle"). A body part is not a finding; if the source describes something abnormal about it, extract that abnormality and put the site in "location".

Output ONLY JSON: {{"findings":[{{"name":"","location":"","character":"","severity":"","onset":"","timing":"","aggravating":"","relieving":""}}]}}'''

# HEAD_PROD collapses severity/onset/timing/aggravating/relieving into ONE rule qualified by
# "ONLY when ... else \"\"", while location gets a dedicated rule ending in "you MUST fill". The
# yields follow the prompt, not the source: location 34% vs aggravating 0.2% -- even though 57% of
# the same documents contain aggravating language and 32% contain relieving language. This variant
# gives the neglected attributes the same treatment location already had: their own rule and a
# vocabulary of admissible VALUES (value lists, not worked input/output examples, so it is not
# few-shot -- character and location already list values this way).
HEAD_PROD_ATTR = HEAD_PROD.replace(
    '- Fill "severity" (mild/moderate/severe or a 0-10 number), "onset" (sudden/gradual), "timing", '
    '"aggravating", "relieving" ONLY when the source states them for THAT finding, else "".',
    '- Fill "aggravating" WHENEVER the source says anything makes the finding worse, brings it on, '
    'triggers, precipitates, provokes or exacerbates it (exertion, lying down/recumbency, meals or '
    'specific foods, deep breathing, coughing, movement, standing, heat/cold, stress, menstruation, '
    'infection, fasting...). Copy the SOURCE\'s own words.\n'
    '- Fill "relieving" WHENEVER the source says anything makes it better, improves, resolves or '
    'alleviates it (rest, sitting up or leaning forward, antacids, a named drug, food, sleep, '
    'position change...). Copy the SOURCE\'s own words.\n'
    '- Fill "timing" whenever the source gives a temporal pattern (nocturnal, postprandial, '
    'episodic, intermittent, exertional, morning, periodic, progressive...).\n'
    '- Fill "severity" (mild/moderate/severe or a 0-10 number) and "onset" (sudden/gradual) only '
    'when the source states them for THAT finding, else "".')

PROMPTS = {"prod": HEAD_PROD, "attr": HEAD_PROD_ATTR}


def windows(text: str, size: int, overlap: int = 800) -> list[str]:
    """Split a long source into overlapping windows so the whole document is read.

    Sources used to be hard-capped (12000 chars) because the IE sent the blob in ONE prompt;
    GeneReviews phenotype sections are ~24k chars, so half the phenotype prose was discarded.
    The overlap keeps a finding that straddles a window boundary readable in one of them."""
    if size <= 0 or len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


_OBJ = re.compile(r"\{[^{}]*\}")

# The KG's candidate attribute schema is location/character/severity/timing/aggravating/relieving
# (CLAUDE.md), and the prompt asks for all of them -- but `v145.clean` keeps only four
# (onset/severity/location/character) because it is frozen for the v144 apples-to-apples
# comparison. Used as the production cleaner it SILENTLY DELETED timing/aggravating/relieving from
# every run, which is why those three read as 0% "missing from the sources". They were extracted
# and then discarded. This cleaner keeps all seven with the same source-grounding rule.
ATTRS = ("onset", "severity", "location", "character", "timing", "aggravating", "relieving")


# A finding is a property of the DISEASE; a number measured in one patient is not. Narrative
# sources report both in the same sentences, so the IE returns "oxygen saturation 84% on room air",
# "high temperature at 38.9 °C", "white cell count of 8200/mm3". These become KG phenotypes, and
# because many such strings normalize onto one concept they turn into spurious hubs -- measured,
# "potassium of 2.8 mmol/L" ended up on 327 diseases, linking unrelated diseases through a single
# patient's lab value. Case-level inspection showed this flipping real rankings.
_MEASURED_VALUE = re.compile(
    r"\d+\s*(?:%|°|mm\s?hg|mmhg|mmol|mg/dl|g/dl|iu/l|u/l|ng/ml|mcg|mm3|cells?/|/ml|/mm|beats?/min)"
    r"|\b\d+(?:\.\d+)?\s*(?:°\s*[cf]|percent)\b"
    r"|\b(?:saturation|count|level|titer|titre|concentration|pressure|index|rate)\b[^.]{0,20}\d"
    r"|\d[^.]{0,20}\b(?:saturation|count|level|titer|titre)\b", re.I)


def is_measured_value(name: str) -> bool:
    """True when the finding names one patient's measurement rather than a clinical property."""
    return bool(_MEASURED_VALUE.search(name))


def clean_full(finds, srcl):
    """Source-grounded cleaner over the FULL attribute schema (see ATTRS)."""
    out = []
    for f in finds:
        if not isinstance(f, dict):
            continue
        nm = str(f.get("name", "")).strip().lower()
        if len(nm) < 2 or not in_src(nm, srcl):
            continue
        if is_measured_value(nm):
            continue
        g = {"name": nm}
        for at in ATTRS:
            v = str(f.get(at, "") or "").strip().lower()
            if at == "onset":
                v = v if v in ("sudden", "gradual") else ""
            elif v and not in_src(v, srcl):
                v = ""                      # keep only attribute text present in the SOURCE
            g[at] = v
        out.append(g)
    return out


def parse_findings(txt: str) -> tuple[list[dict], bool]:
    """Parse the model's findings, salvaging a response that hit the max_new cap.

    An exhaustive list over a long window can exceed the output budget; the JSON is then cut
    mid-array and `json.loads` yields NOTHING -- the whole window is silently lost (measured:
    Spinocerebellar Ataxia Type 3 went to 0 findings this way). Complete objects before the cut
    are still valid findings, so salvage them and report that the window was truncated."""
    findings = parse_json(txt)
    if findings:
        return findings, False
    salvaged = []
    for m in _OBJ.finditer(txt):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and str(o.get("name", "")).strip():
            salvaged.append(o)
    return salvaged, True


def merge_findings(groups: list[list[dict]]) -> list[dict]:
    """Union findings across a document's windows, deduped by name (first non-empty attr wins)."""
    merged: dict[str, dict] = {}
    for g in groups:
        for f in g:
            key = str(f.get("name", "")).strip().lower()
            if not key:
                continue
            cell = merged.setdefault(key, dict(f))
            for k, v in f.items():
                if str(v or "").strip() and not str(cell.get(k, "") or "").strip():
                    cell[k] = v
    return list(merged.values())


def load_names():
    m = {}
    for line in open(SEED, encoding="utf-8"):
        try:
            d = json.loads(line); m[d["cui"]] = d["name"]
        except Exception:
            pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--source", choices=["v105", "scaleup", "prod"], default="scaleup")
    ap.add_argument("--src-dir", default="pilot/cache/prod/sources",
                    help="for --source prod: per-disease {cui}.json from v156_build_sources")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--chunk", type=int, default=1000, help="checkpoint every N diseases")
    ap.add_argument("--win", type=int, default=12000,
                    help="source characters per IE prompt; longer sources are read as overlapping "
                         "windows and their findings merged (0 = one prompt, old behaviour)")
    ap.add_argument("--resume", action="store_true", help="skip diseases already in --out")
    ap.add_argument("--prompt", choices=["prod", "attr"], default="prod",
                    help="'attr' gives aggravating/relieving/timing their own extraction "
                         "rules; 'prod' is the original single collapsed rule")
    ap.add_argument("--out", default="pilot/cache/prod/ie_pred.json")
    a = ap.parse_args()
    docs = []  # (key, disease_name, source_text)
    if a.source == "v105":
        names = load_names()
        for p in sorted(glob.glob(f"{SRC_V105}/*.txt"))[:a.n]:
            cui = os.path.basename(p)[:-4]
            docs.append((cui, names.get(cui, cui), open(p, encoding="utf-8").read()))
    elif a.source == "prod":  # scale-up sources from v156 (keyed by CUI); {disease,cui,text}
        for p in sorted(glob.glob(f"{a.src_dir}/*.json"))[:a.n]:
            d = json.load(open(p, encoding="utf-8"))
            docs.append((d.get("cui", os.path.basename(p)[:-5]), d.get("disease", ""), d.get("text", "")))
    else:  # scaleup: symptom-rich sections, keyed by index; {disease,title,text}
        for p in sorted(glob.glob(f"{SRC_SCALEUP}/*.json"))[:a.n]:
            d = json.load(open(p, encoding="utf-8"))
            key = os.path.basename(p)[:-5]
            docs.append((key, d.get("disease", key), d.get("text", "")))

    prompt = PROMPTS[a.prompt]   # exhaustive + mandatory-location; see HEAD_PROD_ATTR

    # checkpoint/resume: a full 38k run is ~10-20h, so process in chunks and persist after each
    # (atomic temp+rename). On --resume, skip diseases already in the output file.
    preds = {}
    if a.resume and os.path.exists(a.out):
        preds = json.load(open(a.out, encoding="utf-8"))
        docs = [d for d in docs if d[0] not in preds]
        print(f"resume: {len(preds)} done, {len(docs)} pending", flush=True)

    def save():
        tmp = a.out + ".tmp"
        json.dump(preds, open(tmp, "w"))
        os.replace(tmp, a.out)

    chat = build_backend(a.backend, a.bs, a.max_new)
    for ci in range(0, len(docs), a.chunk):
        chunk = docs[ci:ci + a.chunk]
        # one prompt PER WINDOW; keep the doc index so the windows regroup per disease
        jobs = [(di, nm, w)
                for di, (_, nm, s) in enumerate(chunk)
                for w in windows(s, a.win)]
        texts = chat([prompt.replace("{disease}", nm).replace("{src}", w) for _, nm, w in jobs])
        per_doc: dict[int, list[list[dict]]] = {}
        n_trunc = 0
        for (di, _, w), t in zip(jobs, texts):
            fs, truncated = parse_findings(t)
            n_trunc += truncated
            per_doc.setdefault(di, []).append(clean_full(fs, w.lower()))
        for di, (cui, nm, s) in enumerate(chunk):
            preds[cui] = {"disease": nm, "source": s,
                          "findings": merge_findings(per_doc.get(di, []))}
        save()
        print(f"  chunk {ci//a.chunk + 1}/{(len(docs)+a.chunk-1)//a.chunk} -> {len(preds)} total saved"
              f"  ({n_trunc}/{len(jobs)} windows hit the output cap and were salvaged)", flush=True)
    # quick stats
    nf = sum(len(v["findings"]) for v in preds.values())
    filled = {k: 0 for k in ATTRS}
    for v in preds.values():
        for f in v["findings"]:
            for k in filled:
                if str(f.get(k, "")).strip(): filled[k] += 1
    print(f"diseases={len(preds)} findings={nf} ({nf/max(1,len(preds)):.1f}/disease)", flush=True)
    print("attr fill:", {k: f"{filled[k]}/{nf} ({filled[k]/max(1,nf):.0%})" for k in filled}, flush=True)
    print(f"saved -> {a.out}", flush=True)
    for cui, v in list(preds.items())[:2]:
        print(f"\n[{cui}] {v['disease']}: {[f['name'] for f in v['findings']][:12]}")


if __name__ == "__main__":
    main()
