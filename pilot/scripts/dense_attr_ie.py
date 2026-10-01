"""Attribute-DENSE re-IE for the DDXPlus 49 diseases (validation of the 'thin attribute' fix).

The production IE under-extracted character/severity/onset (KG edges: location 71% vs character
14% / severity 7% / onset 6% on pain edges) — so DDXPlus's rich per-symptom attribute questions
(character/intensity/onset) had nothing to match against and were null in the ablation. The
CONTENT is in the sources (StatPearls H&P describes symptom quality/timing in 50-69% of blobs);
the extractor just favored location. This re-IE uses a prompt that MANDATES character/severity/
onset extraction whenever the source hints at symptom quality/intensity/timing, from the best
available source per disease. Output feeds the same v154/v155 normalization + KG build, then the
DDXPlus attribute combination ablation re-runs to see whether character/severity/onset now credit.
"""
from __future__ import annotations
import json, os, sys, argparse
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend
from v145_strategy_scope import parse_json, clean

PROMPT = '''You are a meticulous clinical information extractor. From the SOURCE about "{disease}", list EXHAUSTIVELY every symptom/sign/finding a patient with this disease can have, atomically, one concept per item, USING ONLY facts stated in the SOURCE.

SOURCE:
"""{src}"""

For EACH finding, aggressively fill these Bates attributes WHENEVER the source gives ANY hint of them (do not leave them empty if the source describes the symptom's site, quality, intensity, or timing):
- "location": anatomical site of the finding (strip it off the name: "burning chest pain" -> name "pain", location "chest").
- "character": the QUALITY/nature of the finding — sharp, dull, burning, crushing, colicky, cramping, throbbing, stabbing, gnawing, squeezing, pressure-like, tearing, band-like, aching. Extract this whenever the source describes HOW the symptom feels.
- "severity": mild / moderate / severe (or a 0-10 number) whenever the source conveys intensity.
- "onset": sudden or gradual whenever the source describes how the symptom began (acute/abrupt -> sudden; insidious/gradual -> gradual).
Strip the attribute words off the finding name. EXCLUDE the disease name, causes, epidemiology, treatments, procedures, normal findings.

Output ONLY JSON: {{"findings":[{{"name":"","location":"","character":"","severity":"","onset":""}}]}}'''


def best_source(cui):
    for d in ("sources_sp", "sources_pmc", "sources"):
        p = f"pilot/cache/prod/{d}/{cui}.json"
        if os.path.exists(p):
            t = json.load(open(p)).get("text", "")
            if t.strip():
                return t[:9000]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot/cache/prod/ie_pred_ddx_denseattr.json")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    a = ap.parse_args()
    dm = json.load(open("data/ddxplus/disease_umls_mapping.json"))["mapping"]
    docs = []
    for info in dm.values():
        cu = info.get("umls_cui")
        if not cu:
            continue
        src = best_source(cu)
        if src:
            docs.append((cu, info.get("name_en", cu), src))
    docs = list({d[0]: d for d in docs}.values())
    print(f"dense-attr IE on {len(docs)} DDXPlus diseases", flush=True)
    chat = build_backend("vllm", a.bs, a.max_new)
    preds = {}
    outs = chat([PROMPT.replace("{disease}", nm).replace("{src}", s) for _, nm, s in docs])
    for (cui, nm, s), t in zip(docs, outs):
        preds[cui] = {"disease": nm, "source": s, "findings": clean(parse_json(t), s.lower())}
    json.dump(preds, open(a.out, "w"))
    # attribute-fill stats
    nf = sum(len(v["findings"]) for v in preds.values())
    fill = {k: 0 for k in ("location", "character", "severity", "onset")}
    for v in preds.values():
        for f in v["findings"]:
            for k in fill:
                if str(f.get(k, "")).strip():
                    fill[k] += 1
    print(f"findings={nf}; attr fill: " + ", ".join(f"{k} {100*fill[k]/max(1,nf):.0f}%" for k in fill), flush=True)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
