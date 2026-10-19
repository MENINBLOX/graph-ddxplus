"""Compact name-only IE over the GeneReviews source blobs, for rare-disease phenotype recall.

The full 8-attribute JSON schema overflows the output budget on phenotype-rich metabolic
diseases (findings truncated -> unparseable -> 0 findings). RareBench scoring uses phenotype
CUIs only (attribute channel is disabled for HPO benchmarks), so here we emit finding NAMES
only -- compact enough to capture 40+ findings including the pathognomonic biochemistry.
Output format is v155-compatible: {cui:{disease, source, findings:[{name}]}}.
"""
from __future__ import annotations
import sys, json, re, glob, os, argparse
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend

PROMPT = '''You are a meticulous clinical information extractor. From the SOURCE about "{disease}", list EXHAUSTIVELY every finding a patient with this disease can have that the SOURCE states — atomically, one concept per item.

SOURCE:
"""{src}"""

Include BOTH (1) clinical symptoms/signs (seizures, intellectual disability, hepatomegaly, ataxia, ...) AND (2) diagnostic LABORATORY/BIOCHEMICAL abnormalities, which are often the PATHOGNOMONIC finding — elevated/decreased metabolites, amino acids, organic acids, hormones, enzymes; enzyme deficiencies; "-emia"/"-uria" abnormalities (e.g., "hyperphenylalaninemia", "elevated glutaric acid", "methylmalonic aciduria", "N-acetylaspartic acid accumulation"). Strip site/laterality/intensity modifiers off the name. Exclude the disease name, causes, epidemiology, treatments, procedures, normal findings.
Output ONLY JSON: {{"findings":["...","..."]}}'''


def parse(t):
    m = re.search(r"\{.*", t, re.S)          # tolerate truncated tail
    if not m:
        return []
    frag = m.group(0)
    names = re.findall(r'"([^"\\]{2,120})"', frag.split('"findings"', 1)[-1])
    # drop the token 'findings' if captured and obvious non-values
    return [n for n in names if n.lower() != "findings"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="pilot/cache/prod/sources_gr")
    ap.add_argument("--out", default="pilot/cache/prod/ie_pred_gr.json")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=2048)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src_dir, "*.json")))
    docs = []
    for f in files:
        d = json.load(open(f))
        txt = d.get("text", "")
        if not txt.strip():                    # skip empty blobs (e.g. no OA article found)
            continue
        docs.append((d["cui"], d.get("disease", ""), txt[:9000]))
    print(f"IE on {len(docs)} non-empty of {len(files)} source blobs", flush=True)
    chat = build_backend("vllm", a.bs, a.max_new)
    preds = {}
    for i in range(0, len(docs), 200):
        chunk = docs[i:i + 200]
        outs = chat([PROMPT.replace("{disease}", nm).replace("{src}", s) for _, nm, s in chunk])
        for (cui, nm, s), t in zip(chunk, outs):
            names = parse(t)
            preds[cui] = {"disease": nm, "source": s,
                          "findings": [{"name": n} for n in dict.fromkeys(names)]}
        json.dump(preds, open(a.out, "w"))
        print(f"  {min(i+200,len(docs))}/{len(docs)}", flush=True)
    lens = [len(v["findings"]) for v in preds.values()]
    lens.sort()
    print(f"findings/disease: median {lens[len(lens)//2]} mean {sum(lens)/len(lens):.1f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
