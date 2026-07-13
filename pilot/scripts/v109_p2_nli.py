"""NLI faithfulness of the IMPROVED (P2 / atomized) attribute-IE output.

Reuses the FROZEN v109c method verbatim for comparability with the frozen v106
number (roberta-large-mnli, sentence-level max-entailment, faithful if max P(entail)>0.5).
Adapts only the INPUT reader to the P2 ARRAY schema: each atom in an array attribute
(location/character/radiation/aggravating/relieving/associated) becomes one claim; each
non-empty scalar attribute (onset/duration/severity/timing/course/context/prior_episodes)
becomes one claim; each finding name is one claim.

Hypothesis phrasing is IDENTICAL to v109c so the entailment scores are directly comparable
to the frozen v106=83%. Premise = disease source text, split into sentences; a claim is
faithful if it is entailed by ANY source sentence (max over sentences).

Usage: python v109_p2_nli.py [p2|scaleup]
"""
import json, glob, re, os, sys, statistics
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET = sys.argv[1] if len(sys.argv) > 1 else "p2"
MID = "FacebookAI/roberta-large-mnli"   # SAME model as v109c (frozen 83% number)
THRESH = 0.5                             # SAME threshold as v109c

tok = AutoTokenizer.from_pretrained(MID)
mdl = AutoModelForSequenceClassification.from_pretrained(MID, dtype=torch.float16).to("cuda").eval()
ENT = [i for i, l in mdl.config.id2label.items() if l.lower().startswith("entail")][0]

# hypothesis templates -- v109c templates verbatim, extended for course/prior_episodes
def hyp(name, a, v):
    T = {"location":   f"The {name} is located in the {v}.",
         "onset":      f"The {name} has a {v} onset.",
         "severity":   f"The {name} is {v}.",
         "character":  f"The {name} is {v}.",
         "radiation":  f"The {name} radiates {v}.",
         "timing":     f"The {name} occurs {v}.",
         "aggravating": f"The {name} is aggravated by {v}.",
         "relieving":  f"The {name} is relieved by {v}.",
         "duration":   f"The {name} lasts {v}.",
         "associated": f"The {name} is associated with {v}.",
         "context":    f"The {name} occurs in the context of {v}.",
         "course":     f"The {name} has a {v} course.",
         "prior_episodes": f"The {name} has {v} prior episodes."}
    return T.get(a, f"The {name} is {v}.")

def split_sents(t):
    t = re.sub(r'\s+', ' ', t)
    return [s.strip() for s in re.split(r'(?<=[.;])\s+', t) if len(s.strip()) > 15][:30]

def aslist(v):
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is not None and str(v).strip():
        return [str(v).strip()]
    return []

ARR = ["location", "character", "radiation", "aggravating", "relieving", "associated"]
SCA = ["onset", "duration", "severity", "timing", "course", "context", "prior_episodes"]

# ---- load (IE output, source sentences) pairs -------------------------------
def load_p2():
    src = {fp.split("/")[-1][:-4]: split_sents(open(fp).read()[:2200])
           for fp in glob.glob("pilot/data/cache/v105_sources/*.txt")}
    for fp in sorted(glob.glob("pilot/data/cache/ie_p2/*.json")):
        if os.path.basename(fp).startswith("_"):
            continue
        o = json.load(open(fp))
        ss = src.get(o.get("cui"))
        if ss:
            yield o, ss

def load_scaleup():
    for fp in sorted(glob.glob("pilot/data/cache/ie_scaleup/*.json")):
        if os.path.basename(fp).startswith("_"):
            continue
        o = json.load(open(fp))
        base = os.path.basename(fp)
        sfp = f"pilot/data/cache/scaleup_sources/{base}"
        if not os.path.exists(sfp):
            continue
        s = json.load(open(sfp)).get("text", "")
        ss = split_sents(s[:2200])
        if ss:
            yield o, ss

loader = {"p2": load_p2, "scaleup": load_scaleup}[DATASET]

# ---- build flat NLI pairs, one claim = (attr-bucket) over N source sentences --
P = []; H = []; cid = []; bucket = []; n = 0
for o, ss in loader():
    for f in o.get("findings", []):
        name = str(f.get("name", "")).strip()
        if not name:
            continue
        # finding-name claim
        for s in ss:
            P.append(s); H.append(f"Patients with this condition have {name}."); cid.append(n)
        bucket.append("finding-name"); n += 1
        # attribute claims (one per atom)
        for k in ARR + SCA:
            for atom in aslist(f.get(k)):
                for s in ss:
                    P.append(s); H.append(hyp(name, k, atom)); cid.append(n)
                bucket.append(k); n += 1

print(f"[{DATASET}] claims={n}  NLI-pairs={len(P)}", flush=True)

@torch.no_grad()
def ent(prems, hyps, bs=128):
    out = []
    for i in range(0, len(prems), bs):
        enc = tok(prems[i:i+bs], hyps[i:i+bs], return_tensors="pt",
                  padding="max_length", truncation=True, max_length=96).to("cuda")
        out += [float(x) for x in torch.softmax(mdl(**enc).logits.float(), dim=1)[:, ENT]]
    return out

sc = ent(P, H)
mx = [0.0] * n
for c, x in zip(cid, sc):
    if x > mx[c]:
        mx[c] = x

# ---- aggregate ---------------------------------------------------------------
def report_bucket(names):
    idx = [i for i in range(n) if bucket[i] in names]
    if not idx:
        return None
    vals = [mx[i] for i in idx]
    faithful = sum(1 for v in vals if v > THRESH)
    return len(vals), 100 * faithful / len(vals), statistics.mean(vals)

groups = [
    ("finding-name", ["finding-name"]),
    ("location", ["location"]),
    ("character", ["character"]),
    ("severity", ["severity"]),
    ("timing/onset", ["timing", "onset"]),
    ("aggravating", ["aggravating"]),
    ("relieving", ["relieving"]),
    ("associated", ["associated"]),
    ("radiation", ["radiation"]),
    ("duration", ["duration"]),
    ("course", ["course"]),
    ("context", ["context"]),
    ("prior_episodes", ["prior_episodes"]),
]

overall_n = n
overall_faithful = sum(1 for v in mx if v > THRESH)
overall_pct = 100 * overall_faithful / max(overall_n, 1)

print(f"\n=== P2 faithfulness ({DATASET}) | model={MID} | thresh={THRESH} ===")
print(f"{'attribute':<16}{'N':>7}{'faithful%':>11}{'mean-entail':>13}")
for label, names in groups:
    r = report_bucket(names)
    if r:
        nn, pct, mean = r
        print(f"{label:<16}{nn:>7}{pct:>10.1f}%{mean:>13.3f}")

# finding vs attr split (matches v109c reporting)
fnd = [mx[i] for i in range(n) if bucket[i] == "finding-name"]
att = [mx[i] for i in range(n) if bucket[i] != "finding-name"]
def pct(a):
    return 100 * sum(1 for x in a if x > THRESH) / max(len(a), 1)
print(f"\nfinding-name : faithful%={pct(fnd):.1f}  n={len(fnd)}  mean={statistics.mean(fnd):.3f}")
print(f"attributes   : faithful%={pct(att):.1f}  n={len(att)}  mean={statistics.mean(att):.3f}")
print(f"OVERALL      : faithful%={overall_pct:.1f}  n={overall_n}")
print(f"\nFrozen v106 (v109c, same method) overall = 83%")

# dump json for report
os.makedirs("pilot/data/cache", exist_ok=True)
out = {"dataset": DATASET, "model": MID, "threshold": THRESH,
       "overall_pct": overall_pct, "overall_n": overall_n,
       "finding_pct": pct(fnd), "finding_n": len(fnd),
       "attr_pct": pct(att), "attr_n": len(att),
       "per_attribute": {label: report_bucket(names) for label, names in groups}}
json.dump(out, open(f"pilot/data/cache/v109_p2_nli_{DATASET}.json", "w"), indent=1)
print(f"\nsaved -> pilot/data/cache/v109_p2_nli_{DATASET}.json")
