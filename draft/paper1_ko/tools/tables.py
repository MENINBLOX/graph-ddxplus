"""Tables of the manuscript, generated from the outputs of the evaluation.

    python3 tables.py <repo root> <output dir>

Writes one markdown file per table: table1.md, table2.md and tableS1.md to tableS7.md.
"""

import json
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DEV = ROOT / "paper1/outputs/v3/dev"
EVAL = ROOT / "paper1/outputs/v3/eval/T"
MINUS = "−"
DASH = "–"


def load(p):
    return json.loads(p.read_text())


def f(x, nd=2):
    s = f"{x:.{nd}f}"
    return s.replace("-", MINUS)


def signed(x, nd=2):
    s = f"{x:+.{nd}f}"
    return s.replace("-", MINUS)


def pct_ci(d):
    return f"{f(100 * d['estimate'])} ({f(100 * d['low'])}{DASH}{f(100 * d['high'])})"


def mean_ci(d):
    return f"{f(d['mean'])} ({f(d['low'])}{DASH}{f(d['high'])})"


def diff_ci(d, scale=100.0):
    return f"{signed(scale * d['estimate'])} ({signed(scale * d['low'])} to {signed(scale * d['high'])})"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write(name, title, body, notes):
    text = f"**{title}**\n\n{body}\n\n" + "\n".join(notes) + "\n"
    (OUT / f"{name}.md").write_text(text)


main, base, hier = load(EVAL / "main.json"), load(EVAL / "baselines.json"), load(EVAL / "hierarchy.json")
dmain, dbase, abl = load(DEV / "main.json"), load(DEV / "baselines.json"), load(DEV / "ablations.json")
bed = base["B-BED"]["stats_file"]

METHODS = [
    ("P1", "List-based method (P1)", "Public lists", "Initial evidence", "Yes or no"),
    ("P2-binary", "Frequency-based method (P2)", "Training frequencies", "Initial evidence", "Yes or no"),
    ("P2-binary-agesex", "P2 with age and sex", "Training frequencies", "Age, sex, initial evidence", "Yes or no"),
    ("P2-value", "P2 with answer values", "Training frequencies", "Initial evidence", "Values"),
    ("P2-value-agesex", "P2 with answer values, age and sex", "Training frequencies", "Age, sex, initial evidence", "Values"),
]
SHORT = {k: label for k, label, *_ in METHODS}
SHORT["B-BED"] = "BED re-implementation"


def run(key):
    return bed if key == "B-BED" else main[key]


# Table 1
rows = []
for key, label, knowledge, start, answers in METHODS:
    s = main[key]["stop"]
    rows.append([label, knowledge, start, answers, pct_ci(s["gtpa1"]), mean_ci(s["il"]), f(s["il"]["median"], 0), f(100 * s["il"]["share_at_cap"])])
s = bed["stop"]
rows.append(["BED re-implementation", "Public lists and training frequencies^a)^", "Initial evidence", "Values, BED simulator", pct_ci(s["gtpa1"]), mean_ci(s["il"]), f(s["il"]["median"], 0), f(100 * s["il"]["share_at_cap"])])
write(
    "table1",
    f"Table 1. Accuracy for the true disease and number of evidences collected on the test set ({main['P1']['n']:,} patients)",
    table(["Method", "Knowledge", "Input at start", "Answers", "Accuracy, % (95% CI)", "Evidence collected, mean (95% CI)", "Median", "Stopped at the question limit, %"], rows),
    [
        "CI, confidence interval; BED, Bayesian experimental design baseline.",
        "Evidence collected includes the initial evidence. P1 and P2 stop when the largest posterior probability reaches 0.99; BED stops by its own rule.",
        "^a)^The statistics file of the training patients distributed with the BED code.",
    ],
)

# Table 2: component analysis on validation
p1 = dmain["P1"]


def comp(label, r, against="P1"):
    c = r["against"][against]["stop"]
    return [label, pct_ci(r["stop"]["gtpa1"]), diff_ci(c["gtpa1_diff_interval"]), f(r["stop"]["il"]["mean"])]


b2 = dbase["B2"]["over_seeds"]["stop"]
rows = [
    ["List-based method (P1)", pct_ci(p1["stop"]["gtpa1"]), DASH, f(p1["stop"]["il"]["mean"])],
    ["*Answers used*", "", "", ""],
    comp("Only “yes” answers to questions", abl["A1"]["positive_only"]),
    comp("Only “no” answers to questions", abl["A1"]["negative_only"]),
    ["*Probability of “yes” for an evidence outside the list (ε)*", "", "", ""],
    comp("ε = 0.1", abl["A3"]["0.1"]),
    comp("ε = 10^−2^", abl["A3"]["0.01"]),
    comp("ε = 10^−9^", abl["A3"]["1e-09"]),
    ["*Question selection*", "", "", ""],
    comp("Evidence on the most candidate lists", abl["A5"]["coverage"]),
    ["Random evidence from the candidate lists^a)^", f"{f(100 * b2['gtpa1']['mean'])} ({f(100 * b2['gtpa1']['min'])}{DASH}{f(100 * b2['gtpa1']['max'])})", f"{signed(100 * b2['gtpa1_diff_vs_P1']['mean'])} ({signed(100 * b2['gtpa1_diff_vs_P1']['min'])} to {signed(100 * b2['gtpa1_diff_vs_P1']['max'])})", f(b2["il_mean"]["mean"])],
    comp("Most frequent evidence in the training set first", dbase["B6"]["P1"]),
    ["*Sub-questions*", "", "", ""],
    comp("Answers filled in", load(DEV / "hierarchy.json")["P1"]),
    comp("Asked only after “yes” to the main question", abl["A4"]["parent_first"]),
    comp("Never asked", abl["A4"]["no_children"]),
    ["*Prior probability*", "", "", ""],
    comp("Disease frequencies of the training set", abl["A2"]["train_prior"]),
]
write(
    "table2",
    f"Table 2. Component analysis of the list-based method on the validation set ({p1['n']:,} patients)",
    table(["Changed component", "Accuracy, % (95% CI)", "Difference from P1, percentage points (95% CI)", "Evidence collected, mean"], rows),
    [
        "CI, confidence interval.",
        "Each row changes one component of P1 and keeps the others.",
        "^a)^Mean of five random orders; the range over the five orders is shown in place of the CI.",
    ],
)

# Table S1: additional metrics on test
rows = []
for key in [m[0] for m in METHODS] + ["B-BED"]:
    s = run(key)["stop"]
    rows.append([SHORT[key], f(100 * s["gtpa1"]["estimate"]), f(100 * s["gtpa1_official"]["estimate"]), f(100 * s["gtpa3"]["estimate"]), f(100 * s["gtpa5"]["estimate"]), f(100 * s["ddr"]["estimate"]), f(100 * s["ddp"]["estimate"]), f(100 * s["ddf1"]["estimate"]), f(100 * s["dshm"]["estimate"]), f(100 * s["per_official"]["estimate"])])
write(
    "tableS1",
    "Supplementary Table S1. Additional metrics on the test set (%)",
    table(["Method", "Accuracy", "Accuracy, official code", "GTPA@3", "GTPA@5", "DDR", "DDP", "DDF1", "DSHM", "PER"], rows),
    [
        "GTPA@k, the true disease is among the k most probable diseases; DDR, DDP and DDF1, recall, precision and their harmonic mean against the reference differential diagnosis; DSHM, harmonic mean of recall and precision for severe diseases; PER, share of the positive evidences of the patient that were collected.",
        "Accuracy, official code: the accuracy as the official code computes it (Supplementary Methods 3).",
    ],
)

# Table S2: stop threshold on test
TAUS = ("0.9", "0.95", "0.99", "0.999")
rows = []
for key in [m[0] for m in METHODS]:
    for t in TAUS:
        s = main[key]["tau_sensitivity"][t]
        rows.append([SHORT[key] if t == TAUS[0] else "", t, pct_ci(s["gtpa1"]), mean_ci(s["il"]), f(100 * s["il"]["share_at_cap"])])
write(
    "tableS2",
    "Supplementary Table S2. Results on the test set by stop threshold",
    table(["Method", "Stop threshold", "Accuracy, % (95% CI)", "Evidence collected, mean (95% CI)", "Stopped at the question limit, %"], rows),
    ["CI, confidence interval.", "The main analysis uses the threshold 0.99."],
)

# Table S3: accuracy by number of questions on test
POINTS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30)
series = [(SHORT[k], run(k)["curve"]["gtpa1"]) for k in [m[0] for m in METHODS] + ["B-BED"]]
seeds = base["B2"]["seeds"]
random_curve = [sum(seeds[s]["curve"]["gtpa1"][b] for s in seeds) / len(seeds) for b in range(31)]
series.append(("P1, random evidence from the candidate lists^a)^", random_curve))
series.append(("P1, most frequent evidence first", base["B6"]["P1"]["curve"]["gtpa1"]))
rows = [[label] + [f(100 * curve[b]) for b in POINTS] for label, curve in series]
write(
    "tableS3",
    "Supplementary Table S3. Accuracy (%) on the test set by number of questions asked after the initial evidence",
    table(["Method"] + [str(b) for b in POINTS], rows),
    ["Each method asks exactly the given number of questions without a stop rule.", "^a)^Mean of five random orders."],
)

# Table S4: paired comparisons on test
PAIRS = [
    ("P2-binary - P1", "P2 − P1", "Lists replaced by training frequencies"),
    ("P2-binary-agesex - P2-binary", "P2 with age and sex − P2", "Age and sex added, yes-or-no answers"),
    ("P2-value - P2-binary", "P2 with answer values − P2", "Answer values added"),
    ("P2-value-agesex - P2-value", "P2 with answer values, age and sex − P2 with answer values", "Age and sex added, answer values"),
    ("B-BED - P2-value", "BED re-implementation − P2 with answer values", "BED and its simulator"),
]
ladder = dict(main["ladder"])
rows = []
for key, label, factor in PAIRS:
    r = ladder[key]
    points = [("At the stop", r["stop"])] + [(f"After {b} questions", r[b]) for b in ("5", "10", "20", "30")]
    points += [(f"Stop threshold {t}", c) for t, c in r.get("tau_sensitivity", {}).items() if t != "0.99"]
    for i, (where, c) in enumerate(points):
        il = diff_ci(c["il_diff"], 1.0) if "il_diff" in c else DASH
        m = c["mcnemar"]
        rows.append([f"{label} ({factor})" if i == 0 else "", where, diff_ci(c["gtpa1_diff_interval"]), il, f"{m['only_this']:,} / {m['only_other']:,}"])
write(
    "tableS4",
    "Supplementary Table S4. Paired comparisons on the test set",
    table(["Comparison", "Point", "Accuracy difference, percentage points (95% CI)", "Difference in evidence collected (95% CI)", "Patients correct in only the first / only the second method"], rows),
    [
        "CI, confidence interval.",
        "Differences are the first method minus the second on the same patients. The last column counts patients with ties broken in the listed order of the public data.",
    ],
)

# Table S5: subsets of the test set
rows = []
for key in [m[0] for m in METHODS] + ["B-BED"]:
    subs = run(key)["subsets"]
    d, s = subs["T-dedup"]["stop"], subs["T-streambench"]["stop"]
    rows.append([SHORT[key], pct_ci(d["gtpa1"]), f(d["il"]["mean"]), pct_ci(s["gtpa1"]), f(s["il"]["mean"])])
nd, ns = main["P1"]["subsets"]["T-dedup"]["n"], main["P1"]["subsets"]["T-streambench"]["n"]
write(
    "tableS5",
    "Supplementary Table S5. Results on two subsets of the test set",
    table(["Method", f"No identical patient in the training set (n = {nd:,}): accuracy, % (95% CI)", "Evidence collected, mean", f"StreamBench patients (n = {ns:,}): accuracy, % (95% CI)", "Evidence collected, mean"], rows),
    ["CI, confidence interval."],
)

# Table S6: candidates of P1 on validation
tried = [json.loads(line) for line in (DEV / "configs_tried.jsonl").read_text().splitlines()]
rows = []
for r in tried:
    if r["policy"] == "P1" and r["config"]["tau"] == 0.99:
        cfg = r["config"]
        rows.append([f(cfg["alpha"], 1), f"{cfg['eps']:g}", f(100 * r["stop"]["gtpa1"]["estimate"]), f(r["stop"]["il"]["mean"]), f(100 * r["curve"]["gtpa1"][-1]), "Candidate" if r["role"] == "selection" else "Outside the candidates"])
write(
    "tableS6",
    "Supplementary Table S6. Candidate settings of the list-based method on the validation set",
    table(["α", "ε", "Accuracy, %", "Evidence collected, mean", "Accuracy after 30 questions, %", "Role"], rows),
    ["The setting with the highest accuracy among the nine candidates was selected (α = 0.5, ε = 0.0001)."],
)

# Table S7: remaining component analysis and reference values
rows = []
for eps in ("0.1", "0.01", "0.001", "1e-06", "1e-09"):
    rows.append(comp(f"ε = {float(eps):g}", abl["A3"][eps]))
ref = []
for split, b in (("Validation", dbase), ("Test", base)):
    ref.append([split, pct_ci(b["B0"]["most_frequent"]["stop"]["gtpa1"]), pct_ci(b["B0"]["initial_only"]["stop"]["gtpa1"]), pct_ci(b["B1"]["set_inclusion"]["stop"]["gtpa1"]), pct_ci(b["B1"]["kb_bayes"]["stop"]["gtpa1"]), pct_ci(b["B1"]["naive_bayes/P2-binary"]["stop"]["gtpa1"]), pct_ci(b["B1"]["naive_bayes/P2-value"]["stop"]["gtpa1"])])
body = table(["Changed component", "Accuracy, % (95% CI)", "Difference from P1, percentage points (95% CI)", "Evidence collected, mean"], rows)
body += "\n\n" + table(["Set", "Most frequent disease", "Initial evidence only", "List check on the full record", "P1 probabilities on the full record", "Naive Bayes on the full record, yes or no", "Naive Bayes on the full record, values"], ref)
write(
    "tableS7",
    "Supplementary Table S7. Further component analysis on the validation set and reference values without questions",
    body,
    ["CI, confidence interval.", "The lower part gives the accuracy, % (95% CI), of methods that ask no question."],
)
# Table 3: reported results and conditions, with this study's rows from the outputs
LIT = [
    ["*Official interactive rules, whole test set*", "", "", "", "", "", "", ""],
    ["AARLC [@aarlc]^b)^", "Training patients", "134,529", "Age, sex, initial evidence", "Patient record", "30", "99.21", "6.73"],
    ["BASD [@basd]^b)^", "Training patients", "134,529", "Age, sex, initial evidence", "Patient record", "30", "97.15", "17.99"],
    ["Diaformer [@diaformer]^c)^", "Training patients", "134,529", "Age, sex, initial evidence", "Patient record", "30", "91.81", "18.45"],
    ["CASANDE [@casande]^c)^", "Training patients", "134,529", "Age, sex, initial evidence", "Patient record, sub-question answers filled in", "30", "98.8", "19.53"],
    ["BED [@bed]^c)^", "Public lists and training frequencies", "134,529", "Initial evidence", "BED simulator", "30", "99.47", "5.47"],
]
OTHER = [
    ["*Other conditions*", "", "", "", "", "", "", ""],
    ["CaRDI [@cardi]^d)^", "Training frequencies and an LLM prior", "490 from the test set", "Age, sex, initial evidence", "Patient record with values", "10", "96.12–96.69", "4.54–4.64"],
    ["MEDDxAgent [@meddxagent]^e)^", "LLM", "100 from the StreamBench patients", "Age, sex, chief complaint", "LLM patient simulator", "15", "86", "NR"],
    ["StreamBench [@streambench]^f)^", "LLM", "1,764 from the test set", "Full patient record", "None", "0", "70.64–92.01", "NA"],
]
ours = []
for key, label, knowledge, start, answers in (METHODS[0], METHODS[4]):
    st = main[key]["stop"]
    name = {"P1": "list-based method (P1)", "P2-value-agesex": "frequency-based method (P2) with answer values, age and sex"}[key]
    ours.append([f"This study, {name}", knowledge, f"{main[key]['n']:,}", start, "Patient record" + (" with values" if answers == "Values" else ", yes or no"), "30", f(100 * st["gtpa1"]["estimate"]), f(st["il"]["mean"])])
st = bed["stop"]
ours.append(["This study, BED re-implementation", "Public lists and training frequencies", f"{bed['n']:,}", "Initial evidence", "BED simulator", "30", f(100 * st["gtpa1"]["estimate"]), f(st["il"]["mean"])])
write(
    "table3",
    "Table 3. Results reported on DDXPlus and their evaluation conditions",
    table(["Study", "Knowledge used", "Patients evaluated", "Input at start", "Answers", "Question limit", "Accuracy, %", "Evidence collected, mean^a)^"], LIT + ours + OTHER),
    [
        "LLM, large language model; NR, not reported; NA, not applicable.",
        "^a)^Including the initial evidence.",
        "^b)^Model trained to predict the true disease; values from Table 3 of the DDXPlus paper [@ddxplus].",
        "^c)^Model trained to predict the true disease, and BED; values from Table 1 of the CASANDE paper [@casande].",
        "^d)^Range over three LLMs; the reported 3.54–3.64 questions after the initial evidence are shown with the initial evidence added.",
        "^e)^GPT-4o on the interactive task after 15 questions.",
        "^f)^GPT-4o, from zero-shot prompting to the best reported method.",
    ],
)
print("tables written")
