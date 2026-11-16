"""Tables of the manuscript, generated from the outputs of the evaluation.

    python3 tables.py <repo root> <output dir>

Writes table1.md to table3.md and tableS1.md to tableS7.md. The supplementary tables are numbered in the order the
text first cites them.
"""

import json
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DEV = ROOT / "paper1/outputs/v3/dev"
EVAL = ROOT / "paper1/outputs/v3/eval/T"
MINUS = "−"
DASH = "–"
TAUS = ("0.9", "0.95", "0.99", "0.999")


def load(p):
    return json.loads(p.read_text())


def f(x, nd=2):
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", MINUS)


def signed(x, nd=2):
    s = f"{x:+.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    return s.replace("-", MINUS)


def eps_label(eps):
    if eps >= 0.1:
        return f"{eps:g}"
    return f"10^{MINUS}{round(-__import__('math').log10(eps))}^"


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
    text = f"**{title}**\n\n{body}\n\n" + "\n\n".join(notes) + "\n"
    (OUT / f"{name}.md").write_text(text)


main, base, hier = load(EVAL / "main.json"), load(EVAL / "baselines.json"), load(EVAL / "hierarchy.json")
dmain, dbase, dhier, abl = load(DEV / "main.json"), load(DEV / "baselines.json"), load(DEV / "hierarchy.json"), load(DEV / "ablations.json")
bed = base["B-BED"]["stats_file"]

METHODS = [
    ("P1", "List-based method (P1)", "Public lists", "Initial evidence", "Yes or no"),
    ("P2-binary", "P2, yes-or-no answers", "Training frequencies", "Initial evidence", "Yes or no"),
    ("P2-binary-agesex", "P2, yes-or-no answers, age and sex", "Training frequencies", "Age, sex, initial evidence", "Yes or no"),
    ("P2-value", "P2, answer values", "Training frequencies", "Initial evidence", "Answer values"),
    ("P2-value-agesex", "P2, answer values, age and sex", "Training frequencies", "Age, sex, initial evidence", "Answer values"),
]
NAME = {k: label for k, label, *_ in METHODS}
NAME["B-BED"] = "BED re-implementation"
MAIN = [m[0] for m in METHODS]
ABBR_METHODS = "P1, list-based method; P2, frequency-based method; BED, Bayesian experimental design."


def run(key):
    return bed if key == "B-BED" else main[key]


# Table 1
rows = []
for key, label, knowledge, start, answers in METHODS:
    s = main[key]["stop"]
    rows.append([label, knowledge, start, answers, pct_ci(s["gtpa1"]), mean_ci(s["il"]), f(s["il"]["median"], 0), f(100 * s["il"]["share_at_cap"])])
s = bed["stop"]
rows.append(["BED re-implementation", "Public lists and training frequencies^a)^", "Initial evidence", "Answer values, BED simulator", pct_ci(s["gtpa1"]), mean_ci(s["il"]), f(s["il"]["median"], 0), f(100 * s["il"]["share_at_cap"])])
write(
    "table1",
    f"Table 1. Accuracy for the true disease and number of evidences collected on the test set ({main['P1']['n']:,} patients)",
    table(["Method", "Knowledge", "Input at start", "Answers", "Accuracy, % (95% CI)", "Evidence collected, mean (95% CI)", "Evidence collected, median", "Stopped at the question limit, %"], rows),
    [
        f"CI, confidence interval; {ABBR_METHODS}",
        "Evidence collected includes the initial evidence. P1 and P2 stop when the largest posterior probability reaches 0.99, when no unasked evidence has an expected information gain of at least 10^−6^ nats, or after 30 questions. BED stops when no question has a utility of at least 0.01, or after 30 questions.",
        "^a)^The statistics file of the training patients distributed with the BED code [@casande_code].",
    ],
)

# Table 2: component analysis on validation
p1 = dmain["P1"]


def comp(label, r):
    c = r["against"]["P1"]["stop"]
    return [label, pct_ci(r["stop"]["gtpa1"]), diff_ci(c["gtpa1_diff_interval"]), f(r["stop"]["il"]["mean"])]


b2 = dbase["B2"]["over_seeds"]["stop"]
rows = [["List-based method (P1)", pct_ci(p1["stop"]["gtpa1"]), DASH, f(p1["stop"]["il"]["mean"])], ["*Answers used*", "", "", ""]]
rows += [comp("Only “yes” answers to questions", abl["A1"]["positive_only"]), comp("Only “no” answers to questions", abl["A1"]["negative_only"])]
rows += [["*Probability of “yes” for an evidence outside the list, ε*", "", "", ""]]
rows += [comp(f"ε = {eps_label(float(e))}", abl["A3"][e]) for e in ("0.1", "0.01", "0.001", "1e-06", "1e-09")]
rows += [
    ["*Question selection*", "", "", ""],
    comp("Evidence on the most candidate lists", abl["A5"]["coverage"]),
    ["Random questions^a)^", f"{f(100 * b2['gtpa1']['mean'])} ({f(100 * b2['gtpa1']['min'])}{DASH}{f(100 * b2['gtpa1']['max'])})", f"{signed(100 * b2['gtpa1_diff_vs_P1']['mean'])} ({signed(100 * b2['gtpa1_diff_vs_P1']['min'])} to {signed(100 * b2['gtpa1_diff_vs_P1']['max'])})", f(b2["il_mean"]["mean"])],
    comp("Frequency-ordered questions", dbase["B6"]["P1"]),
    ["*Sub-questions*", "", "", ""],
    comp("Sub-question answers filled in", dhier["P1"]),
    comp("Asked only after “yes” to the parent question", abl["A4"]["parent_first"]),
    comp("Never asked", abl["A4"]["no_children"]),
    ["*Prior probability*", "", "", ""],
    comp("Disease frequencies of the training set", abl["A2"]["train_prior"]),
]
write(
    "table2",
    f"Table 2. Component analysis of the list-based method on the validation set ({p1['n']:,} patients)",
    table(["Changed component", "Accuracy, % (95% CI)", "Difference from P1, percentage points (95% CI)", "Evidence collected, mean"], rows),
    [
        "CI, confidence interval; P1, list-based method; α, probability of “yes” for an evidence on the list; ε, probability of “yes” for an evidence outside the list. P1 uses α = 0.5 and ε = 10^−4^.",
        "Each row changes one component of P1 and keeps the others.",
        "^a)^Mean of five random orders; the range over the five orders is shown in place of the CI.",
    ],
)

# Table 3: reported results and conditions, with this study's rows from the outputs
WHOLE = [
    ["*Whole test set, up to 30 questions after the initial evidence*", "", "", "", "", "", "", ""],
    ["AARLC [@aarlc]^b)^", "Training patients", "134,529", "Age, sex, initial evidence", "Patient record with values", "30", "99.21", "6.73"],
    ["BASD [@ddxplus; @basd]^b)^", "Training patients", "134,529", "Age, sex, initial evidence", "Patient record with values", "30", "97.15", "17.99"],
    ["Diaformer [@diaformer]^c)^", "Training patients", "134,529", "Initial evidence", "Patient record with values", "30", "91.81", "18.45"],
    ["CASANDE [@casande]^c)^", "Training patients", "134,529", "Age, sex, initial evidence", "Patient record with values, sub-question answers filled in", "30", "98.8", "19.53"],
    ["BED [@bed]^c)^", "Public lists and training frequencies", "134,529", "Initial evidence", "BED simulator", "30", "99.47", "5.47"],
]
ours = []
for key, label in (("P1", "list-based method (P1)"), ("P2-value-agesex", "P2, answer values, age and sex")):
    st = main[key]["stop"]
    start = [m[3] for m in METHODS if m[0] == key][0]
    answers = "Patient record, yes or no" if key == "P1" else "Patient record with values"
    knowledge = [m[2] for m in METHODS if m[0] == key][0]
    ours.append([f"This study, {label}", knowledge, f"{main[key]['n']:,}", start, answers, "30", f(100 * st["gtpa1"]["estimate"]), f(st["il"]["mean"])])
st = bed["stop"]
ours.append(["This study, BED re-implementation", "Public lists and training frequencies", f"{bed['n']:,}", "Initial evidence", "BED simulator", "30", f(100 * st["gtpa1"]["estimate"]), f(st["il"]["mean"])])
OTHER = [
    ["*Other conditions*", "", "", "", "", "", "", ""],
    ["COTCAgent [@cotcagent]^d)^", "Public lists", "1,000", "Initial evidence", "Patient record, yes or no", "10", f"39.6{DASH}60.4", f"6.65{DASH}7.80"],
    ["CaRDI [@cardi]^e)^", "Training frequencies and an LLM prior", "490", "Age, sex, initial evidence", "Patient record with values", "10", f"96.12{DASH}96.69", f"4.54{DASH}4.64"],
    ["MEDDxAgent [@meddxagent]^f)^", "LLM with retrieval and example patients", "100", "Age, sex, chief complaint", "LLM patient simulator", "15", "86", "NR"],
    ["StreamBench [@streambench]^g)^", "LLM", "1,764", "Full patient record", "None", "0", f"70.64{DASH}92.01", "NA"],
]
write(
    "table3",
    "Table 3. Results reported on DDXPlus and their evaluation conditions",
    table(["Study", "Knowledge used", "Patients evaluated", "Input at start", "Answers", "Question limit", "Accuracy, %", "Evidence collected, mean^a)^"], WHOLE + ours + OTHER),
    [
        "AARLC, adaptive alignment of reinforcement learning and classification; BASD, baseline automatic symptom detection; BED, Bayesian experimental design; P1, list-based method; P2, frequency-based method; LLM, large language model; NR, not reported; NA, not applicable.",
        "^a)^Including the initial evidence.",
        "^b)^Model trained to predict the true disease; values from Table 3 of the DDXPlus paper [@ddxplus].",
        "^c)^Model trained to predict the true disease, and BED; values from Table 1 of the CASANDE paper [@casande].",
        "^d)^Two baselines re-implemented in that paper that score diseases from the public lists without learned weights (its Table 3 and Table A4); how the 1,000 patients were drawn is not stated; the reported 5.65–6.80 questions after the initial evidence are shown with the initial evidence added.",
        "^e)^Patients from the test set; range over three LLMs; the reported 3.54–3.64 questions after the initial evidence are shown with the initial evidence added.",
        "^f)^GPT-4o on the interactive task with up to 15 questions; the paper does not name the split.",
        "^g)^Patients from the test set, 36 per disease; GPT-4o, from zero-shot prompting to the best reported method, which uses as examples its own earlier answers on test patients that feedback marked as correct.",
    ],
)

# Table S1: candidates of P1 on validation
tried = [json.loads(line) for line in (DEV / "configs_tried.jsonl").read_text().splitlines()]
rows = []
for r in tried:
    if r["policy"] == "P1" and r["config"]["tau"] == 0.99:
        cfg = r["config"]
        rows.append([f(cfg["alpha"], 1), eps_label(cfg["eps"]), f(100 * r["stop"]["gtpa1"]["estimate"]), f(r["stop"]["il"]["mean"]), f(100 * r["curve"]["gtpa1"][-1]), "Candidate" if r["role"] == "selection" else "Outside the candidates"])
write(
    "tableS1",
    "Supplementary Table S1. Candidate settings of the list-based method on the validation set",
    table(["α", "ε", "Accuracy, %", "Evidence collected, mean", "Accuracy after 30 questions, %", "Role"], rows),
    [
        "α, probability of “yes” for an evidence on the list; ε, probability of “yes” for an evidence outside the list.",
        "The setting with the highest accuracy among the nine candidates was selected (α = 0.5, ε = 10^−4^). The rows outside the candidates were evaluated because the selected ε is at the end of the candidates; they were not used for the selection.",
    ],
)

# Table S2: additional metrics on test
rows = []
for key in MAIN + ["B-BED"]:
    s = run(key)["stop"]
    rows.append([NAME[key], f(100 * s["gtpa1"]["estimate"]), f(100 * s["gtpa1_official"]["estimate"]), f(100 * s["gtpa3"]["estimate"]), f(100 * s["gtpa5"]["estimate"]), f(100 * s["ddr"]["estimate"]), f(100 * s["ddp"]["estimate"]), f(100 * s["ddf1"]["estimate"]), f(100 * s["dshm"]["estimate"]), f(100 * s["per_official"]["estimate"])])
rows.append(["BED, reported in the CASANDE paper [@casande]", "NR", "99.47", "NR", "NR", "NR", "NR", "31.01", "10.78", "18.62"])
write(
    "tableS2",
    "Supplementary Table S2. Additional metrics on the test set (%)",
    table(["Method", "Accuracy", "Accuracy, official code", "GTPA@3", "GTPA@5", "DDR", "DDP", "DDF1", "DSHM", "PER"], rows),
    [
        f"{ABBR_METHODS} NR, not reported.",
        "Accuracy, official code: the accuracy as the official code computes it (Supplementary Methods 5); the accuracy reported for BED was computed by the BED code with the same rule. GTPA@k, the true disease is among the k most probable diseases after removing diseases with probability 0.01 or less, with ties shared as for accuracy. DDR and DDP, mean over patients of the recall and precision against the reference differential diagnosis; DDF1, mean over patients of their harmonic mean. DSHM, mean over patients of the harmonic mean of the rule-in and rule-out rates of severe diseases [@casande]. PER, share of the positive evidences of the patient that were collected, the initial evidence included.",
    ],
)

# Table S3: subsets of the test set
rows = []
for key in MAIN + ["B-BED"]:
    subs = run(key)["subsets"]
    d, s = subs["T-dedup"]["stop"], subs["T-streambench"]["stop"]
    rows.append([NAME[key], pct_ci(d["gtpa1"]), f(d["il"]["mean"]), pct_ci(s["gtpa1"]), f(s["il"]["mean"])])
nd, ns = main["P1"]["subsets"]["T-dedup"]["n"], main["P1"]["subsets"]["T-streambench"]["n"]
write(
    "tableS3",
    "Supplementary Table S3. Results on two subsets of the test set",
    table(["Method", f"No training patient with the same age, sex, initial evidence and recorded evidences with their values (n = {nd:,}): accuracy, % (95% CI)", "Evidence collected, mean", f"StreamBench patients (n = {ns:,}): accuracy, % (95% CI)", "Evidence collected, mean"], rows),
    [f"CI, confidence interval; {ABBR_METHODS}"],
)

# Table S4: paired comparisons on test
PAIRS = [
    ("P2-binary - P1", "P2, yes-or-no answers − P1", "Lists replaced by training frequencies"),
    ("P2-binary-agesex - P2-binary", "P2, yes-or-no answers, age and sex − P2, yes-or-no answers", "Age and sex added"),
    ("P2-value - P2-binary", "P2, answer values − P2, yes-or-no answers", "Answer values added"),
    ("P2-value-agesex - P2-value", "P2, answer values, age and sex − P2, answer values", "Age and sex added"),
    ("B-BED - P2-value", "BED re-implementation − P2, answer values", "BED and its simulator"),
]
ladder = dict(main["ladder"])
rows = []


def pair_rows(label, factor, points):
    out = []
    for i, (where, c) in enumerate(points):
        il = diff_ci(c["il_diff"], 1.0) if "il_diff" in c else DASH
        m = c["mcnemar"]
        out.append([label if i == 0 else "", factor if i == 0 else "", where, diff_ci(c["gtpa1_diff_interval"]), il, f"{m['only_this']:,} / {m['only_other']:,}"])
    return out


for key, label, factor in PAIRS:
    r = ladder[key]
    points = [("At the stop", r["stop"])] + [(f"After {b} questions", r[b]) for b in ("5", "10", "20", "30")]
    points += [(f"Stop threshold {t}", c) for t, c in r.get("tau_sensitivity", {}).items() if t != "0.99"]
    rows += pair_rows(label, factor, points)
for key in MAIN:
    rows += pair_rows(f"{NAME[key]}, sub-question answers filled in − {NAME[key]}", "Sub-question answers filled in", [("At the stop", hier[key]["against"][key]["stop"])])
write(
    "tableS4",
    "Supplementary Table S4. Paired comparisons on the test set",
    table(["Comparison", "Factor changed", "Point", "Accuracy difference, percentage points (95% CI)", "Difference in evidence collected (95% CI)", "Patients correct in only the first / only the second method"], rows),
    [
        f"CI, confidence interval; {ABBR_METHODS}",
        "Differences are the first method minus the second on the same patients; the stop uses the threshold 0.99 unless stated. The last column counts patients with ties broken in the listed order of the public data.",
    ],
)

# Table S5: stop threshold on test
rows = []
for key in MAIN:
    for t in TAUS:
        s = main[key]["tau_sensitivity"][t]
        rows.append([NAME[key] if t == TAUS[0] else "", t, pct_ci(s["gtpa1"]), mean_ci(s["il"]), f(100 * s["il"]["share_at_cap"])])
write(
    "tableS5",
    "Supplementary Table S5. Results on the test set by stop threshold",
    table(["Method", "Stop threshold", "Accuracy, % (95% CI)", "Evidence collected, mean (95% CI)", "Stopped at the question limit, %"], rows),
    ["CI, confidence interval; P1, list-based method; P2, frequency-based method.", "The main analysis uses the threshold 0.99."],
)

# Table S6: accuracy without questions and with the full record
ref = []
for split, b in (("Validation", dbase), ("Test", base)):
    ref.append([split, pct_ci(b["B0"]["most_frequent"]["stop"]["gtpa1"]), pct_ci(b["B0"]["initial_only"]["stop"]["gtpa1"]), pct_ci(b["B1"]["set_inclusion"]["stop"]["gtpa1"]), pct_ci(b["B1"]["kb_bayes"]["stop"]["gtpa1"]), pct_ci(b["B1"]["naive_bayes/P2-binary"]["stop"]["gtpa1"]), pct_ci(b["B1"]["naive_bayes/P2-value"]["stop"]["gtpa1"])])
write(
    "tableS6",
    "Supplementary Table S6. Accuracy, % (95% CI), of baselines and reference values that ask no question",
    table(["Set", "Most frequent disease", "Initial evidence only", "List check, all answers", "P1 probabilities, all answers", "P2 probabilities, yes-or-no answers, all answers", "P2 probabilities, answer values, all answers"], ref),
    [
        "CI, confidence interval; P1, list-based method.",
        "“All answers” gives the answers to all 223 evidences without questions. P2 probabilities are those of P2 without age and sex. P2, frequency-based method.",
    ],
)

# Table S7: accuracy by number of questions on test
POINTS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30)
series = [(NAME[k], run(k)["curve"]["gtpa1"]) for k in MAIN + ["B-BED"]]
seeds = base["B2"]["seeds"]
random_curve = [sum(seeds[s]["curve"]["gtpa1"][b] for s in seeds) / len(seeds) for b in range(31)]
series.append(("P1, random questions^a)^", random_curve))
series.append(("P1, frequency-ordered questions", base["B6"]["P1"]["curve"]["gtpa1"]))
rows = [[label] + [f(100 * curve[b]) for b in POINTS] for label, curve in series]
write(
    "tableS7",
    "Supplementary Table S7. Accuracy (%) on the test set by number of questions asked after the initial evidence",
    table(["Method"] + [str(b) for b in POINTS], rows),
    [
        f"{ABBR_METHODS}",
        "P1 and P2 ask exactly the given number of questions without the stop conditions. BED asks up to the given number and asks no further question once no question has a positive utility.",
        "^a)^Mean of five random orders.",
    ],
)
print("tables written")
