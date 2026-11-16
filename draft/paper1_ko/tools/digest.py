"""Every number the manuscript uses, read from the outputs of protocol v3.

    python3 digest.py <repo root> > numbers.md
"""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
DEV = ROOT / "paper1/outputs/v3/dev"
EVAL = ROOT / "paper1/outputs/v3/eval/T"
MAIN = ("P1", "P2-binary", "P2-binary-agesex", "P2-value", "P2-value-agesex")
TAUS = ("0.9", "0.95", "0.99", "0.999")
BUDGETS = ("5", "10", "15", "20", "30")


def load(p):
    return json.loads(p.read_text())


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}"


def ci(d, nd=2):
    return f"{pct(d['estimate'], nd)} [{pct(d['low'], nd)}, {pct(d['high'], nd)}]"


def stop_line(s, asks=True):
    out = f"GTPA@1 {ci(s['gtpa1'])} | official {pct(s['gtpa1_official']['estimate'])} | fixed {pct(s['gtpa1_fixed']['estimate'])}"
    out += f" | @3 {pct(s['gtpa3']['estimate'])} @5 {pct(s['gtpa5']['estimate'])} GTPA {pct(s['gtpa']['estimate'])}"
    if asks and "il" in s:
        il = s["il"]
        out += (
            f" | IL {il['mean']:.2f} [{il['low']:.2f}, {il['high']:.2f}] median {il['median']:.0f} p95 {il['p95']:.0f}"
            f" cap {100 * il['share_at_cap']:.2f}% questions {il['questions_mean']:.2f}"
        )
    return out


def aux_line(s):
    keys = ("ddr", "ddp", "ddf1", "dshm", "per_official", "per")
    return " ".join(f"{k} {pct(s[k]['estimate'])}" for k in keys if k in s) + f" | macro {pct(s['macro']['gtpa1'])}"


def pair_line(c):
    d = c["gtpa1_diff_interval"]
    out = f"dGTPA@1 {100 * d['estimate']:+.2f} [{100 * d['low']:+.2f}, {100 * d['high']:+.2f}] %p"
    out += f" (fixed {100 * c['gtpa1_fixed_diff']:+.2f}, official {100 * c['gtpa1_official_diff']:+.2f})"
    if "il_diff" in c:
        i = c["il_diff"]
        out += f" | dIL {i['estimate']:+.2f} [{i['low']:+.2f}, {i['high']:+.2f}]"
    m = c["mcnemar"]
    return out + f" | discordant {m['only_this']}:{m['only_other']} p={m['p_value']:.2g}"


def reach(curve, level):
    for b, v in zip(curve["budgets"], curve["gtpa1"]):
        if v >= level:
            return b
    return None


def curve_line(r):
    c = r["curve"]
    pts = " ".join(f"{b}:{pct(c['gtpa1'][b])}" for b in (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30))
    return f"{pts} | reach 90%: {reach(c, 0.90)}, 95%: {reach(c, 0.95)}, 98%: {reach(c, 0.98)}, 99%: {reach(c, 0.99)}"


def worst(r, k=3):
    rows = sorted(r["by_disease"].items(), key=lambda kv: kv[1]["gtpa1"])[:k]
    return "; ".join(f"{name} {pct(v['gtpa1'])} (n={v['n']})" for name, v in rows)


def section(title):
    print(f"\n## {title}\n")


def main_block(tag, main, base, hier, props):
    section(f"{tag}: main policies")
    for k in MAIN:
        r = main[k]
        print(f"- **{k}** {r['config']} n={r['n']} failures={r['failures']}")
        print(f"  - stop: {stop_line(r['stop'])}")
        print(f"  - aux: {aux_line(r['stop'])}")
        print("  - tau: " + "; ".join(f"{t}: {pct(r['tau_sensitivity'][t]['gtpa1']['estimate'])}/{r['tau_sensitivity'][t]['il']['mean']:.2f}" for t in TAUS))
        print(f"  - curve: {curve_line(r)}")
        print(f"  - worst diseases: {worst(r)}")
        for name, sub in r.get("subsets", {}).items():
            print(f"  - subset {name} n={sub['n']}: {stop_line(sub['stop'])}")
    section(f"{tag}: ladder")
    ladder = dict(main["ladder"])
    ladder.update(base["B-BED"]["ladder"])
    for name, r in ladder.items():
        if isinstance(r, str):
            continue
        print(f"- **{name}** ({r['factor']})")
        print(f"  - stop: {pair_line(r['stop'])}")
        for b in BUDGETS:
            print(f"  - budget {b}: {pair_line(r[b])}")
        for t, c in r.get("tau_sensitivity", {}).items():
            print(f"  - tau {t}: {pair_line(c)}")
    section(f"{tag}: P2 vs P1")
    for k in MAIN[1:]:
        print(f"- {k} - P1 stop: {pair_line(main[k]['against']['P1']['stop'])}")
    section(f"{tag}: B-BED")
    bed = base["B-BED"]
    for src in ("stats_file", "counted"):
        r = bed[src]
        print(f"- **{src}** n={r['n']} failures={r['failures']}: {stop_line(r['stop'])}")
        print(f"  - aux: {aux_line(r['stop'])}")
        print(f"  - curve: {curve_line(r)}")
        for other, c in r["against"].items():
            print(f"  - vs {other} stop: {pair_line(c['stop'])}")
        for name, sub in r.get("subsets", {}).items():
            print(f"  - subset {name} n={sub['n']}: {stop_line(sub['stop'])}")
    print(f"- answer_differences: {json.dumps(bed['answer_differences'])}")
    print(f"- reported: {json.dumps(bed['reported'])}")
    section(f"{tag}: baselines")
    for k, r in base["B0"].items():
        print(f"- B0 {k}: {stop_line(r['stop'], asks=False)}")
    for k, r in base["B1"].items():
        print(f"- B1 {k}: {stop_line(r['stop'], asks=False)}")
    o = base["B2"]["over_seeds"]
    print(f"- B2 stop: gtpa1 mean {pct(o['stop']['gtpa1']['mean'])} sd {pct(o['stop']['gtpa1']['sd'], 3)} min {pct(o['stop']['gtpa1']['min'])} max {pct(o['stop']['gtpa1']['max'])}; IL mean {o['stop']['il_mean']['mean']:.2f}")
    print("  - budgets: " + " ".join(f"{b}:{pct(o[b]['gtpa1']['mean'])}" for b in BUDGETS))
    seed0 = base["B2"]["seeds"]["0"]
    print(f"  - seed 0 curve: {curve_line(seed0)}")
    for k, r in base["B6"].items():
        print(f"- B6 {k}: {stop_line(r['stop'])}")
        print(f"  - curve: {curve_line(r)}")
        print(f"  - vs P1 stop: {pair_line(r['against']['P1']['stop'])}")
    section(f"{tag}: hierarchy (implicit answers)")
    for k in MAIN:
        r = hier[k]
        print(f"- {k}: {stop_line(r['stop'])}")
        print(f"  - vs own main: {pair_line(r['against'][k]['stop'])}")
    section(f"{tag}: properties")
    print(json.dumps({k: v for k, v in props.items() if k not in ("provenance", "definitions")}))


dev_main, dev_base = load(DEV / "main.json"), load(DEV / "baselines.json")
print("# Numbers of protocol v3")
sel = load(DEV / "selected.json")
section("validation: selection")
print(f"- P1 selected {sel['P1']['config']} gtpa1 {pct(sel['P1']['gtpa1'], 3)} IL {sel['P1']['il_mean']:.3f}; boundary {json.dumps(sel['P1']['boundary'])}")
rows = [json.loads(line) for line in (DEV / "configs_tried.jsonl").read_text().splitlines()]
for r in rows:
    if r["policy"] == "P1" and r["config"]["tau"] == 0.99:
        last = r["curve"]["gtpa1"][-1]
        print(f"  - P1 {r['role']} alpha {r['config']['alpha']} eps {r['config']['eps']}: {pct(r['stop']['gtpa1']['estimate'], 3)} IL {r['stop']['il']['mean']:.2f} | 30 questions {pct(last, 3)}")
for r in rows:
    if r["policy"] != "P1":
        print(f"  - {r['policy']} {r['role']} lam {r['config']['lam']} tau {r['config']['tau']}: {pct(r['stop']['gtpa1']['estimate'], 3)} IL {r['stop']['il']['mean']:.3f}")
main_block("validation", dev_main, dev_base, load(DEV / "hierarchy.json"), load(DEV / "properties.json"))

abl = load(DEV / "ablations.json")
section("validation: component analysis (against P1)")


def abl_line(name, r):
    c = r["against"]["P1"]
    print(f"- {name}: {stop_line(r['stop'])}")
    print(f"  - vs P1 stop: {pair_line(c['stop'])}")
    print("  - budgets: " + " ".join(f"{b}:{pct(r['budgets'][b]['gtpa1']['estimate'])}({100 * c[b]['gtpa1_diff']:+.2f})" for b in BUDGETS))


abl_line("A1 positive only", abl["A1"]["positive_only"])
abl_line("A1 negative only", abl["A1"]["negative_only"])
abl_line("A2 train prior", abl["A2"]["train_prior"])
for eps, r in abl["A3"].items():
    abl_line(f"A3 eps {eps}", r)
for rule in ("parent_first", "no_children"):
    abl_line(f"A4 {rule}", abl["A4"][rule])
abl_line("A5 coverage", abl["A5"]["coverage"])

ans, unq = load(DEV / "answer_rule.json"), load(DEV / "uniqueness.json")
section("validation: answer rule (보완 4)")
for key in ("P2-value", "P2-recorded", "B-BED"):
    print(f"- **{key}** n={ans[key]['n']}: {stop_line(ans[key]['stop'])}")
    if "tau_sensitivity" in ans[key]:
        print("  - tau: " + "; ".join(f"{t}: {pct(ans[key]['tau_sensitivity'][t]['gtpa1']['estimate'])}/{ans[key]['tau_sensitivity'][t]['il']['mean']:.2f}" for t in TAUS))
for name, r in ans["comparisons"].items():
    print(f"- **{name}** ({r['factor']})")
    print(f"  - stop: {pair_line(r['stop'])}")
    for b in BUDGETS:
        print(f"  - budget {b}: {pair_line(r[b])}")
    for t, c in r.get("tau_sensitivity", {}).items():
        print(f"  - tau {t}: {pair_line(c)}")
section("validation: uniqueness strata (보완 4)")
print(json.dumps(unq["strata"]))
for key, v in unq["policies"].items():
    u, o, q = v["strata"]["unique"], v["strata"]["not_unique"], v["question_limit"]
    print(f"- **{key}** unique {ci(u['gtpa1'])} IL {u['il_mean']:.2f} | not unique {ci(o['gtpa1'])} IL {o['il_mean']:.2f} | lost {v['lost_score']['total']:.1f}, share not unique {pct(v['lost_score']['not_unique_share'])} | limit reached n={q['reached']['n']} {pct(q['reached']['gtpa1']['estimate'])}, earlier {pct(q['stopped_earlier']['gtpa1']['estimate'])}")

if (EVAL / "main.json").exists():
    t_main, t_base = load(EVAL / "main.json"), load(EVAL / "baselines.json")
    main_block("test", t_main, t_base, load(EVAL / "hierarchy.json"), load(EVAL / "properties.json"))
    pv = t_main["provenance"]
    section("test: provenance")
    print({k: pv[k] for k in ("commit", "commit_at_end", "protocol_sha256", "protocol_freeze_commit", "n_patients", "elapsed_s", "subsets")})
