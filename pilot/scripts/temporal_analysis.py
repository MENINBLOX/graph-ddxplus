"""Research design T analysis: decay curves by case age, leakage, within-disease contrast, and the
literature-vs-curated comparison at matched snapshots.

    python temporal_analysis.py --out pilot/cache/v2/temporal/analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from datetime import date
from collections import defaultdict

T = "pilot/cache/v2/temporal"
BINS = [(-10 ** 6, -1095, "<-3y"), (-1095, -365, "-3..-1y"), (-365, 0, "-1..0y"),
        (0, 365, "0..+1y"), (365, 730, "+1..+2y"), (730, 10 ** 6, ">+2y")]
# literature cut (year end) paired with the HPOA release closest to it; the HPOA side is always the later
# date, so the pairing favours the curated source
PAIRS = [(2021, "v2022-02-14"), (2022, "v2022-12-15"), (2023, "v2024-01-11"), (2024, "v2024-12-12"),
         (2025, "v2025-11-24")]


# Hugging Face release date: an upper bound on each model's training-data cutoff
RELEASE = {"qwen25_7b": "2024-09-16", "qwen25_14b": "2024-09-16", "qwen3_4b": "2025-08-05", "phi35": "2024-08-16",
           "openbiollm8b": "2024-04-20", "med42_8b": "2024-07-02", "medgemma4b": "2025-05-19",
           "gemma4_e4b": "2026-03-02", "gemma4_12b": "2026-05-23"}


def age_bin(d: int) -> str:
    return next(lab for lo, hi, lab in BINS if lo <= d < hi)


def hit(r, k):
    return r["rank"] is not None and r["rank"] <= k


def hpoa_citations(tag: str) -> dict[str, set[str]]:
    cite = defaultdict(set)
    for line in open(f"data/external_kg/hpoa_history/{tag}.hpoa"):
        f = line.rstrip("\n").split("\t")
        if len(f) > 10 and f[0].startswith("OMIM:") and f[10] == "P":
            for x in f[4].split(";"):
                if x.startswith("PMID:"):
                    cite[f[0]].add(x[5:])
    return cite


# curation status is undefined for snapshots older than the first archived HPOA release
HPOA_HISTORY_START = "2022-02-14"


def first_curated() -> dict[str, str]:
    """OMIM id -> date of the first HPOA release (of the 36 since 2022-02) that annotates it; absent = never."""
    first = {}
    for f in sorted(glob.glob("data/external_kg/hpoa_history/v*.hpoa")):
        d = f.rsplit("/v", 1)[1][:10]
        for line in open(f):
            x = line.split("\t", 1)[0]
            if x.startswith("OMIM:") and x not in first:
                first[x] = d
    return first


def novelty_table(rows, first, k=10) -> dict:
    """Case age (past/future) x disease status at the snapshot (curated / not yet curated): case-level
    and disease-macro Top-k. Disease-macro averages within each disease first, so a single
    many-patient paper cannot dominate a cell."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["snapdate"] < HPOA_HISTORY_START:
            continue
        f = min((first.get(o, "9999") for o in r["omim"]), default="9999")
        new = f > r["snapdate"]
        cells[("future" if r["age_days"] >= 0 else "past") + ("/new" if new else "/curated")][tuple(r["omim"])].append(hit(r, k))
    out = {}
    for c, by in sorted(cells.items()):
        allv = [x for v in by.values() for x in v]
        out[c] = {"cases": len(allv), "diseases": len(by), "case@k": round(100 * sum(allv) / len(allv), 1),
                  "disease_macro@k": round(100 * sum(sum(v) / len(v) for v in by.values()) / len(by), 1)}
    return out


def curve(rows) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[age_bin(r["age_days"])].append(r)
    out = {}
    for _, _, lab in BINS:
        v = g.get(lab, [])
        if v:
            out[lab] = {"n": len(v), "known": round(100 * sum(r.get("gold_known", 0) for r in v) / len(v), 1),
                        "@1": round(100 * sum(hit(r, 1) for r in v) / len(v), 1),
                        "@10": round(100 * sum(hit(r, 10) for r in v) / len(v), 1)}
    return out


def within_disease(rows, k=10) -> dict:
    """Per snapshot, diseases with cases on both sides of the snapshot: mean per-disease Top-k past vs future."""
    by = defaultdict(lambda: ([], []))
    for r in rows:
        by[(r["snap"], tuple(r["omim"]))][r["age_days"] >= 0].append(hit(r, k))
    pairs = [(sum(p) / len(p), sum(f) / len(f)) for p, f in by.values() if p and f]
    if not pairs:
        return {}
    d = [f - p for p, f in pairs]
    rng = random.Random(0)
    boot = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
    return {"disease_snapshot_pairs": len(pairs), "past": round(100 * sum(p for p, _ in pairs) / len(pairs), 1),
            "future": round(100 * sum(f for _, f in pairs) / len(pairs), 1),
            "diff": round(100 * sum(d) / len(d), 1), "ci": [round(100 * boot[50], 1), round(100 * boot[1949], 1)]}


def paired(a_rows, b_rows, k=10) -> dict:
    a = {r["file"]: hit(r, k) for r in a_rows}
    b = {r["file"]: hit(r, k) for r in b_rows}
    fs = sorted(set(a) & set(b))
    if not fs:
        return {}
    d = [b[f] - a[f] for f in fs]
    rng = random.Random(0)
    boot = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
    return {"n": len(fs), "A": round(100 * sum(a[f] for f in fs) / len(fs), 1),
            "B": round(100 * sum(b[f] for f in fs) / len(fs), 1), "diff_B_minus_A": round(100 * sum(d) / len(d), 1),
            "ci": [round(100 * boot[50], 1), round(100 * boot[1949], 1)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(f"{T}/lit_20*.json") + glob.glob(f"{T}/hpoa_v*.json")):
        if f.endswith("_summary.json"):
            continue
        d = json.load(open(f))
        for r in d["rows"]:
            r["snap"] = d["tag"]
        runs[d["tag"]] = d["rows"]
    res = {"curves": {}, "within_disease": {}, "leakage": {}, "lit_vs_hpoa": []}
    for fam in ("lit", "hpoa"):
        rows = [r for t, v in runs.items() if t.startswith(fam) for r in v]
        res["curves"][fam] = curve(rows)
        res["within_disease"][fam] = within_disease(rows)
    # leakage: HPOA already cites the case's own source paper for its gold disease
    leak_rows = []
    for t, v in runs.items():
        if t.startswith("hpoa_"):
            cite = hpoa_citations(t[5:])
            for r in v:
                leak_rows.append({**r, "cited": any(r["pmid"] in cite.get(o, ()) for o in r["omim"])})
    past = [r for r in leak_rows if r["age_days"] < 0]
    for lab, sel in (("past_all", past), ("past_cited", [r for r in past if r["cited"]]),
                     ("past_uncited", [r for r in past if not r["cited"]])):
        res["leakage"][lab] = {"n": len(sel), "@1": round(100 * sum(hit(r, 1) for r in sel) / max(1, len(sel)), 1),
                               "@10": round(100 * sum(hit(r, 10) for r in sel) / max(1, len(sel)), 1)}
    for y, tag in PAIRS:
        lit, hp = runs.get(f"lit_{y}"), runs.get(f"hpoa_{tag}")
        if not lit or not hp:
            continue
        fut = {r["file"] for r in lit if r["age_days"] >= 0}
        pas = {r["file"] for r in lit if r["age_days"] < 0}
        for side, fs in (("future", fut), ("past", pas)):
            for k in (1, 10):
                res["lit_vs_hpoa"].append({"lit_year": y, "hpoa": tag, "cases": side, "k": k,
                                           **paired([r for r in hp if r["file"] in fs],
                                                    [r for r in lit if r["file"] in fs], k)})
    res["llm"] = {}
    for f in sorted(glob.glob(f"{T}/llm_*.json")):
        if f.endswith("_raw.json"):
            continue
        d = json.load(open(f))
        tag = f.rsplit("/llm_", 1)[1][:-5]
        rel = date.fromisoformat(RELEASE[tag])
        rows = [{**r, "age_days": (date.fromisoformat(r["date"]) - rel).days} for r in d["rows"]]
        by_year = defaultdict(list)
        for r in rows:
            by_year[r["date"][:4]].append(hit(r, 10))
        res["llm"][tag] = {"release": RELEASE[tag], "map_rate": round(d["map_rate"], 3), "curve": curve(rows),
                           "by_year@10": {y: [len(v), round(100 * sum(v) / len(v), 1)] for y, v in sorted(by_year.items())}}
    first = first_curated()
    res["novelty"] = {}
    for fam in ("lit", "hpoa"):
        rows = []
        for t, v in runs.items():
            if t.startswith(fam):
                sd = f"{t[4:]}-12-31" if fam == "lit" else t[6:]
                rows += [{**r, "snapdate": sd} for r in v]
        res["novelty"][fam] = novelty_table(rows, first)
    for f in sorted(glob.glob(f"{T}/llm_*.json")):
        if f.endswith("_raw.json"):
            continue
        tag = f.rsplit("/llm_", 1)[1][:-5]
        rel = RELEASE[tag]
        rows = [{**r, "snapdate": rel, "age_days": (date.fromisoformat(r["date"]) - date.fromisoformat(rel)).days}
                for r in json.load(open(f))["rows"]]
        res["novelty"]["llm_" + tag] = novelty_table(rows, first)
        res["within_disease"]["llm_" + tag] = within_disease([{**r, "snap": tag} for r in rows])
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
