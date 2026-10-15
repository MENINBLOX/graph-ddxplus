"""Compare the literature-graph variants of run_graph_variants.sh.

Development sets: per variant, offtopic_w selected by the design D3 rule (macro lineage Top-10 over the
nine sets, ties Top-1), then per-benchmark paired differences against the frozen graph (base_nm at
offtopic_w 0) with Holm correction and a sign test; the CI is given over patients and over gold
diseases (several benchmarks repeat a disease across many patients); the same comparison between variants
that differ in one change (PAIRS). Dated phenopacket cases: the
2022-cut variants on cases published from 2023 on, split by whether HPOA had annotated the gold
disease by the end of 2022, against HPOA v2022-12-15 and the 2022-cut frozen graph. Cases whose
source paper is in the v2 corpus are dropped from every temporal comparison (v3 excluded all case
papers at collection).

    python variants_analysis.py --out pilot/cache/v2/devD/variants/analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import opendx_eval
import v2_ids
from paired_stats import compare, sign_test
from temporal_analysis import first_curated, hit, paired
from tune_dev import dev_sets

O = "pilot/cache/v2/devD/variants"
HPOA_2022 = "pilot/cache/v2/temporal/hpoa_v2022-12-15.json"
CORPUS_PMIDS = f"{O}/v2_corpus_pmids.json"
REF = ("base_nm", "0.0")
GRID = ("0.0", "0.25", "0.5", "1.0")    # an exact tie goes to the lower offtopic_w
# (B, A): one change at a time, each at its own selected offtopic_w
PAIRS = [(f"{g}_nq", g) for g in ("base_split_tg", "v2dict", "v2union", "v2unionu")] + \
        [(f"{g}_ns", g) for g in ("base_nm", "base_split_tg", "v2dict", "v2umls", "v2union", "v2unionu", "v2union_nq")] + \
        [("v2dict", "base_split_tg"), ("v2umls", "base_split_tg"), ("v2umls", "v2dict"), ("v2union", "base_split_tg"),
         ("v2union", "v2dict"), ("v2unionu", "v2umls"), ("v2dict_ns", "base_split_tg_ns"), ("v2umls_ns", "base_split_tg_ns"), ("v2umls_ns", "v2dict_ns"),
         ("v2union_ns", "base_split_tg_ns"), ("v2union_ns", "v2dict_ns"), ("v2unionu_ns", "v2umls_ns"),
         ("v2union_nq_ns", "v2union_ns")]


def corpus_pmids() -> set[str]:
    if not os.path.exists(CORPUS_PMIDS):
        out = set()
        for f in glob.glob("pilot/cache/v2/sources/*.json"):
            for d in json.load(open(f))["docs"]:
                if d.get("pmid"):
                    out.add(str(d["pmid"]))
                if d["doc_id"].startswith("PMID"):
                    out.add(d["doc_id"][4:])
        json.dump(sorted(out), open(CORPUS_PMIDS, "w"))
    return set(json.load(open(CORPUS_PMIDS)))


def macro(res: list[dict]) -> tuple[float, float]:
    return (round(sum(r["lineage@10"] for r in res) / len(res), 2), round(sum(r["lineage@1"] for r in res) / len(res), 2))


def gold_clusters() -> dict[str, list]:
    opendx_eval.RAW_GOLD = True        # the gold sets graph_dx.py evaluated
    return {b: [tuple(sorted(v2_ids.gold_keys(g))) for g, ev in pats if v2_ids.gold_keys(g) and ev]
            for b, pats in dev_sets().items()}


def versus(a: dict, b: dict, clusters: dict) -> dict:
    out = {}
    for k in (1, 10):
        rows = compare(a, b, k, clusters=clusters)
        pos, neg = sum(r["diff"] > 0 for r in rows), sum(r["diff"] < 0 for r in rows)
        out[f"@{k}"] = {"rows": rows, "B_better": pos, "A_better": neg, "sign_p": sign_test(pos, neg)}
    return out


def dev_part() -> tuple[dict, dict]:
    clusters = gold_clusters()
    runs = defaultdict(dict)
    for f in glob.glob(f"{O}/dev_*_ot*.json"):
        g, ot = os.path.basename(f)[4:-5].rsplit("_ot", 1)
        runs[g][ot] = json.load(open(f))
    out = {}
    ref = runs.get(REF[0], {}).get(REF[1])
    for g, by_ot in sorted(runs.items()):
        missing = [ot for ot in GRID if ot not in by_ot or len(by_ot[ot]["results"]) != 9]
        if missing:
            out.setdefault("_incomplete", {})[g] = missing
            continue
        grid = {ot: macro(by_ot[ot]["results"]) for ot in GRID}
        sel = max(GRID, key=lambda ot: grid[ot])
        d = by_ot[sel]
        row = {"grid": grid, "selected_offtopic_w": sel, "macro@10": grid[sel][0], "macro@1": grid[sel][1],
               "per_bench": {r["bench"]: [r["lineage@1"], r["lineage@10"]] for r in d["results"]}}
        if ref is not None and (g, sel) != REF:
            for k, v in versus(ref, d, clusters).items():
                row[f"vs_frozen{k}"] = v
        out[g] = row
    pairs = {f"{b} vs {a}": versus(runs[a][out[a]["selected_offtopic_w"]], runs[b][out[b]["selected_offtopic_w"]], clusters)
             for b, a in PAIRS if a in out and b in out}
    return out, pairs


def temporal_part() -> dict:
    leak = corpus_pmids()
    first = first_curated()
    hp = {r["file"]: r for r in json.load(open(HPOA_2022))["rows"]}
    runs = {}
    for f in glob.glob(f"{O}/temporal_*_ot*.json"):
        runs[os.path.basename(f)[9:-5]] = [r for r in json.load(open(f))["rows"] if r["pmid"] not in leak]
    out = {"dropped_leaked_cases": sum(1 for r in json.load(open(HPOA_2022))["rows"] if r["pmid"] in leak)}

    def group(r):
        new = min((first.get(o, "9999") for o in r["omim"]), default="9999") > "2022-12-31"
        return ("future" if r["date"] >= "2023-01-01" else "past") + ("/new" if new else "/curated")

    for tag, rows in sorted(runs.items()):
        if "_y2022" not in tag:
            continue
        ref = runs.get("base_nm_y2022_ot" + tag.rsplit("_ot", 1)[1])
        cells = {}
        for grp in ("future/new", "future/curated", "past/new", "past/curated"):
            sel = [r for r in rows if group(r) == grp]
            if not sel:
                continue
            fs = {r["file"] for r in sel}
            cell = {"n": len(sel), "diseases": len({tuple(r["omim"]) for r in sel}),
                    "@1": round(100 * sum(hit(r, 1) for r in sel) / len(sel), 1),
                    "@10": round(100 * sum(hit(r, 10) for r in sel) / len(sel), 1)}
            for k in (1, 10):
                cell[f"vs_hpoa2022@{k}"] = paired([hp[f] for f in fs if f in hp], sel, k)
                if ref is not None:
                    cell[f"vs_base2022@{k}"] = paired([r for r in ref if r["file"] in fs], sel, k)
            cells[grp] = cell
        out[tag] = cells
    for tag, rows in sorted(runs.items()):
        if "_y2022" in tag:
            continue
        ref = runs.get("base_nm_ot" + tag.rsplit("_ot", 1)[1])
        out[tag] = {"n": len(rows), "@1": round(100 * sum(hit(r, 1) for r in rows) / len(rows), 1),
                    "@10": round(100 * sum(hit(r, 10) for r in rows) / len(rows), 1)}
        if ref is not None:
            out[tag]["vs_base@10"] = paired(ref, rows, 10)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, pairs = dev_part()
    res = {"dev": dev, "pairs": pairs, "temporal": temporal_part()}
    json.dump(res, open(a.out, "w"), indent=1)
    print("incomplete (not selected):", res["dev"].pop("_incomplete", {}))
    for g, r in res["dev"].items():
        print(f"dev {g:16s} ot={r['selected_offtopic_w']} macro@10 {r['macro@10']} @1 {r['macro@1']} grid {r['grid']}")
        print("    ", {b: v for b, v in r["per_bench"].items()})
        if "vs_frozen@10" in r:
            v = r["vs_frozen@10"]
            print("     vs frozen @10:", [(x["bench"], x["diff"], x["ci"], x.get("ci_cluster"), round(x["p_holm"], 3))
                                       for x in v["rows"]],
                  f"better {v['B_better']} worse {v['A_better']} sign p {v['sign_p']:.3g}")
    for name, v in res["pairs"].items():
        for k, x in v.items():
            print(f"pair {name:34s} {k:3s} better {x['B_better']} worse {x['A_better']} sign p {x['sign_p']:.3g}",
                  [(r["bench"], r["diff"], r["ci"], r.get("ci_cluster"), round(r["p_holm"], 3)) for r in x["rows"]])
    for t, r in res["temporal"].items():
        if isinstance(r, dict) and "future/new" in r:
            for grp, c in r.items():
                print(f"temporal {t:28s} {grp:15s} n={c['n']:5d} ({c['diseases']}) @1 {c['@1']} @10 {c['@10']} "
                      f"vsHPOA {c['vs_hpoa2022@10'].get('diff_B_minus_A')} {c['vs_hpoa2022@10'].get('ci')} "
                      f"vsBase {c.get('vs_base2022@10', {}).get('diff_B_minus_A')} {c.get('vs_base2022@10', {}).get('ci')}")
        else:
            print(f"temporal {t} {r}")


if __name__ == "__main__":
    main()
