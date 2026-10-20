"""BM25 sensitivity on confirmation set E (research design §9.33 V1, POST HOC).

run: every BM25 configuration of the design E tune grid (mode x k x agg, retrieval_baseline.py tune --dev) on the
v2+v3 index, and the design D configuration on the v2 index, scored on DDXPlus-E and SymCat-E exactly as the
confirmation scored S-bm25 (retrieval_baseline.Retriever, graph_dx.evaluate). Only the lineage lookup differs:
retrieval_baseline.py reads it from 7687, which this analysis may not use, so it is read from the Disease IS_A
hierarchy of another instance (read-only); the run stops unless the frozen S-bm25 configuration reproduces the
stored confirmation result (every patient's lineage position and every strict and lineage summary metric).

stats: S-mix - BM25 per set (patient-paired CI and gold-disease cluster CI, paired_stats.compare) and over both sets
(paired_stats.macro_compare), Top-10 and Top-1. The §9.33 V1 rule reads Top-10 per set: when the CI lower bound of
S-mix - BM25 is above 0 on both sets for all twelve v2+v3 configurations, H1 is written "regardless of the BM25
configuration"; otherwise it stays "against the BM25 configuration selected on development". The rule is read on
the patient CI (the design E judgment CI) and on the cluster CI separately. The v2-index BM25 sizes the dilution of
BM25 by the larger corpus on sealed patients; it is not one of the twelve.

    CONFIRM_E_UNSEAL=1 python pilot/scripts/bm25_sensitivity_e.py run --index v3 --lineage-uri bolt://localhost:7689
    CONFIRM_E_UNSEAL=1 python pilot/scripts/bm25_sensitivity_e.py run --index v2 --lineage-uri bolt://localhost:7689
    CONFIRM_E_UNSEAL=1 python pilot/scripts/bm25_sensitivity_e.py stats
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import opendx_eval
import retrieval_baseline as rb
import v2_ids
from graph_dx import GraphDx, evaluate
from paired_stats import compare, macro_compare

CONF = Path("pilot/cache/v2/confirmE")
FROZEN_E = Path("pilot/cache/v2/devD/frozen_config_E.json")
FROZEN_D = Path("pilot/cache/v2/devD/frozen_config_D.json")
OUT = Path("pilot/cache/v3/framing/v1_bm25")
INDEX = {"v3": "pilot/cache/v3/retrieval", "v2": "pilot/cache/v2/retrieval"}
TUNE = {"v3": "pilot/cache/v3/retrieval/tune_bm25.json", "v2": "pilot/cache/v2/devD/tune_bm25.json"}
SETS = ["ConfirmE-DDXPlus", "ConfirmE-SymCat"]
INDEX_FILES = ["bm25/data.csc.index.npy", "bm25/indices.csc.index.npy", "bm25/indptr.csc.index.npy",
               "bm25/params.index.json", "bm25/vocab.index.json", "chunk_doc.npy", "docs.json"]


def cfg_name(index: str, c: dict) -> str:
    return f"{index}_{c['mode']}_k{c['k']}_{c['agg']}"


def sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


class SensRetriever(rb.Retriever):
    """Retriever.rank with per-concept disease scores memoized (same dicts, same summation order) and the
    lineage memoized per gold set from a read-only graph instance."""

    def __init__(self, c: dict, names: dict, lin: GraphDx, dmemo: dict, lmemo: dict):
        super().__init__(c["method"], c["k"], c["agg"], c["mode"], names, lin)
        self.dmemo, self.lmemo = dmemo, lmemo

    def _concept_scores(self, text: str) -> dict[str, float]:
        key = (text, self.k, self.agg)
        if key not in self.dmemo:
            self.dmemo[key] = self._disease_scores(self._search(text))
        return self.dmemo[key]

    def rank(self, ev: set[str], k: int = 50) -> list[str]:
        terms = sorted(self.names[c] for c in ev if c in self.names)
        if not terms:
            return []
        if self.mode == "joint":
            sc = self._disease_scores(self._search(" ".join(terms)))
        else:
            sc = defaultdict(float)
            for t in terms:
                for d, v in self._concept_scores(t).items():
                    sc[d] += v
        return sorted(sc, key=lambda d: (-sc[d], d))[:k]

    def lineage(self, gold):
        key = frozenset(gold)
        if key not in self.lmemo:
            self.lmemo[key] = self.lin.lineage(gold)
        return self.lmemo[key]


def configs(index: str) -> list[dict]:
    if index == "v3":
        grid = [r["cfg"] for r in json.loads(Path(TUNE["v3"]).read_text())["trail"]]
        if len(grid) != 12:
            raise SystemExit(f"{TUNE['v3']}: {len(grid)} configurations, expected 12")
        return grid
    return [json.loads(FROZEN_D.read_text())["bm25"]]


def check_index(index: str) -> dict[str, str]:
    root = Path(INDEX[index])
    got = {f: sha256(root / f) for f in INDEX_FILES}
    if index == "v3":
        want = json.loads(FROZEN_E.read_text())["S-bm25"]["index_sha256"]
        bad = [f for f in INDEX_FILES if got[f] != want[f]]
        if bad:
            raise SystemExit(f"v3 index differs from the design E freeze: {bad}")
    return got


def run(index: str, lineage_uri: str, auth: str) -> None:
    out = OUT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    shas = check_index(index)
    rb.OUT = INDEX[index]
    opendx_eval.RAW_GOLD = True
    sets = opendx_eval.load_confirm_e()
    if sorted(sets) != SETS:
        raise SystemExit(f"unexpected sets {sorted(sets)}")
    cuis = {c for pats in sets.values() for _, ev in pats for c in ev}
    names = rb.cui_names(cuis)
    print(f"{index}: concepts {len(cuis)}, named {len(names)}", flush=True)
    lin = GraphDx(lineage_uri, auth, 0.5, 2.0, 0.0, 0.0)
    lmemo: dict = {}
    frozen = json.loads(FROZEN_E.read_text())["S-bm25"]["selected"]
    grid = configs(index)
    if index == "v3":                      # the reproduction check runs first, then the configs sharing its hits
        grid = sorted(grid, key=lambda c: (c != frozen, (c["mode"], c["k"]) != (frozen["mode"], frozen["k"])))
    prev = None
    for c in grid:
        if prev is not None and (c["mode"], c["k"]) != prev:
            rb._BACKENDS["bm25"]["cache"].clear()     # joint hits are per patient; keep memory flat
        prev = (c["mode"], c["k"])
        name = cfg_name(index, c)
        rt = SensRetriever(c, names, lin, {}, lmemo)
        res = [evaluate(rt, b, sets[b], 1000) for b in SETS]
        doc = {"cfg": c, "index": INDEX[index], "index_sha256": shas, "lineage_uri": lineage_uri, "results": res}
        if index == "v3" and c == frozen:
            stored = {r["bench"]: r for r in json.loads((CONF / "S-bm25.json").read_text())["results"]}
            for r in res:                  # the stored run keeps lineage positions and every summary metric
                want = stored[r["bench"]]
                bad = [f for f in want if f != "sec" and r.get(f) != want[f]]
                if bad:
                    raise SystemExit(f"{r['bench']}: {bad} differ from {CONF / 'S-bm25.json'}")
            doc["reproduces_confirmation_S-bm25"] = True
            print("frozen S-bm25 reproduced for every patient", flush=True)
        (out / f"{name}.json").write_text(json.dumps(doc))
        print(f"saved {name}", flush=True)
    (out / f"{index}.done").write_text(json.dumps({"configs": [cfg_name(index, c) for c in grid]}))


def status(ci: list[float], cl: list[float]) -> str:
    def side(x: list[float]) -> str | None:
        return "S-mix" if x[0] > 0 else "BM25" if x[1] < 0 else None

    p, d = side(ci), side(cl)
    if p and d:
        return f"{d} higher: disease CI excludes 0"
    if p:
        return f"{p} higher: patient CI only"
    if d:
        return f"{d} higher: disease CI only"
    return "both CIs include 0"


def stats() -> None:
    for index in ("v3", "v2"):
        if not (OUT / "runs" / f"{index}.done").exists():
            raise SystemExit(f"{index} runs have not finished")
    opendx_eval.RAW_GOLD = True
    conf = opendx_eval.load_confirm_e()
    clusters = {b: [tuple(sorted(v2_ids.gold_keys(g))) for g, ev in pats if v2_ids.gold_keys(g) and ev]
                for b, pats in conf.items()}
    smix = json.loads((CONF / "S-mix.json").read_text())
    frozen = json.loads(FROZEN_E.read_text())["S-bm25"]["selected"]
    tune_v3 = {cfg_name("v3", r["cfg"]): r for r in json.loads(Path(TUNE["v3"]).read_text())["trail"]}
    systems = [cfg_name("v3", c) for c in configs("v3")] + [cfg_name("v2", c) for c in configs("v2")]
    runs = {s: json.loads((OUT / "runs" / f"{s}.json").read_text()) for s in systems}

    rows, macros = [], []
    for s in systems:
        for k in (10, 1):
            for r in compare(runs[s], smix, k, clusters=clusters, require=SETS):
                if "ci_cluster" not in r:
                    raise SystemExit(f"{s} {r['bench']}: cluster keys do not pair with {r['n']} patients")
                rows.append({"bm25": s, "cfg": runs[s]["cfg"], "k": k, "bench": r["bench"], "n": r["n"],
                             "bm25_pct": r[f"A@{k}"], "smix_pct": r[f"B@{k}"], "diff": r["diff"], "ci": r["ci"],
                             "ci_cluster": r["ci_cluster"], "diseases": r["clusters"], "only_bm25": r["only_A"],
                             "only_smix": r["only_B"], "p": r["p"], "p_holm": r["p_holm"],
                             "status": status(r["ci"], r["ci_cluster"])})
            m = macro_compare(runs[s], smix, k, SETS)
            macros.append({"bm25": s, "k": k, "sets": m["sets"], "bm25_pct": m["A"], "smix_pct": m["B"],
                           "diff": m["diff"], "ci_patient_stratified": m["ci"], "p_bootstrap": m["p"]})

    v3 = [s for s in systems if s.startswith("v3_")]
    top10 = [r for r in rows if r["k"] == 10]

    def rule(ci_key: str, names: list[str]) -> dict:
        fails = [f"{r['bm25']} {r['bench']} lower {r[ci_key][0]:+.2f}"
                 for r in top10 if r["bm25"] in names and r[ci_key][0] <= 0]
        return {"holds": not fails, "failures": fails,
                "min_lower": {b: min((r[ci_key][0], r["bm25"]) for r in top10 if r["bm25"] in names
                                     and r["bench"] == b) for b in SETS}}

    v1 = {"patient_ci": rule("ci", v3), "cluster_ci": rule("ci_cluster", v3),
          "with_v2_index_patient_ci": rule("ci", systems), "with_v2_index_cluster_ci": rule("ci_cluster", systems)}
    both = v1["patient_ci"]["holds"] and v1["cluster_ci"]["holds"]
    v1["verdict"] = ("regardless of the BM25 configuration (patient and cluster CI)" if both else
                     "regardless of the BM25 configuration on the patient CI only; cluster CI qualifies it"
                     if v1["patient_ci"]["holds"] else "against the BM25 configuration selected on development")

    def pct(s: str, b: str, k: int) -> float:
        return next(r[f"lineage@{k}"] for r in runs[s]["results"] if r["bench"] == b)

    v2d = cfg_name("v2", configs("v2")[0])
    same = cfg_name("v3", configs("v2")[0])
    sel = cfg_name("v3", frozen)
    dilution = []
    for b in SETS:
        for k in (10, 1):
            best = max(v3, key=lambda s: pct(s, b, k))
            dilution.append({"bench": b, "k": k, "v2_designD": pct(v2d, b, k), "v3_same_cfg": pct(same, b, k),
                             "v3_selected": pct(sel, b, k), "v3_best": pct(best, b, k), "v3_best_cfg": best,
                             "same_cfg_change": round(pct(same, b, k) - pct(v2d, b, k), 2),
                             "selected_minus_v2": round(pct(sel, b, k) - pct(v2d, b, k), 2),
                             "best_minus_v2": round(pct(best, b, k) - pct(v2d, b, k), 2)})
    # v3 per_concept/k1000/max minus v2 per_concept/k1000/max: the corpus effect at a fixed configuration
    corpus = []
    for k in (10, 1):
        for r in compare(runs[v2d], runs[same], k, clusters=clusters, require=SETS):
            corpus.append({"k": k, "bench": r["bench"], "v2": r[f"A@{k}"], "v3": r[f"B@{k}"], "diff": r["diff"],
                           "ci": r["ci"], "ci_cluster": r["ci_cluster"], "diseases": r["clusters"]})
        m = macro_compare(runs[v2d], runs[same], k, SETS)
        corpus.append({"k": k, "bench": "macro", "v2": m["A"], "v3": m["B"], "diff": m["diff"], "ci": m["ci"]})
    # E-T3 (S-mix - selected v2+v3 BM25) telescoped through the design D systems on the same sealed patients
    chain = {sel: runs[sel], same: runs[same], v2d: runs[v2d],
             "S-D": json.loads((CONF / "S-D.json").read_text()), "S-mix": smix}
    names_ = list(chain)
    decomp = []
    for a, b in itertools.pairwise(names_):
        for k in (10, 1):
            for r in compare(chain[a], chain[b], k, clusters=clusters, require=SETS):
                decomp.append({"step": f"{b} - {a}", "k": k, "bench": r["bench"], "A": r[f"A@{k}"], "B": r[f"B@{k}"],
                               "diff": r["diff"], "ci": r["ci"], "ci_cluster": r["ci_cluster"]})

    res = {"note": "POST HOC on confirmation set E. diff = S-mix - BM25, lineage Top-k in percentage points; ci = "
                   "patient-paired bootstrap 95% (2,000, seed 0); ci_cluster = gold-disease cluster bootstrap 95%; "
                   "p = exact McNemar, p_holm = Holm over the two sets of one configuration at one k; macro = "
                   "equal-weight over the two sets, set-stratified patient bootstrap. Dev macro values are the "
                   "design E selection trail (D3 rule picked v3_joint_k1000_rrf).",
           "s_mix": str(CONF / "S-mix.json"), "frozen_S_bm25": sel, "dev_macro_v3": {
               s: [tune_v3[s]["macro_top10"], tune_v3[s]["macro_top1"]] for s in v3},
           "e_values": {s: {b: {k: pct(s, b, k) for k in (10, 1)} for b in SETS} for s in systems},
           "rule_V1_top10": v1, "rows": rows, "macro": macros, "dilution": dilution,
           "corpus_effect_same_cfg": {"A": v2d, "B": same, "rows": corpus},
           "decomposition_E_T3": {"note": "S-mix - selected BM25 = sum of the steps (B - A) per set and k",
                                  "steps": decomp}}
    (OUT / "v1_result.json").write_text(json.dumps(res, indent=1))

    print(f"{'BM25':28s} {'k':>3s} {'set':17s} {'BM25':>6s} {'S-mix':>6s} {'diff':>6s} {'patient CI':>16s} "
          f"{'disease CI':>16s} {'dis':>4s}  status")
    for r in rows:
        print(f"{r['bm25']:28s} @{r['k']:<2d} {r['bench']:17s} {r['bm25_pct']:6.2f} {r['smix_pct']:6.2f} "
              f"{r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] "
              f"[{r['ci_cluster'][0]:+6.2f},{r['ci_cluster'][1]:+6.2f}] {r['diseases']:4d}  {r['status']}")
    print("\nmacro (both sets):")
    for m in macros:
        print(f"  {m['bm25']:28s} @{m['k']:<2d} {m['bm25_pct']:6.2f} {m['smix_pct']:6.2f} {m['diff']:+6.2f} "
              f"[{m['ci_patient_stratified'][0]:+6.2f},{m['ci_patient_stratified'][1]:+6.2f}]")
    print("\ndilution (E):")
    for d in dilution:
        print(f"  {d['bench']:17s} @{d['k']:<2d} v2 D {d['v2_designD']:5.2f}  v3 same {d['v3_same_cfg']:5.2f}  "
              f"v3 selected {d['v3_selected']:5.2f}  v3 best {d['v3_best']:5.2f} ({d['v3_best_cfg']})")
    print("\ncorpus effect at the design D configuration (v3 - v2):")
    for r in corpus:
        cl = f" disease [{r['ci_cluster'][0]:+6.2f},{r['ci_cluster'][1]:+6.2f}]" if "ci_cluster" in r else ""
        print(f"  @{r['k']:<2d} {r['bench']:17s} {r['v2']:6.2f} -> {r['v3']:6.2f} {r['diff']:+6.2f} "
              f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}]{cl}")
    print("\nE-T3 decomposition (steps sum to S-mix - selected BM25):")
    for r in decomp:
        print(f"  @{r['k']:<2d} {r['bench']:17s} {r['step']:52s} {r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}]"
              f" disease [{r['ci_cluster'][0]:+6.2f},{r['ci_cluster'][1]:+6.2f}]")
    print(f"\nV1 rule (Top-10, 12 v2+v3 configurations): patient CI {v1['patient_ci']['holds']}, "
          f"cluster CI {v1['cluster_ci']['holds']} -> {v1['verdict']}")
    for key in ("patient_ci", "cluster_ci", "with_v2_index_patient_ci", "with_v2_index_cluster_ci"):
        print(f"  {key}: holds {v1[key]['holds']}; min lower {v1[key]['min_lower']}; failures {v1[key]['failures']}")
    print(f"saved {OUT / 'v1_result.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "stats"])
    ap.add_argument("--index", choices=sorted(INDEX))
    ap.add_argument("--lineage-uri", default="bolt://localhost:7689")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    a = ap.parse_args()
    if a.step == "run":
        if not a.index:
            raise SystemExit("run needs --index")
        if a.lineage_uri.endswith(":7687") or a.lineage_uri.endswith(":7688"):
            raise SystemExit("7687 and 7688 are not used by this analysis")
        run(a.index, a.lineage_uri, a.auth)
    else:
        stats()


if __name__ == "__main__":
    main()
