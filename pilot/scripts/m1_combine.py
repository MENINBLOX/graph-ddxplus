"""HPOA + S-mix combinations on the development sets (research design §9.33 M1; development sets only, post hoc).

S-mix is the frozen E3 literature graph (v3llmd_ns at frozen_config_E.json scoring with its offtopic_w), read from
the instance --smix-uri without writing. HPOA is the design-D curated comparator (build_reference_graph.py --source
hpoa, ranked with frozen_config_D.json comparators.hpoa.onehop). HPOA-LOPO is the P0-4 graph without every HPOA row
that cites a paper of an evaluated Phenopacket-store case (hpoa_lopo.py variant lopo), ranked the same way.

Two combinations:
  RRF    reciprocal-rank fusion of the two systems' own rankings, retrieval_baseline.Fusion with HPOA in the
         retriever's place: score(d) = (1 - w) / (60 + rank_S-mix(d)) + w / (60 + rank_HPOA(d)); a disease missing
         from one list gets nothing from it; ties by id. Each list is GraphDx.rank(ev, 200, window=40) on the
         system's own graph and configuration, i.e. the finding scores of the top 40 diseases plus the
         unified-evidence diseases that a Top-10 evaluation reads, so `lists` keeps a list file only when its first
         ten entries reproduce the stored development ranks of every patient. w in {.25, .5, .75} is chosen with HPOA
         on the nine development sets by the D3 rule (macro lineage Top-10, ties Top-1, the smaller w on a full tie)
         and kept for HPOA-LOPO.
  union  one graph with the edges of both, ranked with the frozen S-mix configuration. An HPOA annotation counts as
         one supporting source whose topic is the disease: an edge in both keeps S-mix's counts plus one (n_docs + 1,
         n_topic_docs + 1, so it is no longer down-weighted as off-topic); an HPOA-only edge has n_docs = n_topic_docs
         = 1 as in the HPOA comparator graph. The rest follows build_graph_v2.py on the union profile: n_findings,
         idf = log(diseases with findings / document frequency), Concept IS_A from phenotype_parents.json,
         IS_DISEASE, the universe Disease IS_A. src_chars stays S-mix's collected text size, so a disease known only
         to HPOA keeps the coverage smoothing of a disease without literature. `union-build` stops unless the S-mix
         export has the frozen fingerprint, the merge with no HPOA edge recomputes that fingerprint, and the graph
         read back from --uri has the fingerprint of the merge.

`stats` compares each combination with each component in three strata: common (DDXPlus-test, SymCat), unpublished
hospital cases (PUMCH_L, PUMCH-ADM) and publication-based rare sets (RAMEDIS, LIRICAL, HMS, MME, Phenopacket-store;
again with HPOA-LOPO in place of HPOA). Per set: paired_stats.compare (patient-paired CI, gold-disease cluster CI,
exact McNemar, Holm across the stratum's sets) and the disease-equal difference. Per stratum: macro_compare
(set-stratified patient bootstrap) and macro_cluster (the same macro with whole gold diseases resampled within each
set, and the disease-equal macro). §9.33 M1 rule, judged at Top-10: "complementary" only if, in the hospital
stratum, combination - HPOA and combination - S-mix both have a macro CI lower bound > 0; it is required here of the
patient-stratified CI, the cluster CI and the disease-equal CI alike.

    python pilot/scripts/m1_combine.py lists --system S-mix --uri bolt://localhost:7689
    python pilot/scripts/m1_combine.py union-build --hpoa data/external_kg/phenotype.hpoa --summary X.json
    python pilot/scripts/m1_combine.py fuse
    python pilot/scripts/m1_combine.py stats
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

import build_reference_graph as BRG
import curated_agreement as CA
import hpoa_lopo
import opendx_eval as O
import v2_ids
from build_graph_v2 import fingerprint
from graph_dx import GraphDx
from m2_curated_e import disease_equal, npz_edges
from neo4j import GraphDatabase
from paired_stats import compare, macro_compare
from tune_dev import dev_sets, macro
from variants_analysis import gold_clusters

OUT = Path("pilot/cache/v3/framing/m1")
DEVD = Path("pilot/cache/v2/devD")
LOPO_DIR = Path("pilot/cache/v3/p0/hpoa_lopo")
STORED = {"S-mix": DEVD / "variants" / "dev_v3llmd_ns_ot0.5.json", "HPOA": DEVD / "ref_hpoa" / "dev.json",
          "HPOA-LOPO": LOPO_DIR / "dev_lopo.json"}
DEPTH, WINDOW, TOP = 200, 40, max(O.KS)
WEIGHTS = (0.25, 0.5, 0.75)
STRATA = {"common": ["DDXPlus-test", "SymCat"], "hospital": ["PUMCH-ADM", "PUMCH_L"],
          "published": ["Phenopacket-store", "RareBench-HMS", "RareBench-LIRICAL", "RareBench-MME",
                        "RareBench-RAMEDIS"]}
ALL_SETS = sorted(s for v in STRATA.values() for s in v)
BATCH = 10000
MERGE_RULE = ("edge in both: S-mix n_docs + 1, n_topic_docs + 1 (one on-topic curated source); HPOA-only edge: "
              "n_docs = n_topic_docs = 1; S-mix-only edge unchanged; src_chars from S-mix")


def config(system: str) -> dict:
    if system == "S-mix":
        f = json.loads((DEVD / "frozen_config_E.json").read_text())
        return {**f["scoring"], "offtopic_w": f["graph_systems"]["S-mix"]["offtopic_w"]}
    return json.loads((DEVD / "frozen_config_D.json").read_text())["comparators"]["hpoa"]["onehop"]


def position(ranked: list[str], ok: set[str]) -> int | None:
    return next((i + 1 for i, d in enumerate(ranked[:TOP]) if d in ok), None)


def result(bench: str, n_pats: int, ranked: list[list[str]], gold: list[set[str]], lin: list[set[str]]) -> dict:
    """One set in graph_dx.evaluate's output format."""
    rl = [position(r, x) for r, x in zip(ranked, lin)]
    rs = [position(r, g) for r, g in zip(ranked, gold)]
    m = len(rl)
    res: dict = {"bench": bench, "patients": n_pats, "evaluated": m}
    for rule, rk in (("strict", rs), ("lineage", rl)):
        for k in O.KS:
            res[f"{rule}@{k}"] = round(100 * sum(p is not None and p <= k for p in rk) / max(1, m), 2)
    res["ranks_lineage"], res["ranks_strict"] = rl, rs
    return res


def reproduces(res: dict, stored: dict) -> str | None:
    """None when every patient's lineage position and the strict/lineage Top-k equal the stored run."""
    if len(res["ranks_lineage"]) != len(stored["ranks_lineage"]):
        return f"{len(res['ranks_lineage'])} vs {len(stored['ranks_lineage'])} patients"
    n = sum(x != y for x, y in zip(res["ranks_lineage"], stored["ranks_lineage"]))
    if "ranks_strict" in stored:
        n += sum(x != y for x, y in zip(res["ranks_strict"], stored["ranks_strict"]))
    agg = [key for key in (f"{r}@{k}" for r in ("strict", "lineage") for k in O.KS) if res[key] != stored[key]]
    return f"{n} patient positions differ, aggregates {agg}" if n or agg else None


def lists(system: str, uri: str, auth: str) -> None:
    """Every development patient's ranked list on the graph at uri, saved only if it reproduces the stored ranks."""
    O.RAW_GOLD = True
    dx = GraphDx(uri, auth, **config(system))
    stored = {r["bench"]: r for r in json.loads(STORED[system].read_text())["results"]}
    out: dict = {"system": system, "uri": uri, "config": config(system), "depth": DEPTH, "window": WINDOW,
                 "graph": {"diseases_with_findings": dx.n, "lbar": dx.lbar, "smed": dx.smed},
                 "patients": {}, "sets": {}, "gold": {}, "lineage": {}, "check": {}}
    bad = []
    for bench, pats in dev_sets().items():
        t0 = time.time()
        ranked, gold, lin, keys = [], [], [], []
        for gold_raw, ev in pats:
            g = v2_ids.gold_keys(gold_raw)
            if not g or not ev:
                continue
            key = "|".join(sorted(g))
            if key not in out["lineage"]:
                out["lineage"][key] = sorted(dx.lineage(g))
            ranked.append(dx.rank(ev, DEPTH, window=WINDOW))
            gold.append(g)
            lin.append(set(out["lineage"][key]))
            keys.append(key)
        res = result(bench, len(pats), ranked, gold, lin)
        diff = reproduces(res, stored[bench])
        out["sets"][bench], out["gold"][bench], out["patients"][bench] = ranked, keys, len(pats)
        lens = sorted(len(r) for r in ranked)
        out["check"][bench] = {"patients": len(ranked), "stored_ranks": diff or "reproduced for every patient",
                               "list_length_min_median_max": [lens[0], lens[len(lens) // 2], lens[-1]]}
        print(f"{system} {bench:19s} n {len(ranked):6d} @10 {res['lineage@10']:6.2f} @1 {res['lineage@1']:6.2f} "
              f"list {out['check'][bench]['list_length_min_median_max']} {round(time.time() - t0)}s "
              f"stored: {diff or 'reproduced'}", flush=True)
        if diff:
            bad.append(f"{bench}: {diff}")
    dx.drv.close()
    if bad:
        sys.exit(f"{system}: stored development ranks not reproduced: {bad}")
    path = OUT / f"lists_{system}.json.gz"
    with gzip.open(f"{path}.part", "wt") as fh:
        json.dump(out, fh)
    os.replace(f"{path}.part", path)
    print(f"saved {path}", flush=True)


def lopo_file(out: Path) -> None:
    """The P0-4 lopo phenotype.hpoa, byte-identical to the file whose sha256 deletions.json records."""
    case_pm = [pm for _, pm in hpoa_lopo.evaluated_cases()]
    lines, rows = hpoa_lopo.read_rows()
    dels, _ = hpoa_lopo.deletions(rows, case_pm, hpoa_lopo.fold_split(case_pm))
    text = "".join(ln for i, ln in enumerate(lines) if i not in dels["lopo"])
    want = json.loads((LOPO_DIR / "deletions.json").read_text())["variants"]["lopo"]["sha256"]
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != want:
        sys.exit(f"lopo file sha256 {got} != deletions.json {want}")
    out.write_text(text)
    print(f"lopo file {out}: {len(dels['lopo'])} rows removed, sha256 {got}", flush=True)


class Graph:
    """The rows the ranker reads (build_graph_v2.fingerprint order) plus the non-scoring node properties."""

    def __init__(self) -> None:
        self.dz: dict[str, dict] = {}                 # id -> name, vocab, n, sc
        self.con: dict[str, dict] = {}                # cui -> tui, idf
        self.edge: dict[tuple[str, str], tuple] = {}  # (disease, concept) -> n_mentions, n_docs, n_topic_docs, src
        self.cisa: list[tuple[str, str]] = []
        self.disa: list[tuple[str, str]] = []
        self.isdz: list[str] = []

    def fp(self) -> str:
        return fingerprint(((k, self.dz[k]["n"], self.dz[k]["sc"]) for k in sorted(self.dz)),
                           ((c, None if self.con[c]["idf"] is None else round(self.con[c]["idf"], 9))
                            for c in sorted(self.con)),
                           ((k, self.edge[k][1], self.edge[k][2]) for k in sorted(self.edge)),
                           self.cisa, self.disa, self.isdz)

    def build_fp(self) -> str:
        """The fingerprint build_graph_v2.py writes to its summary. Its idf part divides by the Disease node count
        (n_dz is reassigned to it before the digest), not by the diseases with findings the stored idf uses."""
        df: dict[str, int] = defaultdict(int)
        for _, c in self.edge:
            df[c] += 1
        n = len(self.dz)
        return fingerprint(((k, self.dz[k]["n"], self.dz[k]["sc"]) for k in sorted(self.dz)),
                           ((c, round(math.log(n / df[c]), 9) if df.get(c) else None) for c in sorted(self.con)),
                           ((k, self.edge[k][1], self.edge[k][2]) for k in sorted(self.edge)),
                           self.cisa, self.disa, self.isdz)


def read_graph(uri: str, auth: str) -> Graph:
    g = Graph()
    drv = GraphDatabase.driver(uri, auth=tuple(auth.split("/", 1)))
    with drv.session() as s:
        for r in s.run("MATCH (d:Disease) RETURN d.id AS id, d.name AS name, d.vocab AS vocab, d.n_findings AS n, "
                       "d.src_chars AS sc"):
            g.dz[r["id"]] = {"name": r["name"], "vocab": r["vocab"], "n": r["n"], "sc": r["sc"]}
        for r in s.run("MATCH (c:Concept) RETURN c.cui AS cui, c.tui AS tui, c.idf AS idf"):
            g.con[r["cui"]] = {"tui": r["tui"], "idf": r["idf"]}
        for r in s.run("MATCH (d:Disease)-[e:HAS_FINDING]->(c:Concept) RETURN d.id AS d, c.cui AS c, "
                       "e.n_mentions AS m, e.n_docs AS nd, e.n_topic_docs AS nt, e.src AS src"):
            g.edge[(r["d"], r["c"])] = (r["m"], r["nd"], r["nt"], r["src"])
        g.cisa = sorted((r["c"], r["p"]) for r in s.run("MATCH (c:Concept)-[:IS_A]->(p:Concept) "
                                                        "RETURN c.cui AS c, p.cui AS p"))
        g.disa = sorted((r["c"], r["p"]) for r in s.run("MATCH (c:Disease)-[:IS_A]->(p:Disease) "
                                                        "RETURN c.id AS c, p.id AS p"))
        pairs = [(r["c"], r["d"]) for r in s.run("MATCH (c:Concept)-[:IS_DISEASE]->(d:Disease) "
                                                 "RETURN c.cui AS c, d.id AS d")]
    drv.close()
    if any(c != d for c, d in pairs):
        sys.exit(f"{uri}: IS_DISEASE joins different ids")
    g.isdz = sorted(c for c, _ in pairs)
    return g


def merge(smix: Graph, hpoa: set[tuple[str, str]]) -> tuple[Graph, dict]:
    """S-mix plus HPOA edges under MERGE_RULE, every derived row recomputed as build_graph_v2.py does."""
    u = Graph()
    u.edge = {k: (m, nd, nt, "smix") for k, (m, nd, nt, _) in smix.edge.items()}
    for k in hpoa:
        m, nd, nt, _ = u.edge.get(k, (0, 0, 0, None))
        u.edge[k] = (m, nd + 1, nt + 1, "both" if k in smix.edge else "hpoa")
    prof: dict[str, set[str]] = defaultdict(set)
    for d, c in u.edge:
        prof[d].add(c)
    n_dz = max(1, len(prof))
    df: dict[str, int] = defaultdict(int)
    for cs in prof.values():
        for c in cs:
            df[c] += 1
    cuis = set(df)
    parents = json.loads(Path("pilot/cache/umls_maps/phenotype_parents.json").read_text())
    u.cisa = sorted((c, p) for c in cuis for p in parents.get(c, []))
    cuis |= {p for _, p in u.cisa}
    u.dz = {k: {**v, "n": len(prof.get(k, ()))} for k, v in smix.dz.items()}
    u.con = {c: {"tui": smix.con.get(c, {}).get("tui") or "",
                 "idf": math.log(n_dz / df[c]) if df.get(c) else None} for c in cuis}
    u.disa = list(smix.disa)
    u.isdz = sorted(c for c in cuis if c in u.dz)
    src = defaultdict(int)
    for v in u.edge.values():
        src[v[3]] += 1
    smix_dz = {d for d, _ in smix.edge}
    info = {"merge_rule": MERGE_RULE, "edges": len(u.edge), "edges_by_source": dict(sorted(src.items())),
            "hpoa_edges": len(hpoa), "smix_edges": len(smix.edge), "diseases_with_findings": len(prof),
            "smix_diseases_with_findings": len(smix_dz),
            "diseases_with_hpoa_findings_only": len(set(prof) - smix_dz), "concepts": len(cuis)}
    return u, info


def write_graph(g: Graph, uri: str, auth: str) -> None:
    drv = GraphDatabase.driver(uri, auth=tuple(auth.split("/", 1)))

    def batches(s, query: str, rows: Iterable[dict], label: str) -> None:
        it, n = iter(rows), 0
        while chunk := list(islice(it, BATCH)):
            s.run(query, rows=chunk).consume()
            n += len(chunk)
        print(f"  {label}: {n}", flush=True)

    with drv.session() as s:
        for match, act in (("MATCH ()-[r]->() RETURN r", "DELETE r"), ("MATCH (n) RETURN n", "DETACH DELETE n")):
            r = s.run("CALL apoc.periodic.iterate($m, $a, {batchSize: 2000}) YIELD failedBatches "
                      "RETURN failedBatches", m=match, a=act).single()
            if r["failedBatches"]:
                sys.exit("wipe failed")
        if s.run("MATCH (n) RETURN count(n) AS n").single()["n"]:
            sys.exit("wipe incomplete")
        for q in ("CREATE CONSTRAINT dz IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
                  "CREATE CONSTRAINT cp IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE"):
            s.run(q).consume()
        batches(s, "UNWIND $rows AS r CREATE (d:Disease {id: r.id}) SET d.name = r.name, d.vocab = r.vocab, "
                   "d.n_findings = r.n, d.src_chars = r.sc",
                [{"id": k, **v} for k, v in g.dz.items()], "Disease")
        batches(s, "UNWIND $rows AS r CREATE (c:Concept {cui: r.cui}) SET c.tui = r.tui, c.idf = r.idf",
                [{"cui": c, **v} for c, v in g.con.items()], "Concept")
        batches(s, "UNWIND $rows AS r MATCH (d:Disease {id: r.d}), (c:Concept {cui: r.c}) "
                   "CREATE (d)-[:HAS_FINDING {n_mentions: r.m, n_docs: r.nd, n_topic_docs: r.nt, src: r.src}]->(c)",
                ({"d": d, "c": c, "m": m, "nd": nd, "nt": nt, "src": src}
                 for (d, c), (m, nd, nt, src) in g.edge.items()), "HAS_FINDING")
        batches(s, "UNWIND $rows AS r MATCH (c:Concept {cui: r.c}), (p:Concept {cui: r.p}) CREATE (c)-[:IS_A]->(p)",
                [{"c": c, "p": p} for c, p in g.cisa], "Concept IS_A")
        batches(s, "UNWIND $rows AS r MATCH (c:Disease {id: r.c}), (p:Disease {id: r.p}) CREATE (c)-[:IS_A]->(p)",
                [{"c": c, "p": p} for c, p in g.disa], "Disease IS_A")
        batches(s, "UNWIND $rows AS r MATCH (c:Concept {cui: r.id}), (d:Disease {id: r.id}) "
                   "CREATE (c)-[:IS_DISEASE]->(d)", [{"id": c} for c in g.isdz], "IS_DISEASE")
    drv.close()


def union_build(hpoa_path: str, smix_uri: str, uri: str, auth: str, summary: Path) -> None:
    if uri == smix_uri:
        sys.exit("the union graph is written to --uri, which must not be the S-mix instance")
    frozen = json.loads((DEVD / "frozen_config_E.json").read_text())["graph_systems"]["S-mix"]
    t0 = time.time()
    smix = read_graph(smix_uri, auth)
    fp = smix.build_fp()
    print(f"S-mix export from {smix_uri}: {len(smix.dz)} diseases, {len(smix.con)} concepts, {len(smix.edge)} "
          f"edges, build fingerprint {fp} ({round(time.time() - t0)}s)", flush=True)
    if fp != frozen["fingerprint"] or len(smix.edge) != frozen["edges"]:
        sys.exit(f"S-mix export fingerprint {fp} / {len(smix.edge)} edges != frozen {frozen['fingerprint']}")
    uni = set(smix.dz)
    want_disa = sorted((c, p) for c, p in set(v2_ids.disease_isa()) if c in uni and p in uni)
    if smix.disa != want_disa:
        sys.exit("S-mix Disease IS_A differs from the universe hierarchy")
    again, _ = merge(smix, set())
    if again.fp() != smix.fp():
        sys.exit("merge() with no HPOA edge does not reproduce the S-mix rows (stored idf included)")
    del again
    info: dict = {"smix_uri": smix_uri, "smix_build_fingerprint": fp, "smix_rows_fingerprint": smix.fp()}
    hpoa: set[tuple[str, str]] = set()
    if hpoa_path != "none":
        sha = hashlib.sha256(Path(hpoa_path).read_bytes()).hexdigest()
        known = {v["sha256"]: (name, v["edges"]) for name, v in
                 json.loads((LOPO_DIR / "deletions.json").read_text())["variants"].items()}
        if sha not in known:
            sys.exit(f"{hpoa_path} is neither the HPOA release nor a P0-4 variant (sha256 {sha})")
        variant, n_edges = known[sha]
        CA.HPOA = hpoa_path
        hpoa = set(BRG.pairs_hpoa(dict.fromkeys(uni)))
        if len(hpoa) != n_edges:
            sys.exit(f"{len(hpoa)} HPOA edges, deletions.json records {n_edges} for {variant}")
        info |= {"hpoa": hpoa_path, "hpoa_sha256": sha, "hpoa_variant": variant}
        if variant == "base":
            ref, _ = npz_edges("hpoa")
            if hpoa != ref:
                sys.exit(f"HPOA edges differ from devD/ref_hpoa/graph.npz: {len(hpoa - ref)} extra, "
                         f"{len(ref - hpoa)} missing")
            info["hpoa_equals_ref_hpoa_npz"] = True
    u, minfo = merge(smix, hpoa)
    del smix
    info |= minfo
    want = u.fp()
    t0 = time.time()
    write_graph(u, uri, auth)
    got = read_graph(uri, auth)
    if got.fp() != want or got.edge.keys() != u.edge.keys():
        sys.exit(f"graph read back from {uri} differs from the merge ({got.fp()} vs {want})")
    srcs = defaultdict(int)
    for v in got.edge.values():
        srcs[v[3]] += 1
    if dict(srcs) != minfo["edges_by_source"]:
        sys.exit(f"edge sources read back {dict(srcs)} != {minfo['edges_by_source']}")
    info |= {"fingerprint": want, "load_sec": round(time.time() - t0), "uri": uri}
    summary.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


def check_mirror(run: Path) -> None:
    stored = {r["bench"]: r for r in json.loads(STORED["S-mix"].read_text())["results"]}
    bad = []
    for r in json.loads(run.read_text())["results"]:
        diff = reproduces(r, stored[r["bench"]])
        print(f"{r['bench']:19s} n {r['evaluated']:6d}: {diff or 'stored S-mix ranks reproduced'}", flush=True)
        if diff:
            bad.append(r["bench"])
    if bad:
        sys.exit(f"S-mix loaded by union-build does not reproduce the stored ranks: {bad}")


def load_lists(system: str) -> dict:
    with gzip.open(OUT / f"lists_{system}.json.gz", "rt") as fh:
        return json.load(fh)


def rrf(a: list[str], b: list[str], w: float) -> list[str]:
    sc: dict[str, float] = defaultdict(float)
    for lst, wt in ((a, 1 - w), (b, w)):
        if wt:
            for i, d in enumerate(lst):
                sc[d] += wt / (60 + i + 1)
    return sorted(sc, key=lambda d: (-sc[d], d))[:TOP]


def fused(smix: dict, ref: dict, w: float) -> dict:
    res = []
    for bench in smix["sets"]:
        if smix["gold"][bench] != ref["gold"][bench]:
            sys.exit(f"{bench}: patients of the two list files differ")
        keys = smix["gold"][bench]
        ranked = [rrf(a, b, w) for a, b in zip(smix["sets"][bench], ref["sets"][bench])]
        res.append(result(bench, smix["patients"][bench], ranked, [set(k.split("|")) for k in keys],
                          [set(smix["lineage"][k]) for k in keys]))
    return {"args": {"combination": "RRF", "w_hpoa": w, "lists": [smix["system"], ref["system"]],
                     "depth": DEPTH, "window": WINDOW, "rrf_k": 60}, "results": res}


def fuse() -> None:
    smix, hpoa, lopo = load_lists("S-mix"), load_lists("HPOA"), load_lists("HPOA-LOPO")
    for other in (hpoa, lopo):
        diff = [k for k in smix["lineage"] if smix["lineage"][k] != other["lineage"].get(k)]
        if diff:
            sys.exit(f"lineage of {len(diff)} gold sets differs between S-mix and {other['system']}")
    for w, name in ((0.0, "S-mix"), (1.0, "HPOA")):
        stored = {r["bench"]: r for r in json.loads(STORED[name].read_text())["results"]}
        bad = [f"{r['bench']}: {d}" for r in fused(smix, hpoa, w)["results"] if (d := reproduces(r, stored[r["bench"]]))]
        if bad:
            sys.exit(f"RRF at w = {w} does not reproduce {name}: {bad}")
        print(f"RRF at w = {w:g} reproduces the stored {name} ranks for every patient", flush=True)
    trail, best, best_s = [], None, (-1.0, -1.0)
    for w in WEIGHTS:
        run = fused(smix, hpoa, w)
        (OUT / f"rrf_w{w}.json").write_text(json.dumps(run))
        (OUT / f"rrf_lopo_w{w}.json").write_text(json.dumps(fused(smix, lopo, w)))
        s = macro(run["results"])
        trail.append({"w": w, "macro_top10": s[0], "macro_top1": s[1],
                      "per_set": {r["bench"]: [r["lineage@10"], r["lineage@1"]] for r in run["results"]}})
        print(f"RRF w = {w}: dev macro Top-10 {s[0]}, Top-1 {s[1]}", flush=True)
        if s > best_s:
            best, best_s = w, s
    sel = {"criterion": "D3: dev macro lineage Top-10 over the nine development sets, ties Top-1, smaller w first",
           "selected_w": best, "selected_score": best_s, "trail": trail,
           "lineage_check": "S-mix and HPOA/HPOA-LOPO list files give the same lineage for every gold set",
           "endpoint_check": "w = 0 and w = 1 reproduce the stored S-mix and HPOA ranks for every patient"}
    (OUT / "rrf_select.json").write_text(json.dumps(sel, indent=1))
    print(f"SELECTED w = {best} {best_s}", flush=True)


_RUNS: dict = {}
_CLUSTERS: dict = {}


def only(run: dict, sets: list[str]) -> dict:
    return {"results": [r for r in run["results"] if r["bench"] in sets]}


def paired_job(job: tuple) -> tuple[list[dict], dict]:
    """One comparison in one stratum at one k (run in a forked worker; reads _RUNS and _CLUSTERS)."""
    st, sets, a, b, role, k = job
    ra, rb = only(_RUNS[a], sets), only(_RUNS[b], sets)
    rows = []
    for r in compare(ra, rb, k, clusters=_CLUSTERS, require=sets):
        de = disease_equal(hits(ra, r["bench"], k), hits(rb, r["bench"], k), _CLUSTERS[r["bench"]])
        rows.append({"stratum": st, "comparison": f"{b} - {a}", "role": role, "k": k, "bench": r["bench"],
                     "n": r["n"], "diseases": r["clusters"], "A_pct": r[f"A@{k}"], "B_pct": r[f"B@{k}"],
                     "diff": r["diff"], "ci": r["ci"], "ci_cluster": r["ci_cluster"], "p": r["p"],
                     "p_holm": r["p_holm"], "only_A": r["only_A"], "only_B": r["only_B"], "disease_equal": de})
    m = macro_compare(ra, rb, k, sets)
    mc = macro_cluster(ra, rb, k, sets, _CLUSTERS)
    return rows, {"stratum": st, "comparison": f"{b} - {a}", "role": role, "k": k, "sets": m["sets"], "A": m["A"],
                  "B": m["B"], "diff": m["diff"], "ci_patient": m["ci"], "p_bootstrap": m["p"],
                  "ci_cluster": mc["cluster"]["ci"], "disease_equal": mc["disease_equal"], "diseases": mc["diseases"]}


def hits(run: dict, bench: str, k: int) -> list[bool]:
    rk = next(r["ranks_lineage"] for r in run["results"] if r["bench"] == bench)
    return [p is not None and p <= k for p in rk]


def macro_cluster(a: dict, b: dict, k: int, sets: list[str], clusters: dict[str, list], boot: int = 2000,
                  seed: int = 0) -> dict:
    """Equal-weight macro over sets with gold diseases resampled: the patient-weighted difference (whole clusters
    drawn within each set) and the disease-equal difference (mean of per-cluster differences). One
    random.Random(seed), sets in sorted order, the same draws for both."""
    data = []
    for s in sorted(sets):
        ha, hb = hits(a, s, k), hits(b, s, k)
        if len(ha) != len(hb) or len(ha) != len(clusters[s]):
            sys.exit(f"{s}: {len(ha)} / {len(hb)} ranks, {len(clusters[s])} gold keys")
        groups: dict = {}
        for i, key in enumerate(clusters[s]):
            groups.setdefault(key, []).append(int(hb[i]) - int(ha[i]))
        data.append(list(groups.values()))

    def stat(gls: list[list[list[int]]]) -> tuple[float, float]:
        pw = sum(100 * sum(sum(g) for g in gl) / sum(len(g) for g in gl) for gl in gls) / len(gls)
        de = sum(100 * sum(sum(g) / len(g) for g in gl) / len(gl) for gl in gls) / len(gls)
        return pw, de

    point = stat(data)
    rng = random.Random(seed)
    dist = [stat([[gl[rng.randrange(len(gl))] for _ in gl] for gl in data]) for _ in range(boot)]
    out = {}
    for j, name in enumerate(("cluster", "disease_equal")):
        xs = sorted(d[j] for d in dist)
        out[name] = {"diff": round(point[j], 2), "ci": [round(xs[int(0.025 * boot)], 2),
                                                        round(xs[int(0.975 * boot) - 1], 2)]}
    out["diseases"] = {s: len(gl) for s, gl in zip(sorted(sets), data)}
    return out


def stats() -> None:
    sel = json.loads((OUT / "rrf_select.json").read_text())
    w = sel["selected_w"]
    runs = {name: json.loads(p.read_text()) for name, p in STORED.items()}
    runs["RRF"] = json.loads((OUT / f"rrf_w{w}.json").read_text())
    runs["RRF-LOPO"] = json.loads((OUT / f"rrf_lopo_w{w}.json").read_text())
    runs["union"] = json.loads((OUT / "union.json").read_text())
    runs["union-LOPO"] = json.loads((OUT / "union_lopo.json").read_text())
    other_w = [x for x in WEIGHTS if x != w]
    for x in other_w:
        runs[f"RRF w={x}"] = json.loads((OUT / f"rrf_w{x}.json").read_text())
        runs[f"RRF-LOPO w={x}"] = json.loads((OUT / f"rrf_lopo_w{x}.json").read_text())
    lopo_trail = []
    for x in WEIGHTS:
        m = macro(json.loads((OUT / f"rrf_lopo_w{x}.json").read_text())["results"])
        lopo_trail.append({"w": x, "macro_top10": m[0], "macro_top1": m[1]})
    for name in ("union", "union-LOPO"):
        args = runs[name]["args"]
        if any(args[key] != v for key, v in config("S-mix").items()):
            sys.exit(f"{name}: ranked with {args}, not the frozen S-mix configuration")
    clusters = gold_clusters()                    # sets RAW_GOLD = True
    plan = []                                     # (stratum label, sets, A, B, role)
    for st, sets in STRATA.items():
        for combo in ("RRF", "union"):
            plan += [(st, sets, "HPOA", combo, "rule"), (st, sets, "S-mix", combo, "rule")]
        plan += [(st, sets, "RRF", "union", "auxiliary"), (st, sets, "S-mix", "HPOA", "reference")]
    lsets = STRATA["published"]
    for combo in ("RRF-LOPO", "union-LOPO"):
        plan += [("published LOPO", lsets, "HPOA-LOPO", combo, "rule"), ("published LOPO", lsets, "S-mix", combo,
                                                                          "rule")]
    plan += [("published LOPO", lsets, "RRF-LOPO", "union-LOPO", "auxiliary"),
             ("published LOPO", lsets, "S-mix", "HPOA-LOPO", "reference")]
    for x in other_w:                             # the weights D3 did not select, for reading only
        for st, sets in STRATA.items():
            plan += [(st, sets, "HPOA", f"RRF w={x}", f"auxiliary: unselected w={x}"),
                     (st, sets, "S-mix", f"RRF w={x}", f"auxiliary: unselected w={x}")]
        plan += [("published LOPO", lsets, "HPOA-LOPO", f"RRF-LOPO w={x}", f"auxiliary: unselected w={x}"),
                 ("published LOPO", lsets, "S-mix", f"RRF-LOPO w={x}", f"auxiliary: unselected w={x}")]
    global _RUNS, _CLUSTERS
    _RUNS, _CLUSTERS = runs, clusters
    jobs = [(st, sets, a, b, role, k) for st, sets, a, b, role in plan for k in (10, 1)]
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1), mp_context=get_context("fork")) as ex:
        done = list(ex.map(paired_job, jobs))
    rows = [r for rs, _ in done for r in rs]
    macros = [m for _, m in done]

    def mrow(st: str, comp: str, k: int = 10) -> dict:
        return next(x for x in macros if x["stratum"] == st and x["comparison"] == comp and x["k"] == k)

    def bounds(x: dict) -> dict:
        return {"patient": x["ci_patient"][0], "cluster": x["ci_cluster"][0],
                "disease_equal": x["disease_equal"]["ci"][0]}

    decision: dict = {"rule": "§9.33 M1: 'complementary' only if, in the unpublished hospital stratum (PUMCH_L, "
                              "PUMCH-ADM), combination - HPOA and combination - S-mix both have a macro Top-10 CI "
                              "lower bound > 0 (required of the patient-stratified, the gold-disease cluster and the "
                              "disease-equal CI)", "selected_w": w}
    for combo, lc in (("RRF", "RRF-LOPO"), ("union", "union-LOPO")):
        hh, hs = mrow("hospital", f"{combo} - HPOA"), mrow("hospital", f"{combo} - S-mix")
        ok = all(v > 0 for x in (hh, hs) for v in bounds(x).values())
        cs = mrow("common", f"{combo} - S-mix")
        lh = mrow("published LOPO", f"{lc} - HPOA-LOPO")
        decision[combo] = {
            "hospital": {"vs HPOA": {"diff": hh["diff"], "lower_bounds": bounds(hh)},
                         "vs S-mix": {"diff": hs["diff"], "lower_bounds": bounds(hs)},
                         "complementary_9_33": ok},
            "aux_5_1_hospital": all(v > 0 for v in bounds(hh).values()) and all(v > -1 for v in bounds(hs).values()),
            "aux_5_1_common_no_loss": all(v > -1 for v in bounds(cs).values()),
            "aux_5_1_published_lopo_vs_hpoa": all(v > 0 for v in bounds(lh).values())}
    decision["verdict"] = ("complementary: " + ", ".join(c for c in ("RRF", "union")
                                                          if decision[c]["hospital"]["complementary_9_33"])
                           if any(decision[c]["hospital"]["complementary_9_33"] for c in ("RRF", "union"))
                           else "not complementary: neither combination meets the §9.33 M1 rule")
    res = {"note": "Development sets only (post hoc, §9.33 M1). diff = B - A in percentage points of lineage Top-k. "
                   "ci = patient-paired bootstrap 95% (2,000, seed 0); ci_cluster = gold-disease cluster bootstrap "
                   "95%; disease_equal = mean over gold clusters of the per-cluster difference, clusters resampled; "
                   "p = exact McNemar, p_holm = Holm across the stratum's sets for one comparison at one k. Macro = "
                   "equal-weight mean over the stratum's sets: ci_patient = paired_stats.macro_compare "
                   "(set-stratified patient bootstrap), ci_cluster / disease_equal = macro_cluster (clusters "
                   "resampled within each set). RRF w chosen on the same nine sets (selection optimism).",
           "runs": {**{k: str(v) for k, v in STORED.items()}, "RRF": str(OUT / f"rrf_w{w}.json"),
                    "RRF-LOPO": str(OUT / f"rrf_lopo_w{w}.json"), "union": str(OUT / "union.json"),
                    "union-LOPO": str(OUT / "union_lopo.json")},
           "rrf_selection": sel, "decision": decision,
           "rrf_lopo_trail": {"note": "nine-set dev macro of RRF with HPOA-LOPO at each w (not used to select)",
                              "trail": lopo_trail},
           "per_set": {name: {r["bench"]: [r["lineage@10"], r["lineage@1"]] for r in run["results"]}
                       for name, run in runs.items()},
           "macro": macros, "rows": rows}
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))

    print(f"RRF w selected: {w} ({sel['selected_score']}); trail "
          + ", ".join(f"{t['w']}: {t['macro_top10']}/{t['macro_top1']}" for t in sel["trail"]))
    print(f"\n{'stratum':15s} {'comparison':24s} {'k':>3s} {'A':>6s} {'B':>6s} {'diff':>6s} {'patient CI':>16s} "
          f"{'cluster CI':>16s} | {'dz-equal':>7s} {'CI':>16s}")
    for m in macros:
        de = m["disease_equal"]
        print(f"{m['stratum']:15s} {m['comparison']:24s} @{m['k']:<2d} {m['A']:6.2f} {m['B']:6.2f} {m['diff']:+6.2f} "
              f"[{m['ci_patient'][0]:+6.2f},{m['ci_patient'][1]:+6.2f}] [{m['ci_cluster'][0]:+6.2f},"
              f"{m['ci_cluster'][1]:+6.2f}] | {de['diff']:+7.2f} [{de['ci'][0]:+6.2f},{de['ci'][1]:+6.2f}]")
    print(f"\n{'stratum':15s} {'comparison':24s} {'k':>3s} {'set':19s} {'n':>6s} {'dz':>5s} {'A':>6s} {'B':>6s} "
          f"{'diff':>6s} {'patient CI':>16s} {'cluster CI':>16s} {'Holm p':>8s} | {'dz-equal':>7s} {'CI':>16s}")
    for r in rows:
        de = r["disease_equal"]
        print(f"{r['stratum']:15s} {r['comparison']:24s} @{r['k']:<2d} {r['bench']:19s} {r['n']:6d} "
              f"{r['diseases']:5d} {r['A_pct']:6.2f} {r['B_pct']:6.2f} {r['diff']:+6.2f} "
              f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] [{r['ci_cluster'][0]:+6.2f},{r['ci_cluster'][1]:+6.2f}] "
              f"{r['p_holm']:8.2g} | {de['diff']:+7.2f} [{de['ci'][0]:+6.2f},{de['ci'][1]:+6.2f}]")
    print("\nM1 decision:")
    print(json.dumps(decision, indent=1))
    print(f"\nsaved {OUT / 'stats.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["lists", "lopo-file", "union-build", "check-mirror", "fuse", "stats", "args"])
    ap.add_argument("--system", choices=list(STORED))
    ap.add_argument("--uri", default="bolt://localhost:7688")
    ap.add_argument("--smix-uri", default="bolt://localhost:7689")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--hpoa", default="", help="union-build: phenotype.hpoa to merge, or 'none' for S-mix alone")
    ap.add_argument("--summary", type=Path, help="union-build: where the build summary goes")
    ap.add_argument("--out", type=Path, help="lopo-file: where the filtered phenotype.hpoa goes")
    ap.add_argument("--run", type=Path, help="check-mirror: graph_dx.py output on the S-mix-only union build")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "args":
        if not a.system:
            ap.error("args needs --system")
        c = config(a.system)
        print(" ".join(f"--{key} {c[key]}" for key in ("alpha", "mu", "decay", "unified_w", "support_beta",
                                                         "offtopic_w")))
    elif a.step == "lists":
        if not a.system:
            ap.error("lists needs --system")
        lists(a.system, a.uri, a.auth)
    elif a.step == "lopo-file":
        if not a.out:
            ap.error("lopo-file needs --out")
        lopo_file(a.out)
    elif a.step == "union-build":
        if not (a.hpoa and a.summary):
            ap.error("union-build needs --hpoa and --summary")
        union_build(a.hpoa, a.smix_uri, a.uri, a.auth, a.summary)
    elif a.step == "check-mirror":
        if not a.run:
            ap.error("check-mirror needs --run")
        check_mirror(a.run)
    elif a.step == "fuse":
        fuse()
    else:
        stats()


if __name__ == "__main__":
    main()
