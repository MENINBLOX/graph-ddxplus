"""HippoRAG-style personalized PageRank diagnosis over the literature graph (research design D10).

HippoRAG (NeurIPS 2024) seeds a personalized PageRank (PPR) with the query's entities, weighted by
node specificity, and scores passages by the probability mass that reaches them. Here the passages
are Disease nodes, the entities are finding Concepts, and the walk runs on the undirected graph of
HAS_FINDING, concept IS_A, disease IS_A and IS_DISEASE edges. No LLM at inference.

    python ppr_dx.py export --uri bolt://localhost:7688 --out pilot/cache/v2/devD/graph_gemma.npz
    python ppr_dx.py tune   --graph pilot/cache/v2/devD/graph_gemma.npz --out pilot/cache/v2/devD/tune_ppr_gemma.json
    python ppr_dx.py test   --graph ... --cfg tune.json --bench dev|confirm --out X.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402
from graph_dx import GraphDx, evaluate  # noqa: E402

DEFAULT = {"restart": 0.5, "beta": 0.0, "gamma": 1.0, "h": 1.0, "eta": 0.0, "seed_w": "idf", "skip_seed": 1}
GRID = [("restart", [0.15, 0.3, 0.5, 0.7]), ("beta", [0.0, 0.5, 1.0]), ("gamma", [1.0, 0.0]),
        ("h", [0.0, 0.5, 1.0]), ("eta", [0.0, 0.5, 1.0]), ("seed_w", ["uniform", "idf"]), ("skip_seed", [1, 0])]


def export(uri: str, auth: str, out: str) -> None:
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(uri, auth=tuple(auth.split("/", 1)))
    with drv.session() as s:
        dz = [r["id"] for r in s.run("MATCH (d:Disease) RETURN d.id AS id ORDER BY id")]
        cs = s.run("MATCH (c:Concept) RETURN c.cui AS cui, c.idf AS idf ORDER BY cui").data()
        hf = s.run("MATCH (d:Disease)-[e:HAS_FINDING]->(c:Concept) "
                   "RETURN d.id AS d, c.cui AS c, e.n_docs AS n, coalesce(e.n_topic_docs, e.n_docs) AS t").data()
        cisa = s.run("MATCH (a:Concept)-[:IS_A]->(b:Concept) RETURN a.cui AS a, b.cui AS b").data()
        disa = s.run("MATCH (a:Disease)-[:IS_A]->(b:Disease) RETURN a.id AS a, b.id AS b").data()
        isd = s.run("MATCH (c:Concept)-[:IS_DISEASE]->(d:Disease) RETURN c.cui AS c, d.id AS d").data()
    drv.close()
    nd = len(dz)
    idx = {k: i for i, k in enumerate(dz)}
    for j, r in enumerate(cs):
        idx[("C", r["cui"])] = nd + j

    def c(i):
        return idx[("C", i)]
    np.savez_compressed(
        out, diseases=np.array(dz), concepts=np.array([r["cui"] for r in cs]),
        idf=np.array([r["idf"] if r["idf"] is not None else 0.0 for r in cs], dtype=np.float32),
        hf_d=np.array([idx[r["d"]] for r in hf], dtype=np.int32), hf_c=np.array([c(r["c"]) for r in hf], dtype=np.int32),
        hf_n=np.array([r["n"] for r in hf], dtype=np.float32), hf_t=np.array([r["t"] for r in hf], dtype=np.float32),
        ci_a=np.array([c(r["a"]) for r in cisa], dtype=np.int32), ci_b=np.array([c(r["b"]) for r in cisa], dtype=np.int32),
        di_a=np.array([idx[r["a"]] for r in disa], dtype=np.int32), di_b=np.array([idx[r["b"]] for r in disa], dtype=np.int32),
        id_c=np.array([c(r["c"]) for r in isd], dtype=np.int32), id_d=np.array([idx[r["d"]] for r in isd], dtype=np.int32))
    print(f"exported: diseases {nd}, concepts {len(cs)}, HAS_FINDING {len(hf)}, concept IS_A {len(cisa)}, "
          f"disease IS_A {len(disa)}, IS_DISEASE {len(isd)}", flush=True)


class PPR:
    def __init__(self, path: str):
        g = np.load(path)
        self.g = g
        self.dz = list(g["diseases"])
        self.nd = len(self.dz)
        self.n = self.nd + len(g["concepts"])
        self.cidx = {c: self.nd + j for j, c in enumerate(g["concepts"])}
        self.didx = {d: i for i, d in enumerate(self.dz)}
        self.idf = np.concatenate([np.zeros(self.nd, dtype=np.float32), g["idf"]])
        self._key = None

    def build(self, cfg: dict) -> None:
        key = (cfg["beta"], cfg["gamma"], cfg["h"])
        if key == self._key:
            return
        g = self.g
        w = g["hf_n"] ** cfg["beta"] * np.where(g["hf_t"] > 0, 1.0, cfg["gamma"])
        rows = [g["hf_d"], g["hf_c"]]
        cols = [g["hf_c"], g["hf_d"]]
        vals = [w, w]
        for a, b, wt in ((g["ci_a"], g["ci_b"], cfg["h"]), (g["di_a"], g["di_b"], cfg["h"]),
                         (g["id_c"], g["id_d"], 1.0)):
            if wt > 0 and len(a):
                rows += [a, b]
                cols += [b, a]
                vals += [np.full(len(a), wt, np.float32)] * 2
        A = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        A.eliminate_zeros()
        deg = np.asarray(A.sum(axis=1)).ravel()
        self.deg = deg
        inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0).astype(np.float32)
        self.T = (A.T @ sp.diags(inv)).tocsr()          # column-stochastic transition
        self._key = key

    def seeds(self, ev: set[str], cfg: dict) -> dict[int, float]:
        out = {}
        for cui in ev:
            for i in (self.cidx.get(cui), self.didx.get(cui)):
                if i is None:
                    continue
                wt = 1.0 if cfg["seed_w"] == "uniform" else float(max(self.idf[i], 0.0) or 1.0)
                out[i] = max(out.get(i, 0.0), wt)
        return out

    def rank_batch(self, evs: list[set[str]], cfg: dict, k: int = 50, iters: int = 30) -> list[list[str]]:
        self.build(cfg)
        res: list[list[str]] = []
        r = cfg["restart"]
        hub = np.power(np.maximum(self.deg[: self.nd], 1e-12), cfg["eta"]) if cfg["eta"] else None
        for s0 in range(0, len(evs), 256):
            chunk = evs[s0:s0 + 256]
            S = np.zeros((self.n, len(chunk)), dtype=np.float32)
            seeds = []
            for j, ev in enumerate(chunk):
                sd = self.seeds(ev, cfg)
                seeds.append(sd)
                tot = sum(sd.values())
                for i, v in sd.items():
                    S[i, j] = v / tot
            P = S.copy()
            for _ in range(iters):
                P = r * S + (1 - r) * (self.T @ P)
            D = P[: self.nd]
            if hub is not None:
                D = D / hub[:, None]
            for j, sd in enumerate(seeds):
                col = D[:, j].copy()
                if not sd:
                    res.append([])
                    continue
                if cfg["skip_seed"]:
                    for i in sd:
                        if i < self.nd:
                            col[i] = -1
                top = np.argpartition(-col, k)[:k]
                top = top[np.argsort(-col[top], kind="stable")]
                res.append([self.dz[i] for i in top if col[i] > 0])
        return res


class Memo:
    """Adapter so graph_dx.evaluate can consume precomputed batched rankings."""

    def __init__(self, ranks: dict, lin: GraphDx):
        self.ranks, self.lin = ranks, lin

    def rank(self, ev, k=50):
        return self.ranks[frozenset(ev)][:k]

    def lineage(self, gold):
        return self.lin.lineage(gold)


def run(ppr: PPR, cfg: dict, sets: dict, lin: GraphDx, boot: int = 0) -> list[dict]:
    out = []
    for name, pats in sets.items():
        evs = [ev for g, ev in pats if v2_ids.gold_keys(g) and ev]
        uniq = list({frozenset(e) for e in evs})
        rk = ppr.rank_batch([set(e) for e in uniq], cfg)
        out.append(evaluate(Memo(dict(zip(uniq, rk)), lin), name, pats, boot))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "tune", "test"])
    ap.add_argument("--uri", default="bolt://localhost:7688")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--graph", default="")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--bench", default="dev")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.step == "export":
        export(a.uri, a.auth, a.out)
        return
    O.RAW_GOLD = True
    from tune_dev import dev_sets, macro
    lin = GraphDx(a.uri, a.auth, 0.5, 2.0, 0.0, 0.0)
    ppr = PPR(a.graph)
    if a.step == "tune":
        sets = dev_sets()
        trail = []

        def score(cfg):
            res = run(ppr, cfg, sets, lin)
            m = macro(res)
            trail.append({"cfg": dict(cfg), "macro_top10": m[0], "macro_top1": m[1],
                          "per_bench": {r["bench"]: [r["lineage@1"], r["lineage@10"]] for r in res}})
            print(f"cfg {cfg} -> {m}", flush=True)
            return m
        best = dict(DEFAULT)
        best_s = score(best)
        for _ in range(2):
            changed = False
            for key, vals in GRID:
                for v in vals:
                    if v == best[key]:
                        continue
                    cfg = {**best, key: v}
                    s = score(cfg)
                    if s > best_s:
                        best, best_s, changed = cfg, s, True
                print(f"after {key}: {best} -> {best_s}", flush=True)
            if not changed:
                break
        json.dump({"criterion": "dev macro lineage Top-10 (ties Top-1)", "selected": best,
                   "selected_score": best_s, "trail": trail}, open(a.out, "w"), indent=1)
        print("SELECTED", best, best_s, flush=True)
        return
    cfg = json.load(open(a.cfg))["selected"]
    sets = dev_sets() if a.bench == "dev" else O.load_confirm()
    res = run(ppr, cfg, sets, lin, a.boot)
    json.dump({"cfg": cfg, "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
