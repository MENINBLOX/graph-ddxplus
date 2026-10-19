"""Retrieval-layer baselines on the same literature as the graph (research design §9.21).

Standard medical RAG retrievers (MedRAG/MIRAGE practice) answer the same question as the graph
query: given a patient's findings, which of the 43,575 candidate diseases come back? Each chunk
belongs to an article; an article belongs to the diseases whose name query retrieved it (the same
corpus link the graph uses). Disease score = aggregation of the retrieved chunks' scores over the
diseases each chunk's article is linked to. No LLM, no extraction, no graph.

Same inputs as the graph: patient concepts (CUIs) rendered as their UMLS preferred names; same
universe, same generic-document filter, configuration chosen on DDXPlus validate only.

    python retrieval_baseline.py build                         # chunks + BM25 index + MedCPT embeddings
    python retrieval_baseline.py tune --method bm25 --out X    # validate-only selection
    python retrieval_baseline.py test --method bm25 --cfg X --out Y
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402
from graph_dx import GraphDx, evaluate  # noqa: E402

V2 = "pilot/cache/v2"
OUT = f"{V2}/retrieval"
CHUNK = 1000                 # characters per chunk (MedCPT article encoder: 512 tokens)
MRCONSO = "data/umls_extracted/MRCONSO.RRF"
EXCLUDED = f"{V2}/exclude_nonhuman_diseases.json"
MAX_DOC_DISEASES = 20        # same generic-document rule as build_graph_v2.py


def chunks_of(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + CHUNK)
        if j < len(text):
            k = text.rfind(" ", i + CHUNK // 2, j)
            j = k if k > 0 else j
        out.append(text[i:j].strip())
        i = j
    return [c for c in out if len(c) > 50]


def build_corpus(sources: str) -> None:
    """Two passes over the sources, so only the chunk-to-document map stays in memory: the first
    links documents to diseases, the second streams each kept document's chunks to disk."""
    from build_graph_v2 import article, iter_sources
    uni = set(v2_ids.universe()) - set(json.load(open(EXCLUDED)))
    use, links, title = defaultdict(int), defaultdict(set), {}
    for dz, dz_docs in iter_sources(sources):
        held = {x["doc_id"] for x in dz_docs}
        for x in dz_docs:                  # counted exactly as build_graph_v2.py counts use
            if not (x["doc_id"].startswith("TBL") and article(x["doc_id"]) in held):
                use[article(x["doc_id"])] += 1
            if dz in uni:
                links[x["doc_id"]].add(dz)
                title.setdefault(x["doc_id"], x.get("title") or "")
    docs = sorted(d for d in links if use[article(d)] < MAX_DOC_DISEASES)
    idx = {d: i for i, d in enumerate(docs)}
    os.makedirs(OUT, exist_ok=True)
    chunk_doc, written = [], set()
    with open(f"{OUT}/chunks.jsonl", "w") as fo:
        for dz, dz_docs in iter_sources(sources):
            if dz not in uni:
                continue
            for x in dz_docs:
                d = x["doc_id"]
                if d in idx and d not in written:
                    written.add(d)
                    for c in chunks_of(x["text"]):
                        chunk_doc.append(idx[d])
                        fo.write(json.dumps(c) + "\n")
    json.dump({"docs": docs, "titles": [title[d] for d in docs],
               "doc_diseases": [sorted(links[d]) for d in docs]}, open(f"{OUT}/docs.json", "w"))
    np.save(f"{OUT}/chunk_doc.npy", np.array(chunk_doc, dtype=np.int32))
    print(f"docs {len(docs)}, chunks {len(chunk_doc)}", flush=True)


def load_chunks() -> list[str]:
    return [json.loads(ln) for ln in open(f"{OUT}/chunks.jsonl")]


def build_bm25() -> None:
    import bm25s
    import Stemmer
    corpus = load_chunks()
    stem = Stemmer.Stemmer("english")
    tok = bm25s.tokenize(corpus, stopwords="en", stemmer=stem)
    ret = bm25s.BM25()
    ret.index(tok)
    ret.save(f"{OUT}/bm25")
    print("bm25 index saved", flush=True)


def build_medcpt(batch: int = 256) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    corpus = load_chunks()
    meta = json.load(open(f"{OUT}/docs.json"))
    cdoc = np.load(f"{OUT}/chunk_doc.npy")
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").cuda().half().eval()
    emb = np.zeros((len(corpus), 768), dtype=np.float16)
    order = np.argsort([len(c) for c in corpus])          # length-sorted batches
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([[meta["titles"][cdoc[i]], corpus[i]] for i in idx], truncation=True,
                      padding=True, max_length=512, return_tensors="pt").to("cuda")
            emb[idx] = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy().astype(np.float16)
            if s % (batch * 400) == 0:
                print(f"  medcpt {s}/{len(order)}", flush=True)
    np.save(f"{OUT}/medcpt.npy", emb)
    print("medcpt embeddings saved", flush=True)


def cui_names(cuis: set[str]) -> dict[str, str]:
    """English preferred name per CUI (MRCONSO: TS=P, STT=PF, ISPREF=Y first)."""
    best: dict[str, tuple[int, str]] = {}
    for ln in open(MRCONSO, encoding="utf-8"):
        f = ln.split("|", 15)
        if f[0] not in cuis or f[1] != "ENG":
            continue
        rank = (f[2] != "P") * 4 + (f[4] != "PF") * 2 + (f[6] != "Y")
        if f[0] not in best or rank < best[f[0]][0]:
            best[f[0]] = (rank, f[14])
    return {c: v[1] for c, v in best.items()}


_BACKENDS: dict[str, dict] = {}


def _backend(method: str) -> dict:
    if method not in _BACKENDS:
        b: dict = {"cache": {}}
        if method == "bm25":
            import bm25s
            import Stemmer
            b["stem"] = Stemmer.Stemmer("english")
            b["bm"] = bm25s.BM25.load(f"{OUT}/bm25")
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer
            b["torch"] = torch
            b["qtok"] = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
            b["qenc"] = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").cuda().eval()
            b["E"] = torch.from_numpy(np.load(f"{OUT}/medcpt.npy")).cuda()
        _BACKENDS[method] = b
    return _BACKENDS[method]


class Retriever:
    """rank(ev, k) over the open disease set from retrieved chunks; lineage() via the graph."""

    def __init__(self, method: str, k: int, agg: str, mode: str, names: dict, lineage: GraphDx):
        self.method, self.k, self.agg, self.mode, self.names, self.lin = method, k, agg, mode, names, lineage
        meta = json.load(open(f"{OUT}/docs.json"))
        self.doc_dz = meta["doc_diseases"]
        self.cdoc = np.load(f"{OUT}/chunk_doc.npy")
        b = _backend(method)                    # loaded once per process, shared across configs
        self.cache = b["cache"]
        for key in ("stem", "bm", "torch", "qtok", "qenc", "E"):
            if key in b:
                setattr(self, key, b[key])

    def _search(self, text: str) -> list[tuple[int, float]]:
        if (text, self.k) in self.cache:
            return self.cache[(text, self.k)]
        if self.method == "bm25":
            import bm25s
            q = bm25s.tokenize([text], stopwords="en", stemmer=self.stem, show_progress=False)
            if not q.vocab:
                hits = []
            else:
                ids, sc = self.bm.retrieve(q, k=self.k, show_progress=False)
                hits = [(int(i), float(s)) for i, s in zip(ids[0], sc[0]) if s > 0]
        else:
            t = self.torch
            with t.no_grad():
                enc = self.qtok([text], truncation=True, padding=True, max_length=64,
                                return_tensors="pt").to("cuda")
                q = self.qenc(**enc).last_hidden_state[:, 0, :].half()
                s = (self.E @ q[0]).float()
                v, i = t.topk(s, self.k)
            hits = list(zip(i.tolist(), v.tolist()))
        self.cache[(text, self.k)] = hits
        return hits

    def _disease_scores(self, hits) -> dict[str, float]:
        sc: dict[str, float] = defaultdict(float)
        for r, (c, s) in enumerate(hits):
            w = 1.0 / (60 + r) if self.agg == "rrf" else s
            for d in self.doc_dz[self.cdoc[c]]:
                sc[d] = max(sc[d], w) if self.agg == "max" else sc[d] + w
        return sc

    def rank(self, ev: set[str], k: int = 50) -> list[str]:
        terms = sorted(self.names[c] for c in ev if c in self.names)
        if not terms:
            return []
        if self.mode == "joint":
            sc = self._disease_scores(self._search(" ".join(terms)))
        else:                                   # per-concept retrieval, disease scores summed
            sc = defaultdict(float)
            for t in terms:
                for d, v in self._disease_scores(self._search(t)).items():
                    sc[d] += v
        return sorted(sc, key=lambda d: (-sc[d], d))[:k]

    def lineage(self, gold):
        return self.lin.lineage(gold)


class Fusion:
    """Hybrid layer (post-hoc analysis): reciprocal-rank fusion of the graph ranking and a retriever.

    score(d) = (1 - w) / (60 + rank_graph(d)) + w / (60 + rank_retriever(d)); w chosen on validate.
    Provenance still comes from the graph for every disease the graph ranks.
    """

    def __init__(self, graph: GraphDx, retr: Retriever, w: float, depth: int = 200):
        self.g, self.r, self.w, self.depth = graph, retr, w, depth
        self.memo: dict = {}

    def rank(self, ev: set[str], k: int = 50) -> list[str]:
        key = tuple(sorted(ev))
        if key not in self.memo:
            self.memo[key] = (self.g.rank(ev, self.depth), self.r.rank(ev, self.depth))
        gr, rr = self.memo[key]
        sc: dict[str, float] = defaultdict(float)
        for lst, wt in ((gr, 1 - self.w), (rr, self.w)):
            for i, d in enumerate(lst):
                sc[d] += wt / (60 + i + 1)
        return sorted(sc, key=lambda d: (-sc[d], d))[:k]

    def lineage(self, gold):
        return self.g.lineage(gold)


def benches(names: list[str], ddx_n: int) -> dict:
    sets = {}
    for b in names:
        if b.startswith("ddxplus-"):
            sets.update(O.load_ddxplus(b.split("-", 1)[1], ddx_n))
        elif b == "rarebench":
            sets.update(O.load_rarebench())
        elif b == "pumch":
            sets.update(O.load_pumch())
        elif b == "phenopacket":
            sets.update(O.load_phenopacket())
        elif b == "symcat":
            sets.update(O.load_symcat(10))
        elif b == "confirm":
            sets.update(O.load_confirm())
        elif b == "confirm_e":
            sets.update(O.load_confirm_e())
    return sets


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "bm25", "medcpt", "tune", "test", "fuse_tune", "fuse_test"])
    ap.add_argument("--graph_cfg", default="pilot/cache/v2/eval_qwen3_v21/tune.json")
    ap.add_argument("--graph_uri", default="bolt://localhost:7687")
    ap.add_argument("--dev", action="store_true", help="design D: select on dev macro Top-10")
    ap.add_argument("--method", choices=["bm25", "medcpt"], default="bm25")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--bench", default="ddxplus-test,rarebench,pumch,phenopacket,symcat")
    ap.add_argument("--ddx_n", type=int, default=3000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--sources", default=f"{V2}/sources", help="as build_graph_v2.py --sources")
    ap.add_argument("--index", default=OUT, help="directory of the chunk corpus and indexes")
    a = ap.parse_args()
    OUT = a.index
    if a.step == "build":
        build_corpus(a.sources)
        return
    if a.step == "bm25":
        build_bm25()
        return
    if a.step == "medcpt":
        build_medcpt()
        return
    O.RAW_GOLD = True
    lin = GraphDx("bolt://localhost:7687", a.auth, 0.5, 2.0, 0.0, 0.0)
    if a.step in ("tune", "fuse_tune") and a.dev:
        from tune_dev import dev_sets
        sets = dev_sets(a.ddx_n)
    elif a.step in ("tune", "fuse_tune"):
        sets = benches(["ddxplus-validate"], a.ddx_n)
    else:
        sets = benches(a.bench.split(","), a.ddx_n)
    cuis = {c for pats in sets.values() for _, ev in pats for c in ev}
    names = cui_names(cuis)
    print(f"concepts {len(cuis)}, named {len(names)}", flush=True)
    if a.step.startswith("fuse"):
        gcfg = json.load(open(a.graph_cfg))["selected"]
        g = GraphDx(a.graph_uri, a.auth, **gcfg)
        rcfg = json.load(open(a.cfg))["selected"]
        rt = Retriever(rcfg["method"], rcfg["k"], rcfg["agg"], rcfg["mode"], names, lin)
        if a.step == "fuse_tune":
            trail, best, best_s = [], None, (-1, -1)
            fu = Fusion(g, rt, 0.0)
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                fu.w = w
                if a.dev:
                    from tune_dev import macro
                    res = [evaluate(fu, f"{b} w={w}", v) for b, v in sets.items()]
                    s_ = macro(res)
                    trail.append({"w": w, "macro_top10": s_[0], "macro_top1": s_[1]})
                else:
                    r = evaluate(fu, f"validate fusion w={w}", sets["DDXPlus-validate"])
                    trail.append({"w": w, **{k: v for k, v in r.items() if k != "ranks_lineage"}})
                    s_ = (r["lineage@1"], r["lineage@3"])
                if s_ > best_s:
                    best, best_s = {"w": w, "graph": gcfg, "retriever": rcfg, "graph_uri": a.graph_uri}, s_
            json.dump({"selected": best, "selected_score": best_s, "trail": trail}, open(a.out, "w"), indent=1)
            print("SELECTED", best["w"], best_s, flush=True)
        else:
            sel = json.load(open(a.out.replace("test", "tune")))["selected"]
            fu = Fusion(g, rt, sel["w"])
            res = [evaluate(fu, k, v, a.boot) for k, v in sets.items()]
            json.dump({"cfg": sel, "results": res}, open(a.out, "w"), indent=1)
        return
    if a.step == "tune" and a.dev:
        from tune_dev import macro
        trail, best, best_s = [], None, (-1, -1)
        for mode in ("joint", "per_concept"):
            for k in (100, 1000):
                for agg in ("max", "sum", "rrf"):
                    cfg = {"method": a.method, "mode": mode, "k": k, "agg": agg}
                    rt = Retriever(a.method, k, agg, mode, names, lin)
                    res = [evaluate(rt, b, v) for b, v in sets.items()]
                    m = macro(res)
                    trail.append({"cfg": cfg, "macro_top10": m[0], "macro_top1": m[1]})
                    print("dev", cfg, m, flush=True)
                    if m > best_s:
                        best, best_s = cfg, m
        json.dump({"criterion": "dev macro lineage Top-10 (ties Top-1)", "selected": best,
                   "selected_score": best_s, "trail": trail}, open(a.out, "w"), indent=1)
        print("SELECTED", best, best_s, flush=True)
        return
    if a.step == "tune":
        pats = sets["DDXPlus-validate"]
        trail, best, best_s = [], None, (-1, -1)
        for mode in ("joint", "per_concept"):
            for k in (100, 1000):
                for agg in ("max", "sum", "rrf"):
                    cfg = {"method": a.method, "mode": mode, "k": k, "agg": agg}
                    r = evaluate(Retriever(a.method, k, agg, mode, names, lin), f"validate {cfg}", pats)
                    s = (r["lineage@1"], r["lineage@3"])
                    trail.append({"cfg": cfg, **r})
                    if s > best_s:
                        best, best_s = cfg, s
        json.dump({"criterion": "DDXPlus validate lineage Top-1 (ties Top-3)", "selected": best,
                   "selected_score": best_s, "trail": trail}, open(a.out, "w"), indent=1)
        print("SELECTED", best, best_s, flush=True)
        return
    cfg = json.load(open(a.cfg))["selected"]
    rt = Retriever(cfg["method"], cfg["k"], cfg["agg"], cfg["mode"], names, lin)
    res = [evaluate(rt, k, v, a.boot) for k, v in sets.items()]
    json.dump({"cfg": cfg, "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
