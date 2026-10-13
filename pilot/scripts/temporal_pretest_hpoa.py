"""Pretest: does curated knowledge (HPOA snapshots) show measurable decay on cases published after the snapshot?

For each snapshot S: rank all OMIM diseases annotated in S for every phenopacket case with an
IC-weighted ancestor-closure overlap score; report Top-1/Top-10 by case age relative to S, split by
whether the gold disease is annotated in S at all (knowledge gap) and whether S already cites the
case's own PMID for that disease (provenance leakage, Jiang et al. 2026).
"""
import collections
import json
import math
import os
import re
import sys
import zipfile
from datetime import date

import numpy as np
import scipy.sparse as sp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
SNAPS = ["v2022-02-14", "v2022-10-05", "v2023-06-06", "v2024-01-11", "v2024-06-25", "v2024-12-12",
         "v2025-05-06", "v2025-11-24", "v2026-06-06", "v2026-09-01"]

hp = json.load(open(ROOT + "data/medkg/hp.json"))["graphs"][0]
parents = collections.defaultdict(set)
for e in hp["edges"]:
    if e["pred"] == "is_a":
        s, o = (x.rsplit("/", 1)[-1].replace("_", ":") for x in (e["sub"], e["obj"]))
        parents[s].add(o)
anc_cache = {}


def anc(t):
    if t not in anc_cache:
        out = {t}
        for p in parents.get(t, ()):
            out |= anc(p)
        anc_cache[t] = out
    return anc_cache[t]


sys.setrecursionlimit(10000)
dates = json.load(open(ROOT + "data/external_benchmarks/phenopacket_store/pmid_dates.json"))


def pdate(pmid):
    s = dates.get(pmid, {}).get("sortpubdate", "")[:10]
    return date(*map(int, s.split("/"))) if s else None


cases = []
z = zipfile.ZipFile(ROOT + "data/external_benchmarks/phenopacket_store/all_phenopackets.zip")
srcs = [(n, lambda n=n: z.read(n)) for n in z.namelist() if n.endswith(".json")]
nd = ROOT + "data/confirm/phenopacket_new"
srcs += [(n, lambda n=n: open(f"{nd}/{n}", "rb").read()) for n in os.listdir(nd)]
for name, rd in srcs:
    m = re.search(r"PMID_(\d+)", name)
    p = json.loads(rd())
    terms = {f["type"]["id"] for f in p.get("phenotypicFeatures", []) if not f.get("excluded")}
    gold = {d["term"]["id"] for d in p.get("diseases", [])}
    if m and terms and gold and pdate(m.group(1)):
        cases.append({"pmid": m.group(1), "date": pdate(m.group(1)), "terms": terms, "gold": gold})
print("cases", len(cases))

vocab = {}


def vid(t):
    return vocab.setdefault(t, len(vocab))


Xrows = [{vid(a) for t in c["terms"] for a in anc(t)} for c in cases]


def load_snap(tag):
    ann, cite = collections.defaultdict(set), collections.defaultdict(set)
    for line in open(ROOT + f"data/external_kg/hpoa_history/{tag}.hpoa"):
        if line.startswith("#") or line.startswith("database_id"):
            continue
        f = line.rstrip("\n").split("\t")
        if not f[0].startswith("OMIM:") or f[2] == "NOT" or f[10] != "P":
            continue
        ann[f[0]].add(f[3])
        for r in f[4].split(";"):
            if r.startswith("PMID:"):
                cite[f[0]].add(r[5:])
    return ann, cite


rows = []
for tag in SNAPS:
    sdate = date.fromisoformat(tag[1:])
    ann, cite = load_snap(tag)
    dz = sorted(ann)
    di = {d: i for i, d in enumerate(dz)}
    clos = [{vid(a) for t in ann[d] for a in anc(t)} for d in dz]
    nv = len(vocab)
    df = np.zeros(nv)
    for s in clos:
        df[list(s)] += 1
    ic = np.where(df > 0, -np.log(np.maximum(df, 1) / len(dz)), 0.0)
    Y = sp.lil_matrix((len(dz), nv), dtype=np.float32)
    for i, s in enumerate(clos):
        Y[i, list(s)] = 1
    Y = Y.tocsr()
    norm = np.sqrt(np.asarray(Y @ ic).ravel()) + 1e-9
    X = sp.lil_matrix((len(cases), nv), dtype=np.float32)
    for i, s in enumerate(Xrows):
        X[i, list(s)] = 1
    X = (X.tocsr() @ sp.diags(ic.astype(np.float32))).tocsr()
    for s0 in range(0, len(cases), 1000):
        S = (X[s0:s0 + 1000] @ Y.T).toarray() / norm
        for j, c in enumerate(cases[s0:s0 + 1000]):
            gi = [di[g] for g in c["gold"] if g in di]
            if gi:
                best = S[j, gi].max()
                rank = int((S[j] > best).sum()) + 1
            else:
                rank = None
            rows.append({"snap": tag, "age_days": (c["date"] - sdate).days, "rank": rank,
                         "known": bool(gi), "cited": any(c["pmid"] in cite.get(g, ()) for g in c["gold"])})
    print(tag, "diseases", len(dz), flush=True)
json.dump(rows, open(sys.argv[1], "w"))
