"""Collection-recall pretest: which source expansion brings curated findings into the text?

recall_decomp.py showed that most curated features of phenopacket gold diseases never occur in the
disease's article text. This measures, on a random sample of HPOA-annotated OMIM diseases (seeded,
chosen without benchmark information), the share of curated (disease, HPO) pairs whose HPO
label or synonym occurs in the text under each collection variant:

  base       current corpus (articles used by >= 20 diseases dropped, as in the graph)
  pmc_more   + PMC open-access reviews and case reports by disease name, 30 + 30, clinical sections, 12k cap
  pmc_full   the same articles, whole body, no cap
  abstracts  + PubMed abstracts by disease name (title/abstract), 50
  gene       + PubMed abstracts and PMC open-access full texts found by the causal gene symbol
             (HPO genes_to_disease), which is how monogenic subtypes are reported

Text volume alone raises string presence, so each variant is also scored against the curated
features of a random other sampled disease (control); the gap is the disease-specific gain.
Phenopacket source PMIDs stay excluded.

    set -a; . ./.env; set +a
    python collection_pretest.py --n 300 --out pilot/cache/v2/collect_pretest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curated_agreement as CA  # noqa: E402
import v2_ids  # noqa: E402
from deep_fetch import DROP_SEC, KEEP_SEC, name_variants  # noqa: E402
from fetch_v2 import FetchError, _retry  # noqa: E402
from recall_decomp import hpo_terms, norm  # noqa: E402
from statpearls_fetch import eutil  # noqa: E402

V2 = "pilot/cache/v2"
VARIANTS = ["base", "pmc_more", "pmc_full", "abstracts", "gene"]


def esearch(db: str, term: str, retmax: int) -> list[str]:
    def once():
        try:
            j = json.loads(eutil("esearch.fcgi", {"db": db, "term": term, "retmax": retmax, "retmode": "json",
                                                  "sort": "relevance"}))
        except Exception:
            raise FetchError("esearch")
        if "esearchresult" not in j or "ERROR" in j["esearchresult"]:
            raise FetchError("esearch")
        return j["esearchresult"].get("idlist", [])
    return _retry(once)


def efetch(db: str, ids: list[str]):
    def once():
        try:
            return ET.fromstring(eutil("efetch.fcgi", {"db": db, "id": ",".join(ids), "retmode": "xml"}))
        except ET.ParseError:
            raise FetchError("efetch")
    return _retry(once)


def pmc_docs(ids: list[str], exclude: set[str]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 20):
        for art in efetch("pmc", ids[i:i + 20]).iter("article"):
            aid = {e.get("pub-id-type"): (e.text or "").strip() for e in art.iter("article-id")}
            pmid = aid.get("pmid", "")
            body = art.find(".//body")
            if body is None or (pmid and pmid in exclude):
                continue
            clin, full = [], []
            for sec in body.iter("sec"):
                te = sec.find("title")
                t = "".join(te.itertext()).strip() if te is not None else ""
                ps = [" ".join(" ".join(p.itertext()).split()) for p in sec.findall("p")]
                full += ps
                if not (t and (DROP_SEC.search(t) or not KEEP_SEC.search(t))):
                    clin += ps
            tt = art.find(".//title-group/article-title")
            yr = art.find(".//pub-date/year")
            out.append({"id": "PMC" + aid.get("pmc", aid.get("pmcid", "")).lstrip("PMC"), "pmid": pmid,
                        "title": " ".join("".join(tt.itertext()).split()) if tt is not None else "",
                        "year": yr.text if yr is not None else "",
                        "clinical": " ".join(clin)[:12000], "full": " ".join(full)})
    return out


def pubmed_abstracts(ids: list[str], exclude: set[str]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 100):
        for art in efetch("pubmed", ids[i:i + 100]).iter("PubmedArticle"):
            pmid = (art.findtext(".//PMID") or "").strip()
            ab = " ".join("".join(x.itertext()) for x in art.iter("AbstractText"))
            if pmid in exclude or not ab:
                continue
            out.append({"id": "PMID" + pmid, "pmid": pmid, "title": art.findtext(".//ArticleTitle") or "",
                        "year": art.findtext(".//PubDate/Year") or "", "full": " ".join(ab.split())})
    return out


def first_hits(db: str, names: list[str], tail: str, retmax: int, field: str) -> list[str]:
    for v in names:
        ids = esearch(db, f'"{v}"{field} {tail}', retmax)
        if ids:
            return ids
    return []


def collect(key: str, info: dict, genes: list[str], exclude: set[str]) -> dict:
    base = [info["name"]] if len(info["name"].strip()) >= 3 else []
    syn = info.get("syn") or []
    if isinstance(syn, str):
        import ast
        syn = ast.literal_eval(syn) if syn.startswith("[") else [syn]
    names = list(dict.fromkeys(base + name_variants(info["name"]) + [s for s in syn if s]))[:8]
    oa = "AND open access[filter]"
    rev = first_hits("pmc", names, f'{oa} AND review[Publication Type]', 30, "[Title]") or \
        first_hits("pmc", names, f'{oa} AND review[Publication Type]', 30, "")
    cas = first_hits("pmc", names, f'{oa} AND "case reports"[Publication Type]', 30, "[Title]") or \
        first_hits("pmc", names, f'{oa} AND "case reports"[Publication Type]', 30, "")
    pmc = pmc_docs(list(dict.fromkeys(rev + cas)), exclude)
    abst = pubmed_abstracts(first_hits("pubmed", names, "", 50, "[Title/Abstract]"), exclude)
    gene_ab, gene_pmc = [], []
    for g in genes[:2]:
        gene_ab += pubmed_abstracts(esearch("pubmed", f'{g}[Title/Abstract] AND (patient* OR case OR famil*)', 50), exclude)
        gene_pmc += pmc_docs(esearch("pmc", f'{g}[Title] {oa} AND (patient* OR case OR famil*)', 20), exclude)
    return {"key": key, "name": info["name"], "genes": genes, "pmc": pmc, "abstracts": abst,
            "gene_abstracts": gene_ab, "gene_pmc": gene_pmc}


def texts(rec: dict, base_text: str) -> dict[str, str]:
    t = {"base": base_text}
    t["pmc_more"] = t["base"] + " " + " ".join(d["clinical"] for d in rec["pmc"])
    t["pmc_full"] = t["base"] + " " + " ".join(d["full"] for d in rec["pmc"])
    t["abstracts"] = t["pmc_full"] + " " + " ".join(d["full"] for d in rec["abstracts"])
    t["gene"] = t["abstracts"] + " " + " ".join(d["full"] for d in rec["gene_abstracts"] + rec["gene_pmc"])
    return {k: " " + norm(v) + " " for k, v in t.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/docs", exist_ok=True)
    uni = dict(v2_ids.universe())
    for k in json.load(open(f"{V2}/exclude_nonhuman_diseases.json")):
        uni.pop(k, None)
    cur = defaultdict(set)
    for line in open(CA.HPOA):
        f = line.rstrip("\n").split("\t")
        if len(f) > 10 and f[0].startswith("OMIM:") and f[2] != "NOT" and f[10] == "P":
            cur[f[0]].add(f[3])
    genes = defaultdict(list)
    for line in open("data/external_kg/genes_to_disease.txt"):
        f = line.rstrip("\n").split("\t")
        if f[3].startswith("OMIM:") and f[2] == "MENDELIAN":
            genes[f[3]].append(f[1])
    omim2key = {}
    for o in sorted(cur):
        ks = sorted(v2_ids.gold_keys({o}) & set(uni))
        if ks:
            omim2key[o] = ks[0]
    pool = sorted(omim2key)
    sample = random.Random(a.seed).sample(pool, a.n)
    exclude = set(json.load(open(f"{V2}/exclude_pmids_phenopacket.json")))

    def job(o):
        p = f"{a.out}/docs/{o.replace(':', '_')}.json"
        if os.path.exists(p):
            return
        try:
            rec = collect(omim2key[o], uni[omim2key[o]], genes.get(o, []), exclude)
        except FetchError:
            return
        json.dump(rec, open(p, "w"))
        print(o, len(rec["pmc"]), len(rec["abstracts"]), len(rec["gene_abstracts"]) + len(rec["gene_pmc"]), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, sample))

    from collections import Counter
    use = Counter()
    import glob
    for f in glob.glob(f"{V2}/sources/*.json"):
        for x in json.load(open(f))["docs"]:
            use[x["doc_id"]] += 1
    names = hpo_terms()
    done = [o for o in sample if os.path.exists(f"{a.out}/docs/{o.replace(':', '_')}.json")]
    rng = random.Random(a.seed + 1)
    ctrl = {o: rng.choice([x for x in done if x != o]) for o in done}
    hit = {v: [0, 0] for v in VARIANTS}
    chit = {v: [0, 0] for v in VARIANTS}
    per = []
    for o in done:
        rec = json.load(open(f"{a.out}/docs/{o.replace(':', '_')}.json"))
        p = f"{V2}/sources/{omim2key[o].replace(':', '_')}.json"
        bdocs = [x for x in json.load(open(p))["docs"] if use[x["doc_id"]] < 20] if os.path.exists(p) else []
        tx = texts(rec, " ".join(x["text"] for x in bdocs))
        own, other = cur[o], cur[ctrl[o]] - cur[o]
        row = {"omim": o, "n_base_docs": len(bdocs), "n_pmc": len(rec["pmc"]), "n_abs": len(rec["abstracts"]),
               "n_gene": len(rec["gene_abstracts"]) + len(rec["gene_pmc"])}
        for v in VARIANTS:
            h = sum(any(f" {n} " in tx[v] for n in names.get(x, ())) for x in own)
            c = sum(any(f" {n} " in tx[v] for n in names.get(x, ())) for x in other)
            hit[v][0] += h
            hit[v][1] += len(own)
            chit[v][0] += c
            chit[v][1] += len(other)
            row[v] = round(h / max(1, len(own)), 3)
        per.append(row)
    res = {"diseases": len(done), "sample": a.n, "seed": a.seed,
           "recall": {v: round(100 * hit[v][0] / max(1, hit[v][1]), 1) for v in VARIANTS},
           "control_recall": {v: round(100 * chit[v][0] / max(1, chit[v][1]), 1) for v in VARIANTS},
           "base_zero_doc_diseases": sum(r["n_base_docs"] == 0 for r in per)}
    json.dump({**res, "per_disease": per}, open(f"{a.out}/result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
