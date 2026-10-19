"""Corpus v3 collection: deeper, same caps for every disease (research design §9.27-9.28).

Per disease, by name (and causal gene for Mendelian diseases), with the caps fixed a priori:
  PMC open-access reviews 30 + case reports 30: clinical sections (12k cap), whole body, and tables
  PubMed abstracts 50 by name in title/abstract
  causal gene (HPO genes_to_disease, up to 2 genes): PubMed abstracts 50 + PMC open-access 20
Every phenopacket-store case's source PMID (release 0.1.27 and later) is excluded. Output is one
gzip JSON per disease, so later steps choose which parts to read without fetching again.

    set -a; . ./.env; set +a
    python collect_v3.py --out pilot/cache/v3/raw [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_ids  # noqa: E402
from collection_pretest import efetch, esearch, first_hits, pubmed_abstracts  # noqa: E402
from deep_fetch import DROP_SEC, KEEP_SEC, name_variants  # noqa: E402

V2 = "pilot/cache/v2"
OA = "AND open access[filter]"


def case_pmids() -> set[str]:
    z = zipfile.ZipFile("data/external_benchmarks/phenopacket_store/all_phenopackets.zip")
    names = z.namelist() + os.listdir("data/confirm/phenopacket_new")
    return {m.group(1) for n in names if (m := re.search(r"PMID_(\d+)", n))}


def pmc_full(ids: list[str], exclude: set[str]) -> list[dict]:
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
            tables = []
            for tw in art.iter("table-wrap"):
                cap = " ".join(" ".join(" ".join(x.itertext()) for x in tw.iter("caption")).split())
                rows = [[" ".join("".join(c.itertext()).split()) for c in tr if c.tag in ("td", "th")]
                        for tr in tw.iter("tr")]
                tables.append({"caption": cap, "rows": [r for r in rows if any(r)]})
            tt = art.find(".//title-group/article-title")
            yr = art.find(".//pub-date/year")
            out.append({"id": "PMC" + aid.get("pmc", aid.get("pmcid", "")).lstrip("PMC"), "pmid": pmid,
                        "title": " ".join("".join(tt.itertext()).split()) if tt is not None else "",
                        "year": yr.text if yr is not None else "",
                        "clinical": " ".join(clin)[:12000], "full": " ".join(full), "tables": tables})
    return out


def collect(key: str, info: dict, genes: list[str], exclude: set[str]) -> dict:
    syn = info.get("syn") or []
    if isinstance(syn, str):
        syn = ast.literal_eval(syn) if syn.startswith("[") else [syn]
    base = [info["name"]] if len(info["name"].strip()) >= 3 else []
    names = list(dict.fromkeys(base + name_variants(info["name"]) + [s for s in syn if s]))[:8]
    rev = first_hits("pmc", names, f"{OA} AND review[Publication Type]", 30, "[Title]") or \
        first_hits("pmc", names, f"{OA} AND review[Publication Type]", 30, "")
    cas = first_hits("pmc", names, f'{OA} AND "case reports"[Publication Type]', 30, "[Title]") or \
        first_hits("pmc", names, f'{OA} AND "case reports"[Publication Type]', 30, "")
    rec = {"key": key, "name": info["name"], "names": names, "genes": genes,
           "pmc": pmc_full(list(dict.fromkeys(rev + cas)), exclude),
           "abstracts": pubmed_abstracts(first_hits("pubmed", names, "", 50, "[Title/Abstract]"), exclude),
           "gene_abstracts": [], "gene_pmc": []}
    for g in genes[:2]:
        rec["gene_abstracts"] += pubmed_abstracts(esearch("pubmed", f"{g}[Title/Abstract] AND (patient* OR case OR famil*)", 50), exclude)
        rec["gene_pmc"] += pmc_full(esearch("pmc", f"{g}[Title] {OA} AND (patient* OR case OR famil*)", 20), exclude)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    uni = dict(v2_ids.universe())
    for k in json.load(open(f"{V2}/exclude_nonhuman_diseases.json")):
        uni.pop(k, None)
    genes = defaultdict(list)
    for line in open("data/external_kg/genes_to_disease.txt"):
        f = line.rstrip("\n").split("\t")
        if len(f) > 3 and f[2] == "MENDELIAN":
            for k in v2_ids.gold_keys({f[3]}):
                if f[1] not in genes[k]:
                    genes[k].append(f[1])
    exclude = case_pmids()
    keys = [k for k in sorted(uni) if not os.path.exists(f"{a.out}/{k.replace(':', '_')}.json.gz")]
    if a.limit:
        keys = keys[:a.limit]
    print(f"collect v3: {len(keys)} diseases to fetch, {len(exclude)} case PMIDs excluded", flush=True)

    def job(k):
        try:
            rec = collect(k, uni[k], genes.get(k, []), exclude)
        except Exception as e:  # noqa: BLE001 - one disease's failure must not stop the run; reruns retry it
            print("FAIL", k, type(e).__name__, flush=True)
            return
        tmp = f"{a.out}/{k.replace(':', '_')}.json.gz.tmp"
        with gzip.open(tmp, "wt") as fo:
            json.dump(rec, fo)
        os.replace(tmp, tmp[:-4])
        print(k, len(rec["pmc"]), len(rec["abstracts"]), len(rec["gene_abstracts"]) + len(rec["gene_pmc"]), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, keys))
    print("COLLECT_V3_DONE", flush=True)


if __name__ == "__main__":
    main()
