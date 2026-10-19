#!/usr/bin/env python3
"""Synonym-retry pass for PubMed source gaps.

The main crawl queries each disease by its PREFERRED name only; diseases whose preferred name is
obscure or an administrative ICD bucket return nothing. This pass retries the remaining gap CUIs
using their alternate UMLS names (MRCONSO English synonyms), which often include a more
literature-common term. Same output contract as medkg_pubmed_crawl_fast (per-CUI JSONL, empty
marker = still no hit). Only genuine gaps (0-byte files) are retried, so re-running is idempotent.

Usage: MEDKG_ROOT=... NCBI_API_KEY=... python medkg_pubmed_synonym_retry.py [--workers 6] [--max-syn 4]
"""
from __future__ import annotations
import sys, os, json, time, argparse, threading, requests
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_pubmed_crawl_fast import esearch, efetch, log, PUBMED_DIR

MRCONSO = "data/umls_subset/MRCONSO.RRF"
SEED = str(MEDKG_ROOT / "seeds" / "combined_seed.jsonl")


def load_gap_and_synonyms(seed_path, max_syn):
    seeds = [json.loads(l) for l in open(seed_path) if l.strip()]
    pref = {d["cui"]: d["name"] for d in seeds}
    # gap = seed CUIs whose pubmed file is missing or 0-byte
    gap = set()
    for d in seeds:
        p = PUBMED_DIR / f"{d['cui']}.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            gap.add(d["cui"])
    # collect English synonyms for gap CUIs from MRCONSO (skip the preferred name; keep distinct)
    syns = defaultdict(list)
    seen = defaultdict(set)
    for line in open(MRCONSO, encoding="utf-8"):
        p = line.split("|")
        cui, lat, st = p[0], p[1], p[14]
        if lat != "ENG" or cui not in gap:
            continue
        s = st.strip()
        low = s.lower()
        if not s or low == pref.get(cui, "").lower() or low in seen[cui]:
            continue
        if len(syns[cui]) < max_syn:
            syns[cui].append(s); seen[cui].add(low)
    return gap, syns


def handle(cui, names, k):
    for name in names:
        pmids = esearch(f'"{name}"[Title/Abstract] OR "{name}"[MeSH Terms]', k)
        if not pmids:
            pmids = esearch(name, k)
        arts = [a for a in efetch(pmids) if a.get("abstract", "").strip()] if pmids else []
        if arts:
            out = PUBMED_DIR / f"{cui}.jsonl"; tmp = out.with_suffix(".jsonl.tmp")
            with tmp.open("w") as fh:
                for a in arts:
                    a["cui"] = cui; a["disease_name"] = name
                    fh.write(json.dumps(a, ensure_ascii=False) + "\n")
            tmp.replace(out)
            return "ok"
    return "empty"  # leave existing empty marker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-syn", type=int, default=4)
    ap.add_argument("--abstracts_per_cui", type=int, default=20)
    a = ap.parse_args()
    gap, syns = load_gap_and_synonyms(SEED, a.max_syn)
    targets = [(c, syns[c]) for c in gap if syns.get(c)]
    log(f"SYNONYM retry: gap={len(gap)} with-synonyms={len(targets)} (workers={a.workers})")
    n = {"ok": 0, "empty": 0}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(handle, c, names, a.abstracts_per_cui): c for c, names in targets}
        for i, f in enumerate(as_completed(futs)):
            n[f.result()] += 1
            if (i + 1) % 200 == 0:
                log(f"  {i+1}/{len(targets)} recovered={n['ok']}")
    log(f"Final synonym-retry: recovered={n['ok']} still-empty={n['empty']}")


if __name__ == "__main__":
    main()
