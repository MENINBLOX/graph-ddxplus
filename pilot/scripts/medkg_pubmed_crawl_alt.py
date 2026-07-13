#!/usr/bin/env python3
"""PubMed crawl with multiple alternate search names per CUI (deeper coverage).

Reads seeds with `alt_names` list. For each, runs multiple searches and merges.
Saves to $MEDKG_ROOT/pubmed_alt/{cui}.jsonl with up to 50 abstracts per CUI.

Usage: python medkg_pubmed_crawl_alt.py [--seed FILE] [--abstracts_per_cui K]
"""
from __future__ import annotations
import sys, os, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_pubmed_crawl import esearch, efetch, log

OUT_DIR = MEDKG_ROOT / "pubmed_alt"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=str(MEDKG_ROOT / "seeds" / "ddxplus_expanded_search.jsonl"))
    ap.add_argument("--abstracts_per_cui", type=int, default=50)
    args = ap.parse_args()

    seeds = []
    with open(args.seed) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            seeds.append(json.loads(line))
    log(f"Alt crawl. Seeds: {len(seeds)}, target per CUI: {args.abstracts_per_cui}")

    rate = 0.4
    n_ok = 0
    for s in seeds:
        cui = s["cui"]
        alts = s.get("alt_names") or [s["name"]]
        out_path = OUT_DIR / f"{cui}.jsonl"
        if out_path.exists() and out_path.stat().st_size > 100:
            log(f"  skip existing {cui}")
            continue
        all_pmids = set()
        for q in alts:
            query = f'"{q}"[Title/Abstract] OR "{q}"[MeSH Terms]'
            try:
                pmids = esearch(query, args.abstracts_per_cui)
                all_pmids.update(pmids)
                time.sleep(rate)
                if len(all_pmids) >= args.abstracts_per_cui: break
            except Exception as e:
                log(f"  ERR esearch {cui} ({q}): {e}")
        pmids = list(all_pmids)[:args.abstracts_per_cui]
        if not pmids:
            log(f"  empty {cui}")
            out_path.write_text("")
            continue
        try:
            articles = efetch(pmids)
            time.sleep(rate)
            if articles:
                with out_path.open("w") as o:
                    for a in articles:
                        a["cui"] = cui
                        a["disease_name"] = alts[0]
                        a["search_alts"] = alts
                        o.write(json.dumps(a, ensure_ascii=False) + "\n")
                n_ok += 1
                log(f"  {cui}: {len(articles)} abstracts ({alts[0]})")
        except Exception as e:
            log(f"  ERR efetch {cui}: {e}")
    log(f"Done. ok={n_ok}/{len(seeds)}")


if __name__ == "__main__":
    main()
