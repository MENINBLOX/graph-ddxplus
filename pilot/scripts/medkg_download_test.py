#!/usr/bin/env python3
"""Sanity test for medkg_download_all: 5 representative diseases × 4 sources."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/max/Graph-DDXPlus/pilot/scripts")
from medkg_download_all import crawl_ncbi_book, crawl_medlineplus, crawl_wikipedia, OUT
import time, json

TEST_DISEASES = [
    "Pneumonia",         # common acute (DDXPlus)
    "Atrial fibrillation",  # cardiology common (DDXPlus)
    "Marfan syndrome",   # rare (RareBench / OMIM)
    "Tuberculosis",      # infectious common (DDXPlus)
    "Influenza",         # primary care (SymCat)
]

results = {d: {} for d in TEST_DISEASES}

sp_dir = OUT / "statpearls"
gr_dir = OUT / "genereviews"
mp_dir = OUT / "medlineplus"
wp_dir = OUT / "wikipedia"

for d in TEST_DISEASES:
    print(f"\n=== {d} ===")
    for src, fn in [
        ("statpearls", lambda: crawl_ncbi_book(d, "statpearls", sp_dir)),
        ("genereviews", lambda: crawl_ncbi_book(d, "gene", gr_dir)),
        ("medlineplus", lambda: crawl_medlineplus(d, mp_dir)),
        ("wikipedia",  lambda: crawl_wikipedia(d, wp_dir)),
    ]:
        try:
            r = fn()
            results[d][src] = "OK" if r else "MISS"
            print(f"  {src:12s} {results[d][src]} {r if r else ''}")
        except Exception as e:
            results[d][src] = f"ERR: {e}"
            print(f"  {src:12s} ERR {e}")
        time.sleep(0.5)

print("\n\n=== Summary ===")
for d, sd in results.items():
    print(f"  {d:25s} | " + " | ".join(f"{k}:{v[:3] if isinstance(v,str) else 'NA':>3s}" for k, v in sd.items()))
