#!/usr/bin/env python3
"""v92 — Enhanced UMLS direct mapping with multi-substring strategy.

v87 lesson: 8.5% direct match because LLM produces verbose phrases like
"sudden onset of severe knee pain" that don't match UMLS standardized
"Knee pain" exactly.

v92 strategy:
- Generate ALL meaningful n-grams from LLM phrase
- Strip qualifier prefixes/suffixes recursively
- Try substring match starting from longest
- Stop at first UMLS hit

학술적 정당: UMLS standard ontology lookup, no benchmark coupling.
SymCat/RareBench는 UMLS CUIs를 사용하지만, 우리가 사용하는 lookup table은
UMLS 전체 phenotype CUIs (~500K), not benchmark-specific list.
"""
from __future__ import annotations
import json, re, argparse
from pathlib import Path
from collections import defaultdict


# Qualifier prefixes to strip (recursive)
PREFIX_QUALIFIERS = re.compile(
    r"^(severe|mild|moderate|acute|chronic|sudden onset of|sudden|recurrent|"
    r"persistent|intermittent|episodic|continuous|gradual|progressive|"
    r"transient|early|late|new onset|worsening|"
    r"feeling of|sensation of|reports of|history of|complaints? of|"
    r"sense of|presence of|episode of|episodes of|"
    r"mild to moderate|moderate to severe|"
    r"left|right|bilateral|unilateral|"
    r"increased|decreased|reduced|excessive|"
    r"localized|generalized|diffuse|focal|specific|"
    r"painful|tender|swollen|inflamed|"
    r"visible|palpable|audible|"
    r"a |an |the |"
    r"associated |concurrent |"
    r")\s+", re.IGNORECASE)

SUFFIX_QUALIFIERS = re.compile(
    r"\s+(on (palpation|examination|exertion|movement|coughing|breathing)|"
    r"with (movement|exertion|activity|breathing|coughing)|"
    r"during (rest|exertion|sleep|night|exam|stair climbing|walking|running)|"
    r"at (rest|night|onset)|"
    r"following (injury|trauma|exposure)|"
    r"of the [a-z ]+|"
    r"in the [a-z ]+|"
    r"radiating to [a-z ]+|"
    r"\([^)]*\))\.?\s*$", re.IGNORECASE)


def normalize(s):
    s = s.lower().strip().rstrip(".,;:")
    # Remove parenthetical content
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def generate_candidates(phrase):
    """Generate candidate substrings from longest to shortest, after stripping qualifiers."""
    p = normalize(phrase)
    cands = []
    # Original normalized
    cands.append(p)
    # Strip prefixes/suffixes iteratively
    prev = None
    cur = p
    while cur != prev:
        prev = cur
        cur = PREFIX_QUALIFIERS.sub("", cur).strip()
        cur = SUFFIX_QUALIFIERS.sub("", cur).strip()
        if cur and cur != prev:
            cands.append(cur)
    # Final word-trimming variants (last 2-4 words)
    words = cur.split()
    if len(words) >= 4:
        for n in [4, 3, 2]:
            if len(words) > n:
                cands.append(" ".join(words[-n:]))
                cands.append(" ".join(words[:n]))
    # Dedup preserving order (longest first generally)
    seen = set(); out = []
    for c in cands:
        if c and len(c) >= 3 and c not in seen:
            seen.add(c); out.append(c)
    return out


def load_umls_phenotype_strings(mrconso_path, mrsty_path):
    phen_tuis = {"T033", "T184", "T046", "T037", "T048", "T049", "T191", "T190"}
    phen_cuis = set()
    with open(mrsty_path) as f:
        for line in f:
            parts = line.split("|")
            if len(parts) >= 2 and parts[1] in phen_tuis:
                phen_cuis.add(parts[0])
    print(f"  Phen CUIs: {len(phen_cuis):,}", flush=True)

    print(f"  Scanning MRCONSO...", flush=True)
    str2cuis = defaultdict(set)
    n = 0
    with open(mrconso_path) as f:
        for line in f:
            n += 1
            parts = line.split("|")
            if len(parts) < 15: continue
            c, lang = parts[0], parts[1]
            if lang != "ENG": continue
            if c not in phen_cuis: continue
            s = parts[14].strip().lower()
            if not s or len(s) < 3 or len(s) > 80: continue
            s = re.sub(r"\s+", " ", s)
            str2cuis[s].add(c)
            if n % 2_000_000 == 0:
                print(f"    {n//1_000_000}M lines, {len(str2cuis):,} strings",
                      flush=True)
    print(f"  Done: {len(str2cuis):,} unique phen strings", flush=True)
    return str2cuis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie_path", default="pilot/data/cache/v85_exhaustive_ie.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mrconso", default="/windows/data/umls_subset/MRCONSO.RRF")
    ap.add_argument("--mrsty", default="/windows/data/umls_subset/MRSTY.RRF")
    args = ap.parse_args()

    print("Loading UMLS phenotype strings...", flush=True)
    str2cuis = load_umls_phenotype_strings(args.mrconso, args.mrsty)

    records = [json.loads(l) for l in open(args.ie_path)]
    print(f"IE records: {len(records)}", flush=True)

    name_to_cui = {}
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    for dn, info in icd.items():
        if info.get("cui"): name_to_cui[dn] = info["cui"]
    sym_dis = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    for dn, info in sym_dis.items():
        if info.get("umls_cui") and dn not in name_to_cui:
            name_to_cui[dn] = info["umls_cui"]

    n_total = n_mapped = 0
    n_records = 0; n_edges = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout:
        for r in records:
            dn = r["disease"]
            dcui = name_to_cui.get(dn)
            if not dcui: continue
            edges = {}
            for phen_text, info in r["phenotypes"].items():
                n_total += 1
                cands = generate_candidates(phen_text)
                cui = None
                for c in cands:
                    if c in str2cuis:
                        cuis = str2cuis[c]
                        if not cuis: continue
                        # Smallest CUI for determinism
                        cui = sorted(cuis)[0]
                        break
                if not cui: continue
                if cui == dcui: continue
                p = info.get("prob", 0.0)
                if cui in edges: edges[cui] = max(edges[cui], p)
                else: edges[cui] = p
                n_mapped += 1
            if not edges: continue
            fout.write(json.dumps({
                "disease": dn, "dcui": dcui, "source": r.get("source",""),
                "edges": edges
            }) + "\n")
            n_records += 1
            n_edges += len(edges)
    print(f"\nMapped {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%) phen texts")
    print(f"Saved {n_records} disease records, {n_edges} edges → {args.out}")


if __name__ == "__main__":
    main()
