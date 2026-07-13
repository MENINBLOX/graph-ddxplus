#!/usr/bin/env python3
"""v96 — Aggressive UMLS direct mapping.

v92 vs v96 differences:
1. Strip unclosed parentheses: "moon facies (round face" → "moon facies"
2. Strip trailing comma-list: "infections (e.g., pneumonia, sinusitis" → "infections"
3. Aggressive word truncation: try last 1, 2, 3 words as fallback
4. Simple plural normalization: "lesions" → "lesion"
5. Try first-word and last-word semantic anchor (e.g., "moon facies" → "facies")
"""
from __future__ import annotations
import json, re, argparse
from pathlib import Path
from collections import defaultdict


# Same prefix/suffix qualifiers as v92
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
    r"unexplained|chronic|"
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

# NEW: strip unclosed paren to end of string
UNCLOSED_PAREN = re.compile(r"\s*\([^)]*$")


def normalize(s):
    s = s.lower().strip().rstrip(".,;:")
    # Closed parenthetical
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    # Unclosed parenthetical (NEW)
    s = UNCLOSED_PAREN.sub("", s).strip()
    # Remove "e.g.", "i.e."
    s = re.sub(r"\b(e\.g\.|i\.e\.|etc\.)\b.*$", "", s).strip()
    # Comma-list trailing: "x, y, z" → "x"
    if ',' in s:
        s2 = s.split(',')[0].strip()
        if len(s2) >= 3: s = s2
    s = re.sub(r"\s+", " ", s)
    return s


def maybe_singular(word):
    """Crude singular form."""
    if word.endswith("ies") and len(word) > 4: return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4: return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"): return word[:-1]
    return word


def generate_candidates(phrase):
    p = normalize(phrase)
    cands = [p]
    # Iteratively strip prefixes/suffixes
    prev = None; cur = p
    while cur != prev:
        prev = cur
        cur = PREFIX_QUALIFIERS.sub("", cur).strip()
        cur = SUFFIX_QUALIFIERS.sub("", cur).strip()
        cur = re.sub(r"\s+", " ", cur)
        if cur and cur != prev: cands.append(cur)
    # SKIP word-trimming (caused false positives on PhenoBrain)
    # Singular versions only — safer
    sing_versions = []
    for c in cands:
        ws = c.split()
        if ws:
            ws_sing = [maybe_singular(w) for w in ws]
            if ws_sing != ws:
                sing_versions.append(" ".join(ws_sing))
    cands.extend(sing_versions)
    # Dedup preserving order
    seen = set(); out = []
    for c in cands:
        if c and len(c) >= 3 and c not in seen:
            seen.add(c); out.append(c)
    return out


def load_umls_phenotype_strings(mrconso_path, mrsty_path):
    phen_tuis = {"T033","T184","T046","T037","T048","T049","T191","T190"}
    phen_cuis = set()
    with open(mrsty_path) as f:
        for line in f:
            parts=line.split("|")
            if len(parts)>=2 and parts[1] in phen_tuis: phen_cuis.add(parts[0])
    print(f"  Phen CUIs: {len(phen_cuis):,}", flush=True)

    print(f"  Scanning MRCONSO...", flush=True)
    str2cuis = defaultdict(set)
    n=0
    with open(mrconso_path) as f:
        for line in f:
            n+=1
            parts=line.split("|")
            if len(parts)<15: continue
            c, lang = parts[0], parts[1]
            if lang != "ENG": continue
            if c not in phen_cuis: continue
            s = parts[14].strip().lower()
            if not s or len(s)<3 or len(s)>80: continue
            s = re.sub(r"\s+", " ", s)
            str2cuis[s].add(c)
            if n%2_000_000==0:
                print(f"    {n//1_000_000}M lines, {len(str2cuis):,} strings", flush=True)
    print(f"  Done: {len(str2cuis):,} strings", flush=True)
    return str2cuis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mrconso", default="/windows/data/umls_subset/MRCONSO.RRF")
    ap.add_argument("--mrsty", default="/windows/data/umls_subset/MRSTY.RRF")
    args=ap.parse_args()

    print("Loading UMLS phenotype strings...", flush=True)
    str2cuis = load_umls_phenotype_strings(args.mrconso, args.mrsty)
    records = [json.loads(l) for l in open(args.ie_path)]
    print(f"IE records: {len(records)}", flush=True)

    name_to_cui={}
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd=json.load(f)
    for dn,info in icd.items():
        if info.get("cui"): name_to_cui[dn]=info["cui"]
    sym_dis=json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    for dn,info in sym_dis.items():
        if info.get("umls_cui") and dn not in name_to_cui:
            name_to_cui[dn]=info["umls_cui"]
    try:
        rb_dis=json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]
        for did,info in rb_dis.items():
            dn = info.get("umls_name") or did
            if info.get("umls_cui") and dn and dn not in name_to_cui:
                name_to_cui[dn]=info["umls_cui"]
    except Exception as e: print(f"  rare disease mapping skip: {e}", flush=True)

    n_total=n_mapped=0
    n_records=0; n_edges=0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w") as fout:
        for r in records:
            dn=r["disease"]
            dcui=name_to_cui.get(dn)
            if not dcui: continue
            edges={}
            for phen_text, info in r["phenotypes"].items():
                n_total+=1
                cands=generate_candidates(phen_text)
                cui=None
                for c in cands:
                    if c in str2cuis:
                        cuis=str2cuis[c]
                        if not cuis: continue
                        cui=sorted(cuis)[0]; break
                if not cui: continue
                if cui==dcui: continue
                p = info.get("prob",0)
                if cui in edges: edges[cui]=max(edges[cui], p)
                else: edges[cui]=p
                n_mapped+=1
            if not edges: continue
            fout.write(json.dumps({"disease":dn, "dcui":dcui, "source":r.get("source",""), "edges":edges})+"\n")
            n_records+=1
            n_edges+=len(edges)
    print(f"\nMapped {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%) phen texts")
    print(f"Saved {n_records} disease records, {n_edges} edges → {args.out}")


if __name__=="__main__":
    main()
