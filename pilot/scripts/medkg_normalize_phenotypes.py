#!/usr/bin/env python3
"""Normalize IE'd phenotype strings to HPO terms (with HPO ID where possible).

Strategy:
1. Build HPO term index from data/external_kg/phenotype.hpoa or HPO obo
2. Exact match (lowercase) → HPO ID
3. Fuzzy/synonym match
4. Leave unmatched as raw string (still in KG with provenance, just no HPO ID)

Output: edges_ie.jsonl + 'hpo_id' field where matched.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path("/home/max/Graph-DDXPlus/data/medkg")
EDGES_IE = ROOT / "processed" / "edges_ie.jsonl"
HPO_OBO = Path("/home/max/Graph-DDXPlus/data/external_kg")  # may have hp.obo
ORPHANET_EDGES = ROOT / "processed" / "orphanet_edges.jsonl"
OUT = ROOT / "processed" / "edges_normalized.jsonl"


def build_hpo_index_from_orphanet():
    """Use orphanet_edges.jsonl as a seed: HPO term → HPO ID."""
    idx = {}
    with ORPHANET_EDGES.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            term = e.get("phenotype", "").lower().strip()
            hpo_id = e.get("phenotype_id")
            if term and hpo_id and term not in idx:
                idx[term] = hpo_id
    return idx


def normalize_term(s):
    """Lowercase, strip plurals, common prefixes."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(mild|moderate|severe|acute|chronic)\s+", "", s)
    s = s.rstrip("s")  # crude plural strip
    return s


# Noise patterns — exclude these phenotype strings post-IE
NOISE_PATTERNS = [
    re.compile(r"\b\d+:\d+\b"),           # ratios e.g., 2:1
    re.compile(r"\b\d{4}s?\b"),            # year/decade e.g., 1990s
    re.compile(r"\b(male|female)[-\s]?to[-\s]?(male|female)\b"),
    re.compile(r"\b(ratio|rate|incidence|prevalence|mortality|frequency|odds)\b"),
    re.compile(r"\b\d+\s*(year|month|week|day|%)"),  # numeric durations/percentages
    re.compile(r"\b(more common|less common|usually|often|sometimes|rarely)\b"),
    re.compile(r"\bpredominantly\b"),
    re.compile(r"\b(in|among)\s+(adults|children|elderly|men|women|patients)\b"),
    re.compile(r"\b(predominance|preponderance|distribution|characteristic|feature|presentation|symptom|sign|finding|condition)\b$"),
    re.compile(r"\b(antecedent|preceding|history of|background of)\b"),  # context phrases
]
NOISE_TERMS = {
    "history of present illness", "patient education", "differential diagnosis",
    "clinical features", "physical examination", "history taking",
    "past medical history", "family history", "social history",
    "review of systems", "vital signs", "general appearance",
    "no known", "none", "n/a", "unknown",
    "comparative anatomy", "comparative physiology",
    "male predominance", "female predominance",
    "clinical presentation", "clinical course", "clinical manifestation",
}


def is_noise(term):
    t = term.lower().strip()
    if t in NOISE_TERMS: return True
    if len(t) < 3: return True
    # Tighter length cap — clinical phenotypes are usually short concepts
    if len(t) > 50: return True  # was 80 — too lenient
    # Long phrases (5+ words) are usually descriptive sentences, not phenotype concepts
    word_count = len(t.split())
    if word_count > 6: return True
    for pat in NOISE_PATTERNS:
        if pat.search(t): return True
    # Skip if mostly digits/punctuation
    if sum(1 for ch in t if ch.isalpha()) / max(len(t), 1) < 0.5: return True
    return False


def main():
    if not EDGES_IE.exists():
        print(f"No IE edges yet at {EDGES_IE}; skipping (run medkg_ie_multi_source.py first)")
        return

    print("Building HPO index from Orphanet...")
    hpo_idx = build_hpo_index_from_orphanet()
    print(f"  {len(hpo_idx)} HPO terms indexed")

    n_edges = 0
    n_mapped = 0
    n_filtered = 0
    with EDGES_IE.open() as f, OUT.open("w") as out:
        for line in f:
            line = line.strip()
            if not line: continue
            edge = json.loads(line)
            phen = edge.get("phenotype", "")
            if is_noise(phen):
                n_filtered += 1
                continue
            norm = normalize_term(phen)
            hpo_id = hpo_idx.get(norm) or hpo_idx.get(phen.lower())
            if hpo_id:
                edge["hpo_id"] = hpo_id
                n_mapped += 1
            edge["phenotype_normalized"] = norm
            out.write(json.dumps(edge, ensure_ascii=False) + "\n")
            n_edges += 1
    print(f"\nNormalized {n_edges} edges, filtered noise: {n_filtered}, mapped to HPO: {n_mapped} ({100*n_mapped/max(n_edges,1):.1f}%)")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
