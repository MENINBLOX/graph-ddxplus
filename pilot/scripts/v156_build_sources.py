"""S1 SOURCE ASSEMBLY at scale — one symptom-rich text blob per disease for attribute-IE.

For each seed CUI, assemble a bounded, benchmark-blind clinical text from the free-text sources
we have locally, preferring symptom-bearing passages:
  - PubMed abstracts   data/medkg/pubmed/{cui}.jsonl   (title + abstract per article)
  - encyclopedic sections  processed/sections.jsonl     (statpearls/wikipedia/medlineplus, by CUI)

Selection is deterministic: passages are ranked by a symptom-cue score (count of clinical/finding
cue words) and the top passages are concatenated up to a word budget, so the LLM sees the most
symptom-descriptive text first. Only disease NAMES/텍스트 are used — no benchmark labels (§5).

NOTE: structured HPO annotations (phenotype.hpoa) are a rare-disease phenotype FLOOR but carry no
free-text attributes and raise a RareBench-circularity question; whether to feed them here is a
pending design decision (see docs/source_coverage_audit.md §4). This builder uses free text only.

Output: pilot/cache/prod/sources/{cui}.json = {disease, cui, title, text, provenance}.
Usage: python v156_build_sources.py [--limit N] [--word-budget 1800] [--out-dir ...]"""
from __future__ import annotations
import json, os, re, glob, argparse
from collections import defaultdict

PUBMED_DIR = "data/medkg/pubmed"
SECTIONS = "data/medkg/processed/sections.jsonl"
SEED = "data/medkg/seeds/combined_seed.jsonl"
OUT_DIR = "pilot/cache/prod/sources"

# clinical/finding cue words — passages with more of these describe symptoms, not etiology/genetics.
CUE = set("symptom symptoms sign signs present presents presenting presentation manifest manifestation "
          "manifestations feature features finding findings complain complains complained pain painful "
          "fever swelling swollen rash cough dyspnea dyspnoea fatigue weakness nausea vomiting diarrhea "
          "diarrhoea seizure seizures bleeding numbness tingling dizziness headache discomfort tenderness "
          "characterized experience experiences develops develop developing onset acute chronic severe "
          "mild moderate difficulty loss abnormal impaired reduced increased episodes".split())
# sections that are symptom-oriented get a bonus; etiology/epidemiology get a penalty.
SEC_BONUS = re.compile(r'symptom|sign|clinical|present|manifest|feature|phenotype|physical', re.I)
SEC_PENALT = re.compile(r'epidemiolog|etiolog|patholog|genetic|molecular|treatment|management|'
                        r'prognosis|history|reference|epidemiology|diagnosis|differential', re.I)
_WORD = re.compile(r"[a-z]+")


def cue_score(text: str) -> int:
    return sum(1 for w in _WORD.findall(text.lower()) if w in CUE)


def load_sections_by_cui():
    by = defaultdict(list)
    if not os.path.exists(SECTIONS):
        return by
    for line in open(SECTIONS, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        cui = d.get("umls_cui")
        txt = (d.get("text") or "").strip()
        if cui and txt:
            by[cui].append(d)
    return by


def pubmed_passages(cui: str):
    p = f"{PUBMED_DIR}/{cui}.jsonl"
    out = []
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return out
    for line in open(p, encoding="utf-8"):
        try:
            a = json.loads(line)
        except Exception:
            continue
        ab = (a.get("abstract") or "").strip()
        if ab:
            out.append({"text": ab, "pmid": a.get("pmid"), "kind": "pubmed"})
    return out


def rank_passages(passages, section_recs):
    scored = []
    for pas in passages:
        scored.append((cue_score(pas["text"]), pas))
    for d in section_recs:
        s = cue_score(d["text"])
        name = d.get("section_name", "") + " " + d.get("chapter_title", "")
        if SEC_BONUS.search(name): s += 8
        if SEC_PENALT.search(name): s -= 6
        scored.append((s, {"text": d["text"], "kind": d.get("source", "section"),
                           "ref": d.get("source_id"), "section": d.get("section_name")}))
    # symptom-richest first; stable for reproducibility
    scored.sort(key=lambda x: -x[0])
    return scored


def assemble(name, passages, section_recs, budget):
    scored = rank_passages(passages, section_recs)
    chosen, prov, used = [], [], 0
    for score, pas in scored:
        if score <= 0 and used > 0:
            break  # once we hit non-symptom passages and already have content, stop
        words = pas["text"].split()
        if not words:
            continue
        take = words[: max(0, budget - used)]
        if not take:
            break
        chosen.append(" ".join(take))
        used += len(take)
        prov.append({k: pas.get(k) for k in ("kind", "pmid", "ref", "section") if pas.get(k)})
        if used >= budget:
            break
    return " ".join(chosen).strip(), prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--word-budget", type=int, default=1800)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", default=SEED)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    seeds = [json.loads(l) for l in open(a.seed, encoding="utf-8") if l.strip()]
    if a.limit:
        seeds = seeds[: a.limit]
    print(f"loading sections...", flush=True)
    sec_by = load_sections_by_cui()
    print(f"sections for {len(sec_by):,} CUIs. assembling {len(seeds):,} diseases", flush=True)

    n_written = n_empty = 0
    wlens = []
    for i, d in enumerate(seeds):
        cui, name = d["cui"], d["name"]
        text, prov = assemble(name, pubmed_passages(cui), sec_by.get(cui, []), a.word_budget)
        if not text:
            n_empty += 1
            continue
        json.dump({"disease": name, "cui": cui, "title": name, "text": text, "provenance": prov},
                  open(f"{a.out_dir}/{cui}.json", "w", encoding="utf-8"), ensure_ascii=False)
        n_written += 1
        wlens.append(len(text.split()))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(seeds)}  written={n_written} empty={n_empty}", flush=True)
    med = sorted(wlens)[len(wlens) // 2] if wlens else 0
    print(f"\nDONE. written={n_written} ({n_written/len(seeds):.1%}) empty(no source)={n_empty} "
          f"median_words={med} -> {a.out_dir}")


if __name__ == "__main__":
    main()
