#!/usr/bin/env python3
"""Stage 1: link Wikipedia disease titles to UMLS CUI via scispaCy.

Saves cache to /windows/data/medkg/processed/wikipedia_cui_cache.jsonl
(one JSON per record: {disease, umls_cui, text, source_file, page_id}).

Run this BEFORE the vLLM IE stage so scispaCy's CUDA init does not
collide with vLLM in the same process.
"""
from __future__ import annotations
import os, sys, json, glob
from pathlib import Path

# Force scispaCy onto CPU — vLLM will need GPU later in a separate process
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

WIKI_DIR = Path("/windows/data/medkg/wikipedia")
CACHE_PATH = MEDKG_ROOT / "processed" / "wikipedia_cui_cache.jsonl"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_records():
    out = []
    for fp in sorted(glob.glob(str(WIKI_DIR / "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for pid, page in d.get("query", {}).get("pages", {}).items():
            title = page.get("title")
            extract = page.get("extract", "")
            if not title or not extract or len(extract) < 200:
                continue
            out.append({
                "disease": title,
                "source_file": Path(fp).stem,
                "page_id": pid,
                "text": extract[:2500],
            })
    return out


def main():
    records = extract_records()
    print(f"Loaded {len(records)} Wikipedia records", flush=True)

    print("Loading scispaCy en_core_sci_lg + UMLS linker (CPU)...", flush=True)
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401 — registers factory
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "k": 1,
    })

    mapped = 0
    with CACHE_PATH.open("w") as out:
        for i, r in enumerate(records):
            cui = None
            # 1) try linking the title directly
            doc = nlp(r["disease"])
            for ent in doc.ents:
                if ent._.kb_ents:
                    cui = ent._.kb_ents[0][0]
                    break
            # 2) fallback: NER on text first sentence, find disease-mention
            if not cui:
                doc2 = nlp(r["text"][:500])
                for ent in doc2.ents:
                    if r["disease"].lower() in ent.text.lower() or \
                       ent.text.lower() in r["disease"].lower():
                        if ent._.kb_ents:
                            cui = ent._.kb_ents[0][0]; break
            if cui:
                r["umls_cui"] = cui
                mapped += 1
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(records)}  mapped={mapped}", flush=True)
    print(f"Done. Mapped {mapped}/{len(records)} → {CACHE_PATH}", flush=True)


if __name__ == "__main__":
    main()
