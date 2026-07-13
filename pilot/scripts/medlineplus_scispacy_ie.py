#!/usr/bin/env python3
"""MedlinePlus XML → scispaCy IE.

For each MedlinePlus topic XML:
  1. Extract FullSummary text (rich lay-vocab disease descriptions)
  2. Map filename (topic_slug) → UMLS CUI via MRCONSO search
  3. scispaCy NER + UMLS linker → evidence CUIs
  4. Output edges

Output: /mnt/medkg/processed/edges_medlineplus_scispacy.jsonl
"""
from __future__ import annotations
import re, json, html, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

MLP_DIR = "/mnt/medkg/medlineplus"
OUT = "pilot/data/processed/edges_medlineplus_scispacy.jsonl"
MRCONSO = "/windows/data/umls_subset/MRCONSO.RRF"


def extract_text(xml_str):
    """Extract all FullSummary/title/altTitle content from MedlinePlus XML."""
    # Decode HTML entities first
    xml_str = html.unescape(xml_str)
    # Find FullSummary content
    summaries = re.findall(
        r'<content name="FullSummary">(.*?)</content>',
        xml_str, re.DOTALL
    )
    # Strip remaining tags
    sections = []
    for s in summaries:
        s = re.sub(r'<[^>]+>', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if len(s) > 100:
            sections.append(s)
    return sections


def topic_from_filename(fp):
    """48_xxyy_syndrome → '48 xxyy syndrome'."""
    stem = fp.stem  # e.g. "klinefelter_syndrome"
    return stem.replace('_', ' ')


def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def main():
    t0 = time.time()
    files = sorted(Path(MLP_DIR).glob('*.xml'))
    print(f"MedlinePlus files: {len(files)}", flush=True)

    # Get all topic names from filenames
    topic_to_file = {}
    for fp in files:
        topic = topic_from_filename(fp)
        topic_to_file[norm(topic)] = fp
    print(f"Unique topic names: {len(topic_to_file)}", flush=True)

    # Map topics to CUIs via MRCONSO
    print("Scanning MRCONSO for topic → CUI mapping...", flush=True)
    topic_to_cui = {}
    n = 0
    with open(MRCONSO) as f:
        for line in f:
            n += 1
            parts = line.split('|')
            if len(parts) < 15: continue
            cui, lang, term = parts[0], parts[1], parts[14]
            if lang != 'ENG': continue
            t_norm = norm(term)
            if t_norm in topic_to_file and t_norm not in topic_to_cui:
                topic_to_cui[t_norm] = cui
            if n > 10_000_000 and len(topic_to_cui) == len(topic_to_file):
                break
    print(f"  scanned {n//1_000_000}M lines, mapped {len(topic_to_cui)}/{len(topic_to_file)} topics",
          flush=True)

    # Load scispaCy
    print("Loading scispaCy...", flush=True)
    import spacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "k": 1, "threshold": 0.85
    })

    # Process files
    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    n_processed = 0
    t_proc = time.time()
    with open(OUT, 'w') as fout:
        for fp in files:
            topic = topic_from_filename(fp)
            dcui = topic_to_cui.get(norm(topic))
            if not dcui:
                continue
            try:
                xml_str = open(fp, encoding='utf-8', errors='replace').read()
            except: continue
            sections = extract_text(xml_str)
            if not sections: continue

            seen = set()
            for sect in sections:
                doc = nlp(sect[:5000])
                for ent in doc.ents:
                    if ent._.kb_ents:
                        ec = ent._.kb_ents[0][0]
                        if ec == dcui: continue
                        if ec in seen: continue
                        seen.add(ec)
                        edge = {
                            "disease": topic,
                            "umls_cui": dcui,
                            "evidence_cui": ec,
                            "phenotype": ent.text,
                            "source": "medlineplus_scispacy",
                            "source_id": fp.stem,
                            "extracted_by": "scispacy",
                        }
                        fout.write(json.dumps(edge, ensure_ascii=False) + '\n')
                        n_out += 1
            n_processed += 1
            if n_processed % 100 == 0:
                print(f"  {n_processed} processed, {n_out} edges ({time.time()-t_proc:.0f}s)",
                      flush=True)
    print(f"\nDone. {n_processed} files processed, {n_out} edges → {OUT}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
