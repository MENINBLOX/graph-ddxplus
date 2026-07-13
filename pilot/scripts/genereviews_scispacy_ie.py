#!/usr/bin/env python3
"""GeneReviews IE via scispaCy NER (no LLM needed, fast).

For each GeneReviews HTML:
  1. Extract text (strip tags)
  2. Map disease name → UMLS CUI via MRCONSO lookup
  3. Run scispaCy NER + UMLS linker on clinical sections
  4. Output disease-phenotype edges (with category inference from TUI)

Output: /mnt/medkg/processed/edges_genereviews_scispacy.jsonl
"""
from __future__ import annotations
import re, json, html, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

GR_DIR = "/mnt/medkg/genereviews"
OUT = "/mnt/medkg/processed/edges_genereviews_scispacy.jsonl"
MRCONSO = "/windows/data/umls_subset/MRCONSO.RRF"
MRSTY = "/windows/data/umls_subset/MRSTY.RRF"


def extract_clinical_sections(html_str):
    """Extract text + isolate Clinical/Symptom/Sign/Diagnosis sections."""
    html_str = re.sub(r'<script[^>]*>.*?</script>', '', html_str, flags=re.DOTALL|re.IGNORECASE)
    html_str = re.sub(r'<style[^>]*>.*?</style>', '', html_str, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html_str)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Extract sections that contain phenotype info
    sections = []
    section_keywords = ['Clinical characteristics', 'Clinical features',
                        'Clinical manifestations', 'Signs and symptoms',
                        'Diagnostic features', 'Presentation', 'Phenotype']
    for kw in section_keywords:
        # Find each occurrence
        for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
            start = m.start()
            end = min(start + 5000, len(text))  # 5K chars per section
            sections.append(text[start:end])
            if len(sections) >= 5: break
    if not sections:
        # Fallback: first 10K chars
        sections = [text[:10000]]
    return sections


def name_from_filename(fp):
    stem = fp.stem  # NBK1163_pain_disorder
    parts = stem.split('_', 1)
    return parts[1].replace('_', ' ') if len(parts) > 1 else stem


def main():
    # Disease name → UMLS CUI lookup (via MRCONSO ENG)
    print("Loading GeneReviews files...", flush=True)
    files = sorted(Path(GR_DIR).glob('NBK*.html'))
    print(f"  {len(files)} files", flush=True)

    print("Loading MRCONSO for disease CUI lookup...", flush=True)
    # Normalize names: lowercase + strip non-alpha
    def norm(s):
        return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()

    disease_names = {norm(name_from_filename(fp)): fp.stem for fp in files}
    print(f"  target disease names: {len(disease_names)}", flush=True)

    name_to_cui = {}  # normalized name → CUI
    n_lines = 0
    with open(MRCONSO) as f:
        for line in f:
            n_lines += 1
            parts = line.split('|')
            if len(parts) < 15: continue
            cui, lang, term = parts[0], parts[1], parts[14]
            if lang != 'ENG': continue
            normt = norm(term)
            if normt in disease_names and normt not in name_to_cui:
                name_to_cui[normt] = cui
            if n_lines > 8_000_000 and len(name_to_cui) == len(disease_names):
                break
    print(f"  mapped {len(name_to_cui)}/{len(disease_names)} diseases", flush=True)

    # scispaCy
    print("Loading scispaCy...", flush=True)
    import spacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls", "k": 1, "threshold": 0.85
    })

    # Process each file
    print("Processing...", flush=True)
    t0 = time.time()
    n_out = 0
    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as fout:
        for i, fp in enumerate(files):
            dname = name_from_filename(fp)
            dcui = name_to_cui.get(norm(dname))
            if not dcui:
                continue
            try:
                html_str = open(fp, encoding='utf-8', errors='replace').read()
            except: continue
            sections = extract_clinical_sections(html_str)

            seen_cuis = set()
            for sect in sections:
                doc = nlp(sect[:5000])  # limit per call
                for ent in doc.ents:
                    if ent._.kb_ents:
                        ec = ent._.kb_ents[0][0]
                        if ec == dcui: continue
                        if ec in seen_cuis: continue
                        seen_cuis.add(ec)
                        edge = {
                            "disease": dname,
                            "umls_cui": dcui,
                            "evidence_cui": ec,
                            "phenotype": ent.text,
                            "source": "genereviews_scispacy",
                            "source_id": fp.stem,
                            "extracted_by": "scispacy",
                        }
                        fout.write(json.dumps(edge, ensure_ascii=False) + '\n')
                        n_out += 1
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(files)} edges={n_out} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Done. {n_out} edges → {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
