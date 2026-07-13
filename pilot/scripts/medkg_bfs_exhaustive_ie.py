#!/usr/bin/env python3
"""BFS exhaustive recursive IE across 49 DDXPlus diseases.

Architecture:
1. PubMed crawl (sequential, CUI-cached)
2. vLLM IE batch (single instance, max throughput)
3. scispaCy co-occurrence (CPU, separate stage)
4. Per-disease independent termination (NO global_phen_pool)

Termination criteria (per disease):
- Layer 1: no new phen this depth
- Layer 2: <5 new phens AND coverage >= 0.8 (after depth >= 5)
- Layer 3: MaxDepth = 200

Output: edges_exhaustive_ie.jsonl + per_disease_summary.json
"""
from __future__ import annotations
import os, sys, json, re, time, threading
from pathlib import Path
from collections import defaultdict, deque
from queue import Queue
import requests
from xml.etree import ElementTree as ET

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR
from medkg_ie_pubmed import log

MAX_DEPTH = 200
ABSTRACTS_PER_CUI = 15
SATURATION_NEW_THRESHOLD = 5
SATURATION_COVERAGE = 0.8
MIN_DEPTH_FOR_SATURATION = 5
BEAM_K = 100  # top-K seeds per disease per depth (IDF-weighted)

UA = "MedKG-Research/0.1 (academic; max@meninblox.com)"

OUT_DIR = Path("pilot/data/exhaustive_ie")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EDGE_OUT = OUT_DIR / "edges_exhaustive_ie.jsonl"
SUMMARY_OUT = OUT_DIR / "per_disease_summary.json"
PUBMED_CACHE_DIR = OUT_DIR / "pubmed_cache"
PUBMED_CACHE_DIR.mkdir(exist_ok=True)
IE_CACHE = OUT_DIR / "ie_cache.jsonl"

# ----------------- IE prompt: combined patient-focused + anatomical -----------------
IE_PROMPT = """# Task: Patient-Reportable Phenotype Extraction

Extract symptoms that patients with "{name}" would report in a standard medical questionnaire.

# Categories
- Sensory: pain (location + intensity 1-10 + type), itching, fever, fatigue, dizziness
- Functional: cough (productive/dry), shortness of breath, nausea, vomiting, diarrhea, palpitations
- Visible: rash, swelling (WITH LOCATION), bruising, lumps
- Anatomical location: chest, abdomen, back, neck, shoulder, knee, ankle, etc.
- Triggers: exertion, rest, eating, lying down, exposure

# Exclude
- Clinical examination signs (rhonchi, murmur, Murphy's sign)
- Laboratory values (CBC, troponin)
- Imaging findings (CT/MRI findings)
- Mechanisms, genes, treatments
- The condition name itself

# Output (one phenotype per line, plain text)

# Abstract Title
{title}

# Abstract
{text}

# Patient-reportable phenotypes:
"""


def parse_phen(text: str):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        line = re.sub(r'^[-•\d.)]+\s*', '', line).strip()
        if not line or len(line) > 80: continue
        low = line.lower()
        if any(b in low for b in ['the patient', 'physician', 'doctor', 'clinician', 'should ', 'must ', 'might ']): continue
        out.append(line)
    return out


# ----------------- UMLS resolution -----------------
def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def lemma_word(w):
    if len(w) <= 3: return w
    for s in ["'s", "ies", "sses", "ches", "ses", "es", "ed", "ing", "s"]:
        if w.endswith(s) and len(w) > len(s) + 2:
            b = w[:-len(s)]
            return b + "y" if s == "ies" else b
    return w


def lemmatize(t): return " ".join(lemma_word(w) for w in t.split())


def load_mrconso_index():
    """Build CUI ↔ name index from MRCONSO (ENG only, preferred name)."""
    log("Loading MRCONSO index...")
    t0 = time.time()
    cui_to_name = {}
    PREFERRED = {"HPO": 0, "SNOMEDCT_US": 1, "MSH": 2, "MEDCIN": 3, "NCI": 4, "ICD10CM": 5}
    with open(UMLS_DIR / 'MRCONSO.RRF') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 15 or parts[1] != 'ENG': continue
            cui = parts[0]
            tty = parts[12]
            ts = parts[2]
            ispref = parts[6]
            sab = parts[11]
            text = parts[14]
            prio = PREFERRED.get(sab, 99)
            # Get preferred name
            if (ts == 'P' and ispref == 'Y') or tty in ('PT', 'PF'):
                cur = cui_to_name.get(cui)
                if cur is None or prio < cur[1]:
                    cui_to_name[cui] = (text, prio)
    log(f"  {len(cui_to_name):,} CUIs indexed ({time.time()-t0:.0f}s)")
    return {cui: name for cui, (name, _) in cui_to_name.items()}


# ----------------- PubMed -----------------
def pubmed_cache_path(cui):
    return PUBMED_CACHE_DIR / f"{cui}.jsonl"


def pubmed_fetch(cui, name):
    """Fetch PubMed abstracts for CUI. Cached on disk."""
    cache = pubmed_cache_path(cui)
    if cache.exists():
        return [json.loads(l) for l in cache.open() if l.strip()]
    # Also try main pubmed dir
    main = Path(f'/windows/data/medkg/pubmed/{cui}.jsonl')
    if main.exists() and main.stat().st_size > 0:
        records = [json.loads(l) for l in main.open() if l.strip()]
        if records: return records[:ABSTRACTS_PER_CUI]
    # Fetch
    query = f'"{name}"[Title/Abstract] OR "{name}"[MeSH Terms]'
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": str(ABSTRACTS_PER_CUI),
                    "sort": "relevance", "retmode": "json"},
            timeout=30, headers={"User-Agent": UA})
        if r.status_code != 200: return []
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        time.sleep(0.4)
        if not pmids: cache.write_text(""); return []
        r2 = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            timeout=60, headers={"User-Agent": UA})
        time.sleep(0.4)
        if r2.status_code != 200: return []
        root = ET.fromstring(r2.text)
        records = []
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//PMID"); title_el = art.find(".//ArticleTitle")
            abst_els = art.findall(".//AbstractText")
            if pmid_el is None or title_el is None: continue
            abstract = " ".join((a.text or "") for a in abst_els).strip()
            if len(abstract) < 100: continue
            records.append({"pmid": pmid_el.text, "title": (title_el.text or "").strip(),
                            "abstract": abstract[:2500], "cui": cui, "disease_name": name})
        with cache.open('w') as f:
            for r_ in records:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n")
        return records
    except Exception as e:
        log(f"  pubmed err {cui}: {e}")
        return []


# ----------------- Main BFS expansion -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diseases", type=int, default=49, help="number of diseases (PoC uses 1)")
    ap.add_argument("--max_depth", type=int, default=MAX_DEPTH)
    ap.add_argument("--abstracts_per_cui", type=int, default=ABSTRACTS_PER_CUI)
    ap.add_argument("--beam_k", type=int, default=BEAM_K, help="top-K seeds per disease per depth (0 = no pruning)")
    args = ap.parse_args()
    beam_k = args.beam_k

    cui_to_name = load_mrconso_index()

    # Load DDXPlus 49 disease CUIs
    with open('data/ddxplus/disease_icd10_cui_mapping.json') as f: icd = json.load(f)
    ddx_diseases = [(info['cui'], dn) for dn, info in icd.items() if 'cui' in info][:args.diseases]
    log(f"DDXPlus diseases to expand: {len(ddx_diseases)}")

    # Per-disease state
    disease_phens = {dcui: set() for dcui, _ in ddx_diseases}
    disease_active = {dcui: True for dcui, _ in ddx_diseases}
    disease_seed = {dcui: {dcui} for dcui, _ in ddx_diseases}  # current depth seeds
    disease_depth = {dcui: 0 for dcui, _ in ddx_diseases}
    disease_terminate = {dcui: None for dcui, _ in ddx_diseases}

    # Phen text→CUI cache (for resolving IE output)
    text_to_cui = {}
    # IE result cache: cui → {extracted_cuis, pmids}
    ie_cache = {}

    def resolve_text(text):
        """Resolve phen text to CUI(s) using MRCONSO + lemma + scispaCy."""
        if text in text_to_cui: return text_to_cui[text]
        return None  # placeholder (resolve in batch later)

    # Load vLLM
    log("Loading vLLM (gemma-4-E4B-it on GPU 0,1 — tensor_parallel=2; 8 heads not div by 3)...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.6,
              enforce_eager=True, tensor_parallel_size=2,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=350)

    # All-phen-texts collected; resolve to CUI at end of expansion
    all_phen_texts = set()
    edge_records = []  # (disease, src_cui, dst_text, depth, pmid)

    # Incremental edge flush (append mode): write at end of each depth
    edge_file = EDGE_OUT.open('a')  # append mode for incremental flush
    phen_freq_per_d = defaultdict(lambda: defaultdict(int))  # disease → phen_text → frequency (IE hit count)

    # BFS
    for depth in range(1, args.max_depth + 1):
        active_diseases = [d for d in disease_active if disease_active[d]]
        if not active_diseases:
            log(f"All terminated by depth {depth}")
            break
        log(f"=== Depth {depth}: {len(active_diseases)} active diseases ===")

        # Collect (disease, seed_cui) pairs to fetch
        fetch_tasks = []
        for d in active_diseases:
            for s in disease_seed[d]:
                if s not in ie_cache:
                    fetch_tasks.append((d, s))
        log(f"  PubMed fetch needed: {len(set(t[1] for t in fetch_tasks))} unique CUIs")

        # Sequential fetch (NCBI rate limit)
        seeds_to_fetch = sorted(set(t[1] for t in fetch_tasks))
        t0 = time.time()
        for i, s in enumerate(seeds_to_fetch):
            name = cui_to_name.get(s)
            if not name: ie_cache[s] = []; continue
            ie_cache[s] = pubmed_fetch(s, name)
            if (i + 1) % 50 == 0:
                log(f"    fetched {i+1}/{len(seeds_to_fetch)} ({time.time()-t0:.0f}s)")

        # Build IE batch
        ie_inputs = []  # (disease, seed_cui, pmid, conv)
        for d in active_diseases:
            d_name = cui_to_name.get(d, dict(ddx_diseases).get(d, '?'))
            for s in disease_seed[d]:
                abstracts = ie_cache.get(s, [])
                s_name = cui_to_name.get(s, '?')
                for abs_r in abstracts:
                    conv = [{"role": "user", "content": IE_PROMPT.format(
                        name=s_name, title=abs_r['title'], text=abs_r['abstract'])}]
                    ie_inputs.append((d, s, abs_r['pmid'], conv))
        log(f"  IE inputs: {len(ie_inputs):,}")
        if not ie_inputs:
            for d in active_diseases:
                disease_active[d] = False
                disease_terminate[d] = "no_abstracts"
            continue

        # Batch IE
        t0 = time.time()
        outputs = llm.chat([conv for _, _, _, conv in ie_inputs], sampling)
        log(f"  IE done in {time.time()-t0:.0f}s")

        # Collect per-disease new phen texts (track frequency for beam ranking)
        new_phen_texts_per_d = defaultdict(set)  # disease → set of (text, src_cui, pmid)
        for (d, s, pmid, _), out in zip(ie_inputs, outputs):
            try: text = out.outputs[0].text
            except: continue
            for phen_text in parse_phen(text):
                t_norm = normalize(phen_text)
                new_phen_texts_per_d[d].add((t_norm, s, pmid))
                phen_freq_per_d[d][t_norm] += 1  # frequency for beam ranking
                all_phen_texts.add(t_norm); all_phen_texts.add(lemmatize(t_norm))

        # Per-disease state update (using text-level new for now; CUI resolution batched at end)
        for d in active_diseases:
            new_in_d = {t for (t, _, _) in new_phen_texts_per_d.get(d, set())}
            existing_in_d = {normalize(p) if not isinstance(p, str) or '_' not in p else p
                            for p in disease_phens[d]}  # mix of CUI and text — defer
            # For BFS, use text overlap as proxy for new phen
            truly_new = new_in_d - {p for p in disease_phens[d] if isinstance(p, str)}

            if not truly_new:
                disease_active[d] = False
                disease_terminate[d] = "no_new_phen"
                log(f"    DONE {cui_to_name.get(d, d)[:30]:30s}: depth={depth}, phens={len(disease_phens[d])}, reason=no_new_phen")
                continue
            if depth >= MIN_DEPTH_FOR_SATURATION and len(truly_new) < SATURATION_NEW_THRESHOLD:
                # Saturation: small new + most overlapping with existing
                # Simplified: just terminate if very few new
                if len(truly_new) <= 2:
                    disease_active[d] = False
                    disease_terminate[d] = "saturated"
                    log(f"    DONE {cui_to_name.get(d, d)[:30]:30s}: depth={depth}, phens={len(disease_phens[d])}, reason=saturated")
                    continue

            # Update with beam pruning: keep top-K most frequent new phens
            if beam_k > 0 and len(truly_new) > beam_k:
                ranked = sorted(truly_new, key=lambda t: -phen_freq_per_d[d][t])
                truly_new_pruned = set(ranked[:beam_k])
            else:
                truly_new_pruned = truly_new

            disease_phens[d] |= truly_new  # store ALL discovered phens (for KG content)
            disease_seed[d] = truly_new_pruned  # but only expand top-K (for cost control)
            disease_depth[d] = depth

            # Incremental edge flush
            for (t, src, pmid) in new_phen_texts_per_d[d]:
                rec = {"disease": d, "src": src, "dst_text": t, "depth": depth, "pmid": pmid}
                edge_records.append(rec)
                edge_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        edge_file.flush()
        log(f"  Edges flushed: cumulative {len(edge_records):,}")

        # Convert text seeds to CUIs for next iteration (need CUI for next PubMed fetch)
        # For now, use scispaCy/UMLS resolution on new texts
        # (Implement lazy: resolve only seed-eligible texts each iteration)
        # Simplified: each iter, batch resolve via simple lookup
        unresolved = set()
        for d in disease_active:
            if not disease_active[d]: continue
            for t in disease_seed[d]:
                if t not in text_to_cui and isinstance(t, str):
                    unresolved.add(t); unresolved.add(lemmatize(t))
        # Batch MRCONSO lookup (only newly unresolved)
        # Skip if no unresolved
        if unresolved:
            new_resolved = batch_resolve_texts(unresolved, cui_to_name)
            text_to_cui.update(new_resolved)
            for d in disease_active:
                if not disease_active[d]: continue
                new_seeds = set()
                for t in disease_seed[d]:
                    cuis = text_to_cui.get(t) or text_to_cui.get(lemmatize(t))
                    if cuis:
                        for c in cuis:
                            if c not in disease_phens[d] and c != d:
                                new_seeds.add(c)
                disease_seed[d] = new_seeds

    # Save outputs (edge_file already flushed incrementally; just close)
    edge_file.close()
    log(f"Edges saved: {len(edge_records):,} → {EDGE_OUT}")

    summary = {
        d: {"name": cui_to_name.get(d, '?'), "depth": disease_depth[d],
            "phens": len(disease_phens[d]), "terminate": disease_terminate[d]}
        for d in disease_phens
    }
    json.dump(summary, SUMMARY_OUT.open('w'), indent=2, ensure_ascii=False)
    log(f"Summary → {SUMMARY_OUT}")


def batch_resolve_texts(texts, cui_to_name):
    """Resolve text → CUI batch via MRCONSO ENG lookup."""
    # MRCONSO already loaded; do reverse lookup
    # For perf: pre-build str → CUI list (limited to common ones)
    # Quick version: scan MRCONSO once for these texts
    result = {}
    PREFERRED = {"HPO": 0, "SNOMEDCT_US": 1, "MSH": 2, "MEDCIN": 3, "NCI": 4, "ICD10CM": 5}
    cur = {}
    with open(UMLS_DIR / 'MRCONSO.RRF') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 15 or parts[1] != 'ENG': continue
            text = normalize(parts[14])
            if text not in texts: continue
            cui, sab = parts[0], parts[11]
            prio = PREFERRED.get(sab, 99)
            existing = cur.get(text)
            if existing is None or prio < existing[1]:
                cur[text] = (cui, prio)
    for text, (cui, _) in cur.items():
        result[text] = [cui]
    return result


if __name__ == "__main__":
    main()
