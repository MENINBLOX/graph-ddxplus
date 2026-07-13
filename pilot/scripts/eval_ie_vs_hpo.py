#!/usr/bin/env python3
"""HPO-based IE evaluation: score IE output against curated HPO annotations.

Academic positioning: HPO is used ONLY as evaluation reference (gold standard).
The KG itself is built from raw text via LLM IE — HPO annotations never enter
the KG. This separation mirrors standard practice in biomedical NER/RE
benchmarking (BC5CDR, NCBI-Disease, ChemProt).

Inputs:
  - phenotype.hpoa (disease → HP:NNN annotations, aspect=P only)
  - UMLS MRCONSO (HP:NNN canonical name + synonyms; OMIM/ORPHA → CUI)
  - IE edge file ($MEDKG_ROOT/processed/edges_pubmed_ie.jsonl or any *_ie.jsonl)

Pipeline:
  1. Parse HPO gold:  OMIM:NNN / ORPHA:NNN → {HP:NNN}
  2. Load HPO lex:    HP:NNN → {canonical name, synonyms (lowercased)}
  3. Load OMIM/ORPHA → UMLS CUI mapping
  4. For each IE edge (cui, phenotype_text):
       - phenotype_text → HP:NNN via lex match (lowercase + synonym table)
       - lookup OMIM/ORPHA equivalent of cui → gold HP set
       - per (cui_disease) compute P/R/F1
  5. Aggregate macro-F1 across diseases

Output: $MEDKG_ROOT/processed/ie_eval_vs_hpo.json with per-disease + summary stats.

Usage:
  python eval_ie_vs_hpo.py --edges $MEDKG_ROOT/processed/edges_pubmed_ie.jsonl
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

HPOA = Path("/windows/data/external_kg/phenotype.hpoa")
MRCONSO = UMLS_DIR / "MRCONSO.RRF"


def parse_hpoa(path):
    """Parse phenotype.hpoa → {db_id: {hp_ids}} for aspect=P only."""
    gold = defaultdict(set)
    names = {}
    with path.open() as f:
        for line in f:
            if line.startswith('#') or line.startswith('database_id'): continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 11: continue
            db_id, name, qual, hpo, ref, ev, onset, freq, sex, mod, aspect = parts[:11]
            if aspect != 'P': continue
            gold[db_id].add(hpo)
            names[db_id] = name
    return gold, names


def parse_mrconso(path):
    """Single-pass build of:
      - hp_to_lex:  HP:NNN → set(lowercased names + synonyms)
      - hp_to_cui:  HP:NNN → CUI
      - cui_to_hp:  CUI → set(HP:NNN)
      - omim_to_cui:  OMIM:NNN → CUI
      - orpha_to_cui: ORPHA:NNN → CUI
    """
    hp_to_lex = defaultdict(set)
    hp_to_cui = {}
    cui_to_hp = defaultdict(set)
    omim_to_cui = {}
    orpha_to_cui = {}
    with path.open() as f:
        for line in f:
            p = line.rstrip('\n').split('|')
            if len(p) < 15: continue
            if p[1] != 'ENG': continue
            cui = p[0]; sab = p[11]; code = p[13]; name = p[14].strip()
            if not name: continue
            if sab == 'HPO' and code.startswith('HP:'):
                hp_to_lex[code].add(name.lower())
                if p[2] == 'P' and p[6] == 'Y':  # preferred + suppressible='Y' is canonical
                    hp_to_cui[code] = cui
                cui_to_hp[cui].add(code)
            elif sab == 'OMIM':
                # MRCONSO OMIM has two forms: code='123456' or 'MTHU...'
                # We use only numeric OMIM codes
                if code.isdigit():
                    omim_to_cui[f'OMIM:{code}'] = cui
            elif sab == 'ORPHANET':
                # code like 'ORPHA:123' or '123'
                num = code.replace('ORPHA:', '')
                if num.isdigit():
                    orpha_to_cui[f'ORPHA:{num}'] = cui
    # Fallback: HP CUIs without canonical pick first ATOM
    for hp, lex in hp_to_lex.items():
        if hp not in hp_to_cui:
            # take any cui mapped to this HP (rare cases)
            for cui, hps in cui_to_hp.items():
                if hp in hps:
                    hp_to_cui[hp] = cui; break
    return hp_to_lex, hp_to_cui, cui_to_hp, omim_to_cui, orpha_to_cui


def normalize_phenotype_text(t):
    """Lowercase, trim, light punct strip."""
    t = (t or "").strip().lower()
    t = re.sub(r"[\.,;:]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def build_lex_index(hp_to_lex):
    """Reverse map: lowered name → set of HP IDs (synonym collisions allowed)."""
    idx = defaultdict(set)
    for hp, names in hp_to_lex.items():
        for n in names:
            idx[n].add(hp)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, help="IE edge JSONL (each line: {umls_cui, phenotype})")
    ap.add_argument("--out", default=str(MEDKG_ROOT / "processed" / "ie_eval_vs_hpo.json"))
    ap.add_argument("--min_per_disease", type=int, default=1)
    args = ap.parse_args()

    edges_path = Path(args.edges)
    if not edges_path.exists():
        sys.exit(f"Edges file not found: {edges_path}")

    t0 = time.time()
    print(f"[1] Parsing HPO phenotype.hpoa ...")
    gold_db, gold_name = parse_hpoa(HPOA)
    print(f"    Gold diseases: {len(gold_db):,}")

    print(f"[2] Parsing UMLS MRCONSO ...")
    hp_to_lex, hp_to_cui, cui_to_hp, omim_to_cui, orpha_to_cui = parse_mrconso(MRCONSO)
    lex_idx = build_lex_index(hp_to_lex)
    print(f"    HPO terms: {len(hp_to_lex):,}, with CUI: {len(hp_to_cui):,}")
    print(f"    OMIM CUIs: {len(omim_to_cui):,}, ORPHA CUIs: {len(orpha_to_cui):,}")
    print(f"    Lex index entries: {len(lex_idx):,}")
    print(f"    [elapsed: {time.time()-t0:.0f}s]")

    print(f"[3] Building gold by CUI ...")
    # gold_by_cui[disease_cui] = set(HP:NNN)
    gold_by_cui = defaultdict(set)
    n_omim_match = n_orpha_match = 0
    for db_id, hps in gold_db.items():
        cui = None
        if db_id.startswith('OMIM:'):
            cui = omim_to_cui.get(db_id); n_omim_match += int(cui is not None)
        elif db_id.startswith('ORPHA:'):
            cui = orpha_to_cui.get(db_id); n_orpha_match += int(cui is not None)
        if cui:
            gold_by_cui[cui].update(hps)
    print(f"    Diseases with CUI: OMIM={n_omim_match:,}, ORPHA={n_orpha_match:,}")
    print(f"    Unique gold CUIs: {len(gold_by_cui):,}")

    print(f"[4] Loading IE edges from {edges_path} ...")
    pred_by_cui = defaultdict(set)  # CUI → set of HP:NNN (matched)
    pred_unmatched = defaultdict(int)  # phenotype text → count (no HP match)
    pred_total = defaultdict(int)
    n_edges = 0
    with edges_path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            cui = e.get("umls_cui") or e.get("cui")
            if not cui: continue
            pheno = normalize_phenotype_text(e.get("phenotype") or e.get("phenotype_normalized") or "")
            if not pheno: continue
            n_edges += 1
            pred_total[cui] += 1
            hps = lex_idx.get(pheno)
            if hps:
                pred_by_cui[cui].update(hps)
            else:
                pred_unmatched[pheno] += 1
    print(f"    Edges: {n_edges:,}")
    print(f"    CUIs with predictions: {len(pred_by_cui):,}")
    n_total_pred = sum(len(s) for s in pred_by_cui.values())
    print(f"    HP-matched predictions: {n_total_pred:,}")
    print(f"    Unmatched phenotype strings (top 10):")
    for txt, cnt in sorted(pred_unmatched.items(), key=lambda x: -x[1])[:10]:
        print(f"      {cnt:5d}  {txt[:80]}")

    print(f"[5] Computing per-disease P/R/F1 ...")
    overlap_cuis = set(gold_by_cui.keys()) & set(pred_by_cui.keys())
    print(f"    Overlap (gold ∩ pred): {len(overlap_cuis):,} CUIs")

    per_disease = {}
    for cui in overlap_cuis:
        gold = gold_by_cui[cui]
        pred = pred_by_cui[cui]
        if len(pred) < args.min_per_disease: continue
        tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_disease[cui] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1,
                             "gold_n": len(gold), "pred_n": len(pred)}

    n = len(per_disease) or 1
    macro_p = sum(d["p"] for d in per_disease.values()) / n
    macro_r = sum(d["r"] for d in per_disease.values()) / n
    macro_f1 = sum(d["f1"] for d in per_disease.values()) / n
    micro_tp = sum(d["tp"] for d in per_disease.values())
    micro_fp = sum(d["fp"] for d in per_disease.values())
    micro_fn = sum(d["fn"] for d in per_disease.values())
    micro_p = micro_tp / max(micro_tp + micro_fp, 1)
    micro_r = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

    summary = {
        "edges_file": str(edges_path),
        "total_edges": n_edges,
        "diseases_evaluated": len(per_disease),
        "macro_P": round(macro_p, 4),
        "macro_R": round(macro_r, 4),
        "macro_F1": round(macro_f1, 4),
        "micro_P": round(micro_p, 4),
        "micro_R": round(micro_r, 4),
        "micro_F1": round(micro_f1, 4),
        "micro_tp_fp_fn": [micro_tp, micro_fp, micro_fn],
    }
    print(f"\n=== HPO-gold IE evaluation ===")
    print(f"  Diseases evaluated:  {len(per_disease):,}")
    print(f"  Macro P/R/F1:        {macro_p:.3f} / {macro_r:.3f} / {macro_f1:.3f}")
    print(f"  Micro P/R/F1:        {micro_p:.3f} / {micro_r:.3f} / {micro_f1:.3f}")
    print(f"  TP / FP / FN:        {micro_tp:,} / {micro_fp:,} / {micro_fn:,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_disease": per_disease}, ensure_ascii=False, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
