#!/usr/bin/env python3
"""ER-Reason Complete Profile 평가.

실제 응급실 EHR 데이터(3,984 patients, 1,554 diagnoses)에서
GraphTrace의 KG scoring으로 최종 진단 정확도를 측정.

입력: Chief complaint (+ One_Sentence_Extracted에서 추출한 증상)
출력: Ranked disease predictions vs primaryeddiagnosisname

Usage:
    uv run python scripts/benchmark_er_reason.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

ER_REASON_CSV = Path("data/er_reason/er-reason-a-benchmark-dataset-for-llm-based-clinical-reasoning-in-the-emergency-room-1.0.0/er_reason.csv")
UMLS_DIR = Path("data/umls_extracted")
OUTPUT_DIR = Path("results")


def load_er_reason() -> List[dict]:
    """ER-Reason CSV 로드."""
    print("  CSV 로드...")
    patients = []
    with open(ER_REASON_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cc = row.get("primarychiefcomplaintname", "").strip()
            dx = row.get("primaryeddiagnosisname", "").strip()
            one_sentence = row.get("One_Sentence_Extracted", "").strip()
            if cc and dx and cc != "*Unspecified":
                patients.append({
                    "chief_complaint": cc,
                    "diagnosis": dx,
                    "one_sentence": one_sentence,
                    "age": row.get("Age", ""),
                    "sex": row.get("sex", ""),
                    "acuity": row.get("acuitylevel", ""),
                    "disposition": row.get("eddisposition", ""),
                })
    print(f"  로드 완료: {len(patients)} patients")
    return patients


def build_umls_index() -> Tuple[Dict, Dict]:
    """MRCONSO에서 name→CUI 인덱스."""
    print("  MRCONSO 인덱싱...")
    name_to_cui = {}
    preferred = {}
    with open(UMLS_DIR / "MRCONSO.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, ts, name = parts[0], parts[1], parts[2], parts[14]
            if lang != "ENG":
                continue
            lower = name.lower().strip()
            if lower not in name_to_cui:
                name_to_cui[lower] = (cui, name)
            if ts == "P" and cui not in preferred:
                preferred[cui] = name
    print(f"    {len(name_to_cui)} names indexed")
    return name_to_cui, preferred


def extract_symptoms_from_sentence(one_sentence: str) -> List[str]:
    """One_Sentence_Extracted에서 증상/소견 키워드 추출.

    패턴: "presenting with X, Y, and Z" 또는 "p/w X and Y"
    """
    symptoms = []

    # "presenting with ...", "p/w ..."  이후 텍스트 추출
    patterns = [
        r"(?:presenting (?:to the ED )?with|p/w|presents with|presented with|complaining of|c/o)\s+(.+?)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, one_sentence, re.IGNORECASE)
        if m:
            symptom_text = m.group(1)
            # 쉼표, and로 분리
            parts = re.split(r",\s*|\s+and\s+|\s+with\s+|\s+x\s+\d+", symptom_text)
            for p in parts:
                p = p.strip().rstrip(".")
                if 2 < len(p) < 60:
                    symptoms.append(p)
            break

    # Chief complaint에서 나온 키워드도 추가하지 않음 (별도 처리)
    return symptoms


def map_to_cui(name: str, name_to_cui: Dict) -> str | None:
    """이름 → UMLS CUI 매핑 (여러 전략)."""
    lower = name.lower().strip()

    # 1. Exact
    if lower in name_to_cui:
        return name_to_cui[lower][0]

    # 2. 괄호 제거
    no_paren = re.sub(r"\s*\(.*?\)\s*", " ", lower).strip()
    no_paren = re.sub(r"\s+", " ", no_paren)
    if no_paren != lower and no_paren in name_to_cui:
        return name_to_cui[no_paren][0]

    # 3. "unspecified" 등 qualifier 제거
    simplified = re.sub(
        r",\s*(unspecified|initial encounter|subsequent|sequela|due to|"
        r"unspecified type|unspecified site|unspecified cause).*$",
        "", lower
    ).strip()
    if simplified != lower and simplified in name_to_cui:
        return name_to_cui[simplified][0]

    # 4. CMS code 태그 제거
    no_cms = re.sub(r"\s*\(CMS code\)\s*$", "", lower).strip()
    if no_cms != lower and no_cms in name_to_cui:
        return name_to_cui[no_cms][0]

    # 5. 약어 → 풀네임 (common ER abbreviations)
    abbrevs = {
        "uti": "urinary tract infection",
        "aki": "acute kidney injury",
        "ckd": "chronic kidney disease",
        "copd": "chronic obstructive pulmonary disease",
        "chf": "congestive heart failure",
        "dvt": "deep vein thrombosis",
        "pe": "pulmonary embolism",
        "mi": "myocardial infarction",
        "cva": "cerebrovascular accident",
        "tia": "transient ischemic attack",
        "aids": "acquired immunodeficiency syndrome",
        "pid": "pelvic inflammatory disease",
        "gerd": "gastroesophageal reflux disease",
        "ards": "acute respiratory distress syndrome",
        "sbo": "small bowel obstruction",
        "lle": "left lower extremity",
        "rle": "right lower extremity",
    }

    # Check if any abbreviation is in the name
    for abbr, full in abbrevs.items():
        if abbr in lower.split():
            expanded = lower.replace(abbr, full)
            if expanded in name_to_cui:
                return name_to_cui[expanded][0]
        # Also try the abbreviation in parentheses pattern
        paren_pat = f"({abbr})"
        if paren_pat in lower:
            if full in name_to_cui:
                return name_to_cui[full][0]

    return None


def build_kg_index() -> Tuple[Dict, Dict]:
    """MRREL에서 symptom→disease, disease→symptom 관계."""
    print("  증상-질병 관계 인덱스 구축...")

    disease_stys = {"T047", "T019", "T191", "T046"}
    symptom_stys = {"T184", "T033", "T034", "T048", "T037"}

    cui_stys = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui_stys[parts[0]].add(parts[1])

    symptom_to_diseases = defaultdict(set)
    disease_to_symptoms = defaultdict(set)

    relevant_relas = {
        "has_finding", "finding_site_of", "associated_with",
        "manifestation_of", "has_manifestation", "clinically_associated_with",
        "may_be_finding_of", "disease_has_finding",
    }

    with open(UMLS_DIR / "MRREL.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui1, cui2 = parts[0], parts[4]
            rela = parts[7].lower() if len(parts) > 7 else ""

            if rela not in relevant_relas:
                continue

            stys1 = cui_stys.get(cui1, set())
            stys2 = cui_stys.get(cui2, set())

            if (stys1 & symptom_stys) and (stys2 & disease_stys):
                symptom_to_diseases[cui1].add(cui2)
                disease_to_symptoms[cui2].add(cui1)
            elif (stys2 & symptom_stys) and (stys1 & disease_stys):
                symptom_to_diseases[cui2].add(cui1)
                disease_to_symptoms[cui1].add(cui2)

    print(f"    symptom CUIs: {len(symptom_to_diseases)}, disease CUIs: {len(disease_to_symptoms)}")
    return symptom_to_diseases, disease_to_symptoms


def evaluate(
    patients: List[dict],
    name_to_cui: Dict,
    symptom_to_diseases: Dict,
    disease_to_symptoms: Dict,
) -> dict:
    """Complete profile 평가."""

    results = {
        "total": 0, "hit1": 0, "hit3": 0, "hit5": 0, "hit10": 0,
        "no_symptom_cui": 0, "no_disease_cui": 0, "no_candidates": 0,
    }
    cc_results = defaultdict(lambda: {"total": 0, "hit1": 0, "hit3": 0})
    symptom_map_stats = {"total": 0, "mapped": 0}

    for patient in tqdm(patients, desc="Evaluating"):
        # 1. Chief complaint → CUI
        cc_cui = map_to_cui(patient["chief_complaint"], name_to_cui)

        # 2. One_Sentence_Extracted에서 추가 증상 추출
        extra_symptoms = extract_symptoms_from_sentence(patient["one_sentence"])
        extra_cuis = set()
        for sym in extra_symptoms:
            symptom_map_stats["total"] += 1
            cui = map_to_cui(sym, name_to_cui)
            if cui:
                extra_cuis.add(cui)
                symptom_map_stats["mapped"] += 1

        # Confirmed CUIs = CC + extracted symptoms
        confirmed_cuis = set()
        if cc_cui:
            confirmed_cuis.add(cc_cui)
        confirmed_cuis.update(extra_cuis)

        if not confirmed_cuis:
            results["no_symptom_cui"] += 1
            continue

        # 3. Ground truth diagnosis → CUI
        gt_cui = map_to_cui(patient["diagnosis"], name_to_cui)
        if not gt_cui:
            results["no_disease_cui"] += 1
            continue

        # 4. 후보 질병: confirmed 증상과 연결된 모든 질병
        candidate_diseases = set()
        for s_cui in confirmed_cuis:
            candidate_diseases.update(symptom_to_diseases.get(s_cui, set()))

        if not candidate_diseases:
            results["no_candidates"] += 1
            results["total"] += 1
            cc_results[patient["chief_complaint"]]["total"] += 1
            continue

        # 5. Evidence ratio scoring
        scores = []
        for d_cui in candidate_diseases:
            d_symptoms = disease_to_symptoms.get(d_cui, set())
            c = len(confirmed_cuis & d_symptoms)
            if c == 0:
                continue
            score = (c / (c + 1)) * c
            scores.append((d_cui, score))

        scores.sort(key=lambda x: -x[1])
        top_k = [cui for cui, _ in scores[:10]]

        results["total"] += 1
        cc = patient["chief_complaint"]
        cc_results[cc]["total"] += 1

        if gt_cui in top_k[:1]:
            results["hit1"] += 1
            cc_results[cc]["hit1"] += 1
        if gt_cui in top_k[:3]:
            results["hit3"] += 1
            cc_results[cc]["hit3"] += 1
        if gt_cui in top_k[:5]:
            results["hit5"] += 1
        if gt_cui in top_k[:10]:
            results["hit10"] += 1

    return results, dict(cc_results), symptom_map_stats


def main():
    print("=" * 60)
    print("ER-Reason Complete Profile Evaluation")
    print("=" * 60)

    print("\n[1/4] 데이터 로드...")
    patients = load_er_reason()

    print(f"\n[2/4] UMLS 인덱스...")
    name_to_cui, preferred = build_umls_index()

    print(f"\n[3/4] KG 인덱스...")
    symptom_to_diseases, disease_to_symptoms = build_kg_index()

    print(f"\n[4/4] 평가...")
    start = time.time()
    results, cc_results, sym_stats = evaluate(
        patients, name_to_cui, symptom_to_diseases, disease_to_symptoms
    )
    elapsed = time.time() - start

    # Report
    t = results["total"]
    print(f"\n{'='*60}")
    print(f"ER-REASON COMPLETE PROFILE RESULTS")
    print(f"{'='*60}")
    if t > 0:
        print(f"Evaluated:  {t} / {len(patients)} patients")
        print(f"Hit@1:  {results['hit1']}/{t} ({results['hit1']/t*100:.1f}%)")
        print(f"Hit@3:  {results['hit3']}/{t} ({results['hit3']/t*100:.1f}%)")
        print(f"Hit@5:  {results['hit5']}/{t} ({results['hit5']/t*100:.1f}%)")
        print(f"Hit@10: {results['hit10']}/{t} ({results['hit10']/t*100:.1f}%)")
        print(f"\nExcluded: no_symptom_cui={results['no_symptom_cui']}, "
              f"no_disease_cui={results['no_disease_cui']}, "
              f"no_candidates={results['no_candidates']}")
        if sym_stats["total"] > 0:
            print(f"Symptom extraction: {sym_stats['mapped']}/{sym_stats['total']} "
                  f"({sym_stats['mapped']/sym_stats['total']*100:.1f}%)")

        # Top chief complaints
        print(f"\nChief Complaint별 Hit@1 (n>=20):")
        sorted_cc = sorted(cc_results.items(), key=lambda x: -x[1]["total"])
        for cc, cr in sorted_cc[:15]:
            ct = cr["total"]
            if ct >= 20:
                print(f"  {cc}: {cr['hit1']/ct*100:.1f}% ({cr['hit1']}/{ct})")

    print(f"\nTime: {elapsed:.1f}s")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = {
        "dataset": "er_reason",
        "method": "complete_profile_evidence_ratio",
        "total_patients": len(patients),
        "results": results,
        "symptom_extraction_stats": sym_stats,
        "elapsed_seconds": elapsed,
    }
    with open(OUTPUT_DIR / "er_reason_benchmark.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {OUTPUT_DIR / 'er_reason_benchmark.json'}")


if __name__ == "__main__":
    main()
