#!/usr/bin/env python3
"""전체 프로필 입력 벤치마크.

환자의 모든 양성 증상을 한번에 입력하고 진단 스코어링만 수행.
탐색(inquiry) 과정 없음 — KG 스코어링의 상한선 측정.

사용법:
    uv run python scripts/experiment_complete_profile.py \
        --workers 8 --ports "7687,7688,7689,7690,7691,7692,7693,7694"
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)


def run_single_patient(args):
    patient_data, loader_data, neo4j_port = args

    from src.data_loader import DDXPlusLoader
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        kg = UMLSKG(uri=f"bolt://localhost:{neo4j_port}")
    except Exception:
        return {"error": True}

    try:
        gt_disease_eng = loader.fr_to_eng.get(patient_data["pathology"], patient_data["pathology"])
        gt_cui = loader.get_disease_cui(gt_disease_eng)

        # 환자의 모든 양성 증상을 confirmed로 등록
        confirmed_cuis = set()
        for ev_str in patient_data["evidences"]:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                confirmed_cuis.add(cui)

        # initial evidence도 추가
        initial_cui = loader.get_symptom_cui(patient_data["initial_evidence"])
        if initial_cui:
            confirmed_cuis.add(initial_cui)

        if not confirmed_cuis:
            kg.close()
            return {"error": True}

        # 진단 스코어링 (denied 없음 — 전체 프로필이므로)
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d,
             count(DISTINCT s) AS total_symptoms,
             count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count
        WHERE confirmed_count > 0
        RETURN d.cui AS cui, d.name AS name,
               confirmed_count, total_symptoms
        """
        with kg.driver.session() as session:
            result = session.run(query, confirmed_cuis=list(confirmed_cuis))
            candidates = []
            for r in result:
                c = r["confirmed_count"]
                t = r["total_symptoms"]
                raw = score_v15_ratio(c, 0, t)  # denied=0
                candidates.append({
                    "cui": r["cui"], "name": r["name"], "score": raw,
                    "confirmed_count": c, "total_symptoms": t,
                })

        total = sum(c["score"] for c in candidates)
        for c in candidates:
            c["score"] = c["score"] / total if total > 0 else 0
        candidates.sort(key=lambda x: x["score"], reverse=True)

        correct_at_1 = candidates[0]["cui"] == gt_cui if candidates else False
        correct_at_10 = any(c["cui"] == gt_cui for c in candidates[:10]) if candidates else False

        kg.close()
        return {
            "error": False,
            "correct_at_1": int(correct_at_1),
            "correct_at_10": int(correct_at_10),
            "confirmed": len(confirmed_cuis),
            "pathology": gt_disease_eng,
        }
    except Exception:
        kg.close()
        return {"error": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690,7691,7692,7693,7694")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test")
    print(f"=== Complete Profile Benchmark ({len(patients):,}건) ===")

    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                       for k, v in loader.conditions.items()},
    }
    patients_data = [
        {"age": p.age, "sex": p.sex, "initial_evidence": p.initial_evidence,
         "evidences": p.evidences, "pathology": p.pathology,
         "differential_diagnosis": p.differential_diagnosis}
        for p in patients
    ]

    tasks = [(pd, loader_data, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start = time.time()
    results, errors = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc="complete_profile") as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start
    count = len(results)

    gtpa_1 = sum(r["correct_at_1"] for r in results) / count if count else 0
    gtpa_10 = sum(r["correct_at_10"] for r in results) / count if count else 0

    # Per-disease analysis
    from collections import defaultdict
    disease_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        d = r["pathology"]
        disease_stats[d]["total"] += 1
        disease_stats[d]["correct"] += r["correct_at_1"]

    output = {
        "mode": "complete_profile",
        "description": "All positive symptoms given at once, no inquiry phase",
        "scoring": "v15_ratio (denied=0)",
        "count": count,
        "errors": errors,
        "gtpa_1": gtpa_1,
        "gtpa_10": gtpa_10,
        "avg_confirmed": float(np.mean([r["confirmed"] for r in results])),
        "elapsed": elapsed,
        "per_disease": {
            d: {"count": s["total"], "gtpa_1": s["correct"] / s["total"] if s["total"] else 0}
            for d, s in sorted(disease_stats.items(), key=lambda x: x[1]["correct"] / max(x[1]["total"], 1))
        },
    }

    print(f"\nGTPA@1: {gtpa_1:.2%}, GTPA@10: {gtpa_10:.2%}")
    print(f"Avg Confirmed: {np.mean([r['confirmed'] for r in results]):.1f}")
    print(f"Errors: {errors}, Elapsed: {elapsed:.0f}s")

    path = Path("results") / "complete_profile_134529.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
