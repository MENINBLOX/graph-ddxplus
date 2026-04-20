#!/usr/bin/env python3
"""분기: CUI 쌍 사전 필터링 전략 비교.

67만 CUI 쌍을 LLM에 보내기 전 필터링 전략을 비교한다.
(A) 최소 N건 이상 공출현 필터
(B) 노이즈 CUI 블랙리스트
(C) semantic type 조합 필터 (T047/T184/T034만)
(D) A+B+C 결합
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}

# 진단에 핵심적인 semantic type (disease + sign/symptom + lab + finding)
CORE_STYS = {"T047", "T184", "T033", "T034", "T191"}


def load_cui_stys() -> dict[str, set[str]]:
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui_stys[p[0]].add(p[1])
    return dict(cui_stys)


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, lang, ts, name = p[0], p[1], p[2], p[14]
            if lang == "ENG" and (cui not in names or ts == "P"):
                names[cui] = name
    return names


def main():
    print("=" * 80)
    print("분기: CUI 쌍 사전 필터링 전략 비교")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    with open(DATA_DIR / "step1_cui_pairs.json") as f:
        all_pairs = json.load(f)
    with open(DATA_DIR / "step1_documents.json") as f:
        docs_data = json.load(f)

    cui_stys = load_cui_stys()
    cui_names = load_cui_names()

    # 쌍별 공출현 횟수 계산
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    pair_pmids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in all_pairs:
        key = (pair["cui_a"], pair["cui_b"])
        pair_counts[key] += 1
        pair_pmids[key].add(pair["pmid"])

    print(f"  총 쌍 수 (비고유): {len(all_pairs):,}")
    print(f"  고유 쌍 수: {len(pair_counts):,}")

    # 공출현 횟수 분포
    freq_dist = defaultdict(int)
    for cnt in pair_counts.values():
        freq_dist[cnt] += 1
    print(f"\n  공출현 횟수 분포:")
    for n in sorted(freq_dist.keys())[:10]:
        print(f"    {n}건: {freq_dist[n]:,}쌍")
    if max(freq_dist.keys()) > 10:
        print(f"    11+건: {sum(v for k,v in freq_dist.items() if k > 10):,}쌍")

    # 노이즈 CUI 식별 (비의학적 매칭)
    noise_cuis = set()
    noise_names = [
        "Symptom", "Other Symptom", "No information available",
        "Reduced", "Test Result", "Increased", "Disease, NOS",
        "Present", "Well", "Expression Negative", "High",
        "Increased (finding)", "Decreased (finding)",
        "Performance Status", "Normal", "Negative",
    ]

    # 이름으로 노이즈 CUI 식별
    for cui in set(c for pair in pair_counts for c in pair):
        name = cui_names.get(cui, "")
        if any(noise in name for noise in noise_names):
            noise_cuis.add(cui)

    # 가장 빈번한 CUI 중 노이즈 후보 확인
    cui_total_freq = defaultdict(int)
    for (a, b), cnt in pair_counts.items():
        cui_total_freq[a] += cnt
        cui_total_freq[b] += cnt

    print(f"\n  노이즈 CUI 후보: {len(noise_cuis)}개")
    for cui in sorted(noise_cuis, key=lambda c: -cui_total_freq.get(c, 0))[:15]:
        print(f"    {cui}: {cui_names.get(cui, '?')[:50]} (빈도={cui_total_freq[cui]})")

    # ============================================================
    # 필터링 전략 비교
    # ============================================================
    print(f"\n[2/3] 필터링 전략 비교")
    print("=" * 80)

    original = set(pair_counts.keys())
    print(f"  원본: {len(original):,}쌍")

    # (A) 최소 N건 이상 공출현
    for min_n in [2, 3, 5]:
        filtered = {k for k, v in pair_counts.items() if v >= min_n}
        print(f"  (A) 공출현 >= {min_n}건: {len(filtered):,}쌍 (감소율 {1-len(filtered)/len(original):.1%})")

    # (B) 노이즈 CUI 블랙리스트
    filtered_b = {(a, b) for (a, b) in original if a not in noise_cuis and b not in noise_cuis}
    print(f"  (B) 노이즈 제거: {len(filtered_b):,}쌍 (감소율 {1-len(filtered_b)/len(original):.1%})")

    # (C) Core semantic type만 (T047, T184, T033, T034, T191)
    filtered_c = set()
    for (a, b) in original:
        a_stys = cui_stys.get(a, set())
        b_stys = cui_stys.get(b, set())
        if (a_stys & CORE_STYS) and (b_stys & CORE_STYS):
            filtered_c.add((a, b))
    print(f"  (C) Core STY만: {len(filtered_c):,}쌍 (감소율 {1-len(filtered_c)/len(original):.1%})")

    # (D) A(>=2) + B + C 결합
    filtered_d = set()
    for (a, b) in original:
        if pair_counts[(a, b)] < 2:
            continue
        if a in noise_cuis or b in noise_cuis:
            continue
        a_stys = cui_stys.get(a, set())
        b_stys = cui_stys.get(b, set())
        if not ((a_stys & CORE_STYS) and (b_stys & CORE_STYS)):
            continue
        filtered_d.add((a, b))
    print(f"  (D) A(>=2)+B+C: {len(filtered_d):,}쌍 (감소율 {1-len(filtered_d)/len(original):.1%})")

    # (E) A(>=3) + B + C 결합
    filtered_e = set()
    for (a, b) in original:
        if pair_counts[(a, b)] < 3:
            continue
        if a in noise_cuis or b in noise_cuis:
            continue
        a_stys = cui_stys.get(a, set())
        b_stys = cui_stys.get(b, set())
        if not ((a_stys & CORE_STYS) and (b_stys & CORE_STYS)):
            continue
        filtered_e.add((a, b))
    print(f"  (E) A(>=3)+B+C: {len(filtered_e):,}쌍 (감소율 {1-len(filtered_e)/len(original):.1%})")

    # (D)의 semantic type 조합 분포
    print(f"\n[3/3] 필터링된 쌍 (D)의 semantic type 조합 분포:")
    sty_combo = defaultdict(int)
    for (a, b) in filtered_d:
        a_stys = cui_stys.get(a, set()) & CORE_STYS
        b_stys = cui_stys.get(b, set()) & CORE_STYS
        for sa in sorted(a_stys):
            for sb in sorted(b_stys):
                combo = f"{sa}-{sb}" if sa <= sb else f"{sb}-{sa}"
                sty_combo[combo] += 1
    for combo, cnt in sorted(sty_combo.items(), key=lambda x: -x[1]):
        print(f"    {combo}: {cnt:,}")

    # (D) 필터링된 쌍 중 DDXPlus 5개 질환 관련 CUI 확인
    ddx_disease_cuis = {
        "Pneumonia": "C0032285",
        "Pulmonary embolism": "C0034065",
        "GERD": "C0017168",
        "Panic attack": "C0086769",
        "Bronchitis": "C0006277",
    }
    print(f"\n  DDXPlus 5개 질환의 CUI 쌍 수 (필터 D):")
    for name, dcui in ddx_disease_cuis.items():
        related = [(a, b) for (a, b) in filtered_d if a == dcui or b == dcui]
        print(f"    {name} ({dcui}): {len(related)}쌍")
        # 상위 5개
        for (a, b) in sorted(related, key=lambda p: -pair_counts[p])[:5]:
            other = b if a == dcui else a
            print(f"      → {cui_names.get(other, '?')[:40]} (x{pair_counts[(a,b)]})")

    # 필터링된 쌍 저장 (Step 2 입력)
    filtered_pairs_list = []
    for (a, b) in filtered_d:
        filtered_pairs_list.append({
            "cui_a": a,
            "cui_b": b,
            "n_cooccurrence": pair_counts[(a, b)],
            "pmids": sorted(pair_pmids[(a, b)]),
        })
    filtered_pairs_list.sort(key=lambda x: -x["n_cooccurrence"])

    with open(DATA_DIR / "step1_filtered_pairs.json", "w") as f:
        json.dump(filtered_pairs_list, f, indent=2, ensure_ascii=False)
    print(f"\n필터링된 쌍 저장: {DATA_DIR / 'step1_filtered_pairs.json'}")
    print(f"원본 {len(original):,} → 필터(D) {len(filtered_d):,} ({len(filtered_d)/len(original)*100:.1f}%)")


if __name__ == "__main__":
    main()
