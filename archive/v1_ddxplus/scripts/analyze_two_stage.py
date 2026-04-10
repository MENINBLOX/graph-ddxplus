#!/usr/bin/env python3
"""Two-Stage 불일치 분석 스크립트.

Stage 1 추론에서 언급한 번호와 Stage 2 강제 선택이 일치하는지 분석합니다.
"""

import re
import sys


def extract_inferred_choice(reason: str) -> tuple[str | None, str]:
    """Stage 1 추론에서 LLM이 추천한 번호 추출.

    Returns:
        (추천 번호, 추출 근거 문장)
    """
    reason_lower = reason.lower()

    # 패턴 1: "I would select option X", "the best choice is X"
    patterns = [
        r'(?:i\s+)?(?:would\s+)?(?:select|choose|pick|recommend)\s+(?:option\s+)?(\d)',
        r'(?:the\s+)?(?:best|most\s+informative|optimal)\s+(?:choice|option|answer)\s+(?:is|would\s+be)\s+(?:option\s+)?(\d)',
        r'answer\s*(?:is|:)\s*(\d)',
        r'option\s+(\d)\s+(?:is|would\s+be)\s+(?:the\s+)?(?:best|most)',
        r'select\s+(\d)',
        r'choice\s*(?:is|:)\s*(\d)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, reason_lower)
        if matches:
            # 마지막 매칭이 최종 결론일 가능성 높음
            match = re.search(pattern, reason_lower)
            if match:
                return matches[-1], match.group(0)

    # 패턴 2: 마지막 문장에서 숫자만 추출
    last_sentences = reason[-500:]
    numbers = re.findall(r'\b([1-5])\b', last_sentences)
    if numbers:
        return numbers[-1], f"마지막 부분에서 숫자 '{numbers[-1]}' 발견"

    return None, "추론에서 번호 추출 실패"


def analyze_log_file(log_path: str):
    """로그 파일에서 Two-Stage 불일치 분석."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Debug 블록 추출
    debug_blocks = re.findall(
        r'\[Two-Stage Debug #(\d+)\](.*?)(?=\[Two-Stage Debug|\Z)',
        content,
        re.DOTALL
    )

    if not debug_blocks:
        print("Two-Stage Debug 블록을 찾을 수 없습니다.")
        return

    total_mismatches = 0
    total_analyzed = 0
    mismatch_examples = []

    for block_num, block_content in debug_blocks:
        # 각 케이스 추출
        cases = re.findall(
            r'\[(\d+)\] Candidates: (\[.*?\])\s+Reason \(len=\d+\): \'(.*?)\'',
            block_content,
            re.DOTALL
        )

        # Selection 추출
        selections = re.findall(r'Selection: (\d)', block_content)

        for i, (case_num, candidates, reason) in enumerate(cases):
            if i >= len(selections):
                break

            selection = selections[i]
            inferred, evidence = extract_inferred_choice(reason)
            total_analyzed += 1

            if inferred and inferred != selection:
                total_mismatches += 1
                mismatch_examples.append({
                    'block': block_num,
                    'case': case_num,
                    'candidates': candidates,
                    'inferred': inferred,
                    'selection': selection,
                    'evidence': evidence,
                    'reason_end': reason[-300:] if len(reason) > 300 else reason
                })

    # 결과 출력
    print("=" * 70)
    print("Two-Stage 불일치 분석 결과")
    print("=" * 70)
    print(f"분석된 케이스: {total_analyzed}")
    print(f"불일치 케이스: {total_mismatches}")
    if total_analyzed > 0:
        print(f"불일치율: {100 * total_mismatches / total_analyzed:.1f}%")
    print()

    if mismatch_examples:
        print("=" * 70)
        print("불일치 사례 (최대 5개)")
        print("=" * 70)
        for ex in mismatch_examples[:5]:
            print(f"\n[Block #{ex['block']}, Case #{ex['case']}]")
            print(f"  후보: {ex['candidates']}")
            print(f"  Stage 1 추론 결론: '{ex['inferred']}번' ({ex['evidence']})")
            print(f"  Stage 2 강제 선택: '{ex['selection']}번'")
            print(f"  ❌ 불일치!")
            print(f"  추론 마지막 부분: ...{ex['reason_end']}")

    return total_mismatches, total_analyzed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log_path = "/home/max/Graph-DDXPlus/benchmark_shuffle_2048.log"
    else:
        log_path = sys.argv[1]

    print(f"로그 파일: {log_path}")
    analyze_log_file(log_path)
