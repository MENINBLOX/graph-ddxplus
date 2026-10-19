"""Assemble the redesigned (v2) Korean draft from the paragraph workflow output.
# NOTE (2026-10-10): paper_ko.md is now hand-revised from v2.1c results; rerunning this script overwrites it with the older workflow output.

    python draft/paper_ko_v2/assemble.py WORKFLOW_OUTPUT.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TITLE = ("추론 시점 LLM 없이 근거 문헌을 추적할 수 있는 감별진단: "
         "저사양 LLM 정보추출로 구축한 문헌 그래프의 열린 집합 평가")
HEADINGS = {
    "ABS-1": "## 초록", "INT-1": "## 1. 서론", "REL-1": "## 2. 관련 연구",
    "MET-1": "## 3. 방법\n\n### 3.1 설계 경계", "MET-2": "### 3.2 질환 목록과 문헌 수집",
    "MET-4": "### 3.3 정보추출과 정규화", "MET-6": "### 3.4 그래프 모델",
    "MET-7": "### 3.5 진단 질의", "MET-9": "### 3.6 벤치마크와 평가 프로토콜",
    "RES-1": "## 4. 결과\n\n### 4.1 코퍼스와 그래프", "RES-2": "### 4.2 IE 모델 시범과 문헌 깊이",
    "RES-3": "### 4.3 설정 선택", "RES-4": "### 4.4 열린 집합 진단 정확도",
    "RES-5": "### 4.5 그래프 기능 ablation", "RES-6": "### 4.6 다단계 감별",
    "RES-7": "### 4.7 추적성", "RES-8": "### 4.8 IE 변경의 영향",
    "DIS-1": "## 5. 논의", "DIS-5": "### 5.1 한계", "CON-1": "## 6. 결론",
}


def main() -> None:
    raw = json.load(open(sys.argv[1], encoding="utf-8"))
    res = raw.get("result", raw)
    if isinstance(res, str):
        res = json.loads(res)
    paras = {p["id"]: p for p in res["paragraphs"]}
    out = [f"# {TITLE}", "",
           "> 국문 초안 v2(재설계 연구). 문단별 작성·문장 단위 검토 에이전트 산출물을 조립했다. "
           "`[ref: …]`, `[감사 결과 대기]`, `[RQ4 결과 대기]`는 미해결 항목이다.", ""]
    for pid in res["order"]:
        if pid in HEADINGS:
            out += [HEADINGS[pid], ""]
        p = paras.get(pid)
        out += [(p["revised"] if p else f"({pid} 누락)").strip(), ""]
    text = "\n".join(out)
    (HERE / "paper_ko.md").write_text(text, encoding="utf-8")
    log = ["# 검토 로그 (v2)", "", "## 문단 간 일관성", "",
           "```json\n" + json.dumps(res.get("consistency"), ensure_ascii=False, indent=1) + "\n```", "",
           "## 문단별 플래그와 수정", ""]
    for pid in res["order"]:
        p = paras.get(pid)
        if p and (p.get("flags") or p.get("changes")):
            log.append(f"### {pid}")
            log += [f"- FLAG: {f}" for f in p.get("flags", [])]
            log += [f"- 수정: {c['before']} → {c['after']} ({c['reason']})" for c in p.get("changes", [])]
            log.append("")
    (HERE / "review_log.md").write_text("\n".join(log), encoding="utf-8")
    marks = {m: len(re.findall(re.escape(m), text)) for m in ("[ref:", "[감사 결과 대기]", "[RQ4 결과 대기]")}
    print(f"paragraphs {len(paras)}/{len(res['order'])} chars {len(text)} markers {marks}")


if __name__ == "__main__":
    main()
