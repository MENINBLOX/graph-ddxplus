"""Assemble the Korean draft from the paragraph workflow's output.

The workflow returns one reviewed paragraph per ID plus a cross-paragraph consistency report. This
places each paragraph under its heading (paper_ko.md) and collects the reviewers' flags, sentence
changes and consistency issues (review_log.md).

    python draft/paper_ko/assemble.py WORKFLOW_OUTPUT.json
    python draft/paper_ko/assemble.py REVISE_OUTPUT.json --base PARAS.json   # revision pass

A revision pass returns only the paragraphs it rewrote; --base supplies every other paragraph as
{"order": [...], "paras": {id: {"sec", "text"}}}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TITLE = ("문헌만으로 구축한 속성 정규화 지식그래프의 교차 벤치마크 zero-shot 감별진단: "
         "도달 범위, 한계, 그리고 속성의 검정 가능성")

# heading emitted before the first paragraph of each group
HEADINGS = {
    "ABS-1": "## 초록", "INT-1": "## 1. 서론", "REL-1": "## 2. 관련 연구",
    "MET-1": "## 3. 방법\n\n### 3.1 문제 설정과 설계 제약",
    "MET-3": "### 3.2 근거 텍스트의 획득", "MET-5": "### 3.3 정보추출",
    "MET-9": "### 3.4 정규화, 그래프 조립, 무결성 점검", "MET-11": "### 3.5 진단 알고리즘",
    "MET-18": "### 3.6 속성의 검증 설계", "MET-20": "### 3.7 평가 벤치마크와 통계적 추론",
    "MET-22": "### 3.8 비교 원칙과 동결 설정",
    "RES-1": "## 4. 결과\n\n### 4.1 지식그래프의 규모와 구조", "RES-2": "### 4.2 진단 성능",
    "RES-5": "### 4.3 구성요소 ablation", "RES-7": "### 4.4 속성의 검정 가능성과 기여",
    "RES-9": "### 4.5 지식원 한계의 정량화", "RES-10": "### 4.6 파이프라인의 무증상 실패",
    "DIS-1": "## 5. 논의\n\n### 5.1 문헌 기반 KG의 도달 범위",
    "DIS-2": "### 5.2 한계의 원인: 문헌 어휘와 문진 어휘", "DIS-3": "### 5.3 KG 밀도의 양면성",
    "DIS-4": "### 5.4 속성의 역할", "DIS-5": "### 5.5 연구의 한계", "DIS-7": "### 5.6 후속 연구",
    "CON-1": "## 6. 결론",
}


def main() -> None:
    raw = json.load(open(sys.argv[1], encoding="utf-8"))
    res = raw.get("result", raw)
    if isinstance(res, str):
        res = json.loads(res)
    if "--base" in sys.argv:
        base = json.load(open(sys.argv[sys.argv.index("--base") + 1], encoding="utf-8"))
        paras = {i: {"id": i, "revised": v["text"], "flags": [], "changes": []}
                 for i, v in base["paras"].items()}
        for r in res["revised"]:
            paras[r["id"]] = r
        res = {"order": base["order"], "consistency": res.get("consistency")}
    else:
        paras = {p["id"]: p for p in res["paragraphs"]}

    out = [f"# {TITLE}", "",
           "> 국문 초안. 문단별 작성·문장 단위 검토 에이전트의 산출물을 조립했다. "
           "`[확인 필요]`, `[재측정 필요]`, `[ref: …]`는 미해결 항목이며 목록은 `review_log.md`에 있다.",
           ""]
    for pid in res["order"]:
        if pid in HEADINGS:
            out += [HEADINGS[pid], ""]
        p = paras.get(pid)
        out += [(p["revised"] if p else f"({pid} 누락)").strip(), ""]
    text = "\n".join(out)
    (HERE / "paper_ko.md").write_text(text, encoding="utf-8")

    log = ["# 검토 로그", ""]
    cons = res.get("consistency")
    log += ["## 문단 간 일관성 검토", "",
            cons if isinstance(cons, str) else "```json\n" + json.dumps(cons, ensure_ascii=False, indent=1) + "\n```", ""]
    n_flag = n_chg = 0
    log += ["## 문단별 플래그와 문장 수정", ""]
    for pid in res["order"]:
        p = paras.get(pid)
        if not p:
            continue
        fl, ch = p.get("flags") or [], p.get("changes") or []
        n_flag += len(fl)
        n_chg += len(ch)
        if not fl and not ch:
            continue
        log.append(f"### {pid}")
        log += [f"- FLAG: {f}" for f in fl]
        log += [f"- 수정: {c.get('before', '')} → {c.get('after', '')} ({c.get('reason', '')})" for c in ch]
        log.append("")
    (HERE / "review_log.md").write_text("\n".join(log), encoding="utf-8")

    marks = {m: len(re.findall(re.escape(m), text)) for m in ("[확인 필요", "[재측정 필요", "[ref:")}
    print(f"paragraphs {len(paras)}/{len(res['order'])}  chars {len(text)}  "
          f"flags {n_flag}  changes {n_chg}  markers {marks}")


if __name__ == "__main__":
    main()
