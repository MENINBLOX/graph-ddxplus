"""Assemble the v3 Korean draft (mixed extraction, designs D/E) from the drafting workflow result.

    python draft/paper_ko_v3/assemble.py draft/paper_ko_v3/workflow_result.json

Writes paper_ko.md (sections in reading order), evidence_log.md (per-sentence sources) and
consistency_issues.md (cross-section issues still to apply).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ["ABS-CON:abstract", "INT", "REL", "MET-A", "MET-B", "RES-A", "RES-B", "DIS", "ABS-CON:conclusion"]
CONCLUSION = re.compile(r"^#{1,3} *6\.", re.MULTILINE)
CONTINUED = re.compile(r"^## .*\(계속\)\n+", re.MULTILINE)


def split_abs_con(md: str) -> tuple[str, str]:
    m = CONCLUSION.search(md)
    return (md, "") if m is None else (md[:m.start()].rstrip(), md[m.start():].strip())


def main() -> None:
    res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sec = {s["id"]: s for s in res["sections"]}
    abstract, conclusion = split_abs_con(sec["ABS-CON"]["markdown"])
    body = {"ABS-CON:abstract": abstract, "ABS-CON:conclusion": conclusion}
    out = [f"# {res['title']}", "",
           ("> 국문 초안 v3. 절별 작성 → 문장 단위 사실 확인 → 수정 워크플로 산출물을 조립했다. "
            "`[E 확증 대기: …]`는 설계 E 확증 뒤 채울 자리, `[ref: …]`는 원문 대조 전 인용이다. "
            "문장별 근거는 evidence_log.md, 아직 반영하지 않은 절 간 불일치는 consistency_issues.md에 있다."), ""]
    for key in ORDER:
        text = CONTINUED.sub("", body[key] if key in body else sec[key]["markdown"])
        out += [text.strip(), ""]
    text = "\n".join(out)
    (HERE / "paper_ko.md").write_text(text, encoding="utf-8")
    log = ["# 문장별 근거 (v3)", ""]
    for s in res["sections"]:
        log += [f"## {s['id']}", ""]
        log += [f"- {e['sentence']}\n  - 근거: {e['source']}" for e in s["evidence"]]
        log.append("")
    (HERE / "evidence_log.md").write_text("\n".join(log), encoding="utf-8")
    iss = ["# 절 간 일관성 검토 (v3)", ""]
    for i, x in enumerate(res["consistency"]["issues"], 1):
        iss += [f"## {i}. [{x['severity']}]", "", f"- 인용: {x['quote']}", f"- 문제: {x['problem']}",
                f"- 수정: {x['correct']}", f"- 근거: {x['evidence']}", ""]
    (HERE / "consistency_issues.md").write_text("\n".join(iss), encoding="utf-8")
    marks = {m: len(re.findall(re.escape(m), text)) for m in ("[E 확증 대기", "[ref:", "[E 동결 대기")}
    print(f"sections {len(sec)} chars {len(text)} markers {marks} issues {len(res['consistency']['issues'])}")


if __name__ == "__main__":
    main()
