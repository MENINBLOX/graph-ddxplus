"""Build draft.md from the section sources, the tables and the reference list.

    python3 tools/build_ms.py            # from draft/paper1_ko, with the repository two levels up

Tables come from `tables.py` over the evaluation outputs; {{TABLE1}}-{{TABLE3}} in the sections are replaced by
them and the supplementary tables follow the Supplementary Methods. Citations [@key; @key] are numbered in order
of first appearance ([1,2], runs of three or more as [1-3]) and {{REFERENCES}} becomes the Vancouver list of
`refs.py`. Every output is rewritten; nothing is edited by hand in draft.md, tables/ or figures/.
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent
ROOT = DRAFT.parent.parent
sys.path.insert(0, str(HERE))
from refs import REFS  # noqa: E402

SECTIONS = ("00_front", "10_intro", "20_methods", "30_results", "40_discussion", "50_back", "80_supp_methods", "90_memo")
CITE = re.compile(r"\[(@[^\]]+)\]")
FIGURE1 = "![Figure 1](figures/figure1_accuracy_by_questions.png)"


def compress(numbers: list[int]) -> str:
    numbers = sorted(set(numbers))
    parts, i = [], 0
    while i < len(numbers):
        j = i
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        parts.append(f"{numbers[i]}-{numbers[j]}" if j - i >= 2 else ",".join(str(n) for n in numbers[i : j + 1]))
        i = j + 1
    return ",".join(parts)


def main() -> None:
    tables = DRAFT / "tables"
    tables.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "tables.py"), str(ROOT), str(tables)], check=True)
    parts = {name: (DRAFT / "sections" / f"{name}.md").read_text().strip() for name in SECTIONS}
    for k in ("1", "2", "3"):
        parts["30_results"] = parts["30_results"].replace("{{TABLE%s}}" % k, (tables / f"table{k}.md").read_text().strip())
    supplementary = [(tables / f"tableS{i}.md").read_text().strip() for i in range(1, 8)]
    parts["80_supp_methods"] = "## Supplementary Material\n\n" + parts["80_supp_methods"].replace(
        "## Supplementary Methods", "### Supplementary Methods"
    ) + "\n\n" + "\n\n".join(supplementary)
    text = "\n\n".join(parts[name] for name in SECTIONS)
    if "{{TABLE" in text:
        raise SystemExit("a table placeholder is left")
    if FIGURE1 not in text:
        raise SystemExit("Figure 1 is not placed")
    order: dict[str, int] = {}

    def number(match: re.Match[str]) -> str:
        keys = [k.strip().lstrip("@") for k in match.group(1).split(";")]
        for k in keys:
            if k not in REFS:
                raise SystemExit(f"unknown citation key {k}")
            order.setdefault(k, len(order) + 1)
        return f"[{compress([order[k] for k in keys])}]"

    body = CITE.sub(number, text)
    refs = "\n".join(f"{n}. {REFS[k]}" for k, n in order.items())
    (DRAFT / "draft.md").write_text(body.replace("{{REFERENCES}}", refs) + "\n")
    unused = sorted(set(REFS) - set(order))
    print(f"draft.md: {len(order)} references cited; unused keys {unused}")


main()
