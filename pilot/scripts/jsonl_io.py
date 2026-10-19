"""JSON-lines files, gzip-compressed when the name ends in .gz.

Dictionary outputs repeat each sentence once per match, so they are kept compressed (about 8x smaller).
An interrupted append leaves a truncated last gzip member or a partial last line; complete_lines()
stops before it and repair() rewrites the file without it, so a resumed run can append again.

    python jsonl_io.py count FILE      # number of complete lines
"""
from __future__ import annotations

import gzip
import os
import sys
import zlib
from collections.abc import Iterator
from typing import IO


def open_text(path: str, mode: str = "r") -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", **({"compresslevel": 6} if mode in "wa" else {}))
    return open(path, mode, encoding="utf-8")


def complete_lines(path: str, tail: list[bool] | None = None) -> Iterator[str]:
    """tail, when given, gets True appended if the file ends in a partial line or truncated member."""
    try:
        with open_text(path) as f:
            for ln in f:
                if ln.endswith("\n"):
                    yield ln
                elif tail is not None:
                    tail.append(True)
    except (EOFError, gzip.BadGzipFile, zlib.error):
        if tail is not None:
            tail.append(True)


def _truncate_after_last_newline(path: str) -> None:
    with open(path, "rb+") as fb:
        pos = fb.seek(0, 2)
        while pos > 0:
            step = min(1 << 16, pos)
            pos -= step
            fb.seek(pos)
            i = fb.read(step).rfind(b"\n")
            if i >= 0:
                fb.truncate(pos + i + 1)
                return
        fb.truncate(0)


def repair(path: str) -> None:
    """Drop a partial tail, if any: a plain file is truncated in place, a gzip file is rewritten."""
    if not os.path.exists(path):
        return
    tail: list[bool] = []
    for _ in complete_lines(path, tail):
        pass
    if not tail:
        return
    if not path.endswith(".gz"):
        _truncate_after_last_newline(path)
        return
    tmp = path[:-3] + ".part.gz"
    with open_text(tmp, "w") as fo:
        fo.writelines(complete_lines(path))
    os.replace(tmp, path)


if __name__ == "__main__":
    if sys.argv[1:2] == ["count"]:
        print(sum(1 for _ in complete_lines(sys.argv[2])) if os.path.exists(sys.argv[2]) else 0)
