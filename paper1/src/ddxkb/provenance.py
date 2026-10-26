"""What every output records about its inputs and code: file digests and the git state."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ddxkb.seal import file_sha256

__all__ = ["environment", "file_sha256", "git_changed_since", "git_changes", "git_commits_touching", "git_head"]


def _git(root: Path, *args: str) -> str | None:
    # GIT_OPTIONAL_LOCKS=0 keeps `git status` from refreshing the index, so a run never takes the index
    # lock that a concurrent commit needs.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=env, check=False)
    except FileNotFoundError:
        return None
    return done.stdout if done.returncode == 0 else None


def git_head(root: Path) -> str | None:
    """The checked-out commit, or None outside a git work tree."""
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else None


def git_commits_touching(root: Path, path: str) -> list[str] | None:
    """Hashes of the commits in HEAD's history that change `path`, newest first; None outside a git work tree.

    History is simplified as `git log -- path` does by default, so a merge that only brings in a commit
    already listed is not listed again.
    """
    out = _git(root, "log", "--format=%H", "--", path)
    return None if out is None else out.split()


def git_changes(root: Path, *paths: str, untracked: bool = True) -> list[str] | None:
    """`git status --porcelain` lines for `paths` (the whole tree when none), or None outside a git work tree.

    Ignored files never appear; untracked ones appear only with `untracked`.
    """
    mode = "--untracked-files=all" if untracked else "--untracked-files=no"
    out = _git(root, "status", "--porcelain=v1", mode, "--", *paths)
    return None if out is None else out.splitlines()


def git_changed_since(root: Path, commit: str, *paths: str) -> list[str] | None:
    """Files under `paths` that differ between `commit` and HEAD, or None when either cannot be read."""
    out = _git(root, "diff", "--name-only", commit, "HEAD", "--", *paths)
    return None if out is None else out.splitlines()


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "numpy_simd": numpy_simd(),
    }


def numpy_simd() -> list[str]:
    """The SIMD targets numpy dispatches to on this CPU (`np.show_runtime`'s "found").

    np.argsort orders exact ties by its sort kernel, and the official GTPA@1 reads that order.
    """
    umath = importlib.import_module("numpy._core._multiarray_umath")
    features: dict[str, bool] = getattr(umath, "__cpu_features__", {})
    dispatch: list[str] = getattr(umath, "__cpu_dispatch__", [])
    return sorted(f for f in dispatch if features.get(f, False))
