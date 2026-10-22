"""Map a task over chunks of patients in worker processes; results come back in input order."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.pool import Pool


@contextmanager
def worker_pool(processes: int) -> Iterator[Pool | None]:
    """A pool for `map_items`; None, meaning this process, when processes == 1.

    The workers are spawned, so a calling script needs the `if __name__ == "__main__":` guard.
    """
    if processes < 1:
        raise ValueError("processes must be at least 1")
    if processes == 1:
        yield None
        return
    # spawn, not fork: the parent may hold BLAS or pandas threads, and forking those can deadlock.
    with multiprocessing.get_context("spawn").Pool(processes) as pool:
        yield pool


def map_items[T, R](fn: Callable[[T], R], items: Sequence[T], pool: Pool | None) -> list[R]:
    """fn over items, results in the order of `items`; with a pool, fn and items must pickle."""
    if pool is None:
        return [fn(x) for x in items]
    return pool.map(fn, items, chunksize=1)


def chunks[T](items: Sequence[T], n_chunks: int) -> list[Sequence[T]]:
    """At most n_chunks consecutive slices of nearly equal length that together are `items`."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1")
    size = -(-len(items) // n_chunks) if items else 1
    return [items[i : i + size] for i in range(0, len(items), size)]
