"""
Background chunk generation.

numpy on ~5k-element arrays spends much of its time in Python glue that holds
the GIL, so worker *threads* would still steal frame time from the renderer.
The default backend is a ``spawn`` process pool (workers import no Panda
graphics state). The ``thread`` and ``inline`` backends are kept for debugging
and tests.

The executor keeps only a short queue (``max_inflight``), so it can follow a
moving camera: each :meth:`ChunkExecutor.request` cancels queued work that is
no longer wanted and tops the queue up in the new priority order.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from world_builder.terrain.chunkgen import ChunkData, generate_chunk
from world_builder.terrain.planet import PlanetSpec

Key = tuple[int, int, int]


def default_workers() -> int:
    """:returns: worker count leaving two cores for the renderer and the OS"""
    return max(1, (os.cpu_count() or 2) - 2)


class ChunkExecutor:
    """Generates chunks for one planet spec at a time."""

    def __init__(
        self,
        backend: str = "process",
        workers: int | None = None,
        max_inflight: int | None = None,
        inline_budget: int = 2,
    ):
        """
        :param backend: "process", "thread" or "inline" (synchronous, in
            :meth:`request`)
        :param workers: pool size; defaults to cores - 2
        :param max_inflight: queued plus running tasks; defaults to 2 per worker
        :param inline_budget: chunks generated per request() by the inline
            backend
        """
        self.backend = backend
        self.workers = workers or default_workers()
        self.max_inflight = max_inflight or 2 * self.workers
        self.inline_budget = inline_budget
        if backend == "process":
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        elif backend == "thread":
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
        elif backend == "inline":
            self._pool = None
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self._spec: PlanetSpec | None = None
        self._version = 0
        self._pending: dict[Key, Future] = {}
        self._done: list[ChunkData] = []

    @property
    def version(self) -> int:
        """The current spec version; results from older versions are dropped."""
        return self._version

    @property
    def pending(self) -> int:
        """Tasks queued or running."""
        return len(self._pending)

    def set_spec(self, spec: PlanetSpec) -> int:
        """Switch planets: cancel everything queued and bump the version.

        :param spec: the new planet; resolved here (sea bias) if needed
        :returns: the new version
        """
        self._spec = spec.resolved()
        self._version += 1
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        self._done.clear()
        return self._version

    def request(self, wanted: list[Key]) -> None:
        """Make the queue follow *wanted*.

        :param wanted: keys to generate, most urgent first; keys already
            generated must not be listed
        """
        if self._spec is None:
            return
        if self._pool is None:
            for key in wanted[: self.inline_budget]:
                self._done.append(generate_chunk(self._spec, key, self._version))
            return
        wanted_set = set(wanted)
        for key in list(self._pending):
            if key not in wanted_set and self._pending[key].cancel():
                del self._pending[key]
        for key in wanted:
            if len(self._pending) >= self.max_inflight:
                break
            if key not in self._pending:
                self._pending[key] = self._pool.submit(
                    generate_chunk, self._spec, key, self._version
                )

    def collect(self) -> list[ChunkData]:
        """:returns: chunks finished since the last call, current version only"""
        for key, future in list(self._pending.items()):
            if future.done():
                del self._pending[key]
                if not future.cancelled():
                    self._done.append(future.result())
        done = [c for c in self._done if c.version == self._version]
        self._done = []
        return done

    def shutdown(self) -> None:
        """Stop the workers, dropping queued work."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
