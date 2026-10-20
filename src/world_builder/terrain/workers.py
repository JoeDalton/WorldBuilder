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

Several planets share one pool through *channels*: each has its own spec,
version, wanted list and results. Each free slot goes to the channel with the
fewest tasks in flight, so a distant moon wanting a handful of coarse chunks
is served alongside the planet under the camera rather than after it.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field

from world_builder.terrain.chunkgen import ChunkData, generate_chunk
from world_builder.terrain.cubesphere import Key
from world_builder.terrain.planet import PlanetSpec

#: Channel used when none is named (a single planet).
DEFAULT_CHANNEL = "default"


def default_workers() -> int:
    """:returns: worker count leaving two cores for the renderer and the OS"""
    return max(1, (os.cpu_count() or 2) - 2)


@dataclass
class _Channel:
    spec: PlanetSpec | None = None
    version: int = 0
    #: Keys to generate, most urgent first (a dict: ordered, O(1) removal).
    wanted: dict[Key, None] = field(default_factory=dict)
    pending: dict[Key, Future] = field(default_factory=dict)
    done: list[ChunkData] = field(default_factory=list)


class ChunkExecutor:
    """Generates chunks for one or more planet specs (one per channel)."""

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
        :param max_inflight: queued plus running tasks, all channels together;
            defaults to 2 per worker
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
        self._channels: dict[str, _Channel] = {}
        self._last_version = 0

    @property
    def version(self) -> int:
        """The default channel's spec version; older results are dropped."""
        return self.version_of(DEFAULT_CHANNEL)

    def version_of(self, channel: str) -> int:
        """:returns: the channel's current spec version (0 if it has none)"""
        found = self._channels.get(channel)
        return found.version if found is not None else 0

    @property
    def pending(self) -> int:
        """Tasks queued or running, all channels."""
        return sum(len(c.pending) for c in self._channels.values())

    def pending_for(self, channel: str) -> int:
        """:returns: tasks queued or running for one channel"""
        found = self._channels.get(channel)
        return len(found.pending) if found is not None else 0

    def set_spec(self, spec: PlanetSpec, channel: str = DEFAULT_CHANNEL) -> int:
        """Switch a channel's planet: cancel its queued work, bump its version.

        :param spec: the new planet; resolved here (sea bias) if needed
        :param channel: which planet slot
        :returns: the new version (unique across channels)
        """
        ch = self._channels.setdefault(channel, _Channel())
        ch.spec = spec.resolved()
        self._last_version += 1
        ch.version = self._last_version
        for future in ch.pending.values():
            future.cancel()
        ch.pending.clear()
        ch.done.clear()
        ch.wanted.clear()
        return ch.version

    def request(self, wanted: list[Key], channel: str = DEFAULT_CHANNEL) -> None:
        """Make a channel's queue follow *wanted*.

        :param wanted: keys to generate, most urgent first; keys already
            generated must not be listed
        :param channel: which planet
        """
        ch = self._channels.get(channel)
        if ch is None or ch.spec is None:
            return
        if self._pool is None:
            for key in wanted[: self.inline_budget]:
                ch.done.append(generate_chunk(ch.spec, key, ch.version))
            return
        ch.wanted = dict.fromkeys(wanted)
        for key in list(ch.pending):
            if key not in ch.wanted and ch.pending[key].cancel():
                del ch.pending[key]
        self._fill()

    def collect(self, channel: str = DEFAULT_CHANNEL) -> list[ChunkData]:
        """:returns: a channel's chunks finished since the last call, current
        version only. Freed slots are refilled from the wanted lists."""
        ch = self._channels.get(channel)
        if ch is None:
            return []
        freed = False
        for key, future in list(ch.pending.items()):
            if future.done():
                del ch.pending[key]
                ch.wanted.pop(key, None)
                freed = True
                if not future.cancelled():
                    ch.done.append(future.result())
        done = [c for c in ch.done if c.version == ch.version]
        ch.done = []
        if freed:
            self._fill()
        return done

    def shutdown(self) -> None:
        """Stop the workers, dropping queued work."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _fill(self) -> None:
        """Top the shared queue up: each free slot goes to the channel with
        the fewest tasks in flight, so slots freed one at a time do not all
        go to whichever planet comes first."""
        if self._pool is None:
            return
        total = self.pending
        queues = {
            id(ch): (ch, _unsent(ch))
            for ch in self._channels.values()
            if ch.spec is not None and ch.wanted
        }
        while queues and total < self.max_inflight:
            ch, keys = min(queues.values(), key=lambda entry: len(entry[0].pending))
            key = next(keys, None)
            if key is None:
                del queues[id(ch)]
                continue
            ch.pending[key] = self._pool.submit(
                generate_chunk, ch.spec, key, ch.version
            )
            total += 1


def _unsent(ch: _Channel):
    """:returns: the channel's wanted keys not yet submitted, lazily.

    A function, not a generator expression in the caller's comprehension:
    there ``ch`` would be looked up late, as the last channel, and every
    queue would check that channel's pending set (resubmitting duplicates
    without end)."""
    return (key for key in ch.wanted if key not in ch.pending)
