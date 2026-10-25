"""
Background chunk generation.

numpy on ~5k-element arrays spends much of its time in Python glue that holds
the GIL, so worker *threads* would still steal frame time from the renderer.
The default backend is a ``spawn`` process pool (workers import no Panda
graphics state). The ``thread`` and ``inline`` backends are kept for debugging
and tests. The pool defaults to one worker per physical core but one
(:func:`default_workers`), and its processes run at a lower priority
(:data:`WORKER_NICENESS`), so generation takes only the CPU time the renderer
leaves. They start with the pool, so their start-up (imports, the native
noise library) overlaps the caller's own, rather than waiting for the first
request.

The executor keeps only a short queue (``max_inflight``), so it can follow a
moving camera: each :meth:`ChunkExecutor.request` cancels queued work that is
no longer wanted and tops the queue up in the new priority order.

Several planets share one pool through *channels*: each has its own spec,
version, wanted list and results. Each free slot goes to the channel with the
fewest tasks in flight, so a distant moon wanting a handful of coarse chunks
is served alongside the planet under the camera rather than after it. A
channel may run its own generator (the clouds' tiles share the pool so).
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field

from world_builder.terrain.chunkgen import ChunkData, generate_chunk
from world_builder.terrain.cubesphere import Key
from world_builder.terrain.planet import PlanetSpec

#: Channel used when none is named (a single planet).
DEFAULT_CHANNEL = "default"


#: Fewest workers a pool runs, whatever the machine or the caller asks.
MIN_WORKERS = 1
#: Niceness added to worker processes: background generation yields the CPU
#: to the render loop whenever both want it.
WORKER_NICENESS = 10
#: glibc's malloc settings for the worker processes (Linux). By default it maps
#: and unmaps numpy's ~128 KB temporaries for every operation, so each one
#: page-faults; kept, a numpy chunk took ~24 ms instead of ~40 (see
#: docs/source/performance.md, §2). Set in the environment before the workers
#: spawn (they read it at start), unless already set.
WORKER_MALLOC_ENV = {
    "MALLOC_MMAP_THRESHOLD_": "4194304",
    "MALLOC_TRIM_THRESHOLD_": "67108864",
}


def usable_cpus() -> int:
    """:returns: CPUs this process may run on (affinity and container limits
    included where the platform reports them), at least 1"""
    counter = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if counter is not None:
        return max(1, counter() or 1)
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def physical_cores() -> int | None:
    """:returns: physical cores among the CPUs this process may use (Linux
    sysfs topology), or None where it cannot be told"""
    if not hasattr(os, "sched_getaffinity"):
        return None
    cores = set()
    try:
        for cpu in os.sched_getaffinity(0):
            topology = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
            with open(topology + "physical_package_id") as package:
                with open(topology + "core_id") as core:
                    cores.add((package.read().strip(), core.read().strip()))
    except OSError:
        return None
    return len(cores) or None


def default_workers() -> int:
    """Worker count: one per physical core but one, never below
    :data:`MIN_WORKERS`.

    Not per logical CPU: a worker on the renderer's hyper-threaded sibling
    slows the frame (6 workers on 4 cores / 8 threads: 35 fps flying low,
    3 workers: 47 fps) and adds little throughput (136 against 111 chunks/s,
    the same backlog). Where cores cannot be counted, two logical CPUs are
    left free. On 1-2 core machines one worker shares a core with the
    renderer: slower fills, but terrain still arrives.
    """
    cores = physical_cores()
    spare = cores - 1 if cores is not None else usable_cpus() - 2
    return max(MIN_WORKERS, spare)


@dataclass
class _Channel:
    spec: object | None = None
    #: Top-level (picklable) function(spec, key, version) -> result with
    #: ``key`` and ``version`` attributes; None = terrain chunks
    #: (:func:`generate_chunk`, looked up when called).
    generator: Callable | None = None

    def run(self):
        return self.generator or generate_chunk

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
        niceness: int = WORKER_NICENESS,
    ):
        """
        :param backend: "process", "thread" or "inline" (synchronous, in
            :meth:`request`)
        :param workers: pool size; defaults to :func:`default_workers`;
            raised to :data:`MIN_WORKERS` if lower
        :param max_inflight: queued plus running tasks, all channels together;
            defaults to 2 per worker
        :param inline_budget: chunks generated per request() by the inline
            backend
        :param niceness: added to the worker processes' niceness (lower
            priority), so a busy pool never makes the renderer wait for a
            CPU; 0 = same priority (process backend only)
        """
        self.backend = backend
        if workers is None:
            workers = default_workers()
        self.workers = max(MIN_WORKERS, int(workers))
        self.max_inflight = max(1, max_inflight or 2 * self.workers)
        self.inline_budget = inline_budget
        if backend == "process":
            if sys.platform.startswith("linux"):
                for name, value in WORKER_MALLOC_ENV.items():
                    os.environ.setdefault(name, value)
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_lower_priority,
                initargs=(niceness,),
            )
            # Processes start on demand, one per submission: start them all
            # now. Not counted as in flight.
            for _ in range(self.workers):
                self._pool.submit(_warm_up)
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

    def set_spec(
        self,
        spec: PlanetSpec,
        channel: str = DEFAULT_CHANNEL,
        generator: Callable | None = None,
    ) -> int:
        """Switch a channel's planet: cancel its queued work, bump its version.

        :param spec: the new planet; resolved here (sea bias) if needed. Any
            picklable spec goes with a matching *generator*
        :param channel: which planet slot
        :param generator: what the workers run, ``generator(spec, key,
            version)``, a top-level function; None keeps the channel's
            (terrain chunks by default). Clouds use their own
        :returns: the new version (unique across channels)
        """
        ch = self._channels.setdefault(channel, _Channel())
        if generator is not None:
            ch.generator = generator
        resolve = getattr(spec, "resolved", None)
        ch.spec = resolve() if resolve is not None else spec
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
                ch.done.append(ch.run()(ch.spec, key, ch.version))
            return
        ch.wanted = dict.fromkeys(wanted)
        for key in list(ch.pending):
            if key not in ch.wanted and ch.pending[key].cancel():
                del ch.pending[key]
        self._fill()

    def collect(self, channel: str = DEFAULT_CHANNEL) -> list[ChunkData]:
        """:returns: a channel's chunks (or its generator's results) finished
        since the last call, current version only. Freed slots are refilled
        from the wanted lists. A worker's exception is raised here."""
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
            ch.pending[key] = self._pool.submit(ch.run(), ch.spec, key, ch.version)
            total += 1


def _unsent(ch: _Channel):
    """:returns: the channel's wanted keys not yet submitted, lazily.

    A function, not a generator expression in the caller's comprehension:
    there ``ch`` would be looked up late, as the last channel, and every
    queue would check that channel's pending set (resubmitting duplicates
    without end)."""
    return (key for key in ch.wanted if key not in ch.pending)


def _warm_up() -> None:
    """Worker-process start-up task: load what every chunk needs."""
    from world_builder.terrain import noise

    noise.native_noise()


def _lower_priority(niceness: int) -> None:
    """Worker-process initializer: run below the renderer's priority."""
    if niceness and hasattr(os, "nice"):
        try:
            os.nice(niceness)
        except OSError:  # not permitted here: run at normal priority
            pass
