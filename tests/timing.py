"""
Wall-clock helpers for tests that must hold on slow, shared CI machines.

``WB_TEST_TIME_SCALE`` (default 1) stretches every time limit: CI sets it on
the slow runners, so a fast machine keeps the tight limits as regression guards.
"""

import copy
import math
import os
import time

#: Factor applied to every time limit.
TIME_SCALE = float(os.environ.get("WB_TEST_TIME_SCALE", "1"))


def wait_until(predicate, timeout, msg="condition never met", interval=0.01):
    """Poll *predicate* until it is true; fail after *timeout* s (scaled)."""
    deadline = time.perf_counter() + timeout * TIME_SCALE
    while not predicate():
        assert time.perf_counter() < deadline, msg
        time.sleep(interval)


def unbudgeted(settings):
    """:returns: a copy of terrain *settings* without the wall-clock upload
    budget, so a frame uploads ``uploads_per_frame`` chunks on any machine and
    a fixed frame count does the same work everywhere (issue #28).

    Bypasses the knob's range check (no setting allows it): tests only."""
    settings = copy.copy(settings)
    object.__setattr__(settings, "upload_ms_per_frame", math.inf)
    return settings
