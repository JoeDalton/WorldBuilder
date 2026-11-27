"""
Session-level Panda3D configuration shared by all tests.

Setting window-type none and audio-library-name null before any ShowBase is
ever constructed keeps the test suite headless (no window, no audio device)
while still giving full access to the scene-graph API, loaders, and math types
from panda3d.core.
"""

import os

import pytest
from panda3d.core import loadPrcFileData

# Must be applied before the first ShowBase construction.
loadPrcFileData("", "window-type none")
loadPrcFileData("", "audio-library-name null")
# The depth convention under test (depth.ReversedDepthCamera): OpenGL's 0-1
# through clip control (desktop, the default), or its [-1, 1] (macOS:
# WB_DEPTH_ZERO_TO_ONE=0). Run the suite under both.
ZERO_TO_ONE = os.environ.get("WB_DEPTH_ZERO_TO_ONE", "1") != "0"
loadPrcFileData("", f"gl-depth-zero-to-one {'#t' if ZERO_TO_ONE else '#f'}")


@pytest.fixture(scope="session")
def showbase():
    """
    The one headless ShowBase for the whole test session.

    ShowBase is a per-process singleton (constructing a second instance
    raises), so every test that needs a live app must share this fixture.
    """
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    yield base
    base.destroy()
