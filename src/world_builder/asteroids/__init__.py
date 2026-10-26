"""
Asteroid fields and planet rings (issue #14): rocks scattered from a spec,
with no list of them stored, so a field can be as large as a ring.

:mod:`spec` describes a field (extent, density, sizes, spins, shapes,
materials, keep-outs) and :mod:`distributions` the variables it draws;
:mod:`scatter` gives the rocks of any part of it, the same every time, and
:mod:`motion` how each turns with time. None of these import Panda: they run
in worker processes and tests too. The design and its phases are in
``docs/source/asteroids.md``.
"""
