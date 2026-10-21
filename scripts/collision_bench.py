"""
Benchmark of terrain collision through Panda's own traverser (issue #16).

Settles an earthlike terrain around a camera over mountains and times one
``CollisionTraverser.traverse`` with a SpaceFlight-sized load::

    python scripts/collision_bench.py [--reps N] [--seed S] [--workers N]
        [--stress] [--geoms]

Load (all within 5 km of the camera): 64 ship spheres (5-20 m), half of them
within 50 m of the ground; 3 sensor spheres per ship (30/50/100 m), a sixth of
them active as on a bot's think frames; 300 laser segments (35 m) between 0
and 500 m up. ``--stress``: every ship within 50 m of the ground and every
laser within 60 m.

By default it times the native solid (``Terrain.enable_collision``; build it
with ``scripts/build_native.sh``): the traverse time against a baseline
without terrain, per collider kind, the contacts, and how many colliders get
more than one terrain entry in the frame (each one is an extra collision
event for the game).

``--geoms`` instead times option A of the issue, Panda's collision against
visible geometry: each drawn chunk gets hidden Geom tiles reusing its vertex
buffer. Variants: tiles per chunk side (4, 8, 16, 32), as Geoms in one
GeomNode per chunk or as one GeomNode each, with or without the skirts; it
also prints the build cost per chunk. It is two orders of magnitude slower.

Correctness samples compare the traverser's contacts with
:class:`TerrainQuery` (segment hit points and normals, sphere contacts).
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionSegment,
    CollisionSphere,
    CollisionTraverser,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    NodePath,
    loadPrcFileData,
)

from world_builder.terrain.chunkgen import QUADS, VERTS, skirt_ring
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.terrain import Terrain, TerrainSettings

SHIP_MASK = BitMask32.bit(2)  # SpaceFlight's DESTRUCTIBLE
SENSOR_RADII = ((5.0, 30.0), (50.0, 50.0), (125.0, 100.0))  # (forward, radius)
SHIPS = 64
LASERS = 300
LASER_LENGTH = 35.0
SPREAD = 5000.0
COLLISION_DISTANCE = 100_000.0


# ── prototype of the collision tiles ───────────────────────────────────────────


def tile_index_arrays(per_side: int, skirts: bool) -> list[np.ndarray]:
    """Split the shared index buffer into per_side² tiles; each skirt segment
    goes to the tile holding its edge vertex."""
    size = QUADS // per_side
    tiles = [[] for _ in range(per_side * per_side)]
    for r in range(QUADS):
        for c in range(QUADS):
            v00 = r * VERTS + c
            v01, v10 = v00 + 1, v00 + VERTS
            v11 = v10 + 1
            tile = (r // size) * per_side + c // size
            tiles[tile].append((v00, v01, v11, v00, v11, v10))
    if skirts:
        ring = skirt_ring()
        n = len(ring)
        for k in range(n):
            top_a, top_b = ring[k], ring[(k + 1) % n]
            bot_a, bot_b = VERTS * VERTS + k, VERTS * VERTS + (k + 1) % n
            r, c = divmod(int(ring[k]), VERTS)
            tile = min(r // size, per_side - 1) * per_side + min(
                c // size, per_side - 1
            )
            tiles[tile].append(
                (top_a, bot_a, bot_b, top_a, bot_b, top_b)
                + (top_a, bot_b, bot_a, top_a, top_b, bot_b)
            )
    return [np.array([i for tri in t for i in tri], dtype=np.uint16) for t in tiles]


def shared_primitives(per_side: int, skirts: bool) -> list[GeomTriangles]:
    prims = []
    for indices in tile_index_arrays(per_side, skirts):
        tris = GeomTriangles(Geom.UH_static)
        tris.set_index_type(GeomEnums.NT_uint16)
        tris.add_next_vertices(indices.size)
        memoryview(tris.modify_vertices()).cast("B")[:] = indices.tobytes()
        prims.append(tris)
    return prims


def tile_materials(chunk, per_side: int) -> list[int]:
    """Dominant biome index per tile (-1 = water)."""
    size = QUADS // per_side
    weights = chunk.weights.reshape(VERTS, VERTS, -1)[:QUADS, :QUADS]
    weights = weights.reshape(per_side, size, per_side, size, -1).sum(axis=(1, 3))
    dominant = weights.reshape(per_side * per_side, -1).argmax(axis=1)
    elevation = chunk.elevation[:QUADS, :QUADS].reshape(per_side, size, per_side, size)
    wet = (elevation < 0.0).mean(axis=(1, 3)).ravel() > 0.5
    return np.where(wet, -1, dominant).tolist()


def build_collision(node: NodePath, chunk, prims, per_side, layout) -> NodePath:
    vdata = node.node().get_geom(0).get_vertex_data()
    materials = tile_materials(chunk, per_side)
    # A sibling, not a child: chunk nodes are final (their bounds stand for
    # everything below), so the traverser would not cull tiles under them.
    parent = node.get_parent().attach_new_node("terrain-collision")
    parent.set_pos(node.get_pos())
    if layout == "geoms":
        geom_node = GeomNode("terrain")
        for prim in prims:
            geom = Geom(vdata)
            geom.add_primitive(prim)
            geom_node.add_geom(geom)
        geom_node.set_into_collide_mask(SHIP_MASK)
        tile = parent.attach_new_node(geom_node)
        tile.set_python_tag("owner", materials)
    else:
        for prim, material in zip(prims, materials):
            geom = Geom(vdata)
            geom.add_primitive(prim)
            geom_node = GeomNode("terrain")
            geom_node.add_geom(geom)
            geom_node.set_into_collide_mask(SHIP_MASK)
            tile = parent.attach_new_node(geom_node)
            tile.set_python_tag("owner", material)
    # What the first traverse would otherwise pay: bounds from the vertices.
    parent.node().get_bounds()
    parent.hide()
    return parent


# ── scene ──────────────────────────────────────────────────────────────────────


def mountain_camera(terrain: Terrain, rng) -> np.ndarray:
    """A camera 300 m over high ground."""
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    heights = terrain.planet.sample_dirs(dirs, apply_stamps=True)[0]
    best = int(np.argsort(heights)[int(0.995 * len(heights))])
    return dirs[best] * (terrain.radius + heights[best] + 300.0)


def settle(terrain: Terrain, camera: np.ndarray, limit_s: float = 300.0) -> None:
    start = time.perf_counter()
    quiet = 0
    while time.perf_counter() - start < limit_s:
        terrain.update(camera)
        stats = terrain.stats
        quiet = quiet + 1 if stats["pending"] == 0 and stats["wanted"] == 0 else 0
        if quiet > 30:
            break
        time.sleep(0.01)
    print(
        f"settled in {time.perf_counter() - start:.1f} s: {len(terrain.shown)} "
        f"chunks shown, finest level {terrain.stats['finest']}"
    )


def tangent_frame(up: np.ndarray):
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    return east, np.cross(up, east)


def scatter(terrain, camera, rng, count, low, high):
    """Planet-frame points within SPREAD of the camera, low..high above ground."""
    if count == 0:  # TerrainQuery.height_at fails on an empty array
        return np.zeros((0, 3)), np.zeros((0, 3))
    up = camera / np.linalg.norm(camera)
    east, north = tangent_frame(up)
    offsets = rng.uniform(-SPREAD, SPREAD, size=(count, 2))
    ground = camera + offsets[:, :1] * east + offsets[:, 1:] * north
    dirs = ground / np.linalg.norm(ground, axis=1, keepdims=True)
    heights = terrain.query.height_at(dirs * terrain.radius, generate=False)
    above = rng.uniform(low, high, size=count)
    return dirs * (terrain.radius + heights + above)[:, None], dirs


def make_colliders(terrain, camera, rng, render: NodePath, stress: bool = False):
    """SpaceFlight-like colliders; returns (root, ship nodes, sensors, lasers).

    stress: every ship within 50 m of the ground and every laser within 60 m
    (a dogfight in a canyon), instead of half the ships and lasers up to 500 m.
    """
    root = render.attach_new_node("colliders")
    low = SHIPS if stress else SHIPS // 2
    near, near_up = scatter(terrain, camera, rng, low, -5.0, 50.0)
    far, far_up = scatter(terrain, camera, rng, SHIPS - low, 50.0, 500.0)
    centres = np.concatenate([near, far])
    ups = np.concatenate([near_up, far_up])
    ships, sensors, lasers = [], [], []
    for index, (centre, up) in enumerate(zip(centres, ups)):
        node = CollisionNode("ship")
        node.add_solid(CollisionSphere(0, 0, 0, rng.uniform(5.0, 20.0)))
        node.set_from_collide_mask(SHIP_MASK)
        node.set_into_collide_mask(SHIP_MASK)
        path = root.attach_new_node(node)
        path.set_pos(*terrain.to_render(centre))
        ships.append(path)
        # Sensors look ahead along a random level heading.
        east, north = tangent_frame(up)
        angle = rng.uniform(0, 2 * np.pi)
        forward = np.cos(angle) * east + np.sin(angle) * north
        for ahead, radius in SENSOR_RADII:
            sensor = CollisionNode("sensor")
            sensor.add_solid(CollisionSphere(0, 0, 0, radius))
            active = index % 6 == 0
            sensor.set_from_collide_mask(SHIP_MASK if active else BitMask32.all_off())
            sensor.set_into_collide_mask(BitMask32.all_off())
            spath = root.attach_new_node(sensor)
            spath.set_pos(*terrain.to_render(centre + ahead * forward))
            sensors.append(spath)
    starts, _ = scatter(terrain, camera, rng, LASERS, 0.0, 60.0 if stress else 500.0)
    directions = rng.normal(size=(LASERS, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for start, direction in zip(starts, directions):
        node = CollisionNode("laser")
        end = direction * LASER_LENGTH
        node.add_solid(CollisionSegment(0, 0, 0, *end))
        node.set_from_collide_mask(SHIP_MASK)
        node.set_into_collide_mask(BitMask32.all_off())
        path = root.attach_new_node(node)
        path.set_pos(*terrain.to_render(start))
        lasers.append((path, start, start + end))
    return root, ships, sensors, lasers


def timed_traverse(traverser, queue, render, reps) -> float:
    """Median traverse time, ms; fewer repetitions when one is slow."""
    times = []
    for _ in range(reps):
        if times and sum(times) > 2000.0:
            break
        queue.clear_entries()
        start = time.perf_counter()
        traverser.traverse(render)
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def contact_report(queue) -> dict:
    """Terrain contacts per kind, and colliders touching >1 terrain node."""
    touched: dict = {}
    for entry in queue.entries:
        if entry.into_node.name != "terrain":
            continue
        touched.setdefault(entry.from_node_path, set()).add(entry.into_node_path)
    report = {}
    for kind in ("ship", "sensor", "laser"):
        counts = [len(v) for k, v in touched.items() if k.name == kind]
        report[kind] = (len(counts), sum(c > 1 for c in counts), max(counts, default=0))
    return report


def check_lasers(terrain, queue, lasers) -> tuple[int, float, float]:
    """Compare the traverser's first hit per laser with segment_intersect."""
    first: dict = {}
    for entry in queue.entries:
        if entry.from_node.name != "laser" or entry.into_node.name != "terrain":
            continue
        point = np.array(entry.get_surface_point(entry.from_node_path.parent))
        normal = np.array(entry.get_surface_normal(entry.from_node_path.parent))
        key = entry.from_node_path
        start = np.array(key.get_pos())
        distance = np.linalg.norm(point - start)
        if key not in first or distance < first[key][0]:
            first[key] = (distance, point, normal)
    offset = terrain._camera - terrain._render_camera  # render -> planet
    worst_point, worst_angle, compared = 0.0, 0.0, 0
    for path, p0, p1 in lasers:
        if path not in first:
            continue
        hit = terrain.query.segment_intersect(p0, p1, generate=False)
        if hit is None:
            continue
        _, point, normal = hit
        _, panda_point, panda_normal = first[path]
        compared += 1
        worst_point = max(
            worst_point, float(np.linalg.norm(panda_point + offset - point))
        )
        cos = float(np.dot(panda_normal / np.linalg.norm(panda_normal), normal))
        worst_angle = max(
            worst_angle, float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        )
    return compared, worst_point, worst_angle


def check_spheres(terrain, queue, ships) -> str:
    """Compare the traverser's ship contacts with TerrainQuery.sphere_contact."""
    offset = terrain._camera - terrain._render_camera  # render -> planet
    hits = {}
    for entry in queue.entries:
        if entry.from_node.name == "ship" and entry.into_node.name == "terrain":
            hits[entry.from_node_path] = (
                np.array(entry.get_surface_point(entry.from_node_path.parent)),
                np.array(entry.get_surface_normal(entry.from_node_path.parent)),
            )
    agree = only_native = only_query = 0
    worst_point = worst_angle = 0.0
    for path in ships:
        centre = np.array(path.get_pos()) + offset
        radius = path.node().get_solid(0).get_radius()
        expected = terrain.query.sphere_contact(centre, radius, generate=False)
        got = hits.get(path)
        if expected is None and got is None:
            agree += 1
        elif expected is None:
            only_native += 1
        elif got is None:
            only_query += 1
        else:
            agree += 1
            worst_point = max(
                worst_point, float(np.linalg.norm(got[0] + offset - expected[0]))
            )
            cos = float(np.dot(got[1] / np.linalg.norm(got[1]), expected[1]))
            worst_angle = max(
                worst_angle, float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            )
    return (
        f"ships vs sphere_contact: {agree} agree, {only_native} only native, "
        f"{only_query} only query; worst point {worst_point:.3f} m, worst normal "
        f"{worst_angle:.2f} deg"
    )


def run_native(
    terrain, kinds, kind_base, traverser, queue, baseline, ships, lasers, reps
) -> None:
    """The native solid, through Terrain.enable_collision."""
    start = time.perf_counter()
    collision = terrain.enable_collision(SHIP_MASK)
    print(
        f"\nnative: {len(collision.registered)} chunks registered in "
        f"{(time.perf_counter() - start) * 1e3:.1f} ms"
    )
    per_kind = {
        k: timed_traverse(t, q, render_of(terrain), reps) - kind_base[k]
        for k, (t, q) in kinds.items()
    }
    traverse = timed_traverse(traverser, queue, render_of(terrain), reps)
    report = contact_report(queue)
    compared, worst_point, worst_angle = check_lasers(terrain, queue, lasers)
    multi = ", ".join(
        f"{report[k][1]}/{report[k][0]}" for k in ("ship", "sensor", "laser")
    )
    print(
        f"native: traverse +{traverse - baseline:.3f} ms"
        f" (ships {per_kind['ships']:.3f}, sensors {per_kind['sensors']:.3f},"
        f" lasers {per_kind['lasers']:.3f}); contacts ship/sensor/laser "
        f"{report['ship'][0]}/{report['sensor'][0]}/{report['laser'][0]};"
        f" colliders on >1 node {multi}"
        f"\n    lasers vs segment_intersect: {compared} compared, worst point "
        f"{worst_point:.3f} m, worst normal {worst_angle:.2f} deg"
        f"\n    {check_spheres(terrain, queue, ships)}"
    )
    terrain.disable_collision()


def render_of(terrain) -> NodePath:
    return terrain.root.get_top()


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--geoms",
        action="store_true",
        help="time Panda's collision against Geom tiles (option A, slow) instead "
        "of the native solid (scripts/build_native.sh)",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="every ship within 50 m of the ground, every laser within 60 m",
    )
    args = parser.parse_args()

    loadPrcFileData("", "window-type none")
    loadPrcFileData("", "audio-library-name null")
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    render = base.render
    rng = np.random.default_rng(args.seed)
    terrain = Terrain(
        render, PRESETS["earthlike"], settings=TerrainSettings(workers=args.workers)
    )
    camera = mountain_camera(terrain, rng)
    print(f"camera {terrain.altitude(camera):.0f} m above sea level")
    settle(terrain, camera)

    near = []
    for key in terrain.shown:
        node = terrain.nodes[key]
        chunk = terrain.store[key]
        distance = np.linalg.norm(chunk.centre - camera) - terrain.grid.chunk_size(
            key[0]
        )
        if distance < COLLISION_DISTANCE:
            near.append((key, node, chunk))
    sea = sum(chunk.h_min < 0.0 for _, _, chunk in near)
    print(
        f"{len(near)} drawn chunks within {COLLISION_DISTANCE / 1e3:.0f} km "
        f"({sea} with sea)"
    )

    _, ships, sensors, lasers = make_colliders(
        terrain, camera, rng, render, args.stress
    )
    traverser = CollisionTraverser()
    queue = CollisionHandlerQueue()
    for path in ships + sensors + [laser[0] for laser in lasers]:
        traverser.add_collider(path, queue)
    active = [s for s in sensors if not s.node().get_from_collide_mask().is_zero()]
    kinds = {}
    for kind, paths in (
        ("ships", ships),
        ("sensors", active),
        ("lasers", [laser[0] for laser in lasers]),
    ):
        kind_traverser, kind_queue = CollisionTraverser(), CollisionHandlerQueue()
        for path in paths:
            kind_traverser.add_collider(path, kind_queue)
        kinds[kind] = (kind_traverser, kind_queue)
    kind_base = {
        k: timed_traverse(t, q, render, args.reps) for k, (t, q) in kinds.items()
    }
    baseline = timed_traverse(traverser, queue, render, args.reps)
    print(
        f"baseline traverse (no terrain): {baseline:.3f} ms, "
        f"{queue.get_num_entries()} contacts"
    )
    if not args.geoms:
        run_native(
            terrain,
            kinds,
            kind_base,
            traverser,
            queue,
            baseline,
            ships,
            lasers,
            args.reps,
        )
        terrain.destroy()
        return
    print(
        "\ntiles layout skirts | build ms/chunk | traverse +ms "
        "| contacts ship/sensor/laser"
        " | colliders on >1 node (ship, sensor, laser; max nodes)"
    )
    for skirts in (True, False):
        for per_side in (4, 8, 16, 32):
            prims = shared_primitives(per_side, skirts)
            for layout in ("geoms", "nodes"):
                start = time.perf_counter()
                built = [
                    build_collision(node, chunk, prims, per_side, layout)
                    for _, node, chunk in near
                ]
                build_ms = (time.perf_counter() - start) * 1e3 / max(len(near), 1)
                per_kind = {
                    k: timed_traverse(t, q, render, args.reps) - kind_base[k]
                    for k, (t, q) in kinds.items()
                }
                traverse = timed_traverse(traverser, queue, render, args.reps)
                report = contact_report(queue)
                if per_side == 8 and layout == "nodes" and skirts:
                    compared, worst_point, worst_angle = check_lasers(
                        terrain, queue, lasers
                    )
                    check = (
                        f"\n    lasers vs segment_intersect: {compared} compared, "
                        "worst point "
                        f"{worst_point:.3f} m, worst normal {worst_angle:.2f} deg"
                    )
                else:
                    check = ""
                multi = ", ".join(
                    f"{report[k][1]}/{report[k][0]} (max {report[k][2]})"
                    for k in ("ship", "sensor", "laser")
                )
                print(
                    f"{per_side * per_side:4d} {layout:6s} {'on ' if skirts else 'off'}"
                    f"   | {build_ms:8.3f}       | {traverse - baseline:8.3f}    "
                    f"| {report['ship'][0]}/{report['sensor'][0]}/{report['laser'][0]}"
                    f" | {multi}{check}"
                    + "\n    +ms by kind: "
                    + ", ".join(f"{k} {v:.3f}" for k, v in per_kind.items())
                )
                for path in built:
                    path.remove_node()
    terrain.destroy()


if __name__ == "__main__":
    main()
