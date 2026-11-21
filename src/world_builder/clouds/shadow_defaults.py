"""
"No clouds overhead": the inputs a shader that receives the clouds' shadow
(cloud_field.glsl's ``planetCloudShadow``: the terrain's) needs before any
clouds are bound to it, or without any.
:meth:`march.CloudMarch.bind_shadow_inputs` overrides them.
"""

from __future__ import annotations

import functools

import numpy as np
from panda3d.core import NodePath, PTA_LVecBase4f, Texture, Vec3

from world_builder.clouds.noise import (
    DETAIL_LEVELS,
    GENUS_VEC4S,
    MAX_GENERA,
    build_deck_light_array,
    build_deck_light_texture,
)
from world_builder.clouds.weather import build_coverage_texture, build_shift_texture


@functools.cache
def _no_weather() -> Texture:
    """A 1-texel weather cube map, for shaders that never read it."""
    tex = Texture("no_weather")
    tex.setup_cube_map(1, Texture.T_unsigned_byte, Texture.F_rg)
    tex.set_ram_image(bytes(12))
    return tex


@functools.cache
def _no_deck_light() -> Texture:
    """A 1-texel deck lighting table, for shaders that never read it."""
    return build_deck_light_texture(np.ones((1, 1, 4), dtype=np.float32))


@functools.cache
def _no_coverage() -> Texture:
    """A 1-texel coverage cube (``genusCoverA``, ``genusCoverB``), no cloud."""
    return build_coverage_texture(np.zeros((6, 1, 1, 4), dtype=np.float32))


@functools.cache
def _no_shift() -> Texture:
    """The genera's shift tables (``genusShift``), all 0."""
    return build_shift_texture([])


@functools.cache
def _no_deck_light_array() -> Texture:
    """A 1-texel far table per genus (``genusDeckLight``)."""
    return build_deck_light_array([np.ones((1, 1, 4), dtype=np.float32)] * MAX_GENERA)


@functools.cache
def _no_noise() -> Texture:
    """A 1-texel noise volume, for shaders that never read it."""
    tex = Texture("no_cloud_noise")
    tex.setup_3d_texture(1, 1, 1, Texture.T_unsigned_byte, Texture.F_red)
    tex.set_ram_image(bytes(1))
    return tex


@functools.cache
def _no_cells() -> Texture:
    """A 1x1x1 Worley volume (``cloudCells``), for shaders that never read it."""
    tex = Texture("no_cloud_cells")
    tex.setup_3d_texture(1, 1, 1, Texture.T_float, Texture.F_rg32)
    tex.set_ram_image(bytes(8))
    return tex


def cloud_shadow_defaults(node: NodePath) -> None:
    """Give a node "no clouds overhead" inputs for cloud_field.glsl's
    ``planetCloudShadow`` (``uCloudShadowLayer`` -1), so a shader that
    includes it (the terrain's) runs without clouds, until
    :meth:`march.CloudMarch.bind_shadow_inputs` overrides them.
    """
    params = PTA_LVecBase4f.empty_array((MAX_GENERA + 1) * GENUS_VEC4S)
    node.set_shader_inputs(
        cloudNoise=_no_noise(),
        cloudCells=_no_cells(),
        layerParams=params,
        genusLevelOffset=PTA_LVecBase4f.empty_array(MAX_GENERA * DETAIL_LEVELS // 4),
        detailOctaves=PTA_LVecBase4f.empty_array(DETAIL_LEVELS // 4),
        weatherMap=_no_weather(),
        deckLight=_no_deck_light(),
        deckLightCosMin=0.0,
        planetRadius=0.0,
        sunDir=Vec3(0.0, 0.0, 1.0),
        noiseSize=1.0,
        noiseOffset=Vec3(0.0, 0.0, 0.0),
        uCloudDebug=0,
        uCloudShadowLayer=-1,
        cloudShadowNear=_no_cloud_map(),
        cloudShadowFar=_no_cloud_map(),
        cloudNearChart=PTA_LVecBase4f.empty_array(4),
        cloudFarChart=PTA_LVecBase4f.empty_array(4),
        cloudShadowMapOn=0,
        uCloudGenera=0,
        uSkyBands=PTA_LVecBase4f.empty_array(4),
        uSkyBandCount=0,
        genusCoverA=_no_coverage(),
        genusCoverB=_no_coverage(),
        genusShift=_no_shift(),
        genusDeckLight=_no_deck_light_array(),
        cloudLight=_no_cloud_map(),
        cloudLightChart=PTA_LVecBase4f.empty_array(4),
        cloudLightOn=0,
    )


@functools.cache
def _no_cloud_map() -> Texture:
    """A 1-texel cloud shadow map, for shaders with none (off)."""
    tex = Texture("no_cloud_shadow_map")
    tex.setup_2d_texture(1, 1, Texture.T_float, Texture.F_rgba32)
    tex.set_ram_image(bytes(16))
    return tex


def ground_shadow_defaults(node: NodePath) -> None:
    """Give a node (the terrain's) "no ground shadow pass" inputs
    (terrain_frag.glsl's ``uGroundShadow``: off, the clouds' shadow shaded
    per pixel), until :meth:`march.CloudMarch.bind_ground_shadow` binds one.
    """
    node.set_shader_inputs(
        uGroundShadow=_no_cloud_map(), uGroundShadowOn=0, uGroundShadowScale=1
    )
