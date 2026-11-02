"""
"No clouds overhead": the inputs a shader that receives the clouds' shadow
(cloud_field.glsl's ``cloudShadow``: the terrain's) needs before any clouds
are bound to it, or without any. :meth:`sphere.CloudVolume.bind_shadow_inputs`
overrides them.
"""

from __future__ import annotations

import functools

import numpy as np
from panda3d.core import NodePath, PTA_LVecBase4f, Texture, Vec3

from world_builder.clouds.noise import LAYER_VEC4S, MAX_LAYERS, build_deck_light_texture


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
    return build_deck_light_texture(np.ones((1, 1, 3), dtype=np.float32))


@functools.cache
def _no_noise() -> Texture:
    """A 1-texel noise volume, for shaders that never read it."""
    tex = Texture("no_cloud_noise")
    tex.setup_3d_texture(1, 1, 1, Texture.T_unsigned_byte, Texture.F_red)
    tex.set_ram_image(bytes(1))
    return tex


def cloud_shadow_defaults(node: NodePath) -> None:
    """Give a node "no clouds overhead" inputs for cloud_field.glsl's
    ``cloudShadowDepth`` (``uCloudShadowLayer`` -1), so a shader that
    includes it (the terrain's) runs without clouds, until
    :meth:`sphere.CloudVolume.bind_shadow_inputs` overrides them.
    """
    params = PTA_LVecBase4f.empty_array(MAX_LAYERS * LAYER_VEC4S)
    node.set_shader_inputs(
        cloudNoise=_no_noise(),
        layerParams=params,
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
        uCloudShadowOther=-1,
        otherWeather=_no_weather(),
        otherDeckLight=_no_deck_light(),
        otherShadowNear=_no_cloud_map(),
        otherShadowFar=_no_cloud_map(),
        otherNearChart=PTA_LVecBase4f.empty_array(4),
        otherFarChart=PTA_LVecBase4f.empty_array(4),
        otherShadowMapOn=0,
    )


@functools.cache
def _no_cloud_map() -> Texture:
    """A 1-texel cloud shadow map, for shaders with none (off)."""
    tex = Texture("no_cloud_shadow_map")
    tex.setup_2d_texture(1, 1, Texture.T_float, Texture.F_rgba32)
    tex.set_ram_image(bytes(16))
    return tex
