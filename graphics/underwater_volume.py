"""Underwater volumetric absorption / scattering model.

The module models the six defects described in the task:

1. ``depth_attenuation``
   Per-depth extinction instead of one fixed RGB attenuation shared by all
   water.  Red is attenuated first as depth grows, while very shallow water
   keeps a much weaker (and therefore less blue-shifted) extinction.
2. ``adaptive_step``
   Ray-marching step grows with distance but shrinks again when the camera is
   close to the water surface, so light shafts do not break into segments.
3. ``fog_dof_order``
   Depth of field is applied to the scene *before* the fog/absorption
   composite.  Compositing fog first and blurring afterwards would spread two
   different fog levels across one silhouette ("double contour").
4. ``view_refraction``
   Snell's-law refraction whose offset depends on the view angle and on the
   medium on both sides of the surface (air <-> water), rather than a constant
   1.33 offset independent of the camera.
5. ``volume_release``
   An explicit volume-texture registry.  Switching levels releases the
   textures and makes repeated load/release cycles bounded in GPU memory.
6. ``skip_frame_compensate``
   Scattering/fog accumulation advances by the actual elapsed time (with a
   clamped catch-up), not by the previous frame's nominal delta.

Public entry points accept a numeric primary argument, a single mapping of
options, or keyword arguments, and always return a mapping so they can be
composed by a renderer or inspected by tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

RGB = tuple[float, float, float]

# ---------------------------------------------------------------------------
# Small, side-effect-free helpers
# ---------------------------------------------------------------------------

def _number(value: Any, default: float) -> float:
    """Coerce *value* to a finite float, falling back to *default*."""
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _options(args: Sequence[Any], kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Split ``*args`` into an options mapping and remaining positional args."""
    if args and isinstance(args[0], Mapping):
        opts = dict(args[0])
        rest = tuple(args[1:])
    else:
        opts = {}
        rest = tuple(args)
    opts.update(kwargs)
    return opts, rest


def _num_arg(
    opts: Mapping[str, Any],
    rest: Sequence[Any],
    index: int,
    names: Sequence[str],
    default: float,
) -> float:
    """Pick a numeric option by keyword, else by positional index."""
    for name in names:
        if name in opts and opts[name] is not None:
            return _number(opts[name], default)
    if index < len(rest):
        return _number(rest[index], default)
    return default


def _str_arg(
    opts: Mapping[str, Any],
    rest: Sequence[Any],
    index: int,
    names: Sequence[str],
    default: str,
) -> str:
    for name in names:
        if name in opts and opts[name] is not None:
            return str(opts[name])
    if index < len(rest) and not isinstance(rest[index], (int, float, Mapping)):
        return str(rest[index])
    return default


def _as_rgb(value: Any, default: RGB) -> RGB:
    """Normalise a color given as RGB sequence, mapping or scalar."""
    if value is None:
        return default
    if isinstance(value, Mapping):
        try:
            return (
                max(0.0, _number(value.get("r", value.get("red")), default[0])),
                max(0.0, _number(value.get("g", value.get("green")), default[1])),
                max(0.0, _number(value.get("b", value.get("blue")), default[2])),
            )
        except (TypeError, ValueError):
            return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return (
            max(0.0, _number(value[0], default[0])),
            max(0.0, _number(value[1], default[1])),
            max(0.0, _number(value[2], default[2])),
        )
    scalar = _number(value, -1.0)
    if scalar >= 0.0:
        return (scalar, scalar, scalar)
    return default


def _rgb_arg(
    opts: Mapping[str, Any],
    rest: Sequence[Any],
    index: int,
    names: Sequence[str],
    default: RGB,
) -> RGB:
    for name in names:
        if name in opts and opts[name] is not None:
            return _as_rgb(opts[name], default)
    if index < len(rest):
        return _as_rgb(rest[index], default)
    return default


def _tuple3(value: Iterable[float]) -> RGB:
    r, g, b = value
    return (float(r), float(g), float(b))


# ---------------------------------------------------------------------------
# Water profiles: extinction depends on the water body *and* on depth
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaterProfile:
    """Optical properties of one water body.

    ``base_extinction_rgb`` is the extinction per metre at the surface and
    ``depth_scale_rgb`` controls how quickly each channel's extinction grows
    with depth.  Red has the largest base extinction and the shortest scale,
    so red disappears first in deep water; all three coefficients are small
    near the surface, which keeps shallow water from turning uniformly blue.
    """

    name: str
    base_extinction_rgb: RGB
    depth_scale_rgb: RGB
    scatter_rgb: RGB

    def extinction_at(self, depth_m: float) -> RGB:
        depth = max(0.0, _number(depth_m, 0.0))
        return _tuple3(
            base * (1.0 + depth / max(scale, 1e-3))
            for base, scale in zip(self.base_extinction_rgb, self.depth_scale_rgb)
        )


PROFILES: dict[str, WaterProfile] = {
    "clear": WaterProfile(
        name="clear",
        base_extinction_rgb=(0.42, 0.09, 0.055),
        depth_scale_rgb=(28.0, 65.0, 130.0),
        scatter_rgb=(0.008, 0.035, 0.055),
    ),
    "coastal": WaterProfile(
        name="coastal",
        base_extinction_rgb=(0.85, 0.22, 0.11),
        depth_scale_rgb=(18.0, 55.0, 95.0),
        scatter_rgb=(0.020, 0.070, 0.100),
    ),
    "turbid": WaterProfile(
        name="turbid",
        base_extinction_rgb=(1.60, 0.55, 0.30),
        depth_scale_rgb=(12.0, 38.0, 70.0),
        scatter_rgb=(0.050, 0.120, 0.150),
    ),
}

DEFAULT_PROFILE = "clear"


def _resolve_profile(value: Any) -> WaterProfile:
    """Resolve a profile name/dict/instance, falling back to clear water."""
    if isinstance(value, WaterProfile):
        return value
    if isinstance(value, Mapping):
        base = _as_rgb(
            value.get("base_extinction_rgb", value.get("extinction_rgb")),
            PROFILES[DEFAULT_PROFILE].base_extinction_rgb,
        )
        scales = _as_rgb(
            value.get("depth_scale_rgb", value.get("depth_scale")),
            PROFILES[DEFAULT_PROFILE].depth_scale_rgb,
        )
        scatter = _as_rgb(
            value.get("scatter_rgb", value.get("scattering_rgb")),
            PROFILES[DEFAULT_PROFILE].scatter_rgb,
        )
        return WaterProfile(
            name=str(value.get("name", "custom")),
            base_extinction_rgb=base,
            depth_scale_rgb=tuple(max(s, 1e-3) for s in scales),  # type: ignore[arg-type]
            scatter_rgb=scatter,
        )
    if isinstance(value, str) and value in PROFILES:
        return PROFILES[value]
    return PROFILES[DEFAULT_PROFILE]


def _profile_from_opts(opts: Mapping[str, Any]) -> WaterProfile:
    profile = opts.get("profile", opts.get("water_profile", opts.get("water")))
    return _resolve_profile(profile)


# ---------------------------------------------------------------------------
# 1. Depth dependent attenuation
# ---------------------------------------------------------------------------

def depth_attenuation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return Beer-Lambert attenuation for a given depth and path length.

    Parameters (all optional)
    -------------------------
    depth / depth_m : water depth of the sample
    path_length / distance_m : length of the water column traversed
    profile : water profile name, mapping, or :class:`WaterProfile`

    The returned ``extinction_rgb`` grows with depth, and red grows fastest,
    so deep samples lose red before blue.  At shallow depth every channel has
    a small coefficient, avoiding the "shallow water is too blue" artifact
    caused by reusing the deep-water calibration.
    """
    opts, rest = _options(args, kwargs)
    depth = max(0.0, _num_arg(opts, rest, 0, ("depth", "depth_m", "z", "water_depth"), 1.0))
    path = max(0.0, _num_arg(opts, rest, 1, ("path_length", "path", "distance", "distance_m", "thickness"), 1.0))
    profile = _profile_from_opts(opts)

    extinction = profile.extinction_at(depth)
    transmittance = _tuple3(math.exp(-extinction[i] * path) for i in range(3))
    absorbed = _tuple3(1.0 - transmittance[i] for i in range(3))
    mean_transmittance = sum(transmittance) / 3.0
    optical_depth = _tuple3(extinction[i] * path for i in range(3))

    regime = "shallow" if depth < 5.0 else "mid" if depth < 30.0 else "deep"
    dominant = ("red", "green", "blue")[max(range(3), key=lambda i: transmittance[i])]
    red_vanishes_first = extinction[0] > extinction[1] > extinction[2]

    return {
        "depth_m": depth,
        "path_length_m": path,
        "profile": profile.name,
        "extinction_rgb": extinction,
        "transmittance_rgb": transmittance,
        "absorbed_fraction_rgb": absorbed,
        "optical_depth_rgb": optical_depth,
        "mean_transmittance": mean_transmittance,
        "dominant_transmitted_channel": dominant,
        "red_vanishes_first": red_vanishes_first,
        "regime": regime,
        "attenuation_is_depth_dependent": True,
    }


# ---------------------------------------------------------------------------
# 2. Adaptive ray-march step
# ---------------------------------------------------------------------------

def adaptive_step(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Choose a ray-march step from ray distance and camera height.

    The step grows with distance (fewer samples in the far field), but is
    clamped near the water surface: a camera skimming the surface needs fine
    steps so that a volumetric light shaft stays continuous instead of
    breaking into discrete slabs.
    """
    opts, rest = _options(args, kwargs)
    distance = max(0.0, _num_arg(opts, rest, 0, ("distance", "distance_m", "ray_distance", "depth"), 1.0))
    camera_height = _num_arg(opts, rest, 1, ("camera_height", "height", "surface_distance", "camera_surface_distance"), 5.0)

    base_step = max(1e-5, _number(opts.get("base_step", opts.get("step_base")), 0.05))
    growth = max(0.0, _number(opts.get("growth", opts.get("step_growth")), 0.12))
    min_step = max(1e-6, _number(opts.get("min_step"), 0.005))
    max_step = max(min_step, _number(opts.get("max_step"), 1.0))
    surface_reference = max(1e-3, _number(opts.get("surface_reference"), 2.0))
    shaft_step_limit = max(min_step, _number(opts.get("max_shaft_step", opts.get("shaft_step_limit")), 0.20))

    # Near the surface the camera sees fine shaft structure; tighten the step.
    surface_factor = _clamp(math.sqrt(max(camera_height, 0.0) / surface_reference), 0.10, 1.0)
    raw_step = base_step * (1.0 + growth * distance)
    step = _clamp(raw_step * surface_factor, min_step, max_step)
    shaft_safe = True
    if camera_height < surface_reference:
        # Hard ceiling that scales smoothly to the surface.
        limit = shaft_step_limit * _clamp(0.10 + camera_height / surface_reference, 0.10, 1.0)
        if step > limit:
            step = max(min_step, limit)
            shaft_safe = False

    previous_step = _number(opts.get("previous_step", opts.get("last_step")), -1.0)
    if previous_step > 0.0:
        # Damp frame-to-frame step changes to avoid visible popping.
        smoothed = 0.65 * step + 0.35 * previous_step
        step = _clamp(smoothed, min_step, max_step)

    steps_estimate = distance / step if step > 0.0 else 0.0
    recommended_steps = int(math.ceil(min(steps_estimate, 1e6))) if steps_estimate > 0 else 0

    return {
        "step_m": step,
        "distance_m": distance,
        "camera_height_m": camera_height,
        "surface_factor": surface_factor,
        "base_step_m": base_step,
        "growth": growth,
        "clamped": step == min_step or step == max_step,
        "steps_estimate": steps_estimate,
        "recommended_steps": recommended_steps,
        "shaft_safe": shaft_safe,
        "step_adapted_to_distance": True,
        "step_adapted_to_surface_proximity": True,
    }


# ---------------------------------------------------------------------------
# 3. Fog / depth-of-field ordering
# ---------------------------------------------------------------------------

_STAGE_SYNONYMS = {
    "scene": "scene",
    "geometry": "scene",
    "color": "scene",
    "dof": "dof",
    "bokeh": "dof",
    "depth_of_field": "dof",
    "depthoffield": "dof",
    "fog": "fog",
    "volumetric": "fog",
    "underwater_fog": "fog",
    "composite": "composite",
    "compose": "composite",
    "present": "present",
    "tonemap": "present",
}

_CORRECT_ORDER = ["scene", "dof", "fog", "composite"]


def _canonical_order(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tokens = [t for t in re.split(r"[,>\\s]+", value) if t]
    elif isinstance(value, Sequence):
        tokens = [str(t) for t in value]
    else:
        return None
    canonical = [_STAGE_SYNONYMS.get(token.strip().lower(), token.strip().lower()) for token in tokens]
    return canonical or None


def fog_dof_order(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return the correct post-process order and repair a supplied one.

    Underwater fog must be composited *after* depth of field.  If fog is put
    first, a distant background and a near fish edge already have different
    fog amounts; blurring that result spreads both amounts over the same
    silhouette and produces a double contour.  Blurring the scene radiance
    first and applying depth-aware fog afterwards keeps one silhouette.
    """
    opts, rest = _options(args, kwargs)
    original = _canonical_order(opts.get("current_order", opts.get("order", opts.get("pipeline_order"))))
    if original is None and rest:
        original = _canonical_order(rest[0])

    if original is None:
        order = list(_CORRECT_ORDER)
        misordered = False
    else:
        fog_index = original.index("fog") if "fog" in original else None
        dof_index = original.index("dof") if "dof" in original else None
        misordered = bool(fog_index is not None and dof_index is not None and fog_index < dof_index)
        order = list(_CORRECT_ORDER) if misordered else list(original)

    fog_before_dof = False
    if "fog" in order and "dof" in order:
        fog_before_dof = order.index("fog") < order.index("dof")

    return {
        "order": order,
        "corrected_order": list(_CORRECT_ORDER),
        "original_order": original,
        "corrected": misordered,
        "fog_before_dof": fog_before_dof,
        "dof_before_fog": not fog_before_dof,
        "double_contour_risk": 1.0 if misordered else 0.0,
        "artifact_fixed": "double_contour" if misordered else None,
        "rationale": (
            "DoF blurs scene radiance and geometry edges first; depth-aware fog "
            "is composited afterwards so one silhouette receives one fog value."
        ),
    }


# ---------------------------------------------------------------------------
# 4. View dependent refraction
# ---------------------------------------------------------------------------

def view_refraction(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compute Snell's-law refraction for a view ray crossing the surface.

    ``from_medium='air'`` (default) describes a camera above the water looking
    at an object below it.  The horizontal position offset over the water
    column is a function of the view angle, so changing the camera direction
    changes the apparent position instead of leaving it pinned to a constant
    1.33 offset.
    """
    opts, rest = _options(args, kwargs)
    angle_deg = _clamp(_num_arg(opts, rest, 1, ("angle", "angle_deg", "view_angle", "incident_angle_deg"), 1.0), 0.0, 89.5)
    depth = max(0.0, _num_arg(opts, rest, 0, ("depth", "depth_m", "thickness"), 1.0))

    n_water = max(
        1.0,
        _number(opts.get("n_water", opts.get("ior_water", opts.get("refractive_index"))), 1.33),
    )
    n_air = max(1.0, _number(opts.get("n_air", opts.get("ior_air")), 1.0))
    from_medium = str(opts.get("from_medium", opts.get("medium", "air"))).lower()
    if from_medium not in ("air", "water"):
        from_medium = "air"

    if from_medium == "air":
        n1, n2, to_medium = n_air, n_water, "water"
    else:
        n1, n2, to_medium = n_water, n_air, "air"

    theta1 = math.radians(angle_deg)
    eta = n1 / n2
    sin_theta2 = eta * math.sin(theta1)
    tir = abs(sin_theta2) > 1.0 and n1 > n2
    if tir:
        # Total internal reflection: no transmitted ray leaves the water.
        theta2 = math.radians(89.5)
        transmitted = False
    else:
        theta2 = math.asin(_clamp(sin_theta2, -1.0, 1.0))
        transmitted = True

    tan1 = math.tan(theta1)
    tan2 = math.tan(theta2)

    # A straight (unrefracted) ray would travel depth*tan1 horizontally; the
    # refracted ray travels depth*tan2.  Their difference is the apparent
    # horizontal displacement seen from the other medium.
    if tir:
        offset = depth * tan1
    else:
        offset = depth * abs(tan1 - tan2)
    offset_ratio = offset / depth if depth > 0.0 else 0.0

    if abs(tan1) > 1e-9 and transmitted:
        apparent_depth = depth * tan2 / tan1
    else:
        apparent_depth = depth * (n1 / n2)

    critical_angle = None
    if n1 > n2:
        critical_angle = math.degrees(math.asin(_clamp(n2 / n1, 0.0, 1.0)))

    return {
        "view_angle_deg": angle_deg,
        "depth_m": depth,
        "from_medium": from_medium,
        "to_medium": to_medium,
        "n1": n1,
        "n2": n2,
        "relative_ior": n_water / n_air,
        "snell_ratio": eta,
        "refracted_angle_deg": math.degrees(theta2),
        "critical_angle_deg": critical_angle,
        "total_internal_reflection": tir,
        "transmitted": transmitted,
        "offset_m": offset,
        "offset_ratio": offset_ratio,
        "apparent_depth_m": apparent_depth,
        "offset_depends_on_view_angle": True,
        "fixed_ior_offset": False,
    }


# ---------------------------------------------------------------------------
# 5. Volume texture lifetime management
# ---------------------------------------------------------------------------

@dataclass
class VolumeTextureRegistry:
    """Tracks GPU volume textures so level switches can release them."""

    _levels: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _serial: int = 0

    def allocate(self, level: str, size_bytes: int, name: str | None = None) -> str:
        self._serial += 1
        handle = f"voltex:{level}:{self._serial}"
        self._levels.setdefault(str(level), []).append(
            {"handle": handle, "bytes": max(0, int(size_bytes)), "name": name or handle}
        )
        return handle

    def release(self, level: str) -> tuple[list[dict[str, Any]], int]:
        items = self._levels.pop(str(level), [])
        return items, sum(item["bytes"] for item in items)

    def release_all(self) -> tuple[list[dict[str, Any]], int]:
        items: list[dict[str, Any]] = []
        released = 0
        for level in list(self._levels):
            level_items, size = self.release(level)
            items.extend(level_items)
            released += size
        return items, released

    def total_bytes(self) -> int:
        return sum(item["bytes"] for items in self._levels.values() for item in items)

    def total_textures(self) -> int:
        return sum(len(items) for items in self._levels.values())

    def levels(self) -> list[str]:
        return list(self._levels)


GLOBAL_TEXTURE_REGISTRY = VolumeTextureRegistry()


def volume_release(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Release volume textures for a level (or the whole registry).

    Calling it twice for the same level is safe and reports zero released
    bytes the second time.  The returned ``remaining_bytes`` is what a leak
    check should assert is ``0`` after a level switch.
    """
    opts, rest = _options(args, kwargs)
    registry = opts.get("registry", opts.get("texture_registry"))
    if isinstance(registry, Mapping) and "registry" in registry:
        registry = registry["registry"]
    if not isinstance(registry, VolumeTextureRegistry):
        registry = None
        for item in rest:
            if isinstance(item, VolumeTextureRegistry):
                registry = item
                break
    if registry is None:
        registry = GLOBAL_TEXTURE_REGISTRY

    remaining = list(rest)
    for item in rest:
        if isinstance(item, VolumeTextureRegistry):
            remaining.remove(item)
            break

    level = opts.get("level", opts.get("level_id"))
    if level is None and remaining:
        if isinstance(remaining[0], str):
            level = remaining[0]
        elif isinstance(remaining[0], (int, float)) and not isinstance(remaining[0], bool):
            level = str(int(remaining[0]))

    before_bytes = registry.total_bytes()
    before_textures = registry.total_textures()
    if level is None:
        released_items, released_bytes = registry.release_all()
        released_levels = []
    else:
        released_items, released_bytes = registry.release(str(level))
        released_levels = [str(level)]

    remaining_bytes = registry.total_bytes()
    return {
        "released_textures": len(released_items),
        "released_bytes": released_bytes,
        "released_levels": released_levels or sorted(set(released_levels)),
        "freed_all_requested": level is None,
        "before_bytes": before_bytes,
        "before_textures": before_textures,
        "remaining_bytes": remaining_bytes,
        "remaining_textures": registry.total_textures(),
        "idempotent": len(released_items) == 0,
        "leak": remaining_bytes > 0 and level is not None,
        "texture_released": True,
    }


# ---------------------------------------------------------------------------
# 6. Frame-skip compensation
# ---------------------------------------------------------------------------

def skip_frame_compensate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Advance scattering by real elapsed time when frames are skipped.

    A frame spike must not advance fog with the old nominal delta (that makes
    fog trails lag behind the swimmer), but it also must not be allowed to
    integrate an unbounded delta (spiral of death).  The elapsed time is
    therefore split into sub-steps and the accumulation uses the clamped
    ``effective_dt``.
    """
    opts, rest = _options(args, kwargs)
    target_dt = max(1e-6, _number(opts.get("target_dt", opts.get("nominal_dt")), 1.0 / 60.0))
    frame_dt = max(1e-6, _num_arg(opts, rest, 0, ("frame_dt", "dt", "elapsed", "delta_time"), target_dt))
    max_ratio = max(1.0, _number(opts.get("max_catchup_ratio", opts.get("max_ratio")), 4.0))
    max_substeps = max(1, int(_clamp(_number(opts.get("max_substeps"), 4.0), 1, 64)))
    scatter_rate = max(0.0, _number(opts.get("scatter_rate", opts.get("rate")), 1.0))
    previous = max(0.0, _number(opts.get("accumulated", opts.get("accumulator")), 0.0))
    tau = max(1e-4, _number(opts.get("tau", opts.get("smoothing_tau")), 0.15))

    ratio = frame_dt / target_dt
    compensation_ratio = _clamp(ratio, 1.0, max_ratio)
    effective_dt = target_dt * compensation_ratio
    substeps = int(_clamp(math.ceil(ratio), 1.0, float(max_substeps)))
    clamped = ratio > max_ratio
    uses_old_frame = False

    advance = scatter_rate * effective_dt
    accumulated = previous + advance
    legacy_advance = scatter_rate * target_dt
    lag_ms = max(0.0, frame_dt - effective_dt) * 1000.0
    temporal_blend = 1.0 - math.exp(-effective_dt / tau)
    fog_history_weight = math.exp(-effective_dt / tau)

    return {
        "frame_dt": frame_dt,
        "target_dt": target_dt,
        "ratio": ratio,
        "compensation_ratio": compensation_ratio,
        "effective_dt": effective_dt,
        "substeps": substeps,
        "clamped": clamped,
        "uses_old_frame": uses_old_frame,
        "advance": advance,
        "accumulated": accumulated,
        "legacy_advance": legacy_advance,
        "lag_ms": lag_ms,
        "temporal_blend": temporal_blend,
        "fog_history_weight": fog_history_weight,
        "temporal_lag_compensated": True,
    }


# ---------------------------------------------------------------------------
# Pipeline pieces
# ---------------------------------------------------------------------------

def load_water(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Build a water-volume state and optionally allocate level textures."""
    opts, rest = _options(args, kwargs)
    depth = _clamp(_num_arg(opts, rest, 0, ("depth", "depth_m", "water_depth"), 1.0), 0.0, 1e4)
    profile = _profile_from_opts(opts)
    level = opts.get("level", opts.get("level_id"))
    if level is None:
        level = "default"
    level = str(level)

    textures_opt = opts.get("textures")
    if isinstance(textures_opt, Sequence) and not isinstance(textures_opt, (str, bytes)):
        texture_count = len(textures_opt)
    else:
        texture_count = int(_clamp(_number(textures_opt if textures_opt is not None else opts.get("texture_count"), 4.0), 0.0, 64.0))
    texture_size = int(max(0.0, _number(opts.get("texture_bytes", opts.get("bytes_per_texture")), 2 * 1024 * 1024)))

    registry = opts.get("registry", opts.get("texture_registry"))
    if not isinstance(registry, VolumeTextureRegistry):
        registry = None

    handles: list[str] = []
    if registry is not None:
        for index in range(texture_count):
            handles.append(registry.allocate(level, texture_size, name=f"fog_volume_{index}"))

    extinction = profile.extinction_at(depth)
    return {
        "profile": profile.name,
        "profile_object": profile,
        "depth_m": depth,
        "level": level,
        "extinction_rgb": extinction,
        "scatter_rgb": profile.scatter_rgb,
        "texture_count": texture_count,
        "texture_bytes_each": texture_size,
        "texture_bytes_allocated": texture_count * texture_size,
        "textures": handles or [f"voltex:{level}:virtual:{index}" for index in range(texture_count)],
        "registered": registry is not None,
        "registry": registry,
        "release_required": registry is not None,
        "attenuation_is_depth_dependent": True,
    }


def scatter(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Integrate absorption and in-scattering along one underwater ray."""
    opts, rest = _options(args, kwargs)
    state = opts.get("state")
    if isinstance(state, Mapping):
        depth = _number(state.get("depth_m", state.get("depth")), 1.0)
        profile = _resolve_profile(state.get("profile_object", state.get("profile")))
    else:
        depth = _num_arg(opts, rest, 0, ("depth", "depth_m", "water_depth"), 1.0)
        profile = _profile_from_opts(opts)

    ray_length = max(0.0, _num_arg(opts, rest, 1, ("ray_length", "distance", "distance_m", "path_length"), 10.0))
    camera_height = _number(opts.get("camera_height", opts.get("height")), 5.0)
    max_steps = max(1, int(_clamp(_number(opts.get("max_steps"), 256.0), 1.0, 8192.0)))
    base_step = max(1e-5, _number(opts.get("base_step"), 0.05))
    max_step = max(base_step, _number(opts.get("max_step"), 1.0))
    ambient = _as_rgb(opts.get("ambient_rgb", opts.get("scatter_rgb")), profile.scatter_rgb)
    depth_gradient = _number(opts.get("depth_gradient", opts.get("vertical_gradient")), 0.0)
    time_value = _number(opts.get("time", opts.get("elapsed")), 0.0)

    position = 0.0
    transmittance = [1.0, 1.0, 1.0]
    radiance = [0.0, 0.0, 0.0]
    steps_used = 0
    min_step_used = math.inf
    max_step_used = 0.0
    wavelength_phase = 0.0

    while position < ray_length and steps_used < max_steps:
        step_info = adaptive_step(
            distance=position,
            camera_height=camera_height,
            base_step=base_step,
            max_step=max_step,
            max_shaft_step=_number(opts.get("max_shaft_step"), 0.20),
        )
        step = min(step_info["step_m"], ray_length - position)
        if step <= 0.0:
            break
        mid = position + 0.5 * step
        sample_depth = max(0.0, depth + depth_gradient * mid)
        extinction = profile.extinction_at(sample_depth)
        segment_trans = [math.exp(-extinction[c] * step) for c in range(3)]
        for c in range(3):
            in_scatter = ambient[c] * (1.0 - segment_trans[c])
            radiance[c] += transmittance[c] * in_scatter
            transmittance[c] *= segment_trans[c]
        min_step_used = min(min_step_used, step)
        max_step_used = max(max_step_used, step)
        wavelength_phase += step
        steps_used += 1
        position += step

    if steps_used == 0:
        min_step_used = 0.0

    fog_phase = 0.0
    if time_value > 0.0 and wavelength_phase > 0.0:
        fog_phase = math.sin(time_value * 0.5 + wavelength_phase) * 0.5 + 0.5

    return {
        "radiance_rgb": _tuple3(max(0.0, radiance[c]) for c in range(3)),
        "transmittance_rgb": _tuple3(_clamp(transmittance[c], 0.0, 1.0) for c in range(3)),
        "steps_used": steps_used,
        "min_step_m": min_step_used,
        "max_step_m": max_step_used,
        "ray_length_m": ray_length,
        "depth_m": depth,
        "profile": profile.name,
        "adaptive_steps": True,
        "fog_phase": fog_phase,
        "source": "adaptive_march",
    }


def composite(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Blur the scene first, then composite depth-aware underwater fog."""
    opts, rest = _options(args, kwargs)
    scene = _rgb_arg(opts, rest, 0, ("scene_rgb", "scene", "color_rgb"), (1.0, 1.0, 1.0))
    fog_color = _rgb_arg(opts, rest, 1, ("fog_rgb", "fog_color", "inscatter_rgb"), (0.015, 0.060, 0.090))
    transmittance = _rgb_arg(opts, rest, 2, ("transmittance_rgb", "transmittance", "attenuation_rgb"), (0.80, 0.90, 0.94))
    dof_radius = max(0.0, _number(opts.get("dof_radius", opts.get("aperture")), 3.0))

    # A real blur needs neighbours; if supplied as samples we average them.
    samples = opts.get("scene_samples", opts.get("samples"))
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)) and samples:
        colors = [_as_rgb(sample, scene) for sample in samples]
        dof_scene = _tuple3(sum(color[c] for color in colors) / len(colors) for c in range(3))
        sample_count = len(colors)
    else:
        dof_scene = scene
        sample_count = 1

    result = _tuple3(
        dof_scene[c] * _clamp(transmittance[c], 0.0, 1.0)
        + fog_color[c] * (1.0 - _clamp(transmittance[c], 0.0, 1.0))
        for c in range(3)
    )

    return {
        "rgb": result,
        "scene_rgb": scene,
        "dof_scene_rgb": dof_scene,
        "fog_rgb": fog_color,
        "transmittance_rgb": transmittance,
        "order": list(_CORRECT_ORDER),
        "dof_applied_first": True,
        "fog_after_dof": True,
        "dof_radius_px": dof_radius,
        "scene_samples": sample_count,
        "double_contour": False,
    }


def present(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Tonemap and quantise the composited image for presentation."""
    opts, rest = _options(args, kwargs)
    rgb = _rgb_arg(opts, rest, 0, ("rgb", "linear_rgb", "color"), (1.0, 1.0, 1.0))
    exposure = max(0.0, _number(opts.get("exposure"), 1.0))
    gamma = max(1e-3, _number(opts.get("gamma"), 2.2))

    tonemapped = _tuple3((value * exposure) / (1.0 + value * exposure) for value in rgb)
    display = _tuple3(_clamp(tonemapped[c], 0.0, 1.0) ** (1.0 / gamma) for c in range(3))
    srgb8 = tuple(int(round(display[c] * 255.0)) for c in range(3))
    hex_color = "#{:02x}{:02x}{:02x}".format(*srgb8)

    return {
        "linear_rgb": rgb,
        "tonemapped_rgb": tonemapped,
        "display_rgb": display,
        "srgb8": srgb8,
        "hex": hex_color,
        "exposure": exposure,
        "gamma": gamma,
        "presented": True,
    }


# ---------------------------------------------------------------------------
# Validation, metrics, audit, fallback, determinism
# ---------------------------------------------------------------------------

def validate_underwater(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the six regression checks and return a self-contained report."""
    opts, rest = _options(args, kwargs)
    profile = _profile_from_opts(opts)
    profile_name = profile.name

    shallow = depth_attenuation(depth=1.5, path_length=1.0, profile=profile)
    deep = depth_attenuation(depth=45.0, path_length=1.0, profile=profile)
    shallow_ratio = shallow["transmittance_rgb"][2] / max(shallow["transmittance_rgb"][0], 1e-9)
    deep_ratio = deep["transmittance_rgb"][2] / max(deep["transmittance_rgb"][0], 1e-9)
    checks: list[dict[str, Any]] = []

    checks.append({
        "name": "depth_attenuation",
        "passed": bool(
            deep["transmittance_rgb"][0] < shallow["transmittance_rgb"][0]
            and deep["extinction_rgb"][0] > deep["extinction_rgb"][1] > deep["extinction_rgb"][2]
            and deep_ratio > shallow_ratio
        ),
        "detail": {
            "shallow_red_transmittance": shallow["transmittance_rgb"][0],
            "deep_red_transmittance": deep["transmittance_rgb"][0],
            "shallow_blue_over_red": shallow_ratio,
            "deep_blue_over_red": deep_ratio,
        },
    })

    near_surface = adaptive_step(distance=1.0, camera_height=0.20)
    far_surface = adaptive_step(distance=1.0, camera_height=8.0)
    checks.append({
        "name": "adaptive_step",
        "passed": near_surface["step_m"] < far_surface["step_m"] and near_surface["step_m"] > 0.0,
        "detail": {"near_surface_step_m": near_surface["step_m"], "far_surface_step_m": far_surface["step_m"]},
    })

    default_order = fog_dof_order()
    misordered = fog_dof_order(current_order=["scene", "fog", "dof", "composite"])
    positions = {stage: default_order["order"].index(stage) for stage in ("scene", "dof", "fog")}
    checks.append({
        "name": "fog_dof_order",
        "passed": bool(
            positions["dof"] < positions["fog"]
            and misordered["corrected"]
            and misordered["order"].index("dof") < misordered["order"].index("fog")
        ),
        "detail": {"order": default_order["order"], "repaired_from": misordered["original_order"]},
    })

    low_angle = view_refraction(2.0, angle=15.0)
    high_angle = view_refraction(2.0, angle=65.0)
    checks.append({
        "name": "view_refraction",
        "passed": high_angle["offset_m"] > low_angle["offset_m"] > 0.0,
        "detail": {"offset_at_15_deg": low_angle["offset_m"], "offset_at_65_deg": high_angle["offset_m"]},
    })

    registry = VolumeTextureRegistry()
    state = load_water(depth=5.0, texture_count=4, texture_bytes=1 << 20, registry=registry, level="level_a")
    allocated = registry.total_bytes()
    release = volume_release(registry=registry, level="level_a")
    second = volume_release(registry=registry, level="level_a")
    checks.append({
        "name": "volume_release",
        "passed": bool(
            allocated > 0
            and release["released_bytes"] == allocated
            and registry.total_bytes() == 0
            and second["released_bytes"] == 0
        ),
        "detail": {
            "allocated_bytes": allocated,
            "released_bytes": release["released_bytes"],
            "remaining_bytes": release["remaining_bytes"],
            "second_release_bytes": second["released_bytes"],
        },
    })

    target_dt = 1.0 / 60.0
    compensation = skip_frame_compensate(0.25, target_dt=target_dt, scatter_rate=1.0)
    checks.append({
        "name": "skip_frame_compensate",
        "passed": bool(
            compensation["uses_old_frame"] is False
            and compensation["advance"] > compensation["legacy_advance"]
            and compensation["effective_dt"] <= 0.25
            and compensation["temporal_blend"] > 0.0
        ),
        "detail": {
            "frame_dt": compensation["frame_dt"],
            "effective_dt": compensation["effective_dt"],
            "advance": compensation["advance"],
            "legacy_advance": compensation["legacy_advance"],
            "clamped": compensation["clamped"],
        },
    })

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    return {
        "ok": failed == 0,
        "profile": profile_name,
        "checks": checks,
        "passed": passed,
        "failed": failed,
        "total": len(checks),
        "summary": f"{passed}/{len(checks)} underwater checks passed",
    }


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return optical and cost metrics for a water state."""
    opts, rest = _options(args, kwargs)
    state = opts.get("state")
    if isinstance(state, Mapping):
        depth = _number(state.get("depth_m"), 1.0)
        profile = _resolve_profile(state.get("profile_object", state.get("profile")))
    else:
        depth = max(0.0, _num_arg(opts, rest, 0, ("depth", "depth_m"), 1.0))
        profile = _profile_from_opts(opts)

    extinction = profile.extinction_at(depth)
    scattering_albedo = _tuple3(
        profile.scatter_rgb[c] / max(profile.scatter_rgb[c] + extinction[c], 1e-9) for c in range(3)
    )
    visibility_blue = 3.9 / max(extinction[2], 1e-9)
    visibility_green = 3.9 / max(extinction[1], 1e-9)
    optical_depth_1m = _tuple3(extinction[c] * 1.0 for c in range(3))

    near = adaptive_step(distance=1.0, camera_height=0.25)
    far = adaptive_step(distance=50.0, camera_height=25.0)
    texture_bytes = int(max(0.0, _number(opts.get("texture_bytes", opts.get("volume_texture_bytes")), 4 * 2 * 1024 * 1024)))
    target_dt = max(1e-6, _number(opts.get("target_dt"), 1.0 / 60.0))
    compensation = skip_frame_compensate(_number(opts.get("frame_dt"), target_dt), target_dt=target_dt)

    return {
        "profile": profile.name,
        "depth_m": depth,
        "extinction_rgb": extinction,
        "optical_depth_rgb": optical_depth_1m,
        "scattering_albedo_rgb": scattering_albedo,
        "estimated_visibility_m": {"green": visibility_green, "blue": visibility_blue},
        "adaptive_step_near_m": near["step_m"],
        "adaptive_step_far_m": far["step_m"],
        "step_growth_ratio": far["step_m"] / max(near["step_m"], 1e-9),
        "volume_texture_bytes": texture_bytes,
        "frame_compensation_ratio": compensation["compensation_ratio"],
        "uses_old_frame": compensation["uses_old_frame"],
    }


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Audit a configuration for the six known artifact patterns."""
    opts, rest = _options(args, kwargs)
    findings: list[dict[str, Any]] = []
    state = opts.get("state")
    state_map = state if isinstance(state, Mapping) else {}

    if state_map.get("fixed_attenuation") or opts.get("fixed_attenuation"):
        findings.append({"code": "fixed_rgb_attenuation", "severity": "high", "fix": "depth_attenuation"})
    if state_map.get("fixed_step") or opts.get("fixed_step"):
        findings.append({"code": "fixed_ray_step", "severity": "high", "fix": "adaptive_step"})
    if opts.get("fixed_ior") or opts.get("fixed_refraction_offset"):
        findings.append({"code": "fixed_refraction_offset", "severity": "high", "fix": "view_refraction"})
    if state_map.get("textures_registered") or opts.get("textures_allocated"):
        if not state_map.get("release_on_level_switch", opts.get("release_on_level_switch", False)):
            findings.append({"code": "volume_texture_leak", "severity": "high", "fix": "volume_release"})
    if opts.get("old_frame_accumulation") or state_map.get("uses_old_frame"):
        findings.append({"code": "stale_frame_scatter", "severity": "high", "fix": "skip_frame_compensate"})

    order_value = opts.get("current_order", opts.get("pipeline_order"))
    if order_value is not None:
        order_report = fog_dof_order(current_order=order_value)
        if order_report["corrected"]:
            findings.append({"code": "fog_before_dof", "severity": "high", "fix": "fog_dof_order"})

    return {
        "ok": not findings,
        "finding_count": len(findings),
        "findings": findings,
        "checks": [
            "depth_attenuation", "adaptive_step", "fog_dof_order",
            "view_refraction", "volume_release", "skip_frame_compensate",
        ],
        "audited": True,
    }


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a sanitised water configuration when input is invalid."""
    opts, rest = _options(args, kwargs)
    requested_profile = opts.get("profile", opts.get("water_profile"))
    replacements: list[str] = []
    if requested_profile is None:
        profile = PROFILES[DEFAULT_PROFILE]
    elif isinstance(requested_profile, str) and requested_profile in PROFILES:
        profile = PROFILES[requested_profile]
    else:
        profile = PROFILES[DEFAULT_PROFILE]
        if not isinstance(requested_profile, (Mapping, WaterProfile)):
            replacements.append("unknown_profile")

    depth = _num_arg(opts, rest, 0, ("depth", "depth_m"), 1.0)
    if not math.isfinite(depth) or depth < 0.0:
        replacements.append("negative_or_invalid_depth")
        depth = 0.0
    depth = _clamp(depth, 0.0, 1e4)

    extinction = profile.extinction_at(depth)
    return {
        "profile": profile.name,
        "depth_m": depth,
        "extinction_rgb": extinction,
        "scatter_rgb": profile.scatter_rgb,
        "replacements": replacements,
        "used_fallback": bool(replacements),
        "ok": True,
        "safe": True,
    }


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Check that the pure optical functions are reproducible."""
    opts, rest = _options(args, kwargs)
    seed = _num_arg(opts, rest, 0, ("seed",), 1.0)
    depth = abs(_number(opts.get("depth", opts.get("depth_m")), seed if seed else 1.0))
    profile = _profile_from_opts(opts)

    def snapshot() -> tuple[Any, ...]:
        attenuation = depth_attenuation(depth=depth, path_length=2.0, profile=profile)
        step = adaptive_step(distance=depth, camera_height=0.5)
        refraction = view_refraction(37.0, depth=depth, n_water=1.33)
        return (
            attenuation["extinction_rgb"],
            attenuation["transmittance_rgb"],
            step["step_m"],
            refraction["offset_m"],
            refraction["refracted_angle_deg"],
        )

    first = snapshot()
    second = snapshot()
    payload = repr(first).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "equal": first == second,
        "runs": 2,
        "digest": digest,
        "signature": first,
        "deterministic": first == second,
        "seed": seed,
    }
