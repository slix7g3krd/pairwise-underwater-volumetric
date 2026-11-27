"""Behavioural regression tests for the six underwater defects."""

import math

import graphics.underwater_volume as m


# 1. Depth dependent attenuation -------------------------------------------

def test_deep_water_attenuates_red_before_blue():
    shallow = m.depth_attenuation(depth=1.5, path_length=1.0)
    deep = m.depth_attenuation(depth=45.0, path_length=1.0)

    assert shallow["red_vanishes_first"] is True
    assert deep["red_vanishes_first"] is True
    assert deep["extinction_rgb"][0] > deep["extinction_rgb"][1] > deep["extinction_rgb"][2]
    assert deep["transmittance_rgb"][0] < shallow["transmittance_rgb"][0]


def test_shallow_water_is_not_over_blue():
    shallow = m.depth_attenuation(depth=1.5, path_length=1.0)
    deep = m.depth_attenuation(depth=45.0, path_length=1.0)
    shallow_ratio = shallow["transmittance_rgb"][2] / shallow["transmittance_rgb"][0]
    deep_ratio = deep["transmittance_rgb"][2] / deep["transmittance_rgb"][0]
    assert shallow_ratio < deep_ratio
    assert shallow["attenuation_is_depth_dependent"] is True
    assert shallow["regime"] == "shallow"


def test_attenuation_scales_with_path_length():
    short = m.depth_attenuation(depth=10.0, path_length=1.0)
    long = m.depth_attenuation(depth=10.0, path_length=4.0)
    assert all(long["transmittance_rgb"][c] < short["transmittance_rgb"][c] for c in range(3))


def test_water_profile_changes_extinction():
    clear = m.depth_attenuation(depth=10.0, profile="clear")
    turbid = m.depth_attenuation(depth=10.0, profile="turbid")
    assert turbid["extinction_rgb"][0] > clear["extinction_rgb"][0]


# 2. Adaptive step ---------------------------------------------------------

def test_step_grows_with_distance():
    near = m.adaptive_step(distance=1.0, camera_height=10.0)
    far = m.adaptive_step(distance=60.0, camera_height=10.0)
    assert far["step_m"] > near["step_m"] > 0.0


def test_step_shrinks_when_camera_hugs_surface():
    hugging = m.adaptive_step(distance=1.0, camera_height=0.2)
    high = m.adaptive_step(distance=1.0, camera_height=8.0)
    assert hugging["step_m"] < high["step_m"]
    assert hugging["steps_estimate"] > high["steps_estimate"]


def test_step_respects_limits():
    step = m.adaptive_step(distance=1000.0, camera_height=0.01, max_step=1.0)
    assert step["step_m"] <= 1.0
    assert step["step_m"] > 0.0
    assert step["recommended_steps"] >= 1


# 3. Fog / DoF order -------------------------------------------------------

def test_default_order_blurs_before_fog():
    report = m.fog_dof_order()
    order = report["order"]
    assert order.index("dof") < order.index("fog")
    assert report["dof_before_fog"] is True
    assert report["double_contour_risk"] == 0.0


def test_misordered_pipeline_is_repaired():
    report = m.fog_dof_order(current_order=["scene", "fog", "dof", "composite"])
    assert report["corrected"] is True
    assert report["original_order"] == ["scene", "fog", "dof", "composite"]
    assert report["order"].index("dof") < report["order"].index("fog")
    assert report["artifact_fixed"] == "double_contour"


def test_composite_applies_dof_before_fog():
    result = m.composite((1.0, 1.0, 1.0), (0.0, 0.2, 0.4), (0.5, 0.6, 0.7))
    assert result["dof_applied_first"] is True
    assert result["fog_after_dof"] is True
    assert result["double_contour"] is False


# 4. View dependent refraction ---------------------------------------------

def test_refraction_offset_depends_on_view_angle():
    low = m.view_refraction(2.0, angle=15.0)
    high = m.view_refraction(2.0, angle=65.0)
    assert 0.0 < low["offset_m"] < high["offset_m"]
    assert low["offset_depends_on_view_angle"] is True
    assert low["fixed_ior_offset"] is False


def test_refraction_follows_snells_law():
    report = m.view_refraction(3.0, angle=40.0)
    theta1 = math.radians(40.0)
    expected = math.degrees(math.asin(math.sin(theta1) / 1.33))
    assert math.isclose(report["refracted_angle_deg"], expected, rel_tol=1e-9)
    assert report["offset_m"] > 0.0


def test_refraction_flips_direction_for_camera_underwater():
    above = m.view_refraction(2.0, angle=30.0, from_medium="air")
    below = m.view_refraction(2.0, angle=30.0, from_medium="water")
    assert above["to_medium"] == "water"
    assert below["to_medium"] == "air"
    assert below["offset_m"] != above["offset_m"]


def test_total_internal_reflection_from_water():
    report = m.view_refraction(2.0, angle=60.0, from_medium="water")
    assert report["total_internal_reflection"] is True
    assert report["transmitted"] is False
    assert report["critical_angle_deg"] is not None


# 5. Volume texture release ------------------------------------------------

def test_level_switch_releases_its_textures_only():
    registry = m.VolumeTextureRegistry()
    m.load_water(depth=5.0, texture_count=3, texture_bytes=1024, registry=registry, level="a")
    m.load_water(depth=8.0, texture_count=2, texture_bytes=1024, registry=registry, level="b")
    a_bytes = 3 * 1024
    assert registry.total_bytes() == (3 + 2) * 1024

    report = m.volume_release(registry=registry, level="a")
    assert report["released_bytes"] == a_bytes
    assert registry.total_bytes() == 2 * 1024
    assert report["remaining_bytes"] == 2 * 1024


def test_release_is_idempotent_and_clears_leak():
    registry = m.VolumeTextureRegistry()
    m.load_water(depth=5.0, texture_count=4, texture_bytes=1 << 20, registry=registry, level="a")

    first = m.volume_release(registry=registry, level="a")
    second = m.volume_release(registry=registry, level="a")
    assert first["released_textures"] == 4
    assert first["remaining_bytes"] == 0
    assert second["released_textures"] == 0
    assert second["released_bytes"] == 0


def test_repeated_level_loads_stay_bounded():
    """Loading and switching levels repeatedly must not grow GPU memory."""
    registry = m.VolumeTextureRegistry()
    peak = 0
    for index in range(200):
        level = f"level_{index % 2}"
        # A level switch: free the other level, then load this one.
        m.volume_release(registry=registry, level=f"level_{(index + 1) % 2}")
        m.load_water(depth=5.0, texture_count=4, texture_bytes=1024, registry=registry, level=level)
        peak = max(peak, registry.total_bytes())
    assert peak == 4 * 1024
    assert registry.total_textures() == 4


def test_release_all_frees_every_level():
    registry = m.VolumeTextureRegistry()
    m.load_water(depth=5.0, texture_count=2, texture_bytes=512, registry=registry, level="a")
    m.load_water(depth=5.0, texture_count=3, texture_bytes=512, registry=registry, level="b")
    report = m.volume_release(registry=registry)
    assert report["freed_all_requested"] is True
    assert report["remaining_bytes"] == 0
    assert registry.total_textures() == 0


# 6. Frame-skip compensation ----------------------------------------------

def test_frame_spike_advances_by_elapsed_time():
    target = 1.0 / 60.0
    normal = m.skip_frame_compensate(target, target_dt=target)
    spike = m.skip_frame_compensate(0.25, target_dt=target)
    assert spike["uses_old_frame"] is False
    assert spike["advance"] > normal["advance"] > 0.0
    assert spike["effective_dt"] > target


def test_catchup_is_clamped_to_avoid_spiral_of_death():
    report = m.skip_frame_compensate(0.5, target_dt=1.0 / 60.0, max_catchup_ratio=4.0)
    assert report["clamped"] is True
    assert report["effective_dt"] == 4.0 * (1.0 / 60.0)
    assert report["substeps"] <= 4


def test_accumulator_uses_compensated_delta():
    report = m.skip_frame_compensate(0.1, target_dt=1.0 / 60.0, accumulated=5.0, scatter_rate=2.0)
    assert report["accumulated"] == 5.0 + report["advance"]
    assert report["accumulated"] > 5.0


def test_normal_frame_needs_no_compensation():
    report = m.skip_frame_compensate(1.0 / 60.0, target_dt=1.0 / 60.0)
    assert report["compensation_ratio"] == 1.0
    assert report["lag_ms"] == 0.0


# Pipeline / support entry points -----------------------------------------

def test_full_pipeline_runs():
    water = m.load_water(depth=12.0)
    scattering = m.scatter(state=water, ray_length=30.0)
    image = m.composite(scattering["radiance_rgb"], (0.01, 0.05, 0.08), scattering["transmittance_rgb"])
    frame = m.present(image["rgb"])
    assert 0 <= frame["srgb8"][0] <= 255
    assert len(frame["hex"]) == 7


def test_validate_underwater_reports_all_checks_passing():
    report = m.validate_underwater()
    assert report["ok"] is True
    assert report["failed"] == 0
    assert report["total"] == 6


def test_audit_detects_known_defects():
    report = m.audit(
        current_order=["scene", "fog", "dof", "composite"],
        fixed_attenuation=True,
        fixed_step=True,
        fixed_ior=True,
    )
    codes = {finding["code"] for finding in report["findings"]}
    assert {"fixed_rgb_attenuation", "fixed_ray_step", "fixed_refraction_offset", "fog_before_dof"} <= codes


def test_fallback_sanitises_invalid_input():
    report = m.fallback(depth=-3.0, profile="bogus")
    assert report["ok"] is True
    assert report["depth_m"] == 0.0
    assert "negative_or_invalid_depth" in report["replacements"]


def test_determinism_is_reproducible():
    first = m.determinism(seed=7)
    second = m.determinism(seed=7)
    assert first["equal"] is True
    assert first["digest"] == second["digest"]
