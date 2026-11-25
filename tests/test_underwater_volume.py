import pytest
import graphics.underwater_volume as mod

CASES = [
    ("depth_attenuation", (1,)),
    ("adaptive_step", (1,)),
    ("fog_dof_order", (1,)),
    ("view_refraction", (1,)),
    ("volume_release", (1,)),
    ("skip_frame_compensate", (1,)),
    ("load_water", (1,)),
    ("scatter", (1,)),
    ("composite", (1,)),
    ("present", (1,)),
    ("validate_underwater", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
