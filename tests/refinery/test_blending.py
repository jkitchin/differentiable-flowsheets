"""Tests for the product blending pool (difflow_refinery.blending)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.streams import make_stream
from difflow_refinery import (
    BlendComponent,
    BlendPool,
    BlendSpec,
    BlendCharacterization,
    cetane_index_d4737,
    cetane_index_d976,
    ethyl_rt70,
    flash_point_blend,
    raoult_rvp,
    refutas_blend,
    refutas_vbn,
    refutas_viscosity,
    rvp_index_blend,
    tbp_evaporated,
    tbp_temperature,
    tbp_to_d86,
    temperature_index_blend,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _grid():
    """n-butane plus a naphtha-to-diesel pseudocomponent grid."""
    names = ["nC4"] + [f"pc{i}" for i in range(14)]
    Tb = [272.66] + list(np.linspace(320.0, 640.0, 14))
    SG = [0.584] + list(np.linspace(0.66, 0.88, 14))
    nan = float("nan")
    return BlendCharacterization(
        names=names, Tb=Tb, SG=SG,
        MW=[58.12] + [nan] * 14, Tc=[425.12] + [nan] * 14,
        Pc=[37.96e5] + [nan] * 14, omega=[0.200] + [nan] * 14,
        qualities={
            "S_ppm": [0.0] + list(np.linspace(5.0, 4000.0, 14)),
            "aromatics_vol": [0.0] + list(np.linspace(5.0, 30.0, 14)),
            "olefins_vol": [0.0] * 15,
        })


@pytest.fixture(scope="module")
def char():
    return _grid()


def _stream(char, weights, scale=10.0):
    return make_stream({n: scale * w for n, w in zip(char.names, weights)},
                       T=310.0, P=2e5)


@pytest.fixture(scope="module")
def stream_components(char):
    n = char.n
    light = np.zeros(n); light[1:6] = [3, 4, 4, 3, 2]
    heavy = np.zeros(n); heavy[4:10] = [1, 2, 3, 3, 2, 1]
    butane = np.zeros(n); butane[0] = 1.0
    return [
        BlendComponent.from_stream("lsr", _stream(char, light), char,
                                   RON=70.0, MON=68.0),
        BlendComponent.from_stream("reformate", _stream(char, heavy), char,
                                   RON=98.0, MON=88.0, aromatics_vol=65.0),
        BlendComponent.from_stream("butane", _stream(char, butane, 2.0),
                                   char, RON=93.0, MON=92.0),
    ]


def _gasoline_components(**kw):
    base = dict(
        reformate=dict(SG=0.80, RON=98.0, MON=88.0, RVP_psi=3.5, S_ppm=1.0,
                       olefins_vol=1.0, aromatics_vol=65.0),
        fcc=dict(SG=0.74, RON=92.0, MON=80.0, RVP_psi=7.0, S_ppm=20.0,
                 olefins_vol=30.0, aromatics_vol=25.0),
        alkylate=dict(SG=0.70, RON=96.0, MON=93.5, RVP_psi=4.5, S_ppm=5.0,
                      olefins_vol=0.5, aromatics_vol=0.5),
        butane=dict(SG=0.584, RON=93.0, MON=92.0, RVP_psi=51.6, S_ppm=1.0,
                    olefins_vol=0.0, aromatics_vol=0.0),
    )
    for name, over in kw.items():
        base[name].update(over)
    return [BlendComponent.from_properties(n, **p) for n, p in base.items()]


RECIPE = jnp.array([0.35, 0.35, 0.25, 0.05])


def test_rules_finite():
    v = jnp.array([0.6, 0.4])
    vals = [
        refutas_blend(v, jnp.array([0.95, 0.95]), jnp.array([2.5, 700.0])),
        refutas_viscosity(refutas_vbn(10.0)),
        *ethyl_rt70(v, jnp.array([98.0, 92.0]), jnp.array([88.0, 80.0]),
                    jnp.array([1.0, 30.0]), jnp.array([65.0, 25.0])),
        rvp_index_blend(v, jnp.array([7.0, 51.6])),
        raoult_rvp(jnp.array([3.0]), jnp.array([2.5e5]), jnp.array([1e-4])),
        flash_point_blend(v, jnp.array([320.0, 340.0])),
        temperature_index_blend(v, jnp.array([253.15, 268.15]), 20.0),
        cetane_index_d4737(0.85, 215.0, 260.0, 310.0),
        cetane_index_d976(0.84, 270.0),
        tbp_to_d86(400.0, 50),
        tbp_temperature(0.5, jnp.array([330.0, 400.0]), jnp.array([0.5, 0.5]), 5.0),
        tbp_evaporated(360.0, jnp.array([330.0, 400.0]), jnp.array([0.5, 0.5]), 5.0),
    ]
    assert all(np.isfinite(float(x)) for x in vals)


def test_characterization_rejects_foreign_species(char):
    with pytest.raises(ValueError, match="outside the characterization"):
        char.flows(make_stream({"benzene": 1.0}, T=300.0, P=1e5))


def test_pool_property_mode_smoke():
    res = BlendPool("gasoline")(_gasoline_components(), RECIPE)
    assert res.stream is None
    assert set(res.margins) == {"RON >= 91", "MON >= 82", "RVP_psi <= 9", "S_ppm <= 10"}
    assert all(np.isfinite(float(v)) for v in res.properties.values())
    assert float(res.margins["RON >= 91"]) == pytest.approx(float(res.properties["RON"]) - 91.0)


def test_pool_stream_mode_smoke(char, stream_components):
    res = BlendPool("gasoline")(stream_components, [0.6, 0.3, 0.1])
    assert res.stream is not None
    assert all(np.isfinite(float(v)) for v in res.margins.values())
    assert np.isfinite(float(res.properties["RVP_psi"]))


def test_other_products():
    jet = [BlendComponent.from_properties(
               "kero", SG=0.80, S_ppm=1500.0, freeze_C=-47.0,
               smoke_mm=22.0, flash_C=45.0),
           BlendComponent.from_properties(
               "hck", SG=0.81, S_ppm=10.0, freeze_C=-38.0,
               smoke_mm=26.0, flash_C=40.0)]
    res = BlendPool("jet")(jet, [0.6, 0.4])
    assert -47.0 < float(res.properties["freeze_C"]) < -38.0
    assert 40.0 < float(res.properties["flash_C"]) < 45.0
    fo = [BlendComponent.from_properties(
              "resid", SG=1.00, S_ppm=9000.0, viscosity_cSt=3000.0,
              CCR_wt=20.0),
          BlendComponent.from_properties(
              "cutter", SG=0.86, S_ppm=100.0, viscosity_cSt=3.0,
              CCR_wt=0.1)]
    res = BlendPool("fuel_oil")(fo, [0.6, 0.4])
    assert set(res.margins) == {"S_ppm <= 5000", "viscosity_cSt <= 380",
                                "SG <= 0.991", "CCR_wt <= 18"}


def test_errors():
    comps = _gasoline_components()
    with pytest.raises(ValueError, match="no default specs"):
        BlendPool("asphalt")
    with pytest.raises(ValueError, match="not available"):
        BlendPool(rules={"RON": "refutas"})
    with pytest.raises(ValueError, match="unknown properties"):
        BlendComponent.from_properties("x", SG=0.7, octane=90)
    with pytest.raises(ValueError, match="needs an SG"):
        BlendComponent.from_properties("x", RON=90.0)
    no_olefins = [BlendComponent.from_properties(
        "x", SG=0.7, RON=90.0, MON=80.0, RVP_psi=7.0, S_ppm=5.0)] * 2
    with pytest.raises(KeyError, match="Ethyl RT-70"):
        BlendPool("gasoline")(no_olefins, [0.5, 0.5])
    with pytest.raises(KeyError, match="has no 'cetane_index'"):
        BlendPool("ulsd")(comps, RECIPE)
    with pytest.raises(ValueError, match="recipe needs 4"):
        BlendPool("gasoline")(comps, [0.5, 0.5])


def test_mixed_modes_rejected(stream_components):
    prop = BlendComponent.from_properties(
        "x", SG=0.7, RON=90.0, MON=80.0, olefins_vol=0.0,
        aromatics_vol=0.0, RVP_psi=5.0)
    with pytest.raises(ValueError, match="mix of stream-mode"):
        BlendPool("gasoline", specs=[])(
            list(stream_components) + [prop], [0.25] * 4)


def test_finite_gradient_at_an_active_spec():
    comps = _gasoline_components()
    ron = float(BlendPool("gasoline")(comps, RECIPE).properties["RON"])
    pool = BlendPool("gasoline", specs=[BlendSpec("RON", ">=", ron)])
    m = pool.spec_margins(comps, RECIPE)
    assert abs(float(m[0])) < 1e-12   # the spec is exactly active
    g = jax.grad(lambda r: jnp.sum(pool.spec_violations(
        comps, r, temperature=0.05)))(RECIPE)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.linalg.norm(g)) > 0.0
    # and it is half the margin's own gradient, with the sign flipped
    gm = jax.grad(lambda r: pool.spec_margins(comps, r)[0])(RECIPE)
    np.testing.assert_allclose(np.asarray(g), -0.5 * np.asarray(gm),
                               rtol=1e-10, atol=1e-14)


def test_as_block_builds(stream_components):
    blk = BlendPool("gasoline").as_block(stream_components)
    assert blk.u_names == ["lsr_V", "reformate_V", "butane_V"]
    y = blk.fn(0.5 * jnp.asarray(blk.ub))
    assert y.shape == (len(blk.y_names),) and np.all(np.isfinite(np.asarray(y)))
