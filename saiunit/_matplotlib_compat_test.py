# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from types import SimpleNamespace

import numpy as np
import pytest

import saiunit as u
from saiunit import _matplotlib_compat as mpl_compat

if not mpl_compat.matplotlib_installed:
    pytest.skip("matplotlib is not available", allow_module_level=True)

if not u.enable_matplotlib_support():
    pytest.skip("matplotlib converter could not be registered", allow_module_level=True)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import units as mpl_units
from matplotlib.units import ConversionError


@pytest.fixture
def ax():
    """Provide a fresh Agg-backed axis and guarantee figure cleanup."""
    fig, axis = plt.subplots()
    try:
        yield axis
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_converter_registered_on_import():
    assert mpl_compat.matplotlib_converter_registered is True
    assert isinstance(mpl_units.registry[u.Quantity], mpl_compat.QuantityConverter)


def test_reshape_patch_installed_on_import():
    assert mpl_compat.matplotlib_reshape_patch_installed is True


def test_enable_matplotlib_support_is_idempotent():
    assert u.enable_matplotlib_support()
    assert u.enable_matplotlib_support()
    assert isinstance(mpl_units.registry[u.Quantity], mpl_compat.QuantityConverter)


def test_register_quantity_converter_handles_registry_variants():
    fake_units = SimpleNamespace(registry={})
    assert mpl_compat.register_quantity_converter(fake_units)
    assert isinstance(fake_units.registry[u.Quantity], mpl_compat.QuantityConverter)

    assert not mpl_compat.register_quantity_converter(SimpleNamespace())


# ---------------------------------------------------------------------------
# QuantityConverter unit methods
# ---------------------------------------------------------------------------

def test_convert_scales_values_to_target_unit():
    converted = mpl_compat.QuantityConverter.convert([101, 125, 150] * u.cmeter, u.meter, axis=None)
    np.testing.assert_allclose(np.asarray(converted), np.asarray([1.01, 1.25, 1.5]))


def test_convert_without_unit_returns_mantissa():
    converted = mpl_compat.QuantityConverter.convert([1, 2, 3] * u.meter, None, axis=None)
    np.testing.assert_allclose(np.asarray(converted), np.asarray([1.0, 2.0, 3.0]))


def test_convert_scalar_quantity():
    converted = mpl_compat.QuantityConverter.convert(3 * u.second, u.second, axis=None)
    np.testing.assert_allclose(np.asarray(converted), 3.0)


def test_convert_empty_quantity_array():
    converted = mpl_compat.QuantityConverter.convert(np.array([]) * u.meter, u.meter, axis=None)
    assert np.asarray(converted).size == 0


def test_convert_raises_actionable_error_for_incompatible_units():
    with pytest.raises(ConversionError, match="incompatible with target axis unit"):
        mpl_compat.QuantityConverter.convert([1, 2, 3] * u.second, u.meter, axis=None)


def test_axisinfo_uses_unit_label_and_supports_unitless():
    meter_info = mpl_compat.QuantityConverter.axisinfo(u.meter, axis=None)
    assert meter_info is not None
    assert meter_info.label == u.meter.dispname

    unitless_info = mpl_compat.QuantityConverter.axisinfo(u.UNITLESS, axis=None)
    assert unitless_info is not None


def test_default_units_returns_quantity_unit():
    assert mpl_compat.QuantityConverter.default_units([1, 2, 3] * u.ms, axis=None) == u.ms


# ---------------------------------------------------------------------------
# Core 0-d behavior that matplotlib relies on (regression guard)
# ---------------------------------------------------------------------------

def test_zero_d_quantity_is_not_iterable_like_numpy():
    # Matplotlib's ``_is_natively_supported`` calls ``np.iterable`` then iterates;
    # a 0-d Quantity must report False, matching ``np.iterable(np.array(5))``.
    assert np.iterable(3 * u.meter) is False
    with pytest.raises(TypeError):
        iter(3 * u.meter)


# ---------------------------------------------------------------------------
# Plotting integration
# ---------------------------------------------------------------------------

def test_plot_sets_axis_units_and_labels_from_quantity(ax):
    ax.plot(np.arange(3) * u.second, np.arange(3) * u.meter)
    assert ax.xaxis.get_units() == u.second
    assert ax.yaxis.get_units() == u.meter
    # axisinfo labels the axes with the unit's display name.
    assert ax.xaxis.label.get_text() == u.second.dispname
    assert ax.yaxis.label.get_text() == u.meter.dispname


def test_plot_second_line_keeps_first_axis_unit(ax):
    y = np.arange(3) * u.cmeter
    ax.plot(np.arange(3) * u.second, y)
    # The axis unit is fixed by the first plot (cm); plotting equivalent meter
    # data on the same axis must not raise and must keep the cm unit.
    ax.plot(np.arange(3) * u.second, y.to(u.meter))
    assert ax.yaxis.get_units() == u.cmeter


def test_plot_incompatible_unit_on_same_axis_raises(ax):
    ax.plot(np.arange(3) * u.second, np.arange(3) * u.meter)
    with pytest.raises(ConversionError):
        ax.plot(np.arange(3) * u.second, np.arange(3) * u.second)


def test_scatter_with_quantity_arrays(ax):
    ax.scatter(np.arange(3) * u.second, np.arange(3) * u.meter)
    assert ax.xaxis.get_units() == u.second
    assert ax.yaxis.get_units() == u.meter


def test_set_xlim_accepts_scalar_quantities(ax):
    ax.plot(np.arange(4) * u.second, np.arange(4) * u.meter)
    ax.set_xlim(0 * u.second, 3 * u.second)
    lo, hi = ax.get_xlim()
    np.testing.assert_allclose([lo, hi], [0.0, 3.0])


def test_axhline_accepts_scalar_quantity(ax):
    ax.plot(np.arange(4) * u.second, np.arange(4) * u.meter)
    line = ax.axhline(2 * u.meter)
    ydata = [v.to_decimal(u.meter) for v in line.get_ydata()]
    np.testing.assert_allclose(ydata, [2.0, 2.0])


def test_axvline_accepts_scalar_quantity(ax):
    ax.plot(np.arange(4) * u.second, np.arange(4) * u.meter)
    line = ax.axvline(1 * u.second)
    xdata = [v.to_decimal(u.second) for v in line.get_xdata()]
    np.testing.assert_allclose(xdata, [1.0, 1.0])


def test_plot_supports_jax_backed_quantities(ax):
    jnp = pytest.importorskip("jax.numpy")
    ax.plot(jnp.arange(3) * u.second, jnp.arange(3) * u.meter)
    assert ax.xaxis.get_units() == u.second
    assert ax.yaxis.get_units() == u.meter


# ---------------------------------------------------------------------------
# hist integration (relies on the _reshape_2D patch)
# ---------------------------------------------------------------------------

def test_hist_single_dataset_preserves_units(ax):
    counts, edges, _ = ax.hist(np.arange(10) * u.cmeter, bins=5)
    assert ax.xaxis.get_units() == u.cmeter
    assert counts.sum() == 10


def test_hist_converts_data_to_axis_unit(ax):
    # Data given in cm should be binned on the cm axis (numeric magnitude 0..9).
    _, edges, _ = ax.hist(np.arange(10) * u.cmeter, bins=3)
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(9.0)


def test_hist_multiple_quantity_datasets(ax):
    counts, _, _ = ax.hist([np.arange(10) * u.meter, np.arange(5) * u.meter], bins=4)
    assert len(counts) == 2


# ---------------------------------------------------------------------------
# _reshape_2D patch unit behavior
# ---------------------------------------------------------------------------

def test_reshape_2d_patch_handles_quantity_shapes():
    from matplotlib import cbook

    one_d = np.arange(4) * u.meter
    result = cbook._reshape_2D(one_d, "x")
    assert len(result) == 1
    assert isinstance(result[0], u.Quantity)

    scalar = 3 * u.meter
    result = cbook._reshape_2D(scalar, "x")
    assert len(result) == 1
    assert result[0].shape == (1,)

    datasets = [np.arange(4) * u.meter, np.arange(2) * u.meter]
    result = cbook._reshape_2D(datasets, "x")
    assert len(result) == 2


def test_reshape_2d_patch_delegates_for_plain_arrays():
    from matplotlib import cbook

    result = cbook._reshape_2D(np.arange(6).reshape(3, 2), "x")
    assert len(result) == 2  # columns
    assert all(isinstance(col, np.ndarray) for col in result)
