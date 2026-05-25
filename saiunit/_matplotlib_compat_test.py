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


# ---------------------------------------------------------------------------
# Axes-method wrappers: registration
# ---------------------------------------------------------------------------

def test_axes_wrappers_installed_on_import():
    assert mpl_compat.matplotlib_axes_patch_installed is True


# ---------------------------------------------------------------------------
# Converter passthrough for plain (non-Quantity) values
# ---------------------------------------------------------------------------

def test_convert_passes_plain_values_through():
    # A bare number on a unit-bearing axis is assumed already in axis units.
    assert mpl_compat.QuantityConverter.convert(2.0, u.meter, axis=None) == 2.0
    np.testing.assert_allclose(
        mpl_compat.QuantityConverter.convert([1, 2, 3], u.meter, axis=None),
        np.asarray([1.0, 2.0, 3.0]),
    )


def test_axhspan_accepts_scalar_quantities(ax):
    ax.plot(np.arange(3) * u.second, np.arange(3) * u.meter)
    patch = ax.axhspan(1 * u.meter, 2 * u.meter)
    assert patch is not None


def test_axvspan_accepts_scalar_quantities(ax):
    ax.plot(np.arange(3) * u.second, np.arange(3) * u.meter)
    patch = ax.axvspan(0.5 * u.second, 1.5 * u.second)
    assert patch is not None


def test_plain_number_on_quantity_axis(ax):
    ax.plot(np.arange(3) * u.second, np.arange(3) * u.meter)
    ax.set_ylim(0, 5)  # plain numbers interpreted as meters
    lo, hi = ax.get_ylim()
    np.testing.assert_allclose([lo, hi], [0.0, 5.0])


# ---------------------------------------------------------------------------
# errorbar
# ---------------------------------------------------------------------------

def test_errorbar_sets_axis_units_and_labels(ax):
    ax.errorbar(
        np.arange(5) * u.second,
        np.arange(5) * u.meter,
        yerr=np.ones(5) * 50 * u.cmeter,
        xerr=np.ones(5) * 0.1 * u.second,
    )
    assert ax.xaxis.get_units() == u.second
    assert ax.yaxis.get_units() == u.meter
    assert ax.yaxis.label.get_text() == u.meter.dispname


def test_errorbar_scales_err_into_axis_unit(ax):
    # y in meters, yerr supplied in cm -> err must be rescaled to meters (0.5 m).
    container = ax.errorbar(
        np.arange(3) * u.second,
        np.arange(3) * u.meter,
        yerr=np.ones(3) * 50 * u.cmeter,
    )
    line = container.lines[0]
    np.testing.assert_allclose(np.asarray(line.get_ydata()), [0.0, 1.0, 2.0])


def test_errorbar_incompatible_err_unit_raises(ax):
    with pytest.raises(Exception):
        ax.errorbar(
            np.arange(3) * u.second,
            np.arange(3) * u.meter,
            yerr=np.ones(3) * u.volt,
        )


# ---------------------------------------------------------------------------
# boxplot / violinplot
# ---------------------------------------------------------------------------

def test_boxplot_vertical_sets_value_axis_unit(ax):
    ax.boxplot(np.arange(10) * u.cmeter)
    assert ax.yaxis.get_units() == u.cmeter


def test_boxplot_horizontal_sets_x_axis_unit(ax):
    try:
        ax.boxplot(np.arange(10) * u.cmeter, orientation="horizontal")
    except TypeError:
        ax.boxplot(np.arange(10) * u.cmeter, vert=False)
    assert ax.xaxis.get_units() == u.cmeter


def test_boxplot_converts_into_existing_axis_unit(ax):
    # Establish the y-axis as meters, then a cm boxplot must rescale to meters.
    ax.plot(np.arange(3) * u.second, np.arange(3) * u.meter)
    result = ax.boxplot(np.array([100.0, 200.0, 300.0]) * u.cmeter)
    assert ax.yaxis.get_units() == u.meter
    median = result["medians"][0].get_ydata()
    np.testing.assert_allclose(np.asarray(median), [2.0, 2.0])  # 200 cm -> 2 m


def test_violinplot_sets_value_axis_unit(ax):
    ax.violinplot(np.linspace(0, 1, 50) * u.volt)
    assert ax.yaxis.get_units() == u.volt


# ---------------------------------------------------------------------------
# stackplot
# ---------------------------------------------------------------------------

def test_stackplot_multiple_series_share_unit(ax):
    ax.stackplot(
        np.arange(5) * u.second,
        np.arange(5) * u.meter,
        np.arange(5) * u.meter,
    )
    assert ax.xaxis.get_units() == u.second
    assert ax.yaxis.get_units() == u.meter


# ---------------------------------------------------------------------------
# hexbin / pie (data stripped where units have no axis meaning)
# ---------------------------------------------------------------------------

def test_hexbin_sets_axis_units(ax):
    ax.hexbin(
        np.linspace(0, 1, 50) * u.second,
        np.linspace(0, 1, 50) * u.meter,
        C=np.linspace(0, 1, 50) * u.volt,
        gridsize=5,
    )
    assert ax.xaxis.get_units() == u.second
    assert ax.yaxis.get_units() == u.meter


def test_pie_accepts_quantity_without_crashing(ax):
    wedges, _ = ax.pie(np.arange(1, 5) * u.meter)
    assert len(wedges) == 4


# ---------------------------------------------------------------------------
# Non-Quantity inputs must be untouched (fast-path identity)
# ---------------------------------------------------------------------------

def test_wrappers_leave_plain_inputs_unchanged(ax):
    ax.boxplot([1, 2, 3, 4, 5])
    ax.errorbar([0, 1, 2], [0, 1, 2], yerr=[0.1, 0.1, 0.1])
    ax.pie([1, 2, 3])
    # No unit attached because nothing was a Quantity.
    assert ax.yaxis.get_units() is None


# ---------------------------------------------------------------------------
# Fail-loud signature guard
# ---------------------------------------------------------------------------

def test_signature_guard_detects_missing_params():
    def good(self, x, y):
        return None

    def bad(self, a, b):
        return None

    assert mpl_compat._signature_has_params(good, ("x", "y"))
    assert not mpl_compat._signature_has_params(bad, ("x", "y"))


def test_unsupported_wrapper_raises_on_quantity_only(ax):
    calls = []

    def fake_original(self, x):
        calls.append(x)
        return "called"

    wrapper = mpl_compat._make_axes_wrapper(
        fake_original, mpl_compat._pre_pie, "pie", supported=False
    )
    # Plain input still passes through to the original.
    assert wrapper(ax, [1, 2, 3]) == "called"
    # Quantity input raises a clear, actionable error.
    with pytest.raises(Exception, match="changed its signature"):
        wrapper(ax, np.arange(3) * u.meter)
