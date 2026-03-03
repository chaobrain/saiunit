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

if not mpl_compat.matplotlib_converter_registered:
    pytest.skip("matplotlib converter is not available", allow_module_level=True)

from matplotlib import units as mpl_units
from matplotlib.units import ConversionError


def test_quantity_converter_is_registered():
    assert mpl_compat.register_quantity_converter()
    assert isinstance(mpl_units.registry[u.Quantity], mpl_compat.QuantityConverter)


def test_convert_scales_values_to_target_unit():
    converted = mpl_compat.QuantityConverter.convert([101, 125, 150] * u.cmeter, u.meter, axis=None)
    np.testing.assert_allclose(np.asarray(converted), np.asarray([1.01, 1.25, 1.5]))


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


def test_register_quantity_converter_handles_registry_variants():
    fake_units = SimpleNamespace(registry={})
    assert mpl_compat.register_quantity_converter(fake_units)
    assert isinstance(fake_units.registry[u.Quantity], mpl_compat.QuantityConverter)

    assert not mpl_compat.register_quantity_converter(SimpleNamespace())
