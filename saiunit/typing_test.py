# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Tests for saiunit.typing — unit-aware type annotations."""

import typing

import pytest

import saiunit as u
from saiunit.typing import (
    PhysicalType,
    is_physical_type,
    quantity_type,
    validate_units,
)


# =========================================================================
# PhysicalType
# =========================================================================

class TestPhysicalType:
    def test_known_type(self):
        pt = PhysicalType("length")
        assert pt.physical_type == "length"
        assert pt.dimension == u.meter.dim

    def test_case_insensitive(self):
        pt = PhysicalType("Length")
        assert pt.physical_type == "length"

    def test_whitespace_stripped(self):
        pt = PhysicalType("  speed  ")
        assert pt.physical_type == "speed"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown physical type"):
            PhysicalType("bogus")

    def test_repr(self):
        pt = PhysicalType("voltage")
        assert repr(pt) == "PhysicalType('voltage')"

    def test_equality(self):
        assert PhysicalType("length") == PhysicalType("length")
        assert PhysicalType("length") != PhysicalType("mass")

    def test_hashable(self):
        s = {PhysicalType("length"), PhysicalType("length"), PhysicalType("mass")}
        assert len(s) == 2

    def test_derived_dimensions(self):
        pt = PhysicalType("speed")
        # speed = m / s  →  length=1, time=-1
        assert pt.dimension.get_dimension("m") == 1.0
        assert pt.dimension.get_dimension("s") == -1.0

    def test_force_dimension(self):
        pt = PhysicalType("force")
        assert pt.dimension == u.newton.dim

    def test_energy_dimension(self):
        pt = PhysicalType("energy")
        assert pt.dimension == u.joule.dim

    def test_voltage_dimension(self):
        pt = PhysicalType("voltage")
        assert pt.dimension == u.volt.dim

    def test_is_physical_type_check(self):
        pt = PhysicalType("length")
        assert is_physical_type(pt)
        assert not is_physical_type(42)
        assert not is_physical_type(u.meter)

    def test_cached_identity(self):
        """Same physical type returns the same object."""
        pt1 = PhysicalType("length")
        pt2 = PhysicalType("length")
        assert pt1 is pt2


# =========================================================================
# isinstance with PhysicalType
# =========================================================================

class TestPhysicalTypeIsinstance:
    def test_quantity_matches_dimension(self):
        q = 5.0 * u.meter
        assert isinstance(q, PhysicalType("length"))

    def test_quantity_wrong_dimension(self):
        q = 5.0 * u.meter
        assert not isinstance(q, PhysicalType("mass"))

    def test_different_units_same_dimension(self):
        q_km = 2.0 * u.kmeter
        q_m = 100.0 * u.meter
        assert isinstance(q_km, PhysicalType("length"))
        assert isinstance(q_m, PhysicalType("length"))

    def test_compound_unit(self):
        q = 3.0 * u.meter / u.second
        assert isinstance(q, PhysicalType("speed"))
        assert not isinstance(q, PhysicalType("length"))

    def test_non_quantity_returns_false(self):
        assert not isinstance(42, PhysicalType("length"))
        assert not isinstance("hello", PhysicalType("length"))
        assert not isinstance(3.14, PhysicalType("mass"))

    def test_dimensionless(self):
        q = u.Quantity(5.0)
        assert isinstance(q, PhysicalType("dimensionless"))
        assert not isinstance(q, PhysicalType("length"))


# =========================================================================
# Quantity.__class_getitem__ + isinstance
# =========================================================================

class TestQuantityClassGetitem:
    def test_with_unit(self):
        ann = u.Quantity[u.meter]
        assert isinstance(ann, type)  # it's a class

    def test_with_string(self):
        ann = u.Quantity["length"]
        assert isinstance(ann, type)

    def test_invalid_key_raises(self):
        with pytest.raises(TypeError, match="Quantity\\[...\\] expects"):
            u.Quantity[42]

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown physical type"):
            u.Quantity["nonexistent"]

    def test_isinstance_with_unit(self):
        """isinstance(x, Quantity[u.meter]) checks dimension compatibility."""
        x = 2.0 * u.kmeter
        assert isinstance(x, u.Quantity[u.meter])  # same dim: length
        assert not isinstance(x, u.Quantity[u.second])  # different dim

    def test_isinstance_with_string(self):
        """isinstance(x, Quantity["length"]) checks physical type."""
        x = 100.0 * u.meter
        assert isinstance(x, u.Quantity["length"])
        assert not isinstance(x, u.Quantity["mass"])

    def test_isinstance_compound_unit(self):
        x = 5.0 * u.meter / u.second
        assert isinstance(x, u.Quantity[u.meter / u.second])
        assert not isinstance(x, u.Quantity[u.meter])

    def test_isinstance_non_quantity(self):
        assert not isinstance(42, u.Quantity[u.meter])
        assert not isinstance("hello", u.Quantity["length"])

    def test_isinstance_with_multiple_types(self):
        """Works in tuple form for isinstance."""
        x = 5.0 * u.meter
        assert isinstance(x, (u.Quantity["length"], u.Quantity["mass"]))
        assert not isinstance(x, (u.Quantity["mass"], u.Quantity["time"]))

    def test_quantity_type_helper(self):
        x = 5.0 * u.meter
        assert isinstance(x, quantity_type("length"))
        assert not isinstance(x, quantity_type("time"))

    def test_quantity_type_equivalent_to_subscript(self):
        assert quantity_type("length") is u.Quantity["length"]
        assert quantity_type(u.meter) is u.Quantity[u.meter]

    def test_used_in_function_annotation(self):
        """Annotations are valid in function signatures."""
        def f(x: u.Quantity[u.meter]) -> u.Quantity[u.second]:
            return x

        # Should not raise
        hints = typing.get_type_hints(f)
        assert 'x' in hints


# =========================================================================
# Pre-built aliases
# =========================================================================

class TestPrebuiltAliases:
    def test_length_isinstance(self):
        from saiunit.typing import LENGTH
        assert isinstance(5.0 * u.meter, LENGTH)
        assert not isinstance(5.0 * u.second, LENGTH)

    def test_speed_isinstance(self):
        from saiunit.typing import SPEED
        assert isinstance(3.0 * u.meter / u.second, SPEED)
        assert not isinstance(3.0 * u.meter, SPEED)

    def test_voltage_isinstance(self):
        from saiunit.typing import VOLTAGE
        assert isinstance(5.0 * u.volt, VOLTAGE)
        assert isinstance(500.0 * u.mvolt, VOLTAGE)

    def test_has_unit_is_quantity(self):
        from saiunit.typing import HAS_UNIT
        assert HAS_UNIT is u.Quantity

    def test_all_aliases_accessible(self):
        """All documented aliases are importable."""
        from saiunit import typing as ut
        for name in [
            'LENGTH', 'MASS', 'TIME', 'CURRENT', 'TEMPERATURE',
            'SUBSTANCE', 'LUMINOSITY', 'FREQUENCY', 'FORCE',
            'ENERGY', 'POWER', 'PRESSURE', 'CHARGE', 'VOLTAGE',
            'RESISTANCE', 'CAPACITANCE', 'CONDUCTANCE',
            'MAGNETIC_FLUX', 'MAGNETIC_FIELD', 'INDUCTANCE',
            'SPEED', 'ACCELERATION', 'AREA', 'VOLUME', 'DENSITY',
        ]:
            alias = getattr(ut, name)
            assert isinstance(alias, type), f"{name} is not a type"

    def test_unknown_alias_raises(self):
        with pytest.raises(AttributeError):
            from saiunit import typing as ut
            _ = ut.NONEXISTENT_ALIAS


# =========================================================================
# Type aliases
# =========================================================================

class TestTypeAliases:
    def test_quantity_like_exists(self):
        from saiunit.typing import QuantityLike
        assert QuantityLike is not None

    def test_unit_like_exists(self):
        from saiunit.typing import UnitLike
        assert UnitLike is not None

    def test_dimension_like_exists(self):
        from saiunit.typing import DimensionLike
        assert DimensionLike is not None


# =========================================================================
# validate_units decorator
# =========================================================================

class TestValidateUnits:
    def test_valid_call_passes(self):
        @validate_units
        def add_lengths(a: u.Quantity[u.meter], b: u.Quantity[u.meter]) -> u.Quantity[u.meter]:
            return a + b

        result = add_lengths(1.0 * u.meter, 2.0 * u.meter)
        assert result.mantissa == 3.0

    def test_compatible_unit_passes_default(self):
        """By default, dimensional compatibility is enough."""
        @validate_units
        def f(x: u.Quantity[u.meter]) -> u.Quantity[u.meter]:
            return x

        result = f(1.0 * u.kmeter)
        assert result.unit.has_same_dim(u.meter)

    def test_incompatible_dimension_raises(self):
        @validate_units
        def f(x: u.Quantity[u.meter]):
            return x

        with pytest.raises(Exception):
            f(1.0 * u.second)

    def test_non_quantity_raises(self):
        @validate_units
        def f(x: u.Quantity[u.meter]):
            return x

        with pytest.raises(TypeError, match="expected a Quantity"):
            f(42)

    def test_strict_mode_rejects_different_scale(self):
        @validate_units(strict=True)
        def f(x: u.Quantity[u.meter]):
            return x

        with pytest.raises(Exception):
            f(1.0 * u.kmeter)

    def test_strict_mode_accepts_same_unit(self):
        @validate_units(strict=True)
        def f(x: u.Quantity[u.meter]):
            return x

        result = f(1.0 * u.meter)
        assert result.mantissa == 1.0

    def test_physical_type_annotation(self):
        @validate_units
        def f(x: u.Quantity["length"]):
            return x

        result = f(1.0 * u.meter)
        assert result.mantissa == 1.0

        with pytest.raises(Exception):
            f(1.0 * u.second)

    def test_no_annotation_no_check(self):
        @validate_units
        def f(x, y: u.Quantity[u.meter]):
            return y

        result = f("anything", 1.0 * u.meter)
        assert result.mantissa == 1.0

    def test_none_value_skipped(self):
        @validate_units
        def f(x: u.Quantity[u.meter] = None):
            return x

        result = f(None)
        assert result is None

    def test_preserves_function_metadata(self):
        @validate_units
        def my_func(x: u.Quantity[u.meter]):
            """My docstring."""
            return x

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_no_constraints_returns_original(self):
        def f(x, y):
            return x + y

        wrapped = validate_units(f)
        assert wrapped is f

    def test_multiple_params(self):
        @validate_units
        def ohms_law(V: u.Quantity[u.volt], R: u.Quantity[u.ohm]) -> u.Quantity[u.amp]:
            return V / R

        result = ohms_law(5.0 * u.volt, 100.0 * u.ohm)
        assert abs(result.mantissa - 0.05) < 1e-10

    def test_kwargs_validated(self):
        @validate_units
        def f(x: u.Quantity[u.meter], y: u.Quantity[u.second]):
            return x

        with pytest.raises(Exception):
            f(1.0 * u.meter, y=1.0 * u.meter)


# =========================================================================
# Integration
# =========================================================================

class TestIntegration:
    def test_full_workflow(self):
        """End-to-end: annotate function, validate, compute."""
        @validate_units
        def kinetic_energy(
            m: u.Quantity[u.kilogram],
            v: u.Quantity["speed"],
        ) -> u.Quantity["energy"]:
            return 0.5 * m * v ** 2

        mass = 10.0 * u.kilogram
        velocity = 3.0 * u.meter / u.second
        ke = kinetic_energy(mass, velocity)
        assert abs(ke.mantissa - 45.0) < 1e-10
        assert ke.dim == u.joule.dim

    def test_isinstance_and_validate_together(self):
        """isinstance check + validate_units on same function."""
        @validate_units
        def f(x: u.Quantity["length"]):
            return x

        q_length = 5.0 * u.meter
        q_time = 1.0 * u.second

        # isinstance check
        assert isinstance(q_length, u.Quantity["length"])
        assert not isinstance(q_time, u.Quantity["length"])

        # validate_units check
        assert f(q_length).mantissa == 5.0
        with pytest.raises(Exception):
            f(q_time)

    def test_km_isinstance_meter(self):
        """User's specific request: km is an instance of Quantity[meter]."""
        x = 2.0 * u.kmeter
        assert isinstance(x, u.Quantity[u.meter])

    def test_annotation_readable_in_help(self):
        """Annotations show up in inspect.signature."""
        import inspect

        def f(x: u.Quantity[u.meter]) -> u.Quantity[u.second]:
            return x

        sig = inspect.signature(f)
        ann = sig.parameters['x'].annotation
        assert isinstance(ann, type)
