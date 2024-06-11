# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import itertools
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_equal

import brainstate as bst

array = np.array
bst.environ.set(precision=64)

from brainunit._unit_common import *
from brainunit._base import (
  DIMENSIONLESS,
  DimensionMismatchError,
  Quantity,
  Unit,
  check_units,
  fail_for_dimension_mismatch,
  get_or_create_dimension,
  get_unit,
  get_basic_unit,
  have_same_unit,
  in_unit,
  is_scalar_type,
)
# from braincore.math import ufuncs_integers
from brainunit._unit_shortcuts import Hz, cm, kHz, ms, mV, nS


def assert_allclose(actual, desired, rtol=4.5e8, atol=0, **kwds):
  """
  Thin wrapper around numpy's `~numpy.testing.utils.assert_allclose` function. The tolerance depends on the floating
  point precision as defined by the `core.default_float_dtype` preference.

  Parameters
  ----------
  actual : `numpy.ndarray`
      The results to check.
  desired : `numpy.ndarray`
      The expected results.
  rtol : float, optional
      The relative tolerance which will be multiplied with the machine epsilon of the type set as
      `core.default_float_type`.
  atol : float, optional
      The absolute tolerance
  """
  assert have_same_unit(actual, desired)
  eps = jnp.finfo(np.float32).eps
  rtol = eps * rtol
  jnp.allclose(
    jnp.asarray(actual), jnp.asarray(desired), rtol=rtol, atol=atol, **kwds
  )


def assert_quantity(q, values, unit):
  values = jnp.asarray(values)
  if isinstance(q, Quantity):
    assert have_same_unit(q.unit, unit), f"Dimension mismatch: ({get_unit(q)}) ({get_unit(unit)})"
    if not jnp.allclose(q.value, values):
      raise AssertionError(f"Values do not match: {q.value} != {values}")
  elif isinstance(q, jnp.ndarray):
    if jnp.allclose(q, values):
      return True
    else:
      raise AssertionError(f"Values do not match: {q} != {values}")
  else:
    assert jnp.allclose(q, values), f"Values do not match: {q} != {values}"

  # try:
  #   if jnp.allclose(q, values):
  #     return True
  # except:
  #   pass
  # assert isinstance(q, Quantity) or (
  #     have_same_unit(unit, 1)
  #     and (values.shape == () or isinstance(q, jnp.ndarray))
  # ), q
  # assert_allclose(jnp.asarray(q), values)
  # assert have_same_unit(q.unit, unit), f"Dimension mismatch: ({get_unit(q)}) ({get_unit(unit)})"


def test_construction():
  """Test the construction of Array objects"""
  q = 500 * ms
  assert_quantity(q, 0.5, second)
  q = np.float64(500) * ms
  assert_quantity(q, 0.5, second)
  q = np.array(500) * ms
  assert_quantity(q, 0.5, second)
  q = np.array([500, 1000]) * ms
  assert_quantity(q, np.array([0.5, 1]), second)
  q = Quantity(500)
  assert_quantity(q, 500, 1)
  q = Quantity(500, dim=second.dim)
  assert_quantity(q, 500, second)
  q = Quantity([0.5, 1], dim=second.dim)
  assert_quantity(q, np.array([0.5, 1]), second)
  q = Quantity(np.array([0.5, 1]), dim=second.dim)
  assert_quantity(q, np.array([0.5, 1]), second)
  q = Quantity([500 * ms, 1 * second])
  assert_quantity(q, np.array([0.5, 1]), second)
  q = Quantity.with_units(np.array([0.5, 1]), second=1)
  assert_quantity(q, np.array([0.5, 1]), second)
  q = [0.5, 1] * second
  assert_quantity(q, np.array([0.5, 1]), second)

  # dimensionless quantities
  q = Quantity([1, 2, 3])
  assert_quantity(q, np.array([1, 2, 3]), Unit(1))
  q = Quantity(np.array([1, 2, 3]))
  assert_quantity(q, np.array([1, 2, 3]), Unit(1))
  q = Quantity([])
  assert_quantity(q, np.array([]), Unit(1))

  # Illegal constructor calls
  with pytest.raises(TypeError):
    Quantity([500 * ms, 1])
  with pytest.raises(TypeError):
    Quantity(["some", "nonsense"])
  with pytest.raises(TypeError):
    Quantity([500 * ms, 1 * volt])


def test_get_dimensions():
  """
  Test various ways of getting/comparing the dimensions of a Array.
  """
  q = 500 * ms
  assert get_unit(q) is get_or_create_dimension(q.unit._dims)
  assert get_unit(q) is q.unit
  assert q.has_same_unit(3 * second)
  dims = q.unit
  assert_equal(dims.get_dimension("time"), 1.0)
  assert_equal(dims.get_dimension("length"), 0)

  assert get_unit(5) is DIMENSIONLESS
  assert get_unit(5.0) is DIMENSIONLESS
  assert get_unit(np.array(5, dtype=np.int32)) is DIMENSIONLESS
  assert get_unit(np.array(5.0)) is DIMENSIONLESS
  assert get_unit(np.float32(5.0)) is DIMENSIONLESS
  assert get_unit(np.float64(5.0)) is DIMENSIONLESS
  assert is_scalar_type(5)
  assert is_scalar_type(5.0)
  assert is_scalar_type(np.array(5, dtype=np.int32))
  assert is_scalar_type(np.array(5.0))
  assert is_scalar_type(np.float32(5.0))
  assert is_scalar_type(np.float64(5.0))
  with pytest.raises(TypeError):
    get_unit("a string")
  # wrong number of indices
  with pytest.raises(TypeError):
    get_or_create_dimension([1, 2, 3, 4, 5, 6])
  # not a sequence
  with pytest.raises(TypeError):
    get_or_create_dimension(42)


def test_display():
  """
  Test displaying a Array in different units
  """

  assert_equal(in_unit(3. * volt, mvolt), "3000. mV")
  assert_equal(in_unit(10. * mV, ohm * amp), "0.01 ohm * A")
  with pytest.raises(DimensionMismatchError):
    in_unit(10 * nS, ohm)
    # with bst.environ.context(precision=32):
    #   assert_equal(in_unit(3. * volt, mvolt), "3000. mV")
    #   assert_equal(in_unit(10. * mV, ohm * amp), "0.01 ohm * A")
    #   with pytest.raises(DimensionMismatchError):
    #     in_unit(10 * nS, ohm)

    # A bit artificial...
    assert_equal(in_unit(10.0, Unit(10.0, scale=1)), "1.0")


def test_unary_operations():
  q = Quantity(5, dim=mV)
  assert_quantity(-q, -5, mV)
  assert_quantity(+q, 5, mV)
  assert_quantity(abs(Quantity(-5, dim=mV)), 5, mV)
  assert_quantity(~Quantity(0b101, dim=DIMENSIONLESS), -0b110, DIMENSIONLESS)


def test_operations():
  q1 = Quantity(5, dim=mV)
  q2 = Quantity(10, dim=mV)
  assert_quantity(q1 + q2, 15, mV)
  assert_quantity(q1 - q2, -5, mV)
  assert_quantity(q1 * q2, 50, mV * mV)
  assert_quantity(q2 / q1, 2, DIMENSIONLESS)
  assert_quantity(q2 // q1, 2, DIMENSIONLESS)
  assert_quantity(q2 % q1, 0, mV)
  assert_quantity(divmod(q2, q1)[0], 2, DIMENSIONLESS)
  assert_quantity(divmod(q2, q1)[1], 0, mV)
  assert_quantity(q1 ** 2, 25, mV ** 2)
  assert_quantity(q1 << 1, 10, mV)
  assert_quantity(q1 >> 1, 2, mV)
  assert_quantity(round(q1, 0), 5, mV)
  # matmul
  q1 = Quantity([1, 2], dim=mV)
  q2 = Quantity([3, 4], dim=mV)
  assert_quantity(q1 @ q2, 11, mV ** 2)


def test_numpy_methods():
  q = Quantity([[1, 2], [3, 4]], dim=mV)
  assert q.all()
  assert q.any()
  assert q.nonzero()[0].tolist() == [0, 0, 1, 1]
  assert q.argmax() == 3
  assert q.argmin() == 0
  assert q.argsort(axis=None).tolist() == [0, 1, 2, 3]
  assert_quantity(q.var(), 1.25, mV ** 2)
  assert_quantity(q.round(), [[1, 2], [3, 4]], mV)
  assert_quantity(q.std(), 1.11803398875, mV)
  assert_quantity(q.sum(), 10, mV)
  assert_quantity(q.trace(), 5, mV)
  assert_quantity(q.cumsum(), [1, 3, 6, 10], mV)
  assert_quantity(q.cumprod(), [1, 2, 6, 24], mV ** 4)
  assert_quantity(q.diagonal(), [1, 4], mV)
  assert_quantity(q.max(), 4, mV)
  assert_quantity(q.mean(), 2.5, mV)
  assert_quantity(q.min(), 1, mV)
  assert_quantity(q.ptp(), 3, mV)
  assert_quantity(q.ravel(), [1, 2, 3, 4], mV)


def test_shape_manipulation():
  q = [[1, 2], [3, 4]] * volt

  # Test flatten
  assert_quantity(q.flatten(), [1, 2, 3, 4], volt)

  # Test swapaxes
  assert_quantity(q.swapaxes(0, 1), [[1, 3], [2, 4]], volt)

  # Test take
  assert_quantity(q.take(jnp.array([0, 2])), [1, 3], volt)

  # Test transpose
  assert_quantity(q.transpose(), [[1, 3], [2, 4]], volt)

  # Test tile
  assert_quantity(q.tile(2), [[1, 2, 1, 2], [3, 4, 3, 4]], volt)

  # Test unsqueeze
  assert_quantity(q.unsqueeze(0), [[[1, 2], [3, 4]]], volt)

  # Test expand_dims
  assert_quantity(q.expand_dims(0), [[[1, 2], [3, 4]]], volt)

  # Test expand_as
  expand_as_shape = (1, 2, 2)
  assert_quantity(q.expand_as(jnp.zeros(expand_as_shape).shape), [[[1, 2], [3, 4]]], volt)

  # Test put
  q_put = [[1, 2], [3, 4]] * volt
  q_put.put([[1, 0], [0, 1]], [10, 30] * volt)
  assert_quantity(q_put, [[1, 30], [10, 4]], volt)

  # Test squeeze (no axes to squeeze in this case, so the array remains the same)
  q_squeeze = [[1, 2], [3, 4]] * volt
  assert_quantity(q_squeeze.squeeze(), [[1, 2], [3, 4]], volt)

  # Test array_split
  q_spilt = [[10, 2], [30, 4]] * volt
  assert_quantity(np.array_split(q_spilt, 2)[0], [[10, 2]], volt)


def test_misc_methods():
  q = [5, 10, 15] * volt

  # Test astype
  assert_quantity(q.astype(np.float32), [5, 10, 15], volt)

  # Test clip
  min_val = [6, 6, 6] * volt
  max_val = [14, 14, 14] * volt
  assert_quantity(q.clip(min_val, max_val), [6, 10, 14], volt)

  # Test conj
  assert_quantity(q.conj(), [5, 10, 15], volt)

  # Test conjugate
  assert_quantity(q.conjugate(), [5, 10, 15], volt)

  # Test copy
  assert_quantity(q.copy(), [5, 10, 15], volt)

  # Test dot
  assert_quantity(q.dot(Quantity([2, 2, 2], dim=DIMENSIONLESS)), 60, volt)

  # Test fill
  q_filled = [5, 10, 15] * volt
  q_filled.fill(2 * volt)
  assert_quantity(q_filled, [2, 2, 2], volt)

  # Test item
  assert_quantity(q.item(0), 5, volt)

  # Test prod
  assert_quantity(q.prod(), 750, volt ** 3)

  # Test repeat
  assert_quantity(q.repeat(2), [5, 5, 10, 10, 15, 15], volt)

  # Test clamp (same as clip, but using min and max values directly)
  assert_quantity(q.clamp(6 * volt, 14 * volt), [6, 10, 14], volt)

  # Test sort
  q = [15, 5, 10] * volt
  assert_quantity(q.sort(), [5, 10, 15], volt)


# def test_str_repr():
#   """
#   Test that str representations do not raise any errors and that repr
#   fullfills eval(repr(x)) == x. Also test generating LaTeX representations via sympy.
#   """
#   import sympy
#
#   units_which_should_exist = [
#     metre,
#     meter,
#     kilogram,
#     kilogramme,
#     second,
#     amp,
#     kelvin,
#     mole,
#     candle,
#     radian,
#     steradian,
#     hertz,
#     newton,
#     pascal,
#     joule,
#     watt,
#     coulomb,
#     volt,
#     farad,
#     ohm,
#     siemens,
#     weber,
#     tesla,
#     henry,
#     lumen,
#     lux,
#     becquerel,
#     gray,
#     sievert,
#     katal,
#     gram,
#     gramme,
#     molar,
#     liter,
#     litre,
#   ]
#
#   # scaled versions of all these units should exist (we just check farad as an example)
#   some_scaled_units = [
#     Yfarad,
#     Zfarad,
#     Efarad,
#     Pfarad,
#     Tfarad,
#     Gfarad,
#     Mfarad,
#     kfarad,
#     hfarad,
#     dafarad,
#     dfarad,
#     cfarad,
#     mfarad,
#     ufarad,
#     nfarad,
#     pfarad,
#     ffarad,
#     afarad,
#     zfarad,
#     yfarad,
#   ]
#
#   # some powered units
#   powered_units = [cmetre2, Yfarad3]
#
#   # Combined units
#   complex_units = [
#     (kgram * metre2) / (amp * second3),
#     5 * (kgram * metre2) / (amp * second3),
#     metre * second ** -1,
#     10 * metre * second ** -1,
#     np.array([1, 2, 3]) * kmetre / second,
#     np.ones(3) * nS / cm ** 2,
#     # Made-up unit:
#     Unit(
#       1,
#       unit=get_or_create_dimension(length=5, time=2),
#       dispname="O",
#     ),
#     8000 * umetre ** 3,
#     [0.0001, 10000] * umetre ** 3,
#     1 / metre,
#     1 / (coulomb * metre ** 2),
#     Unit(1) / second,
#     3.0 * mM,
#     5 * mole / liter,
#     7 * liter / meter3,
#     1 / second ** 2,
#     volt ** -2,
#     (volt ** 2) ** -1,
#     (1 / second) / meter,
#     1 / (1 / second),
#   ]
#
#   unitless = [second / second, 5 * second / second, Unit(1)]
#
#   for u in itertools.chain(
#       units_which_should_exist,
#       some_scaled_units,
#       powered_units,
#       complex_units,
#       unitless,
#   ):
#     assert len(str(u)) > 0
#     if not is_unitless(u):
#       assert len(sympy.latex(u))
#     v1 = repr(u)
#     v2 = eval(v1)
#     assert get_unit(eval(repr(u))) == get_unit(u)
#     assert_allclose(eval(repr(u)).value, u.value)
#
#   for ar in [np.arange(10000) * mV, np.arange(100).reshape(10, 10) * mV]:
#     latex_str = sympy.latex(ar)
#     assert 0 < len(latex_str) < 2000  # arbitrary threshold, but see #1425
#
#   # test the `DIMENSIONLESS` object
#   assert str(DIMENSIONLESS) == "1"
#   assert repr(DIMENSIONLESS) == "Dimension()"
#
#   # test DimensionMismatchError (only that it works without raising an error
#   for error in [
#     DimensionMismatchError("A description"),
#     DimensionMismatchError("A description", DIMENSIONLESS),
#     DimensionMismatchError("A description", DIMENSIONLESS, second.unit),
#   ]:
#     assert len(str(error))
#     assert len(repr(error))


def test_format_quantity():
  # Avoid that the default f-string (or .format call) discards units when used without
  # a format spec
  with bst.environ.context(precision=64):
    q = 0.5 * ms
  assert f"{q}" == f"{q!s}" == str(q)
  assert f"{q:g}" == f"{float(q)}"


def test_slicing():
  # Slicing and indexing, setting items
  Array = np.reshape(np.arange(6), (2, 3)) * mV
  assert_allclose(Array[:].value, Array.value)
  assert_allclose(Array[0].value, (np.asarray(Array)[0] * volt).value)
  assert_allclose(Array[0:1].value, (np.asarray(Array)[0:1] * volt).value)
  assert_allclose(Array[0, 1].value, (np.asarray(Array)[0, 1] * volt).value)
  assert_allclose(Array[0:1, 1:].value, (np.asarray(Array)[0:1, 1:] * volt).value)
  bool_matrix = np.array([[True, False, False], [False, False, True]])
  assert_allclose(Array[bool_matrix].value, (np.asarray(Array)[bool_matrix] * volt).value)


def test_setting():
  quantity = np.reshape(np.arange(6), (2, 3)) * mV
  quantity[0, 1] = 10 * mV
  assert quantity[0, 1] == 10 * mV
  quantity[:, 1] = 20 * mV
  assert np.all(quantity[:, 1] == 20 * mV)
  # TODO: jax.numpy ndarray doesn't support this
  # quantity[1, :] = np.ones((1, 3)) * volt
  # assert np.all(quantity[1, :] == 1 * volt)

  quantity[1, 2] = 0 * mV
  assert quantity[1, 2] == 0 * mV

  def set_to_value(key, value):
    quantity[key] = value

  with pytest.raises(DimensionMismatchError):
    set_to_value(0, 1)
  with pytest.raises(DimensionMismatchError):
    set_to_value(0, 1 * second)
  with pytest.raises(DimensionMismatchError):
    set_to_value((slice(2), slice(3)), np.ones((2, 3)))


def test_multiplication_division():
  quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
  q2 = 5 * second

  for q in quantities:
    # Scalars and array scalars
    assert_quantity(q / 3, np.asarray(q) / 3, volt)
    assert_quantity(3 / q, 3 / np.asarray(q), 1 / volt)
    assert_quantity(q * 3, np.asarray(q) * 3, volt)
    assert_quantity(3 * q, 3 * np.asarray(q), volt)
    assert_quantity(q / np.float64(3), np.asarray(q) / 3, volt)
    assert_quantity(np.float64(3) / q, 3 / np.asarray(q), 1 / volt)
    assert_quantity(q * np.float64(3), np.asarray(q) * 3, volt)
    assert_quantity(np.float64(3) * q, 3 * np.asarray(q), volt)
    assert_quantity(q / jnp.array(3), np.asarray(q) / 3, volt)
    assert_quantity(np.array(3) / q, 3 / np.asarray(q), 1 / volt)
    assert_quantity(q * jnp.array(3), np.asarray(q) * 3, volt)
    assert_quantity(np.array(3) * q, 3 * np.asarray(q), volt)

    # (unitless) arrays
    assert_quantity(q / np.array([3]), np.asarray(q) / 3, volt)
    assert_quantity(np.array([3]) / q, 3 / np.asarray(q), 1 / volt)
    assert_quantity(q * np.array([3]), np.asarray(q) * 3, volt)
    assert_quantity(np.array([3]) * q, 3 * np.asarray(q), volt)

    # arrays with units
    assert_quantity(q / q, np.asarray(q) / np.asarray(q), 1)
    assert_quantity(q * q, np.asarray(q) ** 2, volt ** 2)
    assert_quantity(q / q2, np.asarray(q) / np.asarray(q2), volt / second)
    assert_quantity(q2 / q, np.asarray(q2) / np.asarray(q), second / volt)
    assert_quantity(q * q2, np.asarray(q) * np.asarray(q2), volt * second)

    # using unsupported objects should fail
    with pytest.raises(TypeError):
      q / "string"
    with pytest.raises(TypeError):
      "string" / q
    with pytest.raises(TypeError):
      "string" * q
    with pytest.raises(TypeError):
      q * "string"


def test_addition_subtraction():
  quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
  q2 = 5 * volt

  for q in quantities:
    # arrays with units
    assert_quantity(q + q, np.asarray(q) + np.asarray(q), volt)
    assert_quantity(q - q, 0, volt)
    assert_quantity(q + q2, np.asarray(q) + np.asarray(q2), volt)
    assert_quantity(q2 + q, np.asarray(q2) + np.asarray(q), volt)
    assert_quantity(q - q2, np.asarray(q) - np.asarray(q2), volt)
    assert_quantity(q2 - q, np.asarray(q2) - np.asarray(q), volt)

    # mismatching units
    with pytest.raises(DimensionMismatchError):
      q + 5 * second
    with pytest.raises(DimensionMismatchError):
      5 * second + q
    with pytest.raises(DimensionMismatchError):
      q - 5 * second
    with pytest.raises(DimensionMismatchError):
      5 * second - q

    # scalar
    with pytest.raises(DimensionMismatchError):
      q + 5
    with pytest.raises(DimensionMismatchError):
      5 + q
    with pytest.raises(DimensionMismatchError):
      q + np.float64(5)
    with pytest.raises(DimensionMismatchError):
      np.float64(5) + q
    with pytest.raises(DimensionMismatchError):
      q - 5
    with pytest.raises(DimensionMismatchError):
      5 - q
    with pytest.raises(DimensionMismatchError):
      q - np.float64(5)
    with pytest.raises(DimensionMismatchError):
      np.float64(5) - q

    # unitless array
    with pytest.raises(DimensionMismatchError):
      q + np.array([5])
    with pytest.raises(DimensionMismatchError):
      np.array([5]) + q
    with pytest.raises(DimensionMismatchError):
      q + np.array([5], dtype=np.float64)
    with pytest.raises(DimensionMismatchError):
      np.array([5], dtype=np.float64) + q
    with pytest.raises(DimensionMismatchError):
      q - np.array([5])
    with pytest.raises(DimensionMismatchError):
      np.array([5]) - q
    with pytest.raises(DimensionMismatchError):
      q - np.array([5], dtype=np.float64)
    with pytest.raises(DimensionMismatchError):
      np.array([5], dtype=np.float64) - q

    # Check that operations with 0 work
    assert_quantity(q + 0, np.asarray(q), volt)
    assert_quantity(0 + q, np.asarray(q), volt)
    assert_quantity(q - 0, np.asarray(q), volt)
    # Doesn't support 0 - Quantity
    # assert_quantity(0 - q, -np.asarray(q), volt)
    assert_quantity(q + np.float64(0), np.asarray(q), volt)
    assert_quantity(np.float64(0) + q, np.asarray(q), volt)
    assert_quantity(q - np.float64(0), np.asarray(q), volt)
    # assert_quantity(np.float64(0) - q, -np.asarray(q), volt)

    # using unsupported objects should fail
    with pytest.raises(TypeError):
      "string" + q
    with pytest.raises(TypeError):
      q + "string"
    with pytest.raises(TypeError):
      q - "string"
    with pytest.raises(TypeError):
      "string" - q


# def test_unary_operations():
#   from operator import neg, pos
#
#   for op in [neg, pos]:
#     for x in [2, np.array([2]), np.array([1, 2])]:
#       assert_quantity(op(x * kilogram), op(x), kilogram)


def test_binary_operations():
  """Test whether binary operations work when they should and raise
  DimensionMismatchErrors when they should.
  Does not test for the actual result.
  """
  from operator import add, eq, ge, gt, le, lt, ne, sub

  def assert_operations_work(a, b):
    try:
      # Test python builtins
      tryops = [add, sub, lt, le, gt, ge, eq, ne]
      for op in tryops:
        op(a, b)
        op(b, a)

      # Test equivalent numpy functions
      numpy_funcs = [
        np.add,
        np.subtract,
        np.less,
        np.less_equal,
        np.greater,
        np.greater_equal,
        np.equal,
        np.not_equal,
        np.maximum,
        np.minimum,
      ]
      for numpy_func in numpy_funcs:
        numpy_func(a, b)
        numpy_func(b, a)
    except DimensionMismatchError as ex:
      raise AssertionError(f"Operation raised unexpected exception: {ex}")

  def assert_operations_do_not_work(a, b):
    # Test python builtins
    tryops = [add, sub, lt, le, gt, ge, eq, ne]
    for op in tryops:
      with pytest.raises(DimensionMismatchError):
        op(a, b)
      with pytest.raises(DimensionMismatchError):
        op(b, a)

    # Do not support equivalent numpy functions
    # numpy_funcs = [
    #   np.add,
    #   np.subtract,
    #   np.less,
    #   np.less_equal,
    #   np.greater,
    #   np.greater_equal,
    #   np.equal,
    #   np.not_equal,
    #   np.maximum,
    #   np.minimum,
    # ]
    # for numpy_func in numpy_funcs:
    #   with pytest.raises(DimensionMismatchError):
    #     numpy_func(a, b)
    #   with pytest.raises(DimensionMismatchError):
    #     numpy_func(b, a)

  #
  # Check that consistent units work
  #

  # unit arrays
  a = 1 * kilogram
  for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
    assert_operations_work(a, b)

  # dimensionless units and scalars
  a = 1
  for b in [
    2 * kilogram / kilogram,
    np.array([2]) * kilogram / kilogram,
    np.array([1, 2]) * kilogram / kilogram,
  ]:
    assert_operations_work(a, b)

  # dimensionless units and unitless arrays
  a = np.array([1])
  for b in [
    2 * kilogram / kilogram,
    np.array([2]) * kilogram / kilogram,
    np.array([1, 2]) * kilogram / kilogram,
  ]:
    assert_operations_work(a, b)

  #
  # Check that inconsistent units do not work
  #

  # unit arrays
  a = np.array([1]) * second
  for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
    assert_operations_do_not_work(a, b)

  # unitless array
  a = np.array([1])
  for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
    assert_operations_do_not_work(a, b)

  # scalar
  a = 1
  for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
    assert_operations_do_not_work(a, b)

  # Check that comparisons with inf/-inf always work
  values = [
    2 * kilogram / kilogram,
    2 * kilogram,
    np.array([2]) * kilogram,
    np.array([1, 2]) * kilogram,
  ]
  for value in values:
    assert np.all(value < np.inf)
    assert np.all(np.inf > value)
    assert np.all(value <= np.inf)
    assert np.all(np.inf >= value)
    assert np.all(value != np.inf)
    assert np.all(np.inf != value)
    assert np.all(value >= -np.inf)
    assert np.all(-np.inf <= value)
    assert np.all(value > -np.inf)
    assert np.all(-np.inf < value)


def test_power():
  """
  Test raising quantities to a power.
  """
  values = [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]
  for value in values:
    assert_quantity(value ** 3, np.asarray(value) ** 3, kilogram ** 3)
    # Test raising to a dimensionless Array
    assert_quantity(
      value ** (3 * volt / volt), np.asarray(value) ** 3, kilogram ** 3
    )
    with pytest.raises(DimensionMismatchError):
      value ** (2 * volt)
    with pytest.raises(TypeError):
      value ** np.array([2, 3])


def test_inplace_operations():
  q = np.arange(10) * volt
  q_orig = q.copy()
  q_id = id(q)

  # Doesn't support in-place operations which change unit
  # q *= 2
  # assert np.all(q == 2 * q_orig) and id(q) == q_id
  # q /= 2
  # assert np.all(q == q_orig) and id(q) == q_id
  q += 1 * volt
  assert np.all(q == q_orig + 1 * volt) and id(q) == q_id
  q -= 1 * volt
  assert np.all(q == q_orig) and id(q) == q_id

  # q **= 2
  # assert np.all(q == q_orig ** 2) and id(q) == q_id
  # q **= 0.5
  # assert np.all(q == q_orig) and id(q) == q_id

  def illegal_add(q2):
    q = np.arange(10) * volt
    q += q2

  with pytest.raises(DimensionMismatchError):
    illegal_add(1 * second)
  with pytest.raises(DimensionMismatchError):
    illegal_add(1)

  def illegal_sub(q2):
    q = np.arange(10) * volt
    q -= q2

  with pytest.raises(DimensionMismatchError):
    illegal_add(1 * second)
  with pytest.raises(DimensionMismatchError):
    illegal_add(1)

  def illegal_pow(q2):
    q = np.arange(10) * volt
    q **= q2

  # with pytest.raises(DimensionMismatchError):
  #   illegal_pow(1 * volt)
  # with pytest.raises(TypeError):
  #   illegal_pow(np.arange(10))

  # inplace operations with unsupported objects should fail
  for inplace_op in [
    q.__iadd__,
    q.__isub__,
    # q.__imul__,
    # q.__idiv__,
    # q.__itruediv__,
    # q.__ifloordiv__,
    # q.__imod__,
    # q.__ipow__,
  ]:
    try:
      result = inplace_op("string")
      # if it doesn't fail with an error, it should return NotImplemented
      assert result == NotImplemented
    except TypeError:
      pass  # raised on numpy >= 0.10

  # make sure that inplace operations do not work on units/dimensions at all
  for inplace_op in [
    volt.__iadd__,
    volt.__isub__,
    # volt.__imul__,
    # volt.__idiv__,
    # volt.__itruediv__,
    # volt.__ifloordiv__,
    # volt.__imod__,
    # volt.__ipow__,
  ]:
    with pytest.raises(TypeError):
      inplace_op(volt)
  # for inplace_op in [
  #   volt.unit.__imul__,
  #   volt.unit.__idiv__,
  #   volt.unit.__itruediv__,
  #   volt.unit.__ipow__,
  # ]:
  #   with pytest.raises(TypeError):
  #     inplace_op(volt.unit)


def test_unit_discarding_functions():
  """
  Test functions that discard units.
  """
  from brainunit.math import ones_like, zeros_like

  values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(3, 4) * mV]
  for value in values:
    assert_equal(np.sign(value.value), np.sign(np.asarray(value.value)))
    assert_equal(zeros_like(value), np.zeros_like(np.asarray(value.value)))
    assert_equal(ones_like(value), np.ones_like(np.asarray(value.value)))
    # Calling non-zero on a 0d array is deprecated, don't test it:
    if value.ndim > 0:
      assert_equal(np.nonzero(value.value), np.nonzero(np.asarray(value.value)))


def test_unitsafe_functions():
  """
  Test the unitsafe functions wrapping their numpy counterparts.
  """
  from brainunit.math import (
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    cos,
    cosh,
    exp,
    log,
    sin,
    sinh,
    tan,
    tanh,
  )

  # All functions with their numpy counterparts
  funcs = [
    (sin, np.sin),
    (sinh, np.sinh),
    (arcsin, np.arcsin),
    (arcsinh, np.arcsinh),
    (cos, np.cos),
    (cosh, np.cosh),
    (arccos, np.arccos),
    (arccosh, np.arccosh),
    (tan, np.tan),
    (tanh, np.tanh),
    (arctan, np.arctan),
    (arctanh, np.arctanh),
    (log, np.log),
    (exp, np.exp),
  ]

  unitless_values = [
    3 * mV / mV,
    np.array([1, 2]) * mV / mV,
    np.ones((3, 3)) * mV / mV,
  ]
  numpy_values = [3, np.array([1, 2]), np.ones((3, 3))]
  unit_values = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]

  for func, np_func in funcs:
    # make sure these functions raise errors when run on values with dimensions
    for val in unit_values:
      with pytest.raises(DimensionMismatchError):
        func(val)

    # make sure the functions are equivalent to their numpy counterparts
    # when run on unitless values while ignoring warnings about invalid
    # values or divisions by zero
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      for val in unitless_values:
        if hasattr(val, "value"):
          assert_allclose(func(val.value), np_func(val.value))
        else:
          assert_allclose(func(val), np_func(val))

      for val in numpy_values:
        assert_allclose(func(val), np_func(val))


def test_special_case_numpy_functions():
  """
  Test a couple of functions/methods that need special treatment.
  """
  from brainunit.math import diagonal, dot, ravel, trace, where

  quadratic_matrix = np.reshape(np.arange(9), (3, 3)) * mV

  # Temporarily suppress warnings related to the matplotlib 1.3 bug
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Check that function and method do the same
    assert_allclose(ravel(quadratic_matrix).value, quadratic_matrix.ravel().value)
    # Check that function gives the same result as on unitless arrays
    assert_allclose(
      np.asarray(ravel(quadratic_matrix).value), ravel(np.asarray(quadratic_matrix))
    )
    # Check that the function gives the same results as the original numpy
    # function
    assert_allclose(
      np.ravel(np.asarray(quadratic_matrix.value)), ravel(np.asarray(quadratic_matrix.value))
    )

  # Do the same checks for diagonal, trace and dot
  assert_allclose(diagonal(quadratic_matrix).value, quadratic_matrix.diagonal().value)
  assert_allclose(
    np.asarray(diagonal(quadratic_matrix).value), diagonal(np.asarray(quadratic_matrix.value))
  )
  assert_allclose(
    np.diagonal(np.asarray(quadratic_matrix.value)),
    diagonal(np.asarray(quadratic_matrix.value)),
  )

  assert_allclose(trace(quadratic_matrix).value, quadratic_matrix.trace().value)
  assert_allclose(
    np.asarray(trace(quadratic_matrix).value), trace(np.asarray(quadratic_matrix.value))
  )
  assert_allclose(
    np.trace(np.asarray(quadratic_matrix.value)), trace(np.asarray(quadratic_matrix.value))
  )

  assert_allclose(
    dot(quadratic_matrix, quadratic_matrix).value, quadratic_matrix.dot(quadratic_matrix).value
  )
  assert_allclose(
    np.asarray(dot(quadratic_matrix, quadratic_matrix).value),
    dot(np.asarray(quadratic_matrix.value), np.asarray(quadratic_matrix.value)),
  )
  assert_allclose(
    np.dot(np.asarray(quadratic_matrix.value), np.asarray(quadratic_matrix.value)),
    dot(np.asarray(quadratic_matrix.value), np.asarray(quadratic_matrix.value)),
  )
  assert_allclose(
    np.asarray(quadratic_matrix.prod().value), np.asarray(quadratic_matrix.value).prod()
  )
  assert_allclose(
    np.asarray(quadratic_matrix.prod(axis=0).value),
    np.asarray(quadratic_matrix.value).prod(axis=0),
  )

  # Check for correct units
  assert have_same_unit(quadratic_matrix, ravel(quadratic_matrix))
  assert have_same_unit(quadratic_matrix, trace(quadratic_matrix))
  assert have_same_unit(quadratic_matrix, diagonal(quadratic_matrix))
  assert have_same_unit(
    quadratic_matrix[0] ** 2, dot(quadratic_matrix, quadratic_matrix)
  )
  assert have_same_unit(
    quadratic_matrix.prod(axis=0), quadratic_matrix[0] ** quadratic_matrix.shape[0]
  )

  # check the where function
  # pure numpy array
  cond = [True, False, False]
  ar1 = np.array([1, 2, 3])
  ar2 = np.array([4, 5, 6])
  assert_equal(np.where(cond), where(cond))
  assert_equal(np.where(cond, ar1, ar2), where(cond, ar1, ar2))

  # dimensionless Array
  assert_allclose(
    np.where(cond, ar1, ar2), np.asarray(where(cond, ar1 * mV / mV, ar2 * mV / mV))
  )

  # Array with dimensions
  ar1 = ar1 * mV
  ar2 = ar2 * mV
  assert_allclose(
    np.where(cond, np.asarray(ar1), np.asarray(ar2)),
    np.asarray(where(cond, ar1, ar2)),
  )

  # Check some error cases
  with pytest.raises(ValueError):
    where(cond, ar1)
  with pytest.raises(TypeError):
    where(cond, ar1, ar1, ar2)
  with pytest.raises(DimensionMismatchError):
    where(cond, ar1, ar1 / ms)

  # Check setasflat (for numpy < 1.7)
  if hasattr(Quantity, "setasflat"):
    a = np.arange(10) * mV
    b = np.ones(10).reshape(5, 2) * volt
    c = np.ones(10).reshape(5, 2) * second
    with pytest.raises(DimensionMismatchError):
      a.setasflat(c)
    a.setasflat(b)
    assert_equal(a.flatten(), b.flatten())

  # Check cumprod
  a = np.arange(1, 10) * mV / mV
  assert_allclose(a.cumprod(), np.asarray(a).cumprod())
  # with pytest.raises(TypeError):
  #   (np.arange(1, 5) * mV).cumprod()


# Functions that should not change units

def test_numpy_functions_same_dimensions():
  values = [np.array([1, 2]), np.ones((3, 3))]
  units = [volt, second, siemens, mV, kHz]

  # Do not suopport numpy functions
  # keep_dim_funcs = [
  #   np.abs,
  #   np.cumsum,
  #   np.max,
  #   np.mean,
  #   np.min,
  #   np.negative,
  #   ptp,
  #   np.round,
  #   np.squeeze,
  #   np.std,
  #   np.sum,
  #   np.transpose,
  # ]
  #
  # for value, unit in itertools.product(values, units):
  #   q_ar = value * unit
  #   for func in keep_dim_funcs:
  #     test_ar = func(q_ar)
  #     if not get_unit(test_ar) is q_ar.unit:
  #       raise AssertionError(
  #         f"'{func.__name__}' failed on {q_ar!r} -- unit was "
  #         f"{q_ar.unit}, is now {get_unit(test_ar)}."
  #       )
  #
  #       # Python builtins should work on one-dimensional arrays
  #       value = np.arange(5)
  #       builtins = [abs, max, min, sum]
  #       for unit in units:
  #         q_ar = value * unit
  #       for func in builtins:
  #         test_ar = func(q_ar)
  #       if not get_unit(test_ar) is q_ar.unit:
  #         raise AssertionError(
  #           f"'{func.__name__}' failed on {q_ar!r} -- unit "
  #           f"was {q_ar.unit}, is now "
  #           f"{get_unit(test_ar)}"
  #         )


def test_numpy_functions_indices():
  """
  Check numpy functions that return indices.
  """
  values = [np.array([-4, 3, -2, 1, 0]), np.ones((3, 3)), np.array([17])]
  units = [volt, second, siemens, mV, kHz]

  # numpy functions
  keep_dim_funcs = [np.argmin, np.argmax, np.argsort, np.nonzero]

  for value, unit in itertools.product(values, units):
    q_ar = value * unit
    for func in keep_dim_funcs:
      test_ar = func(q_ar)
      # Compare it to the result on the same value without units
      comparison_ar = func(value)
      test_ar = np.asarray(test_ar)
      comparison_ar = np.asarray(comparison_ar)
      assert_equal(
        test_ar,
        comparison_ar,
        (
            "function %s returned an incorrect result when used on quantities "
            % func.__name__
        ),
      )


# Do not support numpy functions
# def test_numpy_functions_dimensionless():
#   """
#   Test that numpy functions that should work on dimensionless quantities only
#   work dimensionless arrays and return the correct result.
#   """
#   unitless_values = [3, np.array([-4, 3, -1, 2]), np.ones((3, 3))]
#   unit_values = [3 * mV, np.array([-4, 3, -1, 2]) * mV, np.ones((3, 3)) * mV]
#   with warnings.catch_warnings():
#     # ignore division by 0 warnings
#     warnings.simplefilter("ignore", RuntimeWarning)
#     for value in unitless_values:
#       for ufunc in ufuncs_dimensionless:
#         result_unitless = eval(f"np.{ufunc}(value)")
#         result_array = eval(f"np.{ufunc}(np.array(value))")
#         assert isinstance(
#           result_unitless, (np.ndarray, np.number)
#         ) and not isinstance(result_unitless, Quantity)
#         assert_equal(result_unitless, result_array)
#       for ufunc in ufuncs_dimensionless_twoargs:
#         result_unitless = eval(f"np.{ufunc}(value, value)")
#         result_array = eval(f"np.{ufunc}(np.array(value), np.array(value))")
#         assert isinstance(
#           result_unitless, (np.ndarray, np.number)
#         ) and not isinstance(result_unitless, Quantity)
#         assert_equal(result_unitless, result_array)
#
#     for value, unitless_value in zip(unit_values, unitless_values):
#       for ufunc in ufuncs_dimensionless:
#         with pytest.raises(DimensionMismatchError):
#           eval(f"np.{ufunc}(value)", globals(), {"value": value})
#       for ufunc in ufuncs_dimensionless_twoargs:
#         with pytest.raises(DimensionMismatchError):
#           eval(
#             f"np.{ufunc}(value1, value2)",
#             globals(),
#             {"value1": value, "value2": unitless_value},
#           )
#         with pytest.raises(DimensionMismatchError):
#           eval(
#             f"np.{ufunc}(value2, value1)",
#             globals(),
#             {"value1": value, "value2": unitless_value},
#           )
#         with pytest.raises(DimensionMismatchError):
#           eval(f"np.{ufunc}(value, value)", globals(), {"value": value})


# Do not support numpy functions
# def test_numpy_functions_change_dimensions():
#   """
#   Test some numpy functions that change the dimensions of the Array.
#   """
#   unit_values = [np.array([1, 2]) * mV, np.ones((3, 3)) * 2 * mV]
#   for value in unit_values:
#     assert_quantity(np.var(value), np.var(np.array(value)), volt ** 2)
#     assert_quantity(np.square(value), np.square(np.array(value)), volt ** 2)
#     assert_quantity(np.sqrt(value), np.sqrt(np.array(value)), volt ** 0.5)
#     assert_quantity(
#       np.reciprocal(value), np.reciprocal(np.array(value)), 1.0 / volt
#     )


# Do not support numpy functions
# def test_numpy_functions_matmul():
#   """
#   Check support for matmul and the ``@`` operator.
#   """
#   no_units_eye = np.eye(3)
#   with_units_eye = no_units_eye * Mohm
#   matrix_no_units = np.arange(9).reshape((3, 3))
#   matrix_units = matrix_no_units * nA
#
#   # First operand with units
#   assert_allclose((no_units_eye @ matrix_units).value, matrix_units.value)
#   assert have_same_unit(no_units_eye @ matrix_units, matrix_units)
#   assert_allclose(np.matmul(no_units_eye, matrix_units.value), matrix_units.value)
#   assert have_same_unit(np.matmul(no_units_eye, matrix_units.value), matrix_units.value)
#
#   # Second operand with units
#   assert_allclose((with_units_eye @ matrix_no_units).value, (matrix_no_units * Mohm).value)
#   assert have_same_unit(
#     with_units_eye @ matrix_no_units, matrix_no_units * Mohm
#   )
#   assert_allclose(np.matmul(with_units_eye.value, matrix_no_units), (matrix_no_units * Mohm).value)
#   assert have_same_unit(
#     np.matmul(with_units_eye, matrix_no_units), matrix_no_units * Mohm
#   )
#
#   # Both operands with units
#   assert_allclose(
#     (with_units_eye @ matrix_units).value, (no_units_eye @ matrix_no_units * nA * Mohm).value
#   )
#   assert have_same_unit(with_units_eye @ matrix_units, nA * Mohm)
#   assert_allclose(
#     np.matmul(with_units_eye.value, matrix_units.value),
#     (np.matmul(no_units_eye, matrix_no_units) * nA * Mohm).value,
#   )
#   assert have_same_unit(np.matmul(with_units_eye, matrix_units), nA * Mohm)


# def test_numpy_functions_typeerror():
#   """
#   Assures that certain numpy functions raise a TypeError when called on
#   quantities.
#   """
#   unitless_values = [
#     3 * mV / mV,
#     np.array([1, 2]) * mV / mV,
#     np.ones((3, 3)) * mV / mV,
#   ]
#   unit_values = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
#   for value in unitless_values + unit_values:
#     for ufunc in ufuncs_integers:
#       if ufunc == "invert":
#         # only takes one argument
#         with pytest.raises(TypeError):
#           eval(f"np.{ufunc}(value)", globals(), {"value": value})
#       else:
#         with pytest.raises(TypeError):
#           eval(f"np.{ufunc}(value, value)", globals(), {"value": value})


# Doesn't support logical functions
# 
# def test_numpy_functions_logical():
#   """
#   Assure that logical numpy functions work on all quantities and return
#   unitless boolean arrays.
#   """
#   unit_values1 = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
#   unit_values2 = [3 * second, np.array([1, 2]) * second, np.ones((3, 3)) * second]
#   for ufunc in ufuncs_logical:
#     for value1, value2 in zip(unit_values1, unit_values2):
#       try:
#         # one argument
#         result_units = eval(f"np.{ufunc}(value1)")
#         result_array = eval(f"np.{ufunc}(np.array(value1))")
#       except (ValueError, TypeError):
#         # two arguments
#         result_units = eval(f"np.{ufunc}(value1, value2)")
#         result_array = eval(f"np.{ufunc}(np.array(value1), np.array(value2))")
#         # assert that comparing to a string results in "NotImplemented" or an error
#         try:
#           result = eval(f'np.{ufunc}(value1, "a string")')
#           assert result == NotImplemented
#         except (ValueError, TypeError):
#           pass  # raised on numpy >= 0.10
#         try:
#           result = eval(f'np.{ufunc}("a string", value1)')
#           assert result == NotImplemented
#         except (ValueError, TypeError):
#           pass  # raised on numpy >= 0.10
#       assert not isinstance(result_units, Quantity)
#       assert_equal(result_units, result_array)


# 
# def test_arange_linspace():
#     # For dimensionless values, the unit-safe functions should give the same results
#     assert_equal(brian2.arange(5), np.arange(5))
#     assert_equal(brian2.arange(1, 5), np.arange(1, 5))
#     assert_equal(brian2.arange(10, step=2), np.arange(10, step=2))
#     assert_equal(brian2.arange(0, 5, 0.5), np.arange(0, 5, 0.5))
#     assert_equal(brian2.linspace(0, 1), np.linspace(0, 1))
#     assert_equal(brian2.linspace(0, 1, 10), np.linspace(0, 1, 10))
#
#     # Make sure units are checked
#     with pytest.raises(DimensionMismatchError):
#         brian2.arange(1 * mV, 5)
#     with pytest.raises(DimensionMismatchError):
#         brian2.arange(1 * mV, 5 * mV)
#     with pytest.raises(DimensionMismatchError):
#         brian2.arange(1, 5 * mV)
#     with pytest.raises(DimensionMismatchError):
#         brian2.arange(1 * mV, 5 * ms)
#     with pytest.raises(DimensionMismatchError):
#         brian2.arange(1 * mV, 5 * mV, step=1 * ms)
#     with pytest.raises(DimensionMismatchError):
#         brian2.arange(1 * ms, 5 * mV)
#
#     # Check correct functioning with units
#     assert_quantity(
#         brian2.arange(5 * mV, step=1 * mV), float(mV) * np.arange(5, step=1), mV
#     )
#     assert_quantity(
#         brian2.arange(1 * mV, 5 * mV, 1 * mV), float(mV) * np.arange(1, 5, 1), mV
#     )
#     assert_quantity(brian2.linspace(1 * mV, 2 * mV), float(mV) * np.linspace(1, 2), mV)
#
#     # Check errors for arange with incorrect numbers of arguments/duplicate arguments
#     with pytest.raises(TypeError):
#         brian2.arange()
#     with pytest.raises(TypeError):
#         brian2.arange(0, 5, 1, 0)
#     with pytest.raises(TypeError):
#         brian2.arange(0, stop=1)
#     with pytest.raises(TypeError):
#         brian2.arange(0, 5, stop=1)
#     with pytest.raises(TypeError):
#         brian2.arange(0, 5, start=1)
#     with pytest.raises(TypeError):
#         brian2.arange(0, 5, 1, start=1)
#     with pytest.raises(TypeError):
#         brian2.arange(0, 5, 1, stop=2)
#     with pytest.raises(TypeError):
#         brian2.arange(0, 5, 1, step=2)


def test_list():
  """
  Test converting to and from a list.
  """
  values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(4, 3) * mV]
  for value in values:
    l = value.tolist()
    from_list = Quantity(l)
    assert have_same_unit(from_list, value)
    assert_allclose(from_list.value, value.value)


def test_check_units():
  """
  Test the check_units decorator
  """

  @check_units(v=volt)
  def a_function(v, x):
    """
    v has to have units of volt, x can have any (or no) unit.
    """
    pass

  # Try correct units
  a_function(3 * mV, 5 * second)
  a_function(5 * volt, "something")
  a_function([1, 2, 3] * volt, None)
  # lists that can be converted should also work
  a_function([1 * volt, 2 * volt, 3 * volt], None)
  # Strings and None are also allowed to pass
  a_function("a string", None)
  a_function(None, None)

  # Try incorrect units
  with pytest.raises(DimensionMismatchError):
    a_function(5 * second, None)
  with pytest.raises(DimensionMismatchError):
    a_function(5, None)
  with pytest.raises(TypeError):
    a_function(object(), None)
  with pytest.raises(TypeError):
    a_function([1, 2 * volt, 3], None)

  @check_units(result=second)
  def b_function(return_second):
    """
    Return a value in seconds if return_second is True, otherwise return
    a value in volt.
    """
    if return_second:
      return 5 * second
    else:
      return 3 * volt

  # Should work (returns second)
  b_function(True)
  # Should fail (returns volt)
  with pytest.raises(DimensionMismatchError):
    b_function(False)

  @check_units(a=bool, b=1, result=bool)
  def c_function(a, b):
    if a:
      return b > 0
    else:
      return b

  assert c_function(True, 1)
  assert not c_function(True, -1)
  with pytest.raises(TypeError):
    c_function(1, 1)
  with pytest.raises(TypeError):
    c_function(1 * mV, 1)
  with pytest.raises(TypeError):
    c_function(False, 1)


def test_get_basic_unit():
  """
  Test get_unit
  """
  values = [
    (volt.dim, volt),
    (mV.dim, volt),
    ((amp / metre ** 2).dim, amp / metre ** 2),
  ]
  for unit, expected_unit in values:
    unit = get_basic_unit(unit)
    assert isinstance(unit, Unit)
    assert unit == expected_unit
    assert float(unit) == 1.0


def test_get_best_unit():
  # get_best_unit should not check all values for long arrays, since it is
  # a function used for display purposes only. Instead, only the first and
  # last few values should matter (see github issue #966)
  long_ar = np.ones(10000) * siemens
  long_ar[:10] = 1 * nS
  long_ar[-10:] = 2 * nS
  values = [
    # (np.arange(10) * mV, mV),
    # ([0.001, 0.002, 0.003] * second, ms),
    (long_ar, nS),
  ]
  for ar, expected_unit in values:
    assert ar.get_best_unit() is expected_unit
    assert str(expected_unit) in ar.repr_in_best_unit()


def test_switching_off_unit_checks():
  """
  Check switching off unit checks (used for external functions).
  """
  from brainunit._base import turn_off_unit_checking

  x = 3 * second
  y = 5 * volt
  with pytest.raises(DimensionMismatchError):
    x + y

  with turn_off_unit_checking():
    # Now it should work
    assert np.asarray(x + y) == np.array(8)
    assert have_same_unit(x, y)
    assert x.has_same_unit(y)


def test_fail_for_dimension_mismatch():
  """
  Test the fail_for_dimension_mismatch function.
  """
  # examples that should not raise an error
  dim1, dim2 = fail_for_dimension_mismatch(3)
  assert dim1 is DIMENSIONLESS
  assert dim2 is DIMENSIONLESS
  dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt)
  assert dim1 is DIMENSIONLESS
  assert dim2 is DIMENSIONLESS
  dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt, 7)
  assert dim1 is DIMENSIONLESS
  assert dim2 is DIMENSIONLESS
  dim1, dim2 = fail_for_dimension_mismatch(3 * volt, 5 * volt)
  assert dim1 is volt.dim
  assert dim2 is volt.dim

  # examples that should raise an error
  with pytest.raises(DimensionMismatchError):
    fail_for_dimension_mismatch(6 * volt)
  with pytest.raises(DimensionMismatchError):
    fail_for_dimension_mismatch(6 * volt, 5 * second)


def test_deepcopy():
  d = {"x": 1 * second}
  from copy import deepcopy

  d_copy = deepcopy(d)
  assert d_copy["x"] == 1 * second
  d_copy["x"] += 1 * second
  assert d_copy["x"] == 2 * second
  assert d["x"] == 1 * second


# Doesn't support copy
# 
# def test_inplace_on_scalars():
#   # We want "copy semantics" for in-place operations on scalar quantities
#   # in the same way as for Python scalars
#   for scalar in [3 * mV, 3 * mV / mV]:
#     scalar_reference = scalar
#     scalar_copy = Quantity(scalar, copy=True)
#     scalar += scalar_copy
#     assert_equal(scalar_copy, scalar_reference)
#     scalar *= 1.5
#     assert_equal(scalar_copy, scalar_reference)
#     scalar /= 2
#     assert_equal(scalar_copy, scalar_reference)
#
#     # also check that it worked correctly for the scalar itself
#     assert_allclose(scalar, (scalar_copy + scalar_copy) * 1.5 / 2)
#
#   # For arrays, it should use reference semantics
#   for vector in [[3] * mV, [3] * mV / mV]:
#     vector_reference = vector
#     vector_copy = Quantity(vector, copy=True)
#     vector += vector_copy
#     assert_equal(vector, vector_reference)
#     vector *= 1.5
#     assert_equal(vector, vector_reference)
#     vector /= 2
#     assert_equal(vector, vector_reference)
#
#     # also check that it worked correctly for the vector itself
#     assert_allclose(vector, (vector_copy + vector_copy) * 1.5 / 2)


def test_units_vs_quantities():
  # Unit objects should stay Unit objects under certain operations
  # (important e.g. in the unit definition of Equations, where only units but
  # not quantities are allowed)
  assert isinstance(meter ** 2, Unit)
  assert isinstance(meter ** -1, Unit)
  assert isinstance(meter ** 0.5, Unit)
  assert isinstance(meter / second, Unit)
  assert isinstance(amp / meter ** 2, Unit)
  assert isinstance(1 / meter, Unit)
  assert isinstance(1.0 / meter, Unit)

  # Using the unconventional type(x) == y since we want to test that
  # e.g. meter**2 stays a Unit and does not become a Array however Unit
  # inherits from Array and therefore both would pass the isinstance test
  assert type(2 / meter) == Quantity
  assert type(2 * meter) == Quantity
  assert type(meter + meter) == Quantity
  assert type(meter - meter) == Quantity


def test_all_units_list():
  from brainunit._unit_common import all_units

  assert meter in all_units
  assert volt in all_units
  assert cm in all_units
  assert Hz in all_units
  assert all(isinstance(u, Unit) for u in all_units)


def test_constants():
  import brainunit._unit_constants as constants

  # Check that the expected names exist and have the correct dimensions
  assert constants.avogadro_constant.dim == (1 / mole).unit
  assert constants.boltzmann_constant.dim == (joule / kelvin).dim
  assert constants.electric_constant.dim == (farad / meter).dim
  assert constants.electron_mass.dim == kilogram.dim
  assert constants.elementary_charge.dim == coulomb.dim
  assert constants.faraday_constant.dim == (coulomb / mole).dim
  assert constants.gas_constant.dim == (joule / mole / kelvin).dim
  assert constants.magnetic_constant.dim == (newton / amp2).dim
  assert constants.molar_mass_constant.dim == (kilogram / mole).dim
  assert constants.zero_celsius.dim == kelvin.dim

  # Check the consistency between a few constants
  assert_allclose(
    constants.gas_constant.value,
    (constants.avogadro_constant * constants.boltzmann_constant).value,
  )
  assert_allclose(
    constants.faraday_constant.value,
    (constants.avogadro_constant * constants.elementary_charge).value,
  )

# if __name__ == "__main__":
#     test_construction()
#     test_get_dimensions()
#     test_display()
#     test_scale()
#     test_power()
#     test_pickling()
#     test_str_repr()
#     test_slicing()
#     test_setting()
#     test_multiplication_division()
#     test_addition_subtraction()
#     test_unary_operations()
#     test_binary_operations()
#     test_inplace_operations()
#     test_unit_discarding_functions()
#     test_unitsafe_functions()
#     test_special_case_numpy_functions()
#     test_numpy_functions_same_dimensions()
#     test_numpy_functions_indices()
#     test_numpy_functions_dimensionless()
#     test_numpy_functions_change_dimensions()
#     test_numpy_functions_typeerror()
#     test_numpy_functions_logical()
#     # test_arange_linspace()
#     test_list()
#     test_check_units()
#     test_get_unit()
#     test_get_best_unit()
#     test_switching_off_unit_checks()
#     test_fail_for_dimension_mismatch()
#     test_deepcopy()
#     test_inplace_on_scalars()
#     test_units_vs_quantities()
#     test_all_units_list()
#     test_constants()
