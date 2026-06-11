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

"""Regression tests from the Quantity string-formatting audit.

Each test below pins one confirmed bug in ``__repr__`` / ``__str__`` /
``__format__`` / ``_format_value`` (saiunit/_base_quantity.py). The docstrings
record the wrong behavior observed before the fix.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import saiunit as u
from saiunit import Quantity


class TestReprLargePythonInt:
    def test_repr_does_not_crash_on_large_python_int(self):
        """``_format_value`` promotes Python scalars via ``jnp.asarray`` and
        caught only TypeError. A Python int that does not fit int32 raises
        OverflowError, so ``repr``/``str`` crashed for values as small as
        2**40.

        Before the fix: OverflowError: Python int 1099511627776 too large to
        convert to int32.
        """
        q = Quantity(2 ** 40, unit=u.mV)
        r = repr(q)
        s = str(q)
        assert "1099511627776" in r
        assert "1099511627776" in s


class TestPercentFormatGuard:
    def test_percent_rejected_for_named_dimensionless_unit(self):
        """The '%' guard in ``__format__`` checked ``unit.is_unitless`` while
        display uses ``unit.should_display_unit``. Radian is unitless by
        magnitude but displays 'rad', so the guard let through exactly the
        nonsense it was written to block.

        Before the fix: format(Quantity(0.5, unit=u.radian), '.1%')
        == '50.0% rad'.
        """
        q = Quantity(0.5, unit=u.radian)
        with pytest.raises(ValueError):
            format(q, ".1%")

    def test_percent_as_fill_character_is_not_rejected(self):
        """The guard tested ``'%' in format_spec``, so '%' used as a
        legitimate fill character (e.g. '%>10') was rejected even though the
        format type is not '%'.

        Before the fix: ValueError instead of '%%%%%%%0.5 mV'
        (plain ``format(0.5, '%>10')`` gives '%%%%%%%0.5').
        """
        q = Quantity(0.5, unit=u.mV)
        assert format(q, "%>10") == "%%%%%%%0.5 mV"

    def test_percent_array_keeps_percent_semantics(self):
        """For non-scalar unitless quantities, '.1%' fell into the
        precision-regex path which treated '%' as a plain rounding spec: no
        x100 scaling and no '%' sign — silently wrong values. Scalars gave
        '50.0%'; arrays gave '[0.5 0.2]'.

        Either honoring percent semantics per element or rejecting the spec
        for arrays is acceptable; silently emitting wrong numbers is not.
        """
        q = Quantity(np.array([0.5, 0.25]))
        try:
            out = format(q, ".1%")
        except ValueError:
            return  # explicit rejection is an acceptable contract
        assert "%" in out, f"percent semantics dropped: {out!r}"


class TestFormatSpecArraySemantics:
    def test_scientific_spec_array_uses_scientific_notation(self):
        """The array path extracted only the precision digit from '.1e' and
        applied decimal rounding (np.round(value, 1)), ignoring the 'e' type
        entirely. Scalars honor it ('1.2e+03'); arrays did not.

        Before the fix: format(Quantity([1234.5], mV), '.1e')
        == '[1234.5] mV'.
        """
        q = Quantity(np.array([1234.5]), unit=u.mV)
        out = format(q, ".1e")
        value_part = out.replace("mV", "")
        assert "e" in value_part, f"scientific notation dropped: {out!r}"


class TestFormatSpecUnderJit:
    def test_format_spec_scalar_under_jit_does_not_crash(self):
        """``__format__`` had no tracer guard (unlike ``_format_value``).
        For 0-d quantities it called ``format(self.mantissa, format_spec)``
        which raises on tracers.

        Before the fix: TypeError: unsupported format string passed to
        DynamicJaxprTracer.__format__.
        """
        captured = []

        @jax.jit
        def f(x):
            q = Quantity(x, unit=u.mV)
            captured.append(f"{q:.2f}")
            return x

        f(jnp.asarray(1.0))
        assert captured and isinstance(captured[0], str)

    def test_format_spec_array_under_jit_does_not_crash(self):
        """For non-scalar quantities ``__format__`` called
        ``np.asarray(self.mantissa)`` which raises
        TracerArrayConversionError on traced arrays.
        """
        captured = []

        @jax.jit
        def f(x):
            q = Quantity(x, unit=u.mV)
            captured.append(f"{q:.2f}")
            return x

        f(jnp.arange(3.0))
        assert captured and isinstance(captured[0], str)


class TestFormatSpecLazyBackends:
    def test_format_spec_dask_does_not_materialize(self):
        """``_format_value`` never materializes dask graphs and
        ``np.asarray(Quantity)`` raises BackendError, but
        ``__format__('.2f')`` called ``np.asarray(self.mantissa)`` directly,
        silently computing the whole graph.

        Either a lazy-safe fallback or BackendError is acceptable; silent
        materialization is not.
        """
        da = pytest.importorskip("dask.array")
        reads = []

        def spy(block):
            if block.size:  # ignore dask's zero-size meta-inference calls
                reads.append(block.size)
            return block

        arr = da.from_array(np.array([1.0, 2.0]), chunks=1).map_blocks(spy)
        q = Quantity(arr, unit=u.mV)
        reads.clear()
        try:
            format(q, ".2f")
        except u.BackendError:
            pass  # explicit rejection is an acceptable contract
        assert not reads, "format(q, '.2f') materialized the dask graph"

    def test_format_spec_ndonnx_no_opaque_crash(self):
        """``__format__('.2f')`` applied ``np.round`` to the ndonnx symbolic
        array, raising an opaque ufunc TypeError. ``repr``/``str`` degrade
        gracefully for the same quantity.

        Before the fix: TypeError: loop of ufunc does not support argument 0
        of type Array which has no callable rint method.
        """
        ndx = pytest.importorskip("ndonnx")
        q = Quantity(ndx.asarray(np.array([1.0, 2.0])), unit=u.mV)
        try:
            out = format(q, ".2f")
        except u.BackendError:
            return  # explicit rejection is an acceptable contract
        assert isinstance(out, str)

    def test_format_spec_torch_requires_grad_does_not_crash(self):
        """``__format__('.2f')`` called ``np.asarray`` on the torch mantissa;
        tensors with ``requires_grad=True`` refuse the conversion.
        ``repr``/``str`` work for the same quantity.

        Before the fix: RuntimeError: Can't call numpy() on Tensor that
        requires grad.
        """
        torch = pytest.importorskip("torch")
        q = Quantity(torch.tensor([1.5], requires_grad=True), unit=u.mV)
        out = format(q, ".2f")
        assert isinstance(out, str)
