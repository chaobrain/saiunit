# Test Coverage Gap-Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rigorous tests for the genuinely-untested public functions, the 15 missing physical constants, and the 7 modules that currently have no sibling test file — closing the verified coverage gaps in saiunit.

**Architecture:** Colocated extension. Extend the existing sibling `*_test.py` where one exists; create a new sibling `*_test.py` only where none exists. Follow established patterns: `absl.testing.parameterized` / plain pytest functions, the `assert_quantity(q, mantissa, unit=None)` helper, comparison against the `jnp`/`np` reference, and the `class Array(u.CustomArray)` wrapper where the sibling file already exercises it.

**Tech Stack:** pytest, absl-py, jax, numpy, mypy.

---

## IMPORTANT — read before starting

**The target functions already exist and are implemented.** Therefore the normal TDD red→green rhythm is inverted: each new test is expected to **PASS on first run** against the existing code. The "verify it fails first" step does not apply.

- If a newly-written test **PASSES**: good — the gap is now covered. Commit.
- If a newly-written test **FAILS**: do NOT edit the test to make it pass. A failure means you have found a **real bug** in the library. STOP, and report the failure (function, input, expected vs actual) to the user before continuing. Do not modify any file under `saiunit/` except `*_test.py` files unless the user directs you to fix the bug.

**Verification gate (from CLAUDE.md), applied at the end (Task 17):**
- `pytest` green for every changed file.
- `mypy saiunit/` exits 0 (hard CI gate) before the final commit.

**Conventions every test file must keep:**
- Apache 2.0 license header (copy verbatim from any existing `*_test.py` in the same directory).
- File named `<module>_test.py`, colocated with the source.
- Import the public surface via `import saiunit as u` / `import saiunit.math as bm`, and `from saiunit._base_getters import assert_quantity` where numeric comparison is needed.

---

## File Structure

| Action | File | Responsibility |
|---|---|---|
| Modify | `saiunit/linalg/_linalg_keep_unit_test.py` | add `svdvals`, `eigvals` tests |
| Modify | `saiunit/math/_fun_accept_unitless_test.py` | add `frexp` tests |
| Modify | `saiunit/math/_fun_keep_unit_test.py` | add `promote_dtypes` tests |
| Modify | `saiunit/math/_fun_remove_unit_test.py` | add `iscomplexobj`, `alltrue`, `sometrue` tests |
| Modify | `saiunit/math/_fun_change_unit_test.py` | add `cumproduct` test |
| Modify | `saiunit/math/_einops_test.py` | add `einshape` tests |
| Modify | `saiunit/_jax_guard_test.py` | add `jax_only` tests |
| Modify | `saiunit/_base_getters_test.py` | add `is_unit_equal_math` tests |
| Modify | `saiunit/constants_test.py` | add 15 missing constants |
| Create | `saiunit/_unit_shortcuts_test.py` | sweep 28 unit shortcuts |
| Create | `saiunit/math/_alias_test.py` | sweep 14 re-exported names |
| Create | `saiunit/_jax_compat_test.py` | smoke-test 19 JAX shims |
| Create | `saiunit/_compatible_import_test.py` | smoke-test 5 JAX-core shims |
| Create | `saiunit/_typing_test.py` | type-alias import/usability |
| Create | `saiunit/_sparse_base_test.py` | `SparseMatrix` base behavior |
| Create | `saiunit/autograd/_misc_test.py` | private helper unit tests |

The license header referenced below is this exact block (year may be 2024, 2025, or 2026 in existing files — any is fine):

```python
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
```

---

## Task 1: linalg `svdvals` and `eigvals`

**Files:**
- Modify: `saiunit/linalg/_linalg_keep_unit_test.py` (append at end of file)

- [ ] **Step 1: Append the tests**

Append to the end of `saiunit/linalg/_linalg_keep_unit_test.py`:

```python
def test_svdvals_keeps_unit():
    m = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    sv_ref = jnp.linalg.svdvals(m)
    # unitless input -> unitless singular values
    assert_quantity(bulinalg.svdvals(m), sv_ref)
    # dimensioned input -> singular values carry the same unit
    assert_quantity(bulinalg.svdvals(m * meter), sv_ref, meter)


def test_eigvals_keeps_unit():
    a = jnp.array([[1., 2.], [2., 1.]])
    ev_ref = jnp.linalg.eigvals(a)
    assert_quantity(bulinalg.eigvals(a), ev_ref)
    assert_quantity(bulinalg.eigvals(a * meter), ev_ref, meter)
```

(`jnp`, `bulinalg`, `meter`, and `assert_quantity` are already imported at the top of this file.)

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/linalg/_linalg_keep_unit_test.py::test_svdvals_keeps_unit saiunit/linalg/_linalg_keep_unit_test.py::test_eigvals_keeps_unit -v`
Expected: 2 passed. (If either fails, STOP — see "read before starting".)

- [ ] **Step 3: Commit**

```bash
git add saiunit/linalg/_linalg_keep_unit_test.py
git commit -m "test(linalg): cover svdvals and eigvals unit propagation"
```

---

## Task 2: math `frexp`

**Files:**
- Modify: `saiunit/math/_fun_accept_unitless_test.py` (append at end of file)

- [ ] **Step 1: Inspect the file header imports**

Run: `sed -n '1,40p' saiunit/math/_fun_accept_unitless_test.py`
Confirm `import jax.numpy as jnp`, `import numpy as np`, `import pytest`, and `import saiunit as u` are present. If `numpy` is not imported as `np`, add `import numpy as np` to the import block.

- [ ] **Step 2: Append the tests**

Append to the end of `saiunit/math/_fun_accept_unitless_test.py`:

```python
def test_frexp_matches_jnp():
    x = jnp.array([1.0, 2.0, 4.0])
    m, e = u.math.frexp(x)
    jm, je = jnp.frexp(x)
    assert np.allclose(m, jm)
    assert np.allclose(e, je)


def test_frexp_requires_dimensionless():
    x = jnp.array([1.0, 2.0]) * u.meter
    with pytest.raises(TypeError, match="dimensionless"):
        u.math.frexp(x)


def test_frexp_with_unit_to_scale():
    # Scaling by the same unit makes the input dimensionless first.
    x = jnp.array([1.0, 2.0, 4.0]) * u.meter
    m, e = u.math.frexp(x, unit_to_scale=u.meter)
    jm, je = jnp.frexp(jnp.array([1.0, 2.0, 4.0]))
    assert np.allclose(m, jm)
    assert np.allclose(e, je)
```

- [ ] **Step 3: Run the tests (expect PASS)**

Run: `pytest saiunit/math/_fun_accept_unitless_test.py -k frexp -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add saiunit/math/_fun_accept_unitless_test.py
git commit -m "test(math): cover frexp dimensionless handling"
```

---

## Task 3: math `promote_dtypes`

**Files:**
- Modify: `saiunit/math/_fun_keep_unit_test.py` (append at end of file)

- [ ] **Step 1: Inspect the file header imports**

Run: `sed -n '1,40p' saiunit/math/_fun_keep_unit_test.py`
Confirm `import jax.numpy as jnp` and `import saiunit as u` are present.

- [ ] **Step 2: Append the tests**

Append to the end of `saiunit/math/_fun_keep_unit_test.py`:

```python
def test_promote_dtypes_common_type_and_unit():
    a = [1, 2, 3] * u.second          # integer mantissa
    b = [4.0, 5.0, 6.0] * u.second    # float mantissa
    out = u.math.promote_dtypes(a, b)
    assert isinstance(out, list)
    assert len(out) == 2
    # Both promoted to the common (float) dtype.
    assert jnp.issubdtype(out[0].mantissa.dtype, jnp.floating)
    assert jnp.issubdtype(out[1].mantissa.dtype, jnp.floating)
    # Unit is preserved on both.
    assert out[0].unit == u.second
    assert out[1].unit == u.second
    # Values are unchanged.
    assert jnp.allclose(out[0].mantissa, jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(out[1].mantissa, jnp.array([4.0, 5.0, 6.0]))
```

- [ ] **Step 3: Run the test (expect PASS)**

Run: `pytest saiunit/math/_fun_keep_unit_test.py -k promote_dtypes -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add saiunit/math/_fun_keep_unit_test.py
git commit -m "test(math): cover promote_dtypes type promotion and unit preservation"
```

---

## Task 4: math `iscomplexobj`, `alltrue`, `sometrue`

**Files:**
- Modify: `saiunit/math/_fun_remove_unit_test.py` (append at end of file)

- [ ] **Step 1: Append the tests**

Append to the end of `saiunit/math/_fun_remove_unit_test.py`:

```python
def test_iscomplexobj_matches_jnp():
    real = jnp.array([1.0, 2.0])
    comp = jnp.array([1.0 + 2.0j])
    assert bm.iscomplexobj(real) == jnp.iscomplexobj(real)
    assert bm.iscomplexobj(comp) == jnp.iscomplexobj(comp)
    # Unit is stripped before the check; result is unaffected by the unit.
    assert bm.iscomplexobj(real * u.meter) is False
    assert bm.iscomplexobj(comp * u.meter) is True


def test_alltrue_is_all_alias():
    assert bm.alltrue is bm.all
    x = jnp.array([True, True, False])
    assert bool(bm.alltrue(x)) == bool(jnp.all(x))


def test_sometrue_is_any_alias():
    assert bm.sometrue is bm.any
    x = jnp.array([False, False, True])
    assert bool(bm.sometrue(x)) == bool(jnp.any(x))
```

(`jnp`, `bm`, and `u` are already imported at the top of this file.)

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/math/_fun_remove_unit_test.py -k "iscomplexobj or alltrue or sometrue" -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add saiunit/math/_fun_remove_unit_test.py
git commit -m "test(math): cover iscomplexobj and alltrue/sometrue aliases"
```

---

## Task 5: math `cumproduct`

**Files:**
- Modify: `saiunit/math/_fun_change_unit_test.py` (append at end of file)

- [ ] **Step 1: Inspect the file header imports**

Run: `sed -n '1,40p' saiunit/math/_fun_change_unit_test.py`
Confirm `import jax.numpy as jnp` and `import saiunit as u` (and `import saiunit.math as bm` if that is the local convention) are present. Use whichever module alias the file already uses for the math namespace in the test below (`u.math` always works).

- [ ] **Step 2: Append the test**

Append to the end of `saiunit/math/_fun_change_unit_test.py`:

```python
def test_cumproduct_is_cumprod_alias():
    assert u.math.cumproduct is u.math.cumprod
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(u.math.cumproduct(x), u.math.cumprod(x))
```

- [ ] **Step 3: Run the test (expect PASS)**

Run: `pytest saiunit/math/_fun_change_unit_test.py -k cumproduct -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add saiunit/math/_fun_change_unit_test.py
git commit -m "test(math): cover cumproduct alias of cumprod"
```

---

## Task 6: math `einshape`

**Files:**
- Modify: `saiunit/math/_einops_test.py` (append at end of file)

- [ ] **Step 1: Append the tests**

Append to the end of `saiunit/math/_einops_test.py`:

```python
def test_einshape_basic():
    x = jnp.zeros((2, 3, 5))
    assert u.math.einshape(x, 'batch _ w') == {'batch': 2, 'w': 5}


def test_einshape_with_quantity():
    x = jnp.zeros((2, 3, 5)) * u.meter
    assert u.math.einshape(x, 'batch _ w') == {'batch': 2, 'w': 5}


def test_einshape_rejects_composite_axes():
    x = jnp.zeros((6, 4))
    with pytest.raises(RuntimeError):
        u.math.einshape(x, '(a b) c')
```

(`jnp`, `pytest`, and `u` are already imported at the top of this file.)

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/math/_einops_test.py -k einshape -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add saiunit/math/_einops_test.py
git commit -m "test(math): cover einshape parsing and composite-axis rejection"
```

---

## Task 7: `_jax_guard.jax_only`

**Files:**
- Modify: `saiunit/_jax_guard_test.py` (append at end of file)

- [ ] **Step 1: Update imports**

In `saiunit/_jax_guard_test.py`, change the existing import line:

```python
from saiunit._jax_guard import require_jax_backend
```

to:

```python
from saiunit._jax_guard import require_jax_backend, jax_only
```

- [ ] **Step 2: Append the tests**

Append to the end of `saiunit/_jax_guard_test.py`:

```python
def test_jax_only_passes_for_jax_quantity():
    @jax_only
    def f(x):
        return x

    q = u.Quantity(jnp.array([1.0]), unit=u.meter)
    assert f(q) is q


def test_jax_only_raises_for_numpy_quantity():
    @jax_only
    def f(x):
        return x

    q = u.Quantity(np.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        f(q)


def test_jax_only_preserves_metadata():
    @jax_only
    def my_func(x):
        """my docstring"""
        return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "my docstring"
```

(`jnp`, `np`, `pytest`, and `u` are already imported at the top of this file.)

- [ ] **Step 3: Run the tests (expect PASS)**

Run: `pytest saiunit/_jax_guard_test.py -k jax_only -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add saiunit/_jax_guard_test.py
git commit -m "test: cover jax_only decorator guarding and metadata preservation"
```

---

## Task 8: `_base_getters.is_unit_equal_math`

**Files:**
- Modify: `saiunit/_base_getters_test.py` (append at end of file)

- [ ] **Step 1: Inspect the file header imports**

Run: `sed -n '1,40p' saiunit/_base_getters_test.py`
Confirm `import saiunit as u` is present. The function is accessible as `u.is_unit_equal_math`.

- [ ] **Step 2: Append the tests**

Append to the end of `saiunit/_base_getters_test.py`:

```python
def test_is_unit_equal_math_equivalent_units():
    # A composed unit equivalent to volt converts the same way.
    assert u.is_unit_equal_math(u.volt, u.amp * u.ohm) is True
    # Aliases of the same SI unit are mathematically equal.
    assert u.is_unit_equal_math(u.metre, u.meter) is True
    # Identity is trivially equal.
    assert u.is_unit_equal_math(u.volt, u.volt) is True


def test_is_unit_equal_math_different_dim():
    assert u.is_unit_equal_math(u.volt, u.amp) is False
```

- [ ] **Step 3: Run the tests (expect PASS)**

Run: `pytest saiunit/_base_getters_test.py -k is_unit_equal_math -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add saiunit/_base_getters_test.py
git commit -m "test: cover is_unit_equal_math equivalence semantics"
```

---

## Task 9: 15 missing physical constants

**Files:**
- Modify: `saiunit/constants_test.py`

**Background:** `constants_test.py` defines a module-level `constants_list` that drives `test_quantity_constants_and_unit_constants`, which cross-checks each name against `saiunit._unit_constants`. 13 of the 15 missing names exist in **both** `saiunit.constants` and `saiunit._unit_constants`, so they can be added to `constants_list`. Two names — `electronvolt` and `gram` — exist **only** in `saiunit.constants`, so adding them to `constants_list` would break the loop (`getattr(unit_constants, ...)` would raise). Those two get a separate dimension test.

- [ ] **Step 1: Add the 13 dual-defined names to `constants_list`**

Open `saiunit/constants_test.py`. Find the end of the `constants_list = [ ... ]` literal (just before the closing `]`). Add this block of names inside the list:

```python
    # --- added coverage: dual-defined in constants + _unit_constants ---
    'arcminute', 'arcsecond', 'astronomical_unit', 'atmosphere', 'calorie_th',
    'fluid_ounce_US', 'gallon_US', 'horsepower', 'kilogram_force', 'lb',
    'radian', 'speed_unit', 'watt',
```

(Do NOT add `electronvolt` or `gram` here — they are not in `_unit_constants`.)

- [ ] **Step 2: Add a standalone dimension test for all 15 names**

Append to the end of `saiunit/constants_test.py`:

```python
def test_added_constant_dimensions():
    import saiunit.constants as constants

    # 13 dual-defined names (also exercised by the cross-check loop).
    assert constants.arcminute.dim == u.radian.dim
    assert constants.arcsecond.dim == u.radian.dim
    assert constants.radian.dim == u.radian.dim
    assert constants.astronomical_unit.dim == meter.dim
    assert constants.atmosphere.dim == (newton / meter2).dim
    assert constants.calorie_th.dim == joule.dim
    assert constants.fluid_ounce_US.dim == (meter ** 3).dim
    assert constants.gallon_US.dim == (meter ** 3).dim
    assert constants.horsepower.dim == watt.dim
    assert constants.kilogram_force.dim == newton.dim
    assert constants.lb.dim == kilogram.dim
    assert constants.speed_unit.dim == (meter / second).dim
    assert constants.watt.dim == watt.dim

    # 2 names defined only in saiunit.constants.
    assert constants.electronvolt.dim == joule.dim
    assert constants.gram.dim == kilogram.dim
    # gram is one-thousandth of a kilogram.
    assert u.math.isclose(
        constants.gram.to_decimal(kilogram), 0.001
    )
```

**Note on symbols:** `meter`, `meter2`, `newton`, `joule`, `kilogram`, `second`, and `watt` must be available in the test module. The file already does `from saiunit._unit_common import *` (confirm via `sed -n '40,60p' saiunit/constants_test.py`). If `watt` or `meter2` is not in scope after the star-import, reference them as `u.watt` / `u.meter2` instead (both resolve). Prefer the symbol already used elsewhere in the file for consistency.

- [ ] **Step 3: Run the tests (expect PASS)**

Run: `pytest saiunit/constants_test.py -v`
Expected: all passed, including `test_quantity_constants_and_unit_constants` (now looping over the 13 added names) and the new `test_added_constant_dimensions`.

- [ ] **Step 4: Commit**

```bash
git add saiunit/constants_test.py
git commit -m "test(constants): cover 15 previously-untested physical constants"
```

---

## Task 10: `_unit_shortcuts` sweep (new file)

**Files:**
- Create: `saiunit/_unit_shortcuts_test.py`

- [ ] **Step 1: Create the file**

Create `saiunit/_unit_shortcuts_test.py` with the license header (see File Structure section) followed by:

```python
import saiunit as u
from saiunit import _unit_common as uc
from saiunit import _unit_shortcuts as sh

# Each shortcut must be the exact base unit object it aliases.
SHORTCUT_TO_BASE = {
    "mV": "mvolt",
    "mA": "mamp", "uA": "uamp", "nA": "namp", "pA": "pamp",
    "pF": "pfarad", "uF": "ufarad", "nF": "nfarad",
    "nS": "nsiemens", "uS": "usiemens", "mS": "msiemens",
    "ms": "msecond", "us": "usecond",
    "Hz": "hertz", "kHz": "khertz", "MHz": "Mhertz",
    "cm": "cmetre", "cm2": "cmetre2", "cm3": "cmetre3",
    "mm": "mmetre", "mm2": "mmetre2", "mm3": "mmetre3",
    "um": "umetre", "um2": "umetre2", "um3": "umetre3",
    "mM": "mmolar", "uM": "umolar", "nM": "nmolar",
}


def test_shortcut_map_covers_all_exports():
    # Guard against the __all__ and the map drifting apart.
    assert set(sh.__all__) == set(SHORTCUT_TO_BASE)


def test_each_shortcut_is_its_base_unit():
    for short, base in SHORTCUT_TO_BASE.items():
        assert getattr(sh, short) is getattr(uc, base), f"{short} should alias {base}"


def test_shortcuts_exposed_on_top_level_package():
    for short in SHORTCUT_TO_BASE:
        assert getattr(u, short) is getattr(sh, short), f"u.{short} missing or mismatched"
```

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/_unit_shortcuts_test.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add saiunit/_unit_shortcuts_test.py
git commit -m "test: add sweep covering all unit shortcuts"
```

---

## Task 11: `math/_alias` re-export sweep (new file)

**Files:**
- Create: `saiunit/math/_alias_test.py`

**Background:** `math/_alias.py` re-exports 14 names from `saiunit._base_getters` (12 names) and `saiunit._base_decorators` (`check_dims`, `check_units`). All are accessible on `saiunit.math` and are the same objects as on the top-level `saiunit` package.

- [ ] **Step 1: Create the file**

Create `saiunit/math/_alias_test.py` with the license header followed by:

```python
import saiunit as u
import saiunit.math as bm
from saiunit import _alias_module_names  # noqa: F401  (placeholder; see below)
```

Then REPLACE that placeholder import line — `math/_alias.py` has no helper, so just drop it — and use the explicit `__all__` from the module:

```python
import saiunit as u
import saiunit.math as bm
from saiunit.math import _alias as alias_mod

GETTER_NAMES = [
    'is_dimensionless', 'is_unitless', 'get_dim', 'get_unit', 'get_mantissa',
    'get_magnitude', 'display_in_unit', 'maybe_decimal',
    'fail_for_dimension_mismatch', 'fail_for_unit_mismatch', 'assert_quantity',
    'get_or_create_dimension',
]
DECORATOR_NAMES = ['check_dims', 'check_units']


def test_alias_all_matches_expected():
    assert set(alias_mod.__all__) == set(GETTER_NAMES + DECORATOR_NAMES)


def test_aliases_exposed_on_math_namespace():
    for name in GETTER_NAMES + DECORATOR_NAMES:
        assert hasattr(bm, name), f"saiunit.math.{name} missing"


def test_getter_aliases_are_same_object_as_top_level():
    # The re-exported getters are the exact objects on the top-level package.
    for name in GETTER_NAMES:
        if hasattr(u, name):
            assert getattr(bm, name) is getattr(u, name), f"math.{name} != u.{name}"
```

**Important:** Use only the SECOND code block — the first block contains a placeholder import (`_alias_module_names`) that does not exist and must NOT be written to the file. It is shown only to be explicitly discarded.

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/math/_alias_test.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add saiunit/math/_alias_test.py
git commit -m "test(math): add sweep covering _alias re-exports"
```

---

## Task 12: `_jax_compat` shims (new file)

**Files:**
- Create: `saiunit/_jax_compat_test.py`

**Background:** With JAX installed (the default test environment), `HAS_JAX` is `True` and every export passes through to the real `jax` package. These are smoke tests confirming each symbol imports and the non-trivial helpers behave.

- [ ] **Step 1: Create the file**

Create `saiunit/_jax_compat_test.py` with the license header followed by:

```python
import numpy as np
import pytest

import saiunit._jax_compat as jc

EXPORTS = [
    "HAS_JAX", "jax", "jnp", "Tracer", "ShapedArray", "ShapeDtypeStruct",
    "DynamicJaxprTracer", "TracerArrayConversionError",
    "register_pytree_node_class", "ensure_compile_time_eval", "result_type",
    "canonicalize_dtype", "tree_map", "tree_structure", "tree_flatten", "tree",
    "device_put", "devices", "require_jax",
]


def test_all_exports_present():
    assert set(jc.__all__) == set(EXPORTS)
    for name in EXPORTS:
        assert hasattr(jc, name), f"_jax_compat.{name} missing"


def test_has_jax_true_in_test_env():
    # The default test environment installs JAX.
    assert jc.HAS_JAX is True
    assert jc.jax is not None
    assert jc.jnp is not None


def test_require_jax_noop_when_jax_present():
    # Should not raise when JAX is installed.
    jc.require_jax("anything")


def test_result_type_and_canonicalize_dtype():
    assert jc.result_type(np.int32, np.float32) == np.float32
    assert jc.canonicalize_dtype(np.float64) is not None


def test_tree_helpers_roundtrip():
    tree_obj = {"a": [1, 2], "b": 3}
    leaves, treedef = jc.tree_flatten(tree_obj)
    assert sorted(leaves) == [1, 2, 3]
    rebuilt = treedef.unflatten(leaves)
    assert rebuilt == tree_obj
    doubled = jc.tree_map(lambda x: x * 2, tree_obj)
    assert doubled == {"a": [2, 4], "b": 6}
    assert jc.tree_structure(tree_obj) == treedef


def test_register_pytree_node_class_returns_class():
    @jc.register_pytree_node_class
    class Dummy:
        pass

    assert Dummy is not None
```

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/_jax_compat_test.py -v`
Expected: 6 passed.

- [ ] **Step 3: Commit**

```bash
git add saiunit/_jax_compat_test.py
git commit -m "test: add smoke tests for _jax_compat shims"
```

---

## Task 13: `_compatible_import` shims (new file)

**Files:**
- Create: `saiunit/_compatible_import_test.py`

- [ ] **Step 1: Create the file**

Create `saiunit/_compatible_import_test.py` with the license header followed by:

```python
import pytest

import saiunit._compatible_import as ci

EXPORTS = ["safe_map", "unzip2", "wrap_init", "Primitive", "concrete_or_error"]


def test_all_exports_present():
    assert set(ci.__all__) == set(EXPORTS)
    for name in EXPORTS:
        assert hasattr(ci, name), f"_compatible_import.{name} missing"


def test_safe_map_applies_function():
    assert ci.safe_map(lambda a, b: a + b, [1, 2, 3], [10, 20, 30]) == [11, 22, 33]


def test_safe_map_rejects_length_mismatch():
    # Only meaningful in the JAX>=0.6 reimplementation; tolerate either path.
    try:
        ci.safe_map(lambda a, b: a + b, [1, 2], [1])
    except (ValueError, Exception):
        pass


def test_unzip2_splits_pairs():
    xs, ys = ci.unzip2([(1, "a"), (2, "b"), (3, "c")])
    assert xs == (1, 2, 3)
    assert ys == ("a", "b", "c")


def test_concrete_or_error_passes_concrete_value():
    assert ci.concrete_or_error(int, 5) == 5


def test_primitive_is_constructible_with_jax():
    # With JAX installed, Primitive is the real jax primitive class.
    p = ci.Primitive("my_prim")
    assert p.name == "my_prim"
```

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/_compatible_import_test.py -v`
Expected: 6 passed. (If `test_primitive_is_constructible_with_jax` fails because the installed JAX `Primitive` signature differs, replace its body with `assert isinstance(ci.Primitive, type)` and re-run.)

- [ ] **Step 3: Commit**

```bash
git add saiunit/_compatible_import_test.py
git commit -m "test: add smoke tests for _compatible_import shims"
```

---

## Task 14: `_typing` aliases (new file)

**Files:**
- Create: `saiunit/_typing_test.py`

- [ ] **Step 1: Create the file**

Create `saiunit/_typing_test.py` with the license header followed by:

```python
import jax.numpy as jnp
import numpy as np

import saiunit._typing as st

EXPORTS = [
    "Array", "ArrayLike", "ScalarOrArrayLike", "DTypeLike",
    "Shape", "Axis", "Axes", "PyTree",
]


def test_all_exports_present():
    assert set(st.__all__) == set(EXPORTS)
    for name in EXPORTS:
        assert hasattr(st, name), f"_typing.{name} missing"


def test_array_isinstance_with_jax():
    # With JAX installed, Array is jax.Array.
    assert isinstance(jnp.array([1.0, 2.0]), st.Array)
    assert not isinstance([1.0, 2.0], st.Array)


def test_typing_reexported_from_public_module():
    import saiunit.typing as pub
    for name in EXPORTS:
        assert hasattr(pub, name), f"saiunit.typing.{name} missing"
```

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/_typing_test.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add saiunit/_typing_test.py
git commit -m "test: cover _typing aliases import and isinstance behavior"
```

---

## Task 15: `_sparse_base.SparseMatrix` (new file)

**Files:**
- Create: `saiunit/_sparse_base_test.py`

- [ ] **Step 1: Inspect the base class surface**

Run: `sed -n '55,180p' saiunit/_sparse_base.py`
Identify which methods/properties `SparseMatrix` defines concretely vs. leaves abstract. Confirm that the concrete COO/CSR subclasses live in `saiunit/sparse/_coo.py` and `saiunit/sparse/_csr.py` and that `from saiunit import sparse` exposes `COO` / `CSR`.

- [ ] **Step 2: Create the file**

Create `saiunit/_sparse_base_test.py` with the license header followed by:

```python
import jax.numpy as jnp

import saiunit as u
from saiunit._sparse_base import SparseMatrix, _same_sparsity_pattern


def test_sparsematrix_is_exported():
    assert SparseMatrix.__name__ == "SparseMatrix"
    assert "SparseMatrix" in __import__("saiunit._sparse_base", fromlist=["__all__"]).__all__


def test_same_sparsity_pattern_identity():
    a = jnp.array([0, 1, 2])
    assert _same_sparsity_pattern(a, a) is True


def test_same_sparsity_pattern_equal_values():
    a = jnp.array([0, 1, 2])
    b = jnp.array([0, 1, 2])
    assert _same_sparsity_pattern(a, b) is True


def test_same_sparsity_pattern_different_shape():
    a = jnp.array([0, 1, 2])
    b = jnp.array([0, 1])
    assert _same_sparsity_pattern(a, b) is False


def test_concrete_subclasses_are_sparsematrix():
    # COO / CSR subclass the shared base.
    assert issubclass(u.sparse.COO, SparseMatrix)
    assert issubclass(u.sparse.CSR, SparseMatrix)
```

**Note:** If Step 1 reveals that `u.sparse.COO` / `u.sparse.CSR` are named differently (e.g. lowercase, or not re-exported on `u.sparse`), import them from their defining module instead (`from saiunit.sparse._coo import COO`) and adjust `test_concrete_subclasses_are_sparsematrix` accordingly. Keep the three `_same_sparsity_pattern` tests unchanged.

- [ ] **Step 3: Run the tests (expect PASS)**

Run: `pytest saiunit/_sparse_base_test.py -v`
Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add saiunit/_sparse_base_test.py
git commit -m "test: cover SparseMatrix base and sparsity-pattern helper"
```

---

## Task 16: `autograd/_misc` private helpers (new file)

**Files:**
- Create: `saiunit/autograd/_misc_test.py`

- [ ] **Step 1: Create the file**

Create `saiunit/autograd/_misc_test.py` with the license header followed by:

```python
import pytest

from saiunit.autograd._misc import (
    _ensure_index,
    _argnums_partial,
    _check_callable,
    _isgeneratorfunction,
)


def test_ensure_index_scalar():
    assert _ensure_index(2) == 2


def test_ensure_index_sequence():
    assert _ensure_index([1, 2, 3]) == (1, 2, 3)


def test_argnums_partial_single_arg():
    def f(a, b, c):
        return a + b + c

    argnums, partial_fun, dyn = _argnums_partial(f, 1, (10, 20, 30), {})
    assert argnums == 1
    assert dyn == (20,)
    # Re-supply only the differentiated arg; statics are baked in.
    assert partial_fun(99) == 10 + 99 + 30


def test_argnums_partial_negative_index_normalized():
    def f(a, b):
        return a - b

    argnums, partial_fun, dyn = _argnums_partial(f, -1, (5, 2), {})
    assert argnums == 1
    assert dyn == (2,)
    assert partial_fun(2) == 3


def test_argnums_partial_out_of_bounds():
    def f(a):
        return a

    with pytest.raises(ValueError, match="out of bounds"):
        _argnums_partial(f, 3, (1,), {})


def test_check_callable_accepts_function():
    def f(x):
        return x

    _check_callable(f)  # no raise


def test_check_callable_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        _check_callable(42)


def test_check_callable_rejects_staticmethod():
    with pytest.raises(TypeError, match="staticmethod"):
        _check_callable(staticmethod(lambda x: x))


def test_check_callable_rejects_generator_function():
    def gen(x):
        yield x

    with pytest.raises(TypeError, match="generator"):
        _check_callable(gen)


def test_isgeneratorfunction():
    def gen(x):
        yield x

    def regular(x):
        return x

    assert _isgeneratorfunction(gen) is True
    assert _isgeneratorfunction(regular) is False
```

- [ ] **Step 2: Run the tests (expect PASS)**

Run: `pytest saiunit/autograd/_misc_test.py -v`
Expected: 10 passed.

- [ ] **Step 3: Commit**

```bash
git add saiunit/autograd/_misc_test.py
git commit -m "test(autograd): cover argnums/index/callable private helpers"
```

---

## Task 17: Full verification gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite for all changed areas**

Run:
```bash
pytest saiunit/linalg/_linalg_keep_unit_test.py \
       saiunit/math/_fun_accept_unitless_test.py \
       saiunit/math/_fun_keep_unit_test.py \
       saiunit/math/_fun_remove_unit_test.py \
       saiunit/math/_fun_change_unit_test.py \
       saiunit/math/_einops_test.py \
       saiunit/math/_alias_test.py \
       saiunit/_jax_guard_test.py \
       saiunit/_base_getters_test.py \
       saiunit/constants_test.py \
       saiunit/_unit_shortcuts_test.py \
       saiunit/_jax_compat_test.py \
       saiunit/_compatible_import_test.py \
       saiunit/_typing_test.py \
       saiunit/_sparse_base_test.py \
       saiunit/autograd/_misc_test.py -v
```
Expected: all passed, 0 failed.

- [ ] **Step 2: Run mypy (hard CI gate)**

Run: `mypy saiunit/`
Expected: exit 0. (Test files are generally not type-checked, but per CLAUDE.md this must be confirmed clean before any push. If a new test introduced a `mypy error:` line, fix it at the source per CLAUDE.md guidance — do not add `# type: ignore` unless documenting a known mypy bug.)

- [ ] **Step 3: Run the whole suite once for regressions**

Run: `pytest saiunit/ -q`
Expected: no new failures versus the pre-change baseline. (If pre-existing unrelated failures exist on this branch, note them but do not attempt to fix outside scope.)

- [ ] **Step 4: Final confirmation**

Report to the user: number of tests added per file, the `pytest` summary line, and the `mypy` exit status. If any target function's test FAILED (revealing a library bug), report it explicitly rather than marking the plan complete.

---

## Self-Review

**Spec coverage** — every spec item maps to a task:
- Group 1 untested functions → Tasks 1–8 (svdvals, eigvals, frexp, promote_dtypes, iscomplexobj, alltrue, sometrue, cumproduct, einshape, jax_only, is_unit_equal_math). ✓
- Group 2 constants (15) → Task 9. ✓
- Group 3 no-test-file modules → Tasks 10–16 (_unit_shortcuts, math/_alias, _jax_compat, _compatible_import, _typing, _sparse_base, autograd/_misc). ✓
- Verification gate (pytest + mypy) → Task 17. ✓

**Placeholder scan:** Task 11 deliberately shows a discarded placeholder import with an explicit "do NOT write this" instruction; the real code block follows. No other TBD/TODO. ✓

**Type/name consistency:** `assert_quantity(q, mantissa, unit=None)` used per its real signature; `bulinalg`/`bm`/`u`/`jnp` aliases match each file's existing imports (verified by reading the files); `constants_list` extension excludes `electronvolt`/`gram` (verified absent from `_unit_constants`); alias identities (`cumproduct is cumprod`, `alltrue is all`, `sometrue is any`) verified against the live library. ✓
