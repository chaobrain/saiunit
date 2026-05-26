# Test Coverage Gap-Fill — Design

**Date:** 2026-05-26
**Scope:** Fill the gaps only — add tests for genuinely-untested public functions,
missing physical constants, and modules that currently have no sibling test file.
Out of scope: deepening already-passing tests, the 2089 unit-object definitions
in `_unit_common.py`, and functions already tested in a non-sibling file.

## Approach

Colocated extension (Approach A): extend the existing sibling `*_test.py` where one
exists; create a new sibling `*_test.py` only where none exists. Follow the project's
established patterns — `absl.testing.parameterized`, the `assert_quantity` helper,
comparison against the `jnp`/`np` reference, and `CustomArray` compatibility checks
where the sibling file already exercises them.

## Gap Inventory (verified)

### Group 1 — Untested public functions (extend sibling test file)

| Function | Source module | Test file to extend | Notes |
|---|---|---|---|
| `svdvals` | `linalg/_linalg_keep_unit.py` | `linalg/_linalg_keep_unit_test.py` | singular values |
| `eigvals` | `linalg/_linalg_keep_unit.py` | `linalg/_linalg_keep_unit_test.py` | eigenvalues |
| `cumproduct` | `math/_fun_change_unit.py` | `math/_fun_change_unit_test.py` | alias of `cumprod` |
| `frexp` | `math/_fun_accept_unitless.py` | `math/_fun_accept_unitless_test.py` | mantissa/exponent split |
| `promote_dtypes` | `math/_fun_keep_unit.py` | `math/_fun_keep_unit_test.py` | dtype promotion |
| `iscomplexobj` | `math/_fun_remove_unit.py` | `math/_fun_remove_unit_test.py` | predicate |
| `alltrue` | `math/_fun_remove_unit.py` | `math/_fun_remove_unit_test.py` | alias of `all` |
| `sometrue` | `math/_fun_remove_unit.py` | `math/_fun_remove_unit_test.py` | alias of `any` |
| `einshape` | `math/_einops.py` | `math/_einops_test.py` | shape rearrange |
| `jax_only` | `_jax_guard.py` | `_jax_guard_test.py` | guard decorator |
| `is_unit_equal_math` | `_base_getters.py` | `_base_getters_test.py` | unit equality |

`jacobian` was initially flagged but is already tested in `autograd/_jacobian_test.py` —
excluded.

### Group 2 — Missing physical constants (extend `constants_test.py`)

Add to `constants_list` and verify: `arcminute`, `arcsecond`, `astronomical_unit`,
`atmosphere`, `calorie_th`, `electronvolt`, `fluid_ounce_US`, `gallon_US`, `gram`,
`horsepower`, `kilogram_force`, `lb`, `radian`, `speed_unit`, `watt`.

Cross-check numeric value against `scipy.constants` where a counterpart exists;
dimension-only check for those without a scipy counterpart (e.g. `speed_unit`, `radian`).

### Group 3 — No-test-file modules (new sibling test files)

| New test file | Target | Test content |
|---|---|---|
| `_unit_shortcuts_test.py` | `_unit_shortcuts.py` (28) | sweep: each shortcut `== scale × base_unit` (e.g. `mV == 0.001*volt`); dimension match |
| `math/_alias_test.py` | `math/_alias.py` (14) | sweep: each alias is same callable / same result as its target |
| `_jax_compat_test.py` | `_jax_compat.py` (19) | smoke: each export imports; basic call/constant behavior |
| `_compatible_import_test.py` | `_compatible_import.py` (5) | smoke: shims import and behave |
| `_typing_test.py` | `_typing.py` (8) | type aliases importable / usable in `isinstance` or annotation |
| `_sparse_base_test.py` | `_sparse_base.py` | light: base behavior not already covered by coo/csr tests |
| `autograd/_misc_test.py` | `autograd/_misc.py` | unit tests for private helpers `_ensure_index`, `_argnums_partial`, `_check_callable`, `_isgeneratorfunction` |

## Rigor per category

- **keep/change/remove/accept-unitless functions:**
  1. value matches `jnp`/`np` reference on unitless input;
  2. unit propagation correct (keep → same unit; change → expected derived unit;
     remove → unitless; accept-unitless → raises on dimensioned input or strips correctly);
  3. error path on dimensionally-invalid input;
  4. `CustomArray` compatibility where the sibling file already exercises it.
- **aliases** (`cumproduct`, `alltrue`, `sometrue`, and all of `math/_alias`):
  equivalence-only — assert identity-or-equivalence to the target; do not re-test the
  target's full behavior.
- **constants / shortcuts:** numeric value (vs scipy where available) + dimension.
- **shims / typing / private helpers:** import + minimal behavioral smoke.

## Verification gate

- `pytest saiunit/<changed dirs>` green.
- `mypy saiunit/` exits 0 (hard CI gate) before any commit.
- Multi-backend: changes are test-only and use the existing `jnp` reference pattern,
  riding the existing backend-parametrized infrastructure without new wiring.

## Out of scope

- Deepening already-passing tests.
- The 2089 unit-object definitions in `_unit_common.py`.
- Functions already tested in a non-sibling file: `qr`, `eig`, `eigh`, `trace`,
  `diagonal`, `matrix_transpose`, `kron`, `matrix_power`.
