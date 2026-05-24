# Per-backend CI and `promote_integers` cleanup

**Date:** 2026-05-24
**Status:** Draft for review

## Problem

saiunit supports six array backends — `numpy`, `jax`, `cupy`, `torch`, `dask`, `ndonnx` — but CI installs all of them together in one environment and runs `pytest saiunit/`. This hides two classes of bugs:

1. **Pure-backend installs are never exercised.** A user who runs `pip install saiunit` (no extras) gets numpy only; nothing in CI proves that path works end-to-end. Same for `pip install saiunit[torch]`, `[dask]`, `[ndonnx]`.
2. **Backend-specific kwargs leak through the saiunit API.** `saiunit.math.sum` and `saiunit.math.prod` accept `promote_integers=True` and forward it to the active backend's `sum`/`prod`. Only `jax.numpy` accepts that kwarg — every other backend raises `TypeError`. The same shape of bug likely exists for other JAX-only kwargs.

## Goal

Add CI jobs that install **only one backend** at a time and run the full test suite, so that:
- the `pip install saiunit[<backend>]` install path is proven on every PR;
- backend-specific kwargs that leak through saiunit's public API fail loudly on CI instead of at user runtime;
- a future contributor adding a kwarg like `promote_integers` to a saiunit function will see a red build the same day.

Fix the `promote_integers` failures the new jobs surface, by removing the kwarg from the saiunit public API.

## Non-goals

- A cupy CI job. GitHub free runners have no GPU; covered by maintainers locally.
- macOS / Windows variants of per-backend jobs. The existing `test_linux` / `test_macos` / `test_windows` jobs (which install all backends) keep cross-platform smoke coverage.
- A new JAX-version matrix. `CI-daily.yml` already does that.
- A generic "backend kwarg translation" layer. The handful of leaking kwargs are removed at the source instead.

## Design

### 1. CI architecture

Extend `.github/workflows/CI.yml` (do not create new workflow files). Add five new jobs alongside `test_no_jax` / `test_linux` / `test_macos` / `test_windows`:

| Job name | Backend | Extra install |
|---|---|---|
| `test_pure_numpy` | numpy | none (numpy is in `requirements.txt`) |
| `test_pure_jax` | jax | `pip install jax` |
| `test_pure_torch` | torch | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `test_pure_dask` | dask | `pip install 'dask[array]'` |
| `test_pure_ndonnx` | ndonnx | `pip install ndonnx` |

Each job:

- Runs on `ubuntu-latest`, Python 3.13.
- Installs `requirements.txt` + the single extra backend library above. **No other backend libraries are installed.**
- Sets env `SAIUNIT_DEFAULT_BACKEND=<name>` before running pytest.
- Runs `pytest saiunit/`.

The five existing jobs are kept verbatim — they cover the cross-platform "all installed" smoke that pure-isolation jobs intentionally do not.

### 2. Conftest changes

In `conftest.py`, at `pytest_configure`, read env var `SAIUNIT_DEFAULT_BACKEND`. When set, call `u.set_default_backend(name)` so that every test that does **not** use the `backend` fixture runs under the configured backend instead of jax-or-numpy auto-pick.

The existing `backend` fixture (`params=["numpy", "jax", "cupy", "torch", "dask", "ndonnx"]`) is unchanged — it still parametrizes per-test and overrides the session default via `using_backend`. Its `pytest.importorskip(...)` calls already skip params whose library is not installed, so in a pure-X job only the matching param runs.

### 3. Code fixes — drop `promote_integers` from the saiunit API

Remove the `promote_integers` parameter from:

- `saiunit.math.sum` (`saiunit/math/_fun_keep_unit.py:1985`)
- `saiunit.math.nansum` (same file, nearby)
- `saiunit.math.prod` (`saiunit/math/_fun_change_unit.py:341`)
- `saiunit.math.nanprod` (same file, nearby)
- The corresponding `Quantity` methods (`.sum`, `.nansum`, `.prod`, `.nanprod`) if they accept it.

Concretely: delete the parameter from the function signature, the docstring section that describes it, and the internal forwarding (e.g., the `promote_integers=promote_integers` argument passed to `_fun_keep_unit_unary` / `jnp.prod`). Don't silently ignore it — let Python raise `TypeError: unexpected keyword argument` for callers that still pass it, since that's the discoverability we want from a breaking change. Users who specifically need JAX's integer-promotion behavior call `jax.numpy.sum` / `jax.numpy.prod` directly.

This is a **breaking change**. Note it in `changelog.md` under the next version's section.

### 4. Test fixes

The pure-backend jobs will surface failures during implementation. Triage each into one of four buckets and apply the listed fix:

1. **Test passes a removed kwarg** (the `promote_integers` cases) → delete the kwarg from the test call site.
2. **Test asserts a JAX-only type or calls a JAX-only function** (e.g., `jnp.array`, `jax.jit`) → mark with `@pytest.mark.requires_jax` (the marker is already wired up in `conftest.py`).
3. **Test exercises a feature the backend genuinely does not implement** (e.g., dask laziness vs. eager `.item()`, ndonnx lacking `concatenate`) → inside the test, skip with `pytest.skip(f"<backend> does not support <feature>")` gated by an `is_<backend>_array` check or a backend-name check.
4. **Genuine cross-backend bug in saiunit's own code** → fix at the source (most often in `_backend.py` dispatch, in a `xp.` call that should have routed through `get_backend`, or in a function that has a hidden `jnp.` reference). These are the failures the new CI exists to expose.

The implementation plan will enumerate the actual failing tests by running each pure-X locally and bucketing.

### 5. Audit for other leaking JAX-only kwargs

Before declaring the work done, run a quick grep for other suspects forwarded from saiunit reductions / unary ops to the backend: `weights=`, `fill_value=`, `unique_indices=`, `mode=`, `dtype=` overloads. Any that fail on a non-JAX backend get the same treatment as `promote_integers` (remove from saiunit's API, document in changelog).

This audit is bounded — only kwargs that block the pure-backend CI from going green are in scope. Broader cleanup is a follow-up.

## Success criteria

- All five new `test_pure_*` jobs are green on `main`.
- The four existing CI jobs (`test_no_jax`, `test_linux`, `test_macos`, `test_windows`) remain green.
- `grep -rn "promote_integers" saiunit/` returns no hits — neither in source nor in tests. (The only allowed remaining reference is the changelog entry at the repo root.)
- A developer can reproduce a pure-backend run locally:
  ```
  pip install -r requirements.txt
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install .
  SAIUNIT_DEFAULT_BACKEND=torch pytest saiunit/
  ```
- `changelog.md` documents the `promote_integers` removal as a breaking change.

## Open questions

None. (User decisions captured during brainstorming: scope = CI + fixes; isolation = pure; OS = Linux only for per-backend; cupy = skipped; fix style = drop kwarg from saiunit API; workflow layout = extend existing `CI.yml`; torch install = CPU wheel.)
