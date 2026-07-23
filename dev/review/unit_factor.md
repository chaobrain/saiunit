# `Unit.factor` system review (2026-07-23)

Focused inspection of the factor subsystem: `Unit._factor` and every code
path that consumes it — construction/validation, `magnitude`, arithmetic
(`__mul__`/`__div__`/`__pow__`/`reverse`), `factorless()`, the standard-unit
registry, conversion paths in `Quantity` (`to_decimal`, `in_unit`,
constructor list-reconciliation, `_zoom_values_with_units`), the non-SI
catalog (`_unit_constants.py`), and the FFT frequency helpers.

All findings below were **reproduced on this branch**
(`fix/quantity-unit-dimension-review`, Python 3.13, JAX CPU).

> **Status (2026-07-23, same day):** F1–F4 and F6–F8 are **fixed** on this
> branch; F5, F9, F10 are documented (docstrings/comments) as proposed.
> Verified by the full test suite (4038 passed) and `mypy saiunit/`
> (exit 0). Notes recorded during implementation:
>
> - F1: the doc's prediction that `(90*u.degree)**2` would return
>   `Quantity(8100, "°^2")` was wrong — `maybe_decimal` folds any
>   dimensionless-*dimension* result (factor included) to a plain
>   number, for `*` and `**` alike. Both now return `2.4674...`, which
>   is the actual consistency goal (pow matches mul).
> - F3: the repro's "true: 1e-63" was an arithmetic slip — Δscale for
>   `ym¹⁴ → zm¹⁴` is −42, so the true value is `1e-42` (corrected
>   below). Additionally, `_conversion_ratio` saturates genuinely
>   out-of-range ratios to `inf` (IEEE style, matching
>   `Unit.magnitude`) instead of letting Python's `**` raise
>   `OverflowError`.
> - F6: `Unit.__pow__`, `_validate_scale_and_factor` (both scale and
>   factor), and the `parse_unit` numeric-power branch now raise
>   `ValueError` with actionable messages; `Quantity.__pow__` catches
>   `(OverflowError, ValueError)` from the unit power and falls back to
>   `factorless()`.

## Background: how the factor system works

A unit's magnitude is `factor * base ** scale` (base is always 10).
The non-SI catalog (`saiunit/_unit_constants.py`) keeps `factor` in
`[1, 10)` and pushes the decade into `scale`, e.g.
`hour = 3.600 × 10³ s`, `day = 8.64 × 10⁴ s`, `degree = (π/180) × 10⁰ rad`.
Some catalog entries are scale-only with `factor=1.` (`bar`, `erg`, `dyn`,
`hectare`, `angstrom`, `micron`).

Equality, hashing, and the standard-unit registry key are all
**component-wise** on `(dim, scale, base, factor)` — `Unit(scale=3)` and
`Unit(factor=1000.)` have equal magnitude but are distinct units. This is
documented and intentional (`has_same_magnitude`, `__eq__`).

## Summary

| # | Severity | Issue |
|---|----------|-------|
| F1 | **High** | `Quantity.__pow__` strips the factor: `(1*foot)**1` silently becomes `0.3048 m`; inconsistent with `*`, `u.math.sqrt`, and `Unit.__pow__` |
| F2 | Medium | `Unit.factorless()` substitutes arbitrary dimensionless registry aliases: `arcmin.factorless()` → `Unit("crad^2")`, percent-like units → `Unit("rad")` |
| F3 | Medium | Magnitude-ratio conversion silently saturates for combined scales beyond ~±308: `to_decimal` returns `0.0`/`inf`, and `in_unit` returns the mantissa **unchanged** when both magnitudes overflow to `inf` |
| F4 | Medium | `fftfreq`/`rfftfreq` fallback registers ad-hoc units in the global registry, and their display name omits the factor (frequencies misread by the factor) |
| F5 | Low | Float drift in factor arithmetic: `1*inch == 25.4*mm` is `False`; `hour.reverse().reverse() != hour` |
| F6 | Low | Overflow ergonomics: `u.calorie ** 500`, `Unit(factor=2)**2000`, and `parse_unit('2^1024')` raise raw `OverflowError` with confusing messages |
| F7 | Low | A concrete 0-d JAX array is accepted as `factor`, producing a Unit whose `hash()` crashes |
| F8 | Info | `Unit(factor=True)` keeps the `bool`; huge exact `int` factors propagate — both interact with F6 |
| F9 | Info | `(1.0*cal) ** 500` raises `OverflowError` for Python-scalar mantissas but returns `inf` for array mantissas |
| F10 | Info | Catalog notes: radian-power aliases pollute the dimensionless registry keyspace (root cause of F2); `micron`/`umetre` share a physical key (harmless) |

---

## F1 (High) — `Quantity.__pow__` silently rewrites factor units to SI

```python
(1 * u.foot) ** 1        # Quantity(0.3048, "m")   — unit changed by **1 (!)
(1 * u.foot) ** 2        # Quantity(0.09290304, "m^2")
(1 * u.foot) * (1 * u.foot)   # Quantity(1, "ft^2") — mul keeps the factor
u.math.sqrt(Quantity(4.0, unit=u.acre))  # Quantity(2., "acre^0.5")
Quantity(4.0, unit=u.acre) ** 0.5        # Quantity(4.02336, "10.0^1.5 * m")
```

`Quantity.__pow__` opens with `self = self.factorless()`
(`_base_quantity.py:2380`), folding the factor into the mantissa and
switching the unit to the registered factor-1 standard (usually SI)
*before* exponentiation. Every neighbouring operation preserves the
factor: `Quantity.__mul__`/`__div__` compound it through
`Unit.__mul__`/`__div__`, `Unit.__pow__` computes `factor ** other`
(`_base_unit.py:1852`), and the `saiunit.math` change-unit wrappers
(`sqrt`, `cbrt`, …) apply `unit_fun` to the original unit. The result:
`q ** 2` and `q * q` produce the same physical value in **different
representations**, and `q ** 1` is not representation-preserving.

All values are physically correct — this is a representation-consistency
bug, but a loud one: it is the only place in the operator surface where
a factor unit silently converts to SI.

The line predates this repository (inherited from brainunit); its
plausible motivation is preventing `factor ** n` overflow for large
exponents — which it does not actually achieve for Python-scalar
mantissas (see F9).

**Fix** — remove the `factorless()` call from `__pow__` and let
`self.unit ** oc` carry the factor, matching `*`, `/`, and
`saiunit.math`:

```python
def __pow__(self, oc):
    if isinstance(oc, Quantity):
        ...
```

Guard the rare extreme-exponent overflow by wrapping the unit power:

```python
try:
    new_unit = self.unit ** oc
except OverflowError:
    # factor ** oc exceeds float range — fold the factor into the
    # mantissa and retry with the factor-1 unit.
    self = self.factorless()
    new_unit = self.unit ** oc
```

`prod`/`nanprod` (`_base_quantity.py:3004`, `:3058`) should **keep**
their `factorless()` call: there the exponent is the element count, so
`factor ** n` overflow is realistic for ordinary arrays (a 1000-element
calorie product would need `4.184 ** 1000`). Document that reduction
products of factor units are returned in the factor-1 standard unit.

Tests to add: `(1*foot)**1` preserves `ft`; `(1*foot)**2 == (1*foot)*(1*foot)`
including display; `u.math.sqrt(q) == q ** 0.5` display parity for a
factor unit; `(1*cal)**500` does not change behaviour class (see F9).

---

## F2 (Medium) — `factorless()` substitutes wrong dimensionless aliases

```python
u.arcmin.factorless()                    # Unit("crad^2")  — an angle¹ shown as rad²
Unit(factor=0.01).factorless()           # Unit("rad")     — a percent-like ratio shown as rad
Quantity(50.0, unit=Unit(factor=0.01)).factorless()  # Quantity(0.5, "rad")
u.degree.factorless()                    # Unit("rad")     — intended, tested contract
```

`Unit.factorless()` (`_base_unit.py:1275-1277`) does a *raw*
`_standard_units[key]` lookup on `(dim, scale, base, 1.)`. For
dimensionless dims this bypasses both the DIMENSIONLESS guard in
`_find_standard_unit` (`_base_unit.py:99-105`) and the ambiguity check.
The dimensionless keyspace is polluted (see F10): every power of every
prefixed radian is dimensionless, so scale −4 resolves to `cradian2`
(“crad^2”), which happens to be arcminute's scale. Physically all these
labels equal the same pure number, but the rendered unit is
nonsensical for the input (an angle¹ labelled as an angle²).

Note `degree.factorless() → radian` **is a tested contract**
(`_base_unit_test.py:1583`, `TestFactorlessIdentity`), so the fix must
not disable dimensionless substitution wholesale.

**Fix** — reject registry candidates whose display name encodes an
exponent, reusing the existing `_RE_DISPNAME_EXP` regex
(`_base_unit.py:315`):

```python
key = (self.dim, self.scale, self.base, 1.)
if key in _standard_units:
    candidate = _standard_units[key]
    # Dimensionless keys collapse radian powers (rad^2, rad^3, ...)
    # onto plain ratio scales; substituting a power alias would
    # mislabel a first-power unit (arcmin -> "crad^2").  Only accept
    # first-power display names.
    if not (self.dim.is_dimensionless
            and isinstance(candidate.dispname, str)
            and _RE_DISPNAME_EXP.match(candidate.dispname)):
        return candidate
```

Effects: `degree.factorless()` stays `radian` (contract preserved),
`arcsec.factorless()` stays `uradian` (first-power, fine),
`arcmin.factorless()` falls through to the anonymous
`Unit(dim, scale=-4)` path (renders `10.0^-4`). The percent→rad case
remains — defensible since `radian == UNITLESS` by design — but should
be called out in the `factorless()` docstring: *dimensionless results
may carry an angular label because saiunit does not distinguish angle
from ratio.*

---

## F3 (Medium) — magnitude-ratio conversion saturates at extreme scales

```python
Quantity(1.0, unit=u.ymetre ** 14).to_decimal(u.zmetre ** 14)   # 0.0   (true: 1e-42)
Quantity(1.0, unit=u.Zmetre ** 13) + Quantity(1.0, unit=u.Ymetre ** 13)
                                          # Quantity(inf, "Zm^13") (true: 1e39 + 1)
big1 = Unit(u.metre.dim, scale=312); big2 = Unit(u.metre.dim, scale=313)
Quantity(1.0, unit=big1).in_unit(big2)    # Quantity(1., ...) — UNCHANGED (true: 0.1)
```

Every conversion site computes the ratio of **materialized magnitudes**:

- `to_decimal` — `self.mantissa * (self.unit.magnitude / unit.magnitude)` (`_base_quantity.py:1093`)
- `in_unit` — same, plus a `self_mag == target_mag` short-circuit (`_base_quantity.py:1137-1142`)
- constructor list path — (`_base_quantity.py:649`)
- `_zoom_values_with_units` — (`_base_quantity.py:212`)

`Unit.magnitude` saturates to `inf` above ~1e308 and to `0.0` below
~1e-324 (`_base_unit.py:1081-1088`, deliberately IEEE-style). When both
magnitudes saturate the same way, the ratio is `nan`-adjacent garbage —
and `in_unit`'s `inf == inf` short-circuit is worse: it concludes the
units have equal magnitude and returns the mantissa **unchanged**, a
silently wrong number. These scales arise from high powers of prefixed
units (`ym ** 14` has scale −336), so the trigger is exotic but real,
and the failure is silent.

**Fix** — add one helper and use it at all four sites:

```python
def _conversion_ratio(u_from: Unit, u_to: Unit):
    """(f1 * b**s1) / (f2 * b**s2), computed as (f1/f2) * b**(s1-s2)
    so representable ratios of unrepresentable magnitudes stay finite."""
    return (u_from.factor / u_to.factor) * u_from.base ** (u_from.scale - u_to.scale)
```

`10.0 ** Δscale` is exact for |Δscale| ≤ 22 and correctly rounded
beyond, so precision for ordinary conversions is unchanged (verified:
hour→second still yields exactly `3600.0`). Ratios that genuinely
exceed the float range still yield `inf`/`0.0` — that is then honest.
Replace the `self_mag == target_mag` short-circuit in `in_unit` with
the component-wise `unit.has_same_magnitude(self.unit)` check that
`to_decimal` already uses (it cannot false-positive on saturation).

Tests to add: the three repros above, plus `hour → second == 3600.0`
exactness as a regression anchor.

---

## F4 (Medium) — FFT frequency fallback: registry pollution + lying display

```python
import saiunit.fft as sufft
d = Quantity(1.0, unit=Unit(u.second.dim, scale=28, factor=2.0))
sufft.fftfreq(4, d)   # Quantity([...], "10^-28 Hz")  — unit factor is 0.5,
                      # so values misread by 2×; and '10^-28 hertz' is now
                      # permanently in the global name registry
```

When the spacing unit's scale is farther than 3 from every entry in
`_time_freq_map` (i.e. |scale| ≥ 28), `fftfreq`/`rfftfreq` build an
ad-hoc frequency unit via **`Unit.create`**
(`fft/_fft_change_unit.py:832-837`, `:898-903`), which calls
`add_standard_unit` and permanently registers the unit (name and
dispname) in the global registries. Two defects:

1. The generated name `f'10^{-scale} hertz'` ignores the factor: for a
   factor-`f` spacing unit the frequency unit has `factor = 1/f`, so
   the rendered `"10^-28 Hz"` overstates every value by `f`. The
   *numbers* are correct against the true unit; the *label* is wrong.
2. Global registration is a side effect of a query function — the
   registry (and `parse_unit`) permanently learns misleading names, and
   `_unit_name_registry.setdefault` maps that name to whichever factor
   variant arrived first.

**Fix** — the fallback is exactly `1 / d.unit`; use `reverse()`:

```python
except KeyError:
    time_unit = d.unit
    freq_unit = d.unit.reverse()
```

`reverse()` (`_base_unit.py:1764`) negates the scale, inverts the
factor, does *not* register anything, and renders honestly through
`_canonical_str` (e.g. `"0.5 * 10.0^-28 * 1 / s"` — parseable and
correct). Verified the mainstream paths are already factor-correct:
`fftfreq(4, 1*u.hour)` → `mHz` via `to_decimal(ksecond)` (factor folded
into the values), `fftfreq(4, 1*u.minute)` → `dHz`.

---

## F5 (Low) — float drift in factor arithmetic; exact `==` surprises

```python
1 * u.inch == 25.4 * u.mm        # False  (ratio = 25.400000000000002)
u.hour.reverse().reverse() == u.hour   # False (1/(1/3.6) != 3.6)
(u.calorie / u.hour) * u.hour == u.calorie   # True — drift is data-dependent
1 * u.day == 24 * u.hour         # True  (this one happens to be exact)
```

Factor arithmetic in `__mul__`/`__div__`/`reverse` composes factors by
plain float ops; equality is exact (`_base_unit.py:1952`,
`Quantity.__eq__` after conversion). Whether a round-trip survives is
bit-luck: `4.184` survives inversion twice, `3.6` does not. This is
inherent to binary floating point — pint and astropy behave the same —
and is **not fixable** without rational/decimal factors, which would be
a rewrite.

**Proposal** — document rather than change code:

- In the `Unit` docstring's factor section and in `has_same_magnitude`:
  cross-unit `==` after conversion is exact-float; recommended
  comparison for factor units is `u.math.isclose`
  (`u.math.isclose(1*u.inch, 25.4*u.mm)` → `True`, verified).
- A `reverse()` docstring note that `u.reverse().reverse()` may differ
  from `u` in the last ulp of `factor`.

Optionally, `Quantity` equality/comparison conversion could use the
single-rounding `_conversion_ratio` from F3 (marginally fewer
roundings), but it does not rescue the inch/mm case; do not oversell it.

---

## F6 (Low) — overflow ergonomics: raw `OverflowError` in three paths

```python
u.calorie ** 500        # OverflowError: (34, 'Numerical result out of range')
Unit(u.metre.dim, factor=2) ** 2000   # OverflowError: int too large to convert to float
parse_unit('2^1024')    # OverflowError: (34, 'Numerical result out of range')
```

Three call sites let `OverflowError` escape with libc-level messages:

1. `Unit.__pow__` — `self.factor ** other` (`_base_unit.py:1852`).
2. `_validate_scale_and_factor` — `math.isfinite(factor)` raises for
   huge exact ints before the finiteness check can reject them
   (`_base_unit.py:666-674`); reachable because int factors propagate
   exactly through `**` (F8).
3. `parse_unit` numeric-power branch — `base_num ** exp`
   (`_base_unit.py:545`).

The `magnitude` property already treats this correctly (catches
`OverflowError` → `inf`, `_base_unit.py:1082-1088`); these three sites
predate that convention.

**Fix** — convert to the library's own error type with actionable text:

- In `_validate_scale_and_factor`, guard the huge-int case first:
  ```python
  if isinstance(factor, int) and not isinstance(factor, bool):
      try:
          float(factor)
      except OverflowError:
          raise ValueError(
              f"Unit factor overflows the float range: {factor!r}."
          ) from None
  ```
- In `Unit.__pow__` and the `parse_unit` numeric branch, wrap the
  power in `try/except OverflowError` and raise
  `ValueError(f"unit factor {f} ** {exp} overflows the float range; "
  "strip the factor first (factorless()) or reduce the exponent")`.

(If F1's fix lands, `Quantity.__pow__` catches the `Unit.__pow__`
overflow and falls back to `factorless()`, so quantity-level powers
degrade gracefully; the `ValueError` then only reaches users doing bare
`unit ** big`.)

---

## F7 (Low) — concrete array factors produce unhashable Units

```python
un = Unit(factor=jnp.asarray(2.0))   # constructs fine, str -> "2.0"
hash(un)                             # TypeError: unhashable type: ArrayImpl
```

`_normalise_scalar` (`_base_unit.py:632-645`) converts `numbers.Real`
scalars but passes arrays through so **tracers** keep working under
`jit`. A *concrete* 0-d JAX array also passes through, yielding a Unit
that breaks hashing, dict registries, and `__eq__`'s bool contract.

**Fix** — in `_normalise_scalar`, unwrap concrete 0-d arrays before the
pass-through:

```python
if getattr(x, 'ndim', None) == 0 and not _is_tracer(x):
    try:
        return _normalise_scalar(x.item())
    except (AttributeError, TypeError):
        pass
return x
```

(`_is_tracer` lives in `_base_quantity.py`; either import lazily or
duplicate the 3-line check to avoid a cycle.) Tracers still pass
through untouched, preserving the documented jit behaviour.

---

## F8 (Info) — bool and huge-int factors

- `Unit(factor=True)` keeps the `bool` (`True == 1`, hashes with 1, so
  the unit *is* unitless and registry-compatible). Harmless; coercing
  `bool → int` in `_normalise_scalar` would be one line of hygiene.
- Int factors are preserved exactly and compound exactly:
  `(Unit(factor=2) ** 200).factor == 2**200` (exact int). This is why
  F6's `math.isfinite` overflow is reachable. No behaviour change
  needed beyond F6; worth a comment on `_factor`'s type annotation that
  `int` factors are intentional.

## F9 (Info) — backend-dependent overflow in `Quantity.__pow__`

`(1.0 * u.calorie) ** 500` raises `OverflowError` today because
`factorless()` makes the mantissa the Python float `4.184`, and Python
float `**` raises where NumPy/JAX return `inf`. With an array mantissa
the same expression returns `inf`. This asymmetry is general Quantity
behaviour (Python-scalar mantissas follow Python semantics), not
factor-specific; noted here because the factor fold is what plants the
Python scalar. F1's fix removes the fold; the asymmetry remains for
genuinely scalar quantities and belongs to the earlier review's scope
if it is to be addressed.

## F10 (Info) — catalog and registry observations

- **Radian powers pollute the dimensionless keyspace** (root cause of
  F2): the generated catalog registers `radian2`, `cradian2`,
  `Yradian3`, … all with `dim == DIMENSIONLESS`, so dimensionless
  registry keys are shared between ratios, angles, and angle powers.
  Any future registry consumer must treat dimensionless hits as
  *labels*, not identities (this is what bit `factorless()`).
- `micron` (`µm`) and `umetre` (`um`) share the physical key
  `(metre, -6, 10, 1.0)` with different dispnames, so the key is
  ambiguity-flagged; composition falls back to parts-based rendering,
  which happens to produce `um` anyway. No action needed; documented
  here so the duplicate registration is not "fixed" into a behaviour
  change.
- The `[1, 10)` factor normalization convention in
  `_unit_constants.py` is undocumented; a module-level comment stating
  the invariant (and that `Unit.create` does *not* enforce it) would
  prevent drive-by "fixes" that re-encode e.g. `hour` as
  `factor=3600, scale=0` — which would change `==`/hash/registry
  identity for pickled data.

## Verified correct (worth keeping under test)

- `u.math.sin(90 * u.degree) == 1.0`, `float(90*u.degree)` = π/2,
  `np.sin` ufunc dispatch on degree quantities — factor folded via
  `to_decimal` everywhere.
- `(1*u.hour).to_decimal(u.second) == 3600.0` exactly;
  `1*day == 24*hour`, `1*hour == 60*minute == 3600*second`,
  `1*bar == 100*kpascal` all exact.
- `Quantity([1*u.hour, 30*u.minute])` → `[1., 0.5] h`;
  `Quantity([1*u.foot, 1*u.metre])` → `[1., 3.28] ft` (list
  reconciliation folds factors).
- `(1*u.hour) // (7*u.minute)` → `8.0`; `%` and `divmod` align factor
  units before evaluating.
- `str`/`parse_unit` round-trips: `parse_unit(str(u.calorie/u.hour))`,
  anonymous factor units (`"2.0 * m^2 * kg / s^2"`), registered
  unicode dispnames (`°`, `′`, `″`, `µm`).
- `add_standard_unit` refuses non-plain-number factors (tracer safety);
  `factor <= 0` and non-finite factors rejected at construction;
  `parse_unit('0^2')` correctly surfaces the positivity error.
- `fftfreq`/`rfftfreq` mainstream path (|scale| ≤ 27) is factor-correct
  (`1*u.hour` → values in `mHz`, factor folded via `to_decimal`).
