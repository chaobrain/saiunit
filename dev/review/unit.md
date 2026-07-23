# `Unit` issues (`saiunit/_base_unit.py`)

All reproductions run against saiunit 0.5.2. The Unit class held up
well overall: equality/hash contract, factor-unit conversion, display
round-trips (`parse_unit(str(u)) == u` for compound, anonymous, and
fractional-power units), ambiguity handling, and pickling were all
verified correct. Remaining findings:

---

## U1 (Medium) — `parse_unit` cannot parse parenthesised powers

```python
u.parse_unit("(m / s)^2")
# ValueError: Unknown unit token: '(m / s)' in '(m / s)^2'
```

`_parse_term` (`_base_unit.py:443`) finds the caret, but the atom
`"(m / s)"` is neither a number nor a registry name, so it fails. The
canonical formatter never *emits* this form today, but it is the
natural way for users to write squared compound units, and the parser
already handles parenthesised groups everywhere else.

**Fix** — in `_parse_term`, before the numeric/registry lookup of
`atom`, recurse into a paren-group atom:

```python
caret_idx = s.rfind('^')
if caret_idx > 0:
    atom = s[:caret_idx].strip()
    exp_str = s[caret_idx + 1:].strip()
    ...  # exp parsing unchanged
    if atom.startswith('(') and atom.endswith(')'):
        inner = atom[1:-1].strip()
        if not inner:
            raise ValueError(f"Empty parenthesised group in {s!r}")
        return _parse_expression(inner) ** exp
    ...
```

(Verify the paren-balance the same way the existing outer-paren strip
does, so `"(m) * (s)^2"` — where the atom is not a single group — is
not mis-handled. In practice the caret split already isolates one term,
so a simple balanced check on `atom` suffices.)

---

## U2 (Low) — comparison and arithmetic disagree about bare `Unit` operands

```python
Quantity(1.0, unit=u.mV) == u.mV   # True   (unit promoted to 1*mV)
Quantity(1.0, unit=u.mV) + u.mV    # TypeError: Cannot add ... a bare Unit
```

Arithmetic deliberately rejects bare units (`_reject_bare_unit`), while
`_comparison` accepts them via `_to_quantity`, implicitly reading `mV`
as `1*mV`. Either behaviour is defensible; having both is confusing.

**Proposal** — keep comparisons permissive (comparing against a unit as
"one of that unit" is a common idiom and is side-effect free) but state
the asymmetry in the `Unit` docstring and in `Quantity.__eq__` docs.
If strictness is preferred instead, make `_comparison` return
`NotImplemented` for `Unit` operands — but note that would silently
change `q == u.mV` to `False` (identity fallback), which is worse than
either current behaviour; so documenting is the recommended option.

---

## U3 (Low) — display-part merging keys on `dispname`

`_normalise_display_parts` (`_base_unit.py:314`) merges entries whose
*display* symbol matches, regardless of the underlying unit name:

```python
merged: dict[str, tuple] = {}
for name, disp, exp in result:
    if disp in merged: ...   # keyed by dispname
```

Two distinct registered units that share a display symbol (a user unit
registered with `dispname='m'` alongside metre, say) will have their
exponents summed in rendered compound strings. This is display-only —
the physical `dim/scale/factor` math is computed independently and
stays correct — but the rendered string can misrepresent the
composition.

**Fix** — key the merge on `(name, disp)` (falling back to `disp` only
when names are missing), so same-symbol-different-name parts stay
separate:

```python
key = (name, disp)
```

`_merge_display_parts` above it already keys on `name`; aligning the
two removes the inconsistency. Registration already discourages
symbol collisions (`_unit_name_registry.setdefault` keeps the first),
so severity is low.

---

## G1 (Low) — `is_unit_equal_math` docstring contradicts `Unit.__eq__`

`_base_getters.py:360` says:

> This is distinct from ``Unit.__eq__``, which also requires
> ``name``/``dispname`` to match — that operator treats two aliases as
> different units.

`Unit.__eq__` (`_base_unit.py:1891`) does **not** compare names — its
own comment says names are explicitly *not* part of equality, and
empirically `u.metre == u.meter` is `True`. The two functions perform
the identical component-wise comparison.

**Fix** — correct the docstring (and the mirrored claim in
`has_same_unit`): `is_unit_equal_math(u1, u2)` is equivalent to
`u1 == u2` for `Unit` operands, differing only in that it returns
`False` (rather than `NotImplemented`) for non-`Unit` input. Optionally
implement it as a thin wrapper over `__eq__` to keep them from
drifting.

---

## U4 (Info) — registry growth characteristics

- `_unit_registration_index` maps `id(unit) → index` and never evicts.
  Entries are safe (the alias list holds a strong reference, so ids are
  never recycled while an entry exists) but the dict grows with every
  registered unit for the process lifetime. Fine for realistic usage;
  worth a comment. If it ever matters, key by the unit object itself in
  a `dict` after making registration order part of `Unit` state, or use
  an `id → (ref, index)` scheme.
- Compound-unit auto-registration (each new `Unit.__mul__`/`__div__`
  result that hits `add_standard_unit` via user code) is dedup'd by
  `(name, dispname)` — verified no unbounded alias-list growth on
  repeated `Unit.create` of the same unit.

## Verified-correct behaviours worth keeping under test

- `(u.mV ** 0.5) ** 2 == u.mV`, including standard-name restoration via
  the float-scale key (`-3.0` hashing equal to `-3` in the registry).
- `str(u.mV ** (1/3))` → `"mV^0.3333333333333333"` round-trips through
  `parse_unit` exactly (Python float repr round-trip).
- `u.radian == UNITLESS` is `True` by design (factor-1 dimensionless);
  the hash contract holds.
- User `Unit.create(dim, 'my_j', 'V', factor=2.0)` cannot hijack
  `parse_unit('V')` (first registration wins in `_unit_name_registry`).
