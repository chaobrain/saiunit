# `Dimension` issues (`saiunit/_base_dimension.py`)

The `Dimension` class is in good shape. The singleton cache is
thread-safe (double-checked locking), the hash/eq contract holds across
mixed exponent dtypes (`hash(1) == hash(1.0)`, `-0.0` vs `0.0` verified
to hit the same cache slot), pickling round-trips through
`get_or_create_dimension` preserving identity, and non-finite `**`
exponents are rejected before they can poison the cache.

Findings are informational:

---

## D1 (Info) — `__getstate__` / `__setstate__` are dead code

`Dimension` defines `__reduce__` (`_base_dimension.py:650`), which takes
precedence over `__getstate__`/`__setstate__` (`:621`/`:635`) for every
pickle protocol in use. The state methods are therefore never called by
pickle; they would only matter if `__reduce__` were removed. Verified:
`pickle.loads(pickle.dumps(u.mV.dim)) is u.mV.dim` (i.e. the
`__reduce__` singleton path runs).

**Proposal** — delete `__getstate__`/`__setstate__`, or add a one-line
comment that they are intentionally shadowed by `__reduce__` and kept
only as documentation of the state layout. If deleted, note that
`__setstate__`'s re-freeze (`flags.writeable = False`) is the only
behaviour lost, and it is unreachable today.

---

## D2 (Info) — cache growth under arbitrary float exponents

Every distinct exponent tuple interns a `Dimension` forever. Ordinary
usage produces a handful; code that computes data-dependent fractional
powers in a loop (`q ** (1/n)` for varying `n`) grows the cache without
bound. Non-finite exponents are already rejected (the worst failure
mode — NaN keys that never hit the cache — is closed). No action
proposed; if it ever becomes a problem the cache could become an LRU
for non-integer keys, but that would weaken the `is`-identity guarantee
and is not worth it now.

---

## D3 (Info) — `get_dim_for_display` redundant branch

```python
if (isinstance(d, int) and d == 1) or d is DIMENSIONLESS:
    return "1"
if isinstance(d, Dimension):
    return str(d)
return str(d)          # identical to the branch above
```

The last two branches are the same; harmless, collapse when touched.

---

## Verified-correct behaviours worth keeping under test

- `get_or_create_dimension` rejects non-numeric sequences (a 7-char
  string passes the length check but is caught by the dtype guard),
  rejects mixing positional + keyword forms, and gives a helpful error
  for unknown dimension names.
- `Dimension.__eq__` returns `NotImplemented` for foreign types (so
  reflected comparison runs), and `__ne__` mirrors it.
- `Dimension ** value` rejects tracers, multi-element exponents, empty
  arrays, and non-finite values with distinct, accurate messages.
- `Unit * Dimension` and `Unit / Dimension` raise `TypeError`
  explicitly rather than producing a nonsense object.
