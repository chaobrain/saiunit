#!/usr/bin/env python
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
"""Render the rst feature-support matrix from ``dev/backend_support_data.json``.

Output: ``docs/backends/feature_support_matrix.rst``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dev" / "backend_support_data.json"
OUT_PATH = ROOT / "docs" / "backends" / "feature_support_matrix.rst"

ALL_BACKENDS = ["numpy", "jax", "cupy", "torch", "dask", "ndonnx"]

GLYPHS = {
    "pass": "✓",
    "fail": "✗",
    "skip": "⊘",
    "warn": "⚠",
    "unmapped": "?",
    "na": "—",
    "guard": "🅙",   # JAX-only guard fired (BackendError on non-jax)
    "cupy": "?",
}


def _cell(cell: dict[str, str]) -> str:
    return GLYPHS.get(cell.get("status", "?"), "?")


def _make_footnote(notes: dict[str, str], detail: str) -> str:
    """Register a footnote and return its rst reference."""
    if not detail:
        return ""
    if detail not in notes:
        notes[detail] = f"fn-{len(notes) + 1}"
    return f" [#{notes[detail]}]_"


def _table_header(swept: list[str]) -> list[str]:
    widths = ["40"] + ["10"] * len(ALL_BACKENDS)
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        f"   :widths: {' '.join(widths)}",
        "",
        "   * - Function",
    ]
    for b in ALL_BACKENDS:
        lines.append(f"     - {b}")
    return lines


def _row(fq_short: str, row: dict[str, dict[str, str]], notes: dict[str, str]) -> list[str]:
    """Emit rst list-table rows for one function. Includes a footnote where
    the cell carries useful detail."""
    lines = [f"   * - ``{fq_short}``"]
    for b in ALL_BACKENDS:
        cell = row.get(b)
        if cell is None:
            # cupy column (or any unswept backend) — always '?'
            lines.append(f"     - {GLYPHS['cupy']}")
            continue
        glyph = _cell(cell)
        # Footnote only for fail/skip/warn with non-empty detail.
        detail = cell.get("detail", "")
        if cell["status"] in ("fail", "skip", "warn") and detail:
            ref = _make_footnote(notes, detail)
            lines.append(f"     - {glyph}{ref}")
        else:
            lines.append(f"     - {glyph}")
    return lines


def _render_table(
    title: str,
    items: list[tuple[str, dict[str, dict[str, str]]]],
    swept: list[str],
    notes: dict[str, str],
) -> list[str]:
    """Render one ``list-table`` for a group of (function-name, row-data)."""
    if not items:
        return [f"*{title}: no functions in this group.*", ""]
    lines = [f".. list-table:: {title}",
             "   :header-rows: 1",
             f"   :widths: 40 {' '.join(['10'] * len(ALL_BACKENDS))}",
             "",
             "   * - Function"]
    for b in ALL_BACKENDS:
        lines.append(f"     - {b}")
    for name, row in items:
        lines.extend(_row(name, row, notes))
    lines.append("")
    return lines


def _escape_rst_inline(s: str) -> str:
    """Escape rst inline markup characters (``*``, ``|``) that appear inside
    backend error messages — torch errors contain ``*`` list bullets and pipe
    chars that docutils otherwise parses as emphasis or table separators."""
    return s.replace("\\", "\\\\").replace("*", "\\*").replace("|", "\\|")


def _render_footnotes(notes: dict[str, str]) -> list[str]:
    """Emit a single block of footnotes in registration order."""
    lines: list[str] = []
    # Reverse map: id -> message
    items = sorted(notes.items(), key=lambda kv: int(kv[1].split("-")[1]))
    for msg, fid in items:
        # Sanitize message: collapse newlines, truncate, escape rst inline markup.
        clean = " ".join(msg.split())
        if len(clean) > 200:
            clean = clean[:197] + "..."
        lines.append(f".. [#{fid}] {_escape_rst_inline(clean)}")
    if lines:
        lines.insert(0, "")
        lines.insert(0, "Footnotes")
        lines.insert(1, "---------")
        lines.insert(2, "")
    return lines


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _subpackage_summary(
    rows: list[tuple[str, dict[str, dict[str, str]]]],
    swept: list[str],
) -> dict[str, str]:
    """For one subpackage, compute a per-backend high-level rating."""
    out: dict[str, str] = {}
    for b in ALL_BACKENDS:
        if b == "cupy" or b not in swept:
            out[b] = "?"
            continue
        n = sum(1 for _, r in rows if r.get(b))
        if not n:
            out[b] = "?"
            continue
        passes = sum(1 for _, r in rows if r.get(b, {}).get("status") == "pass")
        fails = sum(1 for _, r in rows if r.get(b, {}).get("status") == "fail")
        skips = sum(1 for _, r in rows if r.get(b, {}).get("status") == "skip")
        rate = passes / n
        if rate >= 0.95 and fails == 0:
            out[b] = "Full ✓"
        elif rate >= 0.80:
            out[b] = "Mostly ⚠"
        elif rate >= 0.30:
            out[b] = "Partial ⚠"
        else:
            out[b] = "Limited ✗"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    swept = data["swept_backends"]
    function_results = data["function_results"]
    quantity_results = data["quantity_results"]
    jax_only_results = data["jax_only_results"]
    jax_only_inventory = data["jax_only_inventory"]
    source_map = data["source_map"]
    coverage = data["coverage"]
    non_dispatched_math = data.get("non_dispatched_math", [])

    notes: dict[str, str] = {}

    # ---- Bucket math by source module ----
    math_by_submodule: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for fq, row in function_results.items():
        if not fq.startswith("saiunit.math."):
            continue
        name = fq.rsplit(".", 1)[1]
        if name in non_dispatched_math:
            continue
        submod = source_map.get(fq, "?")
        # Normalize known module names to a friendlier label.
        label = submod.replace("saiunit.math.", "").replace("_fun_", "")
        math_by_submodule[label].append((name, row))

    for label in math_by_submodule:
        math_by_submodule[label].sort()

    linalg_rows = sorted(
        ((fq.rsplit(".", 1)[1], row) for fq, row in function_results.items()
         if fq.startswith("saiunit.linalg.")),
        key=lambda kv: kv[0],
    )
    fft_rows = sorted(
        ((fq.rsplit(".", 1)[1], row) for fq, row in function_results.items()
         if fq.startswith("saiunit.fft.")),
        key=lambda kv: kv[0],
    )
    quantity_rows = sorted(
        ((fq.rsplit(".", 1)[1], row) for fq, row in quantity_results.items()),
        key=lambda kv: kv[0],
    )

    # ---- Top-level summary table ----
    summary_rows: list[tuple[str, dict[str, str]]] = []
    summary_rows.append(("saiunit.math",
                          _subpackage_summary(
                              [r for rows in math_by_submodule.values() for r in rows],
                              swept)))
    summary_rows.append(("saiunit.linalg",
                          _subpackage_summary(linalg_rows, swept)))
    summary_rows.append(("saiunit.fft",
                          _subpackage_summary(fft_rows, swept)))
    summary_rows.append(("Quantity methods",
                          _subpackage_summary(quantity_rows, swept)))
    # JAX-only subpackages: derive from the probe results.
    for subpkg_label, label_short in (
        ("saiunit.lax (*)",      "saiunit.lax"),
        ("saiunit.autograd (*)", "saiunit.autograd"),
        ("saiunit.sparse (*)",   "saiunit.sparse"),
    ):
        probe = jax_only_results.get(subpkg_label, {})
        row: dict[str, str] = {}
        for b in ALL_BACKENDS:
            if b == "cupy" or b not in swept:
                row[b] = "?"
            elif b == "jax":
                row[b] = "Full ✓" if probe.get("jax", {}).get("status") == "pass" else "Limited ✗"
            else:
                row[b] = "JAX-only 🅙"
        summary_rows.append((label_short, row))

    # ---- Begin rst output ----
    out: list[str] = []

    out += [
        "Feature support matrix",
        "======================",
        "",
        ".. note::",
        "   This page is generated from ``dev/backend_support_data.json``,",
        "   which is produced by ``dev/backend_support_sweep.py`` —",
        "   an automated sweep that invokes every public function in",
        "   ``saiunit.math``, ``saiunit.linalg``, ``saiunit.fft``, and every",
        "   public ``Quantity`` method across each locally-installed backend",
        "   and records the outcome.  Re-run the sweep and the renderer to",
        "   refresh this page.",
        "",
        "Cell legend",
        "-----------",
        "",
        "==========  ==========================================================",
        "Glyph       Meaning",
        "==========  ==========================================================",
        "``✓``       Verified: the call returned a value of the expected backend kind.",
        "``⊘``       Skipped: the backend's array-API surface does not expose the underlying op,",
        "            or it rejects a keyword saiunit forwards (e.g. JAX-only ``precision=``).",
        "``✗``       Failed: the call raised an unexpected exception on this backend.",
        "``⚠``       Works with a caveat (e.g. lazy result on dask, expected ``BackendError`` for",
        "            materialization on dask).",
        "``🅙``       JAX-only by design — gated by ``saiunit._jax_guard.require_jax_backend``.",
        "            Raises :class:`~saiunit.BackendError` on any non-jax backend.",
        "``—``       Not applicable to backend dispatch (dtype factories, dimension predicates).",
        "``?``       Not tested in this report. Cupy is always ``?`` because no CUDA backend",
        "            was available when this sweep ran. The single unmapped Quantity method",
        "            (``tree_unflatten``) is also ``?`` because automated invocation requires",
        "            a hand-crafted aux/children pair.",
        "==========  ==========================================================",
        "",
        f"**Sweep environment.**  Backends invoked: ``{'``, ``'.join(swept)}``.  ",
        f"Backends shown but not tested: ``{'``, ``'.join(data['untested_backends'])}``.",
        "",
    ]

    # ---- Summary table ----
    out += [
        "High-level summary",
        "------------------",
        "",
        ".. list-table:: Per-subpackage rating",
        "   :header-rows: 1",
        f"   :widths: 30 {' '.join(['12'] * len(ALL_BACKENDS))}",
        "",
        "   * - Subpackage",
    ]
    for b in ALL_BACKENDS:
        out.append(f"     - {b}")
    for name, row in summary_rows:
        out.append(f"   * - **{name}**")
        for b in ALL_BACKENDS:
            out.append(f"     - {row.get(b, '?')}")
    out += [
        "",
        "Rating thresholds: **Full** ≥ 95 % pass and zero fail; **Mostly** ≥ 80 % pass;",
        "**Partial** ≥ 30 % pass; **Limited** < 30 % pass; **JAX-only** = gated by",
        "``require_jax_backend``.",
        "",
    ]

    # ---- Per-backend notes ----
    out += [
        "Backend-specific notes",
        "----------------------",
        "",
        "- **jax** — full feature set; default backend.  All JAX-only subpackages",
        "  (``saiunit.lax``, ``saiunit.autograd``, ``saiunit.sparse``) require this",
        "  backend.",
        "- **numpy** — eager CPU computation through ``array_api_compat.numpy``.",
        "  A handful of reductions (``amax``, ``amin``, ``mean``, ``nan*`` variants)",
        "  fail when saiunit forwards a ``where=None`` kwarg numpy can't interpret.",
        "  These are listed with footnotes in the math tables below.",
        "- **cupy** — *not tested in this report* (no CUDA toolkit in the sweep",
        "  environment).  Cells are ``?``.  Cupy's array-API surface tracks numpy",
        "  closely, so support is expected to mirror the numpy column, but this",
        "  document does not claim it.",
        "- **torch** — through ``array_api_compat.torch``.  The torch array-API",
        "  surface lacks several ops saiunit dispatches to (``cbrt``, ``digamma``,",
        "  some ``einops`` reductions, ``axes=`` for n-D FFTs) and rejects",
        "  JAX-flavored kwargs (``precision``, ``symmetrize_input``, ``tol``).",
        "  Affected calls are recorded as skip rather than fail.",
        "- **dask** — lazy arrays.  Reductions and most array ops succeed but the",
        "  result remains lazy until ``.compute()``.  Per ``saiunit._base_quantity``,",
        "  the Python casts ``float(q)`` / ``int(q)`` / ``operator.index(q)`` /",
        "  ``np.asarray(q)`` / ``hash(q)`` and the ``Quantity.tolist`` method raise",
        "  :class:`~saiunit.BackendError` to avoid silent materialization — these",
        "  cells are ``⚠`` with the BackendError text in the footnote.",
        "  ``Quantity.item`` on dask raises a different error (the dask Array has",
        "  no ``.item()`` method) so it appears as ``⊘`` rather than ``⚠``.",
        "  Methods like ``Quantity.float`` / ``.double`` are ``.astype`` in disguise",
        "  and stay lazy on dask, so they pass.",
        "- **ndonnx** — symbolic graph-building backend.  Many array-API ops",
        "  (``fft.*``, several ``linalg.*``, complex / specialty math) are not",
        "  implemented and surface as ``⊘`` skip rows.  Saiunit does not encode",
        "  unit information into the ONNX graph.",
        "",
    ]

    # ---- JAX-only subpackages ----
    out += [
        "JAX-only subpackages",
        "--------------------",
        "",
        "These subpackages dispatch directly to JAX primitives that have no",
        "array-API equivalent.  Each entry point is wrapped with",
        "``saiunit._jax_guard.require_jax_backend``, which raises",
        ":class:`~saiunit.BackendError` on any non-jax mantissa.",
        "",
    ]
    for label, inv_key in (
        ("saiunit.lax", "saiunit.lax"),
        ("saiunit.autograd", "saiunit.autograd"),
        ("saiunit.sparse", "saiunit.sparse"),
    ):
        funcs = jax_only_inventory.get(inv_key, [])
        probe = jax_only_results.get(f"{label} (*)", {})
        out += [
            f"**{label}** — {len(funcs)} public callable(s); all require ``jax``.",
            "",
        ]
        # Per-backend status from probe.
        out += [
            ".. list-table::",
            "   :header-rows: 1",
            f"   :widths: 16 {' '.join(['12'] * len(ALL_BACKENDS))}",
            "",
            "   * - Probe result",
        ]
        for b in ALL_BACKENDS:
            out.append(f"     - {b}")
        out.append("   * - all functions")
        for b in ALL_BACKENDS:
            if b == "cupy" or b not in swept:
                out.append(f"     - {GLYPHS['cupy']}")
                continue
            st = probe.get(b, {}).get("status", "?")
            if b == "jax":
                out.append("     - ✓" if st == "pass" else f"     - {GLYPHS.get(st, '?')}")
            else:
                # Non-jax should raise BackendError -> status 'guard'.
                out.append("     - 🅙" if st == "guard" else f"     - {GLYPHS.get(st, '?')}")
        out.append("")
        # Folded function list.
        out += [
            ".. dropdown:: List of " + label + " functions",
            "",
            "   .. hlist::",
            "      :columns: 4",
            "",
        ]
        for fn_name in funcs:
            out.append(f"      * ``{fn_name}``")
        out.append("")

    # ---- saiunit.math by submodule ----
    out += [
        "saiunit.math",
        "------------",
        "",
        "Public callables in ``saiunit.math`` that go through the multi-backend",
        "dispatcher.  Grouped by source submodule for readability.",
        "",
    ]
    submodule_order = [
        ("array_creation",  "Array creation"),
        ("keep_unit",       "Unit-preserving"),
        ("change_unit",     "Unit-changing"),
        ("accept_unitless", "Dimensionless-only"),
        ("remove_unit",     "Unit-removing"),
    ]
    seen_labels: set[str] = set()
    for key, title in submodule_order:
        rows = math_by_submodule.get(key, [])
        seen_labels.add(key)
        header = f"``{key}`` — {title}"
        out += [header, "^" * len(header), ""]
        out += _render_table(f"saiunit.math — {title}", rows, swept, notes)

    # Any remaining math buckets we didn't pre-label.
    for key, rows in math_by_submodule.items():
        if key in seen_labels:
            continue
        header = f"``{key}``"
        out += [header, "^" * len(header), ""]
        out += _render_table(f"saiunit.math — {key}", rows, swept, notes)

    # Non-dispatched math (dtype factories, predicates).
    out += [
        "Non-dispatched helpers",
        "^^^^^^^^^^^^^^^^^^^^^^",
        "",
        "These names live under ``saiunit.math`` for convenience but do not",
        "dispatch on backend — they are dtype factories (re-exported from",
        "``jax.numpy``) or pure-Python predicates / introspection helpers over",
        "``Quantity`` / ``Unit`` objects.  Behavior is identical on every",
        "backend.",
        "",
    ]
    # Render 6 per row in a hlist.
    out += [
        ".. hlist::",
        "   :columns: 4",
        "",
    ]
    for name in sorted(non_dispatched_math):
        out.append(f"   * ``{name}``")
    out.append("")

    # ---- saiunit.linalg ----
    out += [
        "saiunit.linalg",
        "--------------",
        "",
    ]
    out += _render_table("saiunit.linalg", linalg_rows, swept, notes)

    # ---- saiunit.fft ----
    out += [
        "saiunit.fft",
        "-----------",
        "",
        "Routing varies inside ``saiunit.fft``: ``_fft_change_unit.py`` calls",
        "``saiunit._backend.get_backend()`` directly (e.g. for ``fftfreq`` /",
        "``rfftfreq``), while ``_fft_keep_unit.py`` delegates to the math",
        "package's ``_fun_keep_unit_unary`` helper and inherits its dispatch.",
        "",
    ]
    out += _render_table("saiunit.fft", fft_rows, swept, notes)

    # ---- Quantity methods ----
    out += [
        "Quantity methods",
        "----------------",
        "",
        "Methods on ``saiunit.Quantity`` itself.  ``.to_<backend>()`` methods",
        "ignore the *current* backend and convert to the named one — cells show",
        "``⊘`` when the target backend isn't installed in the sweep environment.",
        "",
        "Materialization is documented above (see *Backend-specific notes*).",
        "``Quantity.tolist`` on dask is the one method that raises",
        ":class:`~saiunit.BackendError` from saiunit's own guard (``⚠``).",
        "``.item`` reports ``⊘`` on dask / ndonnx because the underlying array",
        "object does not expose ``.item()``.",
        "",
    ]
    out += _render_table("Quantity public methods", quantity_rows, swept, notes)

    # ---- Coverage stat ----
    out += [
        "Coverage statistic",
        "------------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 15 15 15 15",
        "",
        "   * - Subpackage",
        "     - Mapped",
        "     - Non-dispatched",
        "     - Unmapped",
        "     - Total",
    ]
    for label, key in [("saiunit.math", "math"),
                        ("saiunit.linalg", "linalg"),
                        ("saiunit.fft", "fft"),
                        ("Quantity", "quantity")]:
        c = coverage[key]
        mapped = c["mapped"]
        na = c.get("na", 0)
        total = c["total"]
        unmapped = total - mapped - na
        out.append(f"   * - {label}")
        out.append(f"     - {mapped}")
        out.append(f"     - {na}")
        out.append(f"     - {unmapped}")
        out.append(f"     - {total}")
    out.append("")
    out.append("*Mapped* = functions the sweep actually invoked.  ")
    out.append("*Non-dispatched* = type factories / predicates that don't go ")
    out.append("through backend dispatch.  ")
    out.append("*Unmapped* = no call pattern registered (will appear as ``?`` in tables).")
    out.append("")

    # ---- How this was produced ----
    out += [
        "How this was produced",
        "---------------------",
        "",
        "``dev/backend_support_sweep.py`` walks every public callable in the",
        "subpackages above, picks a calling pattern from an in-script registry,",
        "and invokes the function under ``with saiunit.using_backend(b)`` for",
        "each backend ``b`` in the local environment.  Outcomes are classified",
        "as ``pass`` / ``skip`` / ``fail`` / ``warn`` / ``unmapped`` / ``na`` and",
        "written to ``dev/backend_support_data.json``.",
        "",
        "``dev/backend_support_render.py`` (this script's source) reads that",
        "JSON and emits the rst file you are currently reading.  To refresh:",
        "",
        ".. code-block:: bash",
        "",
        "   PYTHONPATH=. python dev/backend_support_sweep.py",
        "   PYTHONPATH=. python dev/backend_support_render.py",
        "",
        "JAX-only subpackages are probed with one representative function per",
        "subpackage rather than enumerated — every entry point in",
        "``saiunit.lax`` / ``.autograd`` / ``.sparse`` is gated identically.",
        "",
    ]

    # ---- Footnotes ----
    out += _render_footnotes(notes)

    OUT_PATH.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT_PATH} ({len(out)} lines, {len(notes)} footnotes)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
