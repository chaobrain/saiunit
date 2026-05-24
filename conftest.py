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

"""Shared pytest fixtures for saiunit tests.

The :func:`backend` fixture is auto-parameterized to run each consuming test
twice — once with the NumPy default backend and once with the JAX default.
Tests that want backend coverage should add ``backend`` to their signature.

When the environment variable ``SAIUNIT_DEFAULT_BACKEND`` is set (to one of
``numpy``, ``jax``, ``cupy``, ``torch``, ``dask``, ``ndonnx``), the session
default backend is set to that value before any test runs. This is what the
per-backend CI jobs use to run the whole suite under a single backend without
modifying individual tests.
"""

import os
from pathlib import Path

import pytest

import saiunit as u
from saiunit._jax_compat import HAS_JAX


def _scan_jax_only_test_files():
    """Return absolute paths of test files that import JAX at module level.

    Used to populate ``collect_ignore`` when JAX is not installed — those
    modules would otherwise crash at import time during collection. We treat
    any line beginning with ``import jax`` or ``from jax`` as a top-level
    JAX dependency; this matches every current usage in the suite.
    """
    root = Path(__file__).parent / "saiunit"
    ignored = []
    for path in root.rglob("*_test.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            # Only count truly top-level imports — an indented ``import jax``
            # inside an ``if HAS_JAX:`` block or a function body must not flag
            # the whole file as JAX-dependent.
            if line.startswith("import jax") or line.startswith("from jax"):
                ignored.append(str(path))
                break
    return ignored


# When JAX isn't installed (e.g., the pure-numpy / pure-torch / pure-dask /
# pure-ndonnx CI jobs), skip collection of test files that import JAX at the
# top. ``_no_jax_test.py`` and other JAX-free test files still run.
if not HAS_JAX:
    collect_ignore = _scan_jax_only_test_files()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_jax: mark test as requiring an installed JAX (skipped otherwise)",
    )
    env_backend = os.environ.get("SAIUNIT_DEFAULT_BACKEND")
    if env_backend:
        u.set_default_backend(env_backend)


def pytest_collection_modifyitems(config, items):
    skip_jax = pytest.mark.skip(reason="JAX not installed")
    for item in items:
        if "requires_jax" in item.keywords and not HAS_JAX:
            item.add_marker(skip_jax)


@pytest.fixture(params=["numpy", "jax", "cupy", "torch", "dask", "ndonnx"])
def backend(request):
    """Set the saiunit default backend for the duration of the test.

    Skips automatically when the requested backend's library isn't installed.
    """
    if request.param == "jax":
        if not HAS_JAX:
            pytest.skip("JAX not installed")
    elif request.param == "cupy":
        pytest.importorskip("cupy")
    elif request.param == "torch":
        pytest.importorskip("torch")
    elif request.param == "dask":
        pytest.importorskip("dask.array")
    elif request.param == "ndonnx":
        pytest.importorskip("ndonnx")
    with u.using_backend(request.param):
        yield request.param
