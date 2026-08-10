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

import numpy as np

from dev.backend_support_sweep import _classify_outcome


def test_classify_outcome_rejects_wrong_result_backend():
    outcome = _classify_outcome(
        "saiunit.math.square",
        "cupy",
        lambda fn, backend: np.asarray([1.0]),
        object(),
    )

    assert outcome["status"] == "fail"
    assert "expected cupy" in outcome["detail"]
    assert "numpy" in outcome["detail"]


def test_classify_outcome_checks_arrays_nested_in_tuples():
    outcome = _classify_outcome(
        "saiunit.linalg.slogdet",
        "torch",
        lambda fn, backend: (1.0, np.asarray(2.0)),
        object(),
    )

    assert outcome["status"] == "fail"
    assert "expected torch" in outcome["detail"]


def test_classify_outcome_accepts_explicit_numpy_target():
    outcome = _classify_outcome(
        "Quantity.to_numpy",
        "cupy",
        lambda fn, backend: np.asarray([1.0]),
        object(),
    )

    assert outcome == {"status": "pass", "detail": ""}
