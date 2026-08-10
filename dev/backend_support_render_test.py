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

from dev.backend_support_render import _subpackage_summary


def test_subpackage_summary_uses_measured_cupy_results():
    rows = [("square", {"cupy": {"status": "pass", "detail": ""}})]

    summary = _subpackage_summary(rows, ["cupy"])

    assert summary["cupy"].startswith("Full ")
    assert summary["cupy"] != "?"


def test_subpackage_summary_marks_unswept_backend_unknown():
    rows = [("square", {"numpy": {"status": "pass", "detail": ""}})]

    summary = _subpackage_summary(rows, ["numpy"])

    assert summary["cupy"] == "?"
