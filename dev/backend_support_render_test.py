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

import json
import re

import pytest

from docs.auto_generater import MATH_API_SECTIONS
from dev import backend_support_render
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


def test_group_math_results_rejects_unclassified_dispatched_function():
    function_results = {
        "saiunit.math.not_in_manifest": {
            "numpy": {"status": "pass", "detail": ""}
        }
    }

    with pytest.raises(ValueError, match="not_in_manifest"):
        backend_support_render._group_math_results(function_results, [])


def test_feature_matrix_uses_the_math_api_taxonomy(tmp_path, monkeypatch):
    output = tmp_path / "feature_support_matrix.rst"
    monkeypatch.setattr(backend_support_render, "OUT_PATH", output)

    backend_support_render.main()

    page = output.read_text()
    math_section = page.split("\nsaiunit.math\n------------\n", 1)[1].split(
        "\nNon-dispatched helpers\n", 1
    )[0]
    titles = [title for title, _ in MATH_API_SECTIONS]
    for title in titles:
        assert title in math_section

    for old_heading in (
        "``array_creation``",
        "``keep_unit``",
        "``change_unit``",
        "``accept_unitless``",
        "``remove_unit``",
    ):
        assert old_heading not in math_section
    assert not re.search(
        r"^``saiunit\.math``$", math_section, re.MULTILINE
    )
    assert "no functions in this group" not in math_section

    data = json.loads(backend_support_render.DATA_PATH.read_text())
    non_dispatched = set(data["non_dispatched_math"])
    expected_names = {
        fq.rsplit(".", 1)[1]
        for fq in data["function_results"]
        if fq.startswith("saiunit.math.")
        and fq.rsplit(".", 1)[1] not in non_dispatched
    }
    rendered_names = re.findall(r"^   \* - ``([^`]+)``$", math_section, re.MULTILINE)
    assert len(rendered_names) == len(set(rendered_names))
    assert set(rendered_names) == expected_names

    section_by_name = {
        name: title for title, names in MATH_API_SECTIONS for name in names
    }
    selected = ("correlate", "cov", "trapezoid", "ldexp", "angle", "exp", "sign")
    starts = [math_section.index(title) for title in titles]
    for name in selected:
        index = titles.index(section_by_name[name])
        end = starts[index + 1] if index + 1 < len(starts) else len(math_section)
        section = math_section[starts[index]:end]
        assert f"   * - ``{name}``" in section
