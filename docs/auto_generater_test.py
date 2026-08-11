from collections import Counter
import inspect

import saiunit.math as math

from docs import auto_generater


EXPECTED_SECTIONS = (
    "Array Creation and Conversion",
    "Unit-preserving Operations",
    "Unit-changing Operations",
    "Dimensionless-input Operations",
    "Angle and Phase Operations",
    "Unit-removing Operations",
    "Activation Functions",
    "Einstein Operations",
    "Dtypes, Constants, and Utilities",
)


def _math_sections():
    return tuple(getattr(auto_generater, "MATH_API_SECTIONS", ()))


def test_math_api_has_the_approved_taxonomy_in_order():
    assert tuple(title for title, _ in _math_sections()) == EXPECTED_SECTIONS


def test_math_api_documents_every_public_export_exactly_once():
    documented = [name for _, names in _math_sections() for name in names]
    counts = Counter(documented)
    duplicates = sorted(name for name, count in counts.items() if count != 1)
    public = set(math.__all__) - {"fft", "linalg"}

    assert not duplicates
    assert set(documented) == public


def test_math_api_reallocates_functions_by_actual_unit_semantics():
    section_by_name = {
        name: title for title, names in _math_sections() for name in names
    }

    assert section_by_name["correlate"] == "Unit-changing Operations"
    assert section_by_name["cov"] == "Unit-changing Operations"
    assert section_by_name["trapezoid"] == "Unit-changing Operations"
    assert section_by_name["ldexp"] == "Unit-preserving Operations"
    assert section_by_name["angle"] == "Angle and Phase Operations"
    assert section_by_name["deg2rad"] == "Angle and Phase Operations"
    assert section_by_name["radians"] == "Angle and Phase Operations"


def test_generated_math_page_uses_function_templates_without_module_duplication(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    auto_generater.main("saiunit")
    page = (tmp_path / "apis" / "saiunit.math.rst").read_text()

    assert page.startswith("Mathematical Functions\n======================")
    assert ".. automodule:: saiunit.math" not in page

    functions = {
        name
        for _, names in _math_sections()
        for name in names
        if inspect.isfunction(getattr(math, name))
    }
    for block in page.split(".. autosummary::")[1:]:
        entries = {
            line.strip()
            for line in block.splitlines()
            if line.startswith("   ") and not line.strip().startswith(":")
        }
        if entries & functions:
            assert ":template: classtemplate.rst" not in block
    assert ":template: classtemplate.rst" in page
    positions = [page.index(title) for title in EXPECTED_SECTIONS]
    assert positions == sorted(positions)
