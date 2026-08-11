import json
import re
from pathlib import Path

import pytest


BACKEND_NOTEBOOKS = (
    "cupy.ipynb",
    "dask.ipynb",
    "jax.ipynb",
    "ndonnx.ipynb",
    "numpy.ipynb",
    "overview.ipynb",
    "torch.ipynb",
)
BACKEND_DIR = Path(__file__).parent


def _load_notebook(filename):
    return json.loads((BACKEND_DIR / filename).read_text())


@pytest.mark.parametrize("filename", BACKEND_NOTEBOOKS)
def test_backend_notebooks_save_output_for_every_code_cell(filename):
    notebook = _load_notebook(filename)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell.get("outputs") for cell in code_cells)


@pytest.mark.parametrize("filename", BACKEND_NOTEBOOKS)
def test_backend_notebooks_have_clean_sequential_execution(filename):
    notebook = _load_notebook(filename)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert [cell.get("execution_count") for cell in code_cells] == list(
        range(1, len(code_cells) + 1)
    )
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", ())
    )
    assert not any(
        output.get("output_type") == "stream" and output.get("name") == "stderr"
        for cell in code_cells
        for output in cell.get("outputs", ())
    )


@pytest.mark.parametrize("filename", BACKEND_NOTEBOOKS)
def test_backend_notebooks_do_not_capture_machine_specific_output(filename):
    notebook = _load_notebook(filename)
    serialized = json.dumps(notebook)

    assert "/home/acer" not in serialized
    assert not re.search(r"20\\d{2}-\\d{2}-\\d{2}[ T]\\d{2}:\\d{2}", serialized)


@pytest.mark.parametrize("filename", BACKEND_NOTEBOOKS)
def test_backend_notebooks_cover_creation_operations_and_conversion(filename):
    notebook = _load_notebook(filename)
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "u.Quantity(" in source
    assert any(token in source for token in (" + ", "u.math.", "autograd"))
    assert ".to_" in source
