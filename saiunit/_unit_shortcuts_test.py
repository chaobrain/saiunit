# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import saiunit as u
from saiunit import _unit_common as uc
from saiunit import _unit_shortcuts as sh

# Each shortcut must be the exact base unit object it aliases.
SHORTCUT_TO_BASE = {
    "mV": "mvolt",
    "mA": "mamp", "uA": "uamp", "nA": "namp", "pA": "pamp",
    "pF": "pfarad", "uF": "ufarad", "nF": "nfarad",
    "nS": "nsiemens", "uS": "usiemens", "mS": "msiemens",
    "ms": "msecond", "us": "usecond",
    "Hz": "hertz", "kHz": "khertz", "MHz": "Mhertz",
    "cm": "cmetre", "cm2": "cmetre2", "cm3": "cmetre3",
    "mm": "mmetre", "mm2": "mmetre2", "mm3": "mmetre3",
    "um": "umetre", "um2": "umetre2", "um3": "umetre3",
    "mM": "mmolar", "uM": "umolar", "nM": "nmolar",
}


def test_shortcut_map_covers_all_exports():
    # Guard against the __all__ and the map drifting apart.
    assert set(sh.__all__) == set(SHORTCUT_TO_BASE)


def test_each_shortcut_is_its_base_unit():
    for short, base in SHORTCUT_TO_BASE.items():
        assert getattr(sh, short) is getattr(uc, base), f"{short} should alias {base}"


def test_shortcuts_exposed_on_top_level_package():
    for short in SHORTCUT_TO_BASE:
        assert getattr(u, short) is getattr(sh, short), f"u.{short} missing or mismatched"
