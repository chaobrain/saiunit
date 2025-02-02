# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import pytest

import saiunit as u

try:
    import matplotlib.pyplot as plt
    from matplotlib.units import ConversionError
except ImportError:
    pytest.skip("matplotlib is not installed", allow_module_level=True)


def test_matplotlib_compat():
    plt.figure()
    plt.plot([1, 2, 3])
    plt.show()

    plt.cla()
    plt.plot([1, 2, 3] * u.meter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.cmeter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150])
    plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160])
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160] * u.cmeter)
    plt.show()

    with pytest.raises(ConversionError):
        plt.cla()
        plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
        plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160] * u.cmeter)
        plt.plot([0.1, 0.15, 0.2] * u.second, [131, 155, 180] * u.mA)
        plt.show()

    # with pytest.raises(ConversionError):
    plt.cla()
    plt.plot([101, 125, 150], [101, 125, 151] * u.cmeter)
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150])
    plt.show()
