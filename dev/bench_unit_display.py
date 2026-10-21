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

"""Micro-benchmarks for unit composition and display.

Usage: python dev/bench_unit_display.py
"""

import subprocess
import sys
import timeit

SETUP = "import saiunit as u; mV, ms, mS, nA, cm2 = u.mV, u.ms, u.mS, u.nA, u.cm2; q = 3 * u.mV"
CASES = {
    'mul (mV * ms)': "mV * ms",
    'div (mV / ms)': "mV / ms",
    'pow (mV ** 2)': "mV ** 2",
    'reverse (mV)': "mV.reverse()",
    'str(mS * nA / cm2)': "str(mS * nA / cm2)",
    'str(3 * mV)': "str(q)",
}


def main():
    for label, stmt in CASES.items():
        timer = timeit.Timer(stmt, setup=SETUP)
        n, _ = timer.autorange()
        best = min(timer.repeat(repeat=5, number=n)) / n
        print(f"{label:22s} {best * 1e6:8.2f} us")
    code = "import time; t = time.perf_counter(); import saiunit; print(time.perf_counter() - t)"
    times = [float(subprocess.check_output([sys.executable, '-c', code]).decode().split()[-1]) for _ in range(5)]
    print(f"{"import saiunit":22s} {min(times) * 1e3:8.2f} ms")


if __name__ == '__main__':
    main()
