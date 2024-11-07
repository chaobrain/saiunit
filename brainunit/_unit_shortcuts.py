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

"""
Optional short unit names

This module defines the following short unit names:

mV, mA, uA (micro_amp), nA, pA, mF, uF, nF, nS, mS, uS, ms,
Hz, kHz, MHz, cm, cm2, cm3, mm, mm2, mm3, um, um2, um3
"""

from ._unit_common import (
    mvolt,
    mamp,
    uamp,
    namp,
    pamp,
    pfarad,
    ufarad,
    nfarad,
    nsiemens,
    usiemens,
    msiemens,
    msecond,
    usecond,
    hertz,
    khertz,
    Mhertz,
    cmetre,
    cmetre2,
    cmetre3,
    mmetre,
    mmetre2,
    mmetre3,
    umetre,
    umetre2,
    umetre3,
    mmolar,
    umolar,
    nmolar,
)

__all__ = [
    "mV",
    "mA",
    "uA",
    "nA",
    "pA",
    "pF",
    "uF",
    "nF",
    "nS",
    "uS",
    "mS",
    "ms",
    "us",
    "Hz",
    "kHz",
    "MHz",
    "cm",
    "cm2",
    "cm3",
    "mm",
    "mm2",
    "mm3",
    "um",
    "um2",
    "um3",
    "mM",
    "uM",
    "nM",
]

mV = mvolt

mA = mamp
uA = uamp
nA = namp
pA = pamp

pF = pfarad
uF = ufarad
nF = nfarad

nS = nsiemens
uS = usiemens
mS = msiemens

ms = msecond
us = usecond

Hz = hertz
kHz = khertz
MHz = Mhertz

cm = cmetre
cm2 = cmetre2
cm3 = cmetre3
mm = mmetre
mm2 = mmetre2
mm3 = mmetre3
um = umetre
um2 = umetre2
um3 = umetre3

mM = mmolar
uM = umolar
nM = nmolar
