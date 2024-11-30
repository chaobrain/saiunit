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

__version__ = "0.0.3"

from . import _matplotlib_compat
from . import lax
from . import math
from . import linalg
from . import autograd
from ._base import *
from ._base import __all__ as _base_all
from ._celsius import *
from ._celsius import __all__ as _celsius_all
from ._unit_common import *
from ._unit_common import __all__ as _common_all
from ._unit_constants import *
from ._unit_constants import __all__ as _constants_all
from ._unit_shortcuts import *
from ._unit_shortcuts import __all__ as _std_units_all

__all__ = ['math', 'linalg', 'autograd'] + _common_all + _std_units_all + _constants_all + _base_all + _celsius_all
del _common_all, _std_units_all, _constants_all, _base_all, _celsius_all, _matplotlib_compat
