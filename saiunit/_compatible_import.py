# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import jax

__all__ = [
    'safe_map',
]


if jax.__version_info__ < (0, 6, 0):
    from jax.util import safe_map, safe_zip, unzip2, unzip3, wraps

else:

    def safe_map(f, *args):
        args = list(map(list, args))
        n = len(args[0])
        for arg in args[1:]:
            assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
        return list(map(f, *args))

