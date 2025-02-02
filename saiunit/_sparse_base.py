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

from __future__ import annotations

from typing import Sequence

from jax.experimental.sparse import JAXSparse

__all__ = [
    "SparseMatrix"
]


class SparseMatrix(JAXSparse):

    # Not abstract methods because not all sparse classes implement them

    def with_data(self, data):
        raise NotImplementedError(f"{self.__class__}.assign_data")

    def sum(self, axis: int | Sequence[int] = None):
        if axis is not None:
            raise NotImplementedError("CSR.sum with axis is not implemented.")
        return self.data.sum()

    def __abs__(self):
        raise NotImplementedError(f"{self.__class__}.__abs__")

    def __neg__(self):
        raise NotImplementedError(f"{self.__class__}.__neg__")

    def __pos__(self):
        raise NotImplementedError(f"{self.__class__}.__pos__")

    def __matmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__matmul__")

    def __rmatmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmatmul__")

    def __mul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__mul__")

    def __rmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmul__")

    def __add__(self, other):
        raise NotImplementedError(f"{self.__class__}.__add__")

    def __radd__(self, other):
        raise NotImplementedError(f"{self.__class__}.__radd__")

    def __sub__(self, other):
        raise NotImplementedError(f"{self.__class__}.__sub__")

    def __rsub__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rsub__")

    def __div__(self, other):
        raise NotImplementedError(f"{self.__class__}.__div__")

    def __rdiv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rdiv__")

    def __truediv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__truediv__")

    def __rtruediv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rtruediv__")

    def __floordiv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__floordiv__")

    def __rfloordiv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rfloordiv__")

    def __mod__(self, other):
        raise NotImplementedError(f"{self.__class__}.__mod__")

    def __rmod__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmod__")

    def __getitem__(self, item):
        raise NotImplementedError(f"{self.__class__}.__getitem__")
