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

import numbers
from abc import ABC
from typing import Sequence, Union

import jax
import numpy as np
from jax.experimental.sparse import JAXSparse

__all__ = [
    "SparseMatrix"
]


class SparseMatrix(JAXSparse, ABC):
    """
    Base class for sparse matrices in ``saiunit``.

    This class is a subclass of ``jax.experimental.sparse.JAXSparse`` and adds some methods that are not implemented
    in the original class.
    """

    def with_data(
        self,
        data: Union[jax.Array, np.ndarray, numbers.Number, 'Quantity']
    ):
        """
        Create a new sparse matrix with the same structure but different data.

        Args:
            data: The new data.

        Returns:

        """
        raise NotImplementedError(f"{self.__class__}.assign_data")

    def sum(self, axis: Union[int, Sequence[int]] = None):
        """
        Sum of the elements of the sparse matrix.

        Args:
            axis: Axis or axes along which the sum is computed. The default is to compute the sum of the flattened array.
                Only None is supported.

        Returns:
            The sum of the elements of the sparse matrix.

        """
        if axis is not None:
            raise NotImplementedError("CSR.sum with axis is not implemented.")
        return self.data.sum()

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, 'Quantity'],
        w_dim_arr: Union[jax.Array, np.ndarray, 'Quantity']
    ) -> Union[jax.Array, 'Quantity']:
        """
        The protocol method to convert the product of the sparse matrix and a vector to the sparse matrix data.

        This protocol method is primarily used in `brainscale <https://github.com/chaobrain/brainscale>`_.

        Args:
            y_dim_arr: The first vector.
            w_dim_arr: The second vector.

        Returns:
            The outer product of the two vectors.

        """
        raise NotImplementedError(f"{self.__class__}.yw_to_y is not implemented.")

    def __abs__(self):
        raise NotImplementedError(f"{self.__class__}.__abs__ is not implemented.")

    def __neg__(self):
        raise NotImplementedError(f"{self.__class__}.__neg__ is not implemented.")

    def __pos__(self):
        raise NotImplementedError(f"{self.__class__}.__pos__ is not implemented.")

    def __matmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__matmul__ is not implemented.")

    def __rmatmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmatmul__ is not implemented.")

    def __mul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__mul__ is not implemented.")

    def __rmul__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmul__ is not implemented.")

    def __add__(self, other):
        raise NotImplementedError(f"{self.__class__}.__add__ is not implemented.")

    def __radd__(self, other):
        raise NotImplementedError(f"{self.__class__}.__radd__ is not implemented.")

    def __sub__(self, other):
        raise NotImplementedError(f"{self.__class__}.__sub__ is not implemented.")

    def __rsub__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rsub__ is not implemented.")

    def __div__(self, other):
        raise NotImplementedError(f"{self.__class__}.__div__ is not implemented.")

    def __rdiv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rdiv__ is not implemented.")

    def __truediv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__truediv__ is not implemented.")

    def __rtruediv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rtruediv__ is not implemented.")

    def __floordiv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__floordiv__ is not implemented.")

    def __rfloordiv__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rfloordiv__ is not implemented.")

    def __mod__(self, other):
        raise NotImplementedError(f"{self.__class__}.__mod__ is not implemented.")

    def __rmod__(self, other):
        raise NotImplementedError(f"{self.__class__}.__rmod__ is not implemented.")

    def __getitem__(self, item):
        raise NotImplementedError(f"{self.__class__}.__getitem__ is not implemented.")
