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


import jax

CustomArray = None

__all__ = [
    'maybe_custom_array',
    'maybe_custom_array_tree',
]


def set_module_as(module: str):
    """
    A decorator that changes the __module__ attribute of a function.

    This utility decorator is useful for making functions appear as if they belong
    to a different module than where they are defined, which can help organize
    the public API of a package.

    Parameters
    ----------
    module : str
        The module name to set as the function's __module__ attribute.

    Returns
    -------
    callable
        A decorator function that modifies the __module__ attribute of the
        decorated function.

    Examples
    --------
    >>> @set_module_as('saiunit.public')
    ... def my_function():
    ...     pass
    ...
    >>> my_function.__module__
    'saiunit.public'
    """

    def wrapper(fun: callable):
        fun.__module__ = module
        return fun

    return wrapper


def maybe_custom_array(x):
    """
    Unwrap a :class:`~saiunit.CustomArray` to its underlying data.

    If ``x`` is a :class:`~saiunit.CustomArray` instance, return its
    ``.data`` attribute. Otherwise return ``x`` unchanged. The
    :class:`~saiunit.CustomArray` class is lazily imported to avoid
    circular dependencies.

    Parameters
    ----------
    x : Any
        The input value, which may be a :class:`~saiunit.CustomArray`.

    Returns
    -------
    Any
        ``x.data`` if ``x`` is a :class:`~saiunit.CustomArray`,
        otherwise ``x`` unchanged.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su._misc.maybe_custom_array(5)
        5
        >>> import numpy as np
        >>> class MyArray(su.CustomArray):
        ...     def __init__(self, value):
        ...         self.data = value
        >>> arr = MyArray(np.array([1, 2, 3]))
        >>> su._misc.maybe_custom_array(arr)
        array([1, 2, 3])

    """
    global CustomArray
    if CustomArray is None:
        from saiunit.custom_array import CustomArray
    if isinstance(x, CustomArray):
        return x.data
    else:
        return x


def maybe_custom_array_tree(x):
    """
    Recursively unwrap :class:`~saiunit.CustomArray` instances in a pytree.

    Traverses a JAX-compatible pytree and replaces every
    :class:`~saiunit.CustomArray` leaf with its ``.data`` attribute using
    :func:`jax.tree.map`. Non-CustomArray leaves are left unchanged.

    Parameters
    ----------
    x : Any
        A pytree (nested lists, tuples, dicts, etc.) that may contain
        :class:`~saiunit.CustomArray` instances as leaves.

    Returns
    -------
    Any
        A new pytree of the same structure with every
        :class:`~saiunit.CustomArray` replaced by its underlying data.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import numpy as np
        >>> class MyArray(su.CustomArray):
        ...     def __init__(self, value):
        ...         self.data = value
        >>> tree = [MyArray(np.array([1, 2])), 3, {'k': MyArray(np.array([4]))}]
        >>> result = su._misc.maybe_custom_array_tree(tree)
        >>> result[0]
        array([1, 2])
        >>> result[1]
        3
        >>> result[2]['k']
        array([4])

    """
    global CustomArray
    if CustomArray is None:
        from saiunit.custom_array import CustomArray
    return jax.tree.map(maybe_custom_array, x, is_leaf=lambda a: isinstance(a, CustomArray))
