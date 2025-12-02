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


import jax

CustomArray = None


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
    Convert a CustomArray to its underlying value if needed.

    This function checks if the input is an instance of CustomArray and extracts
    its value attribute if so. If the input is not a CustomArray, it returns the
    input unchanged. CustomArray is lazily imported to avoid circular dependencies.

    Parameters
    ----------
    x : Any
        The input value which may be a CustomArray instance.

    Returns
    -------
    ArrayLikje
        The underlying value if x is a CustomArray, otherwise x unchanged.

    Examples
    --------
    >>> from saiunit.custom_array import CustomArray
    >>> regular_value = 5
    >>> custom_arr = CustomArray(10)  # Assuming CustomArray wraps values
    >>> maybe_custom_array(regular_value)
    5
    >>> maybe_custom_array(custom_arr)
    10
    """
    global CustomArray
    if CustomArray is None:
        from saiunit.custom_array import CustomArray
    if isinstance(x, CustomArray):
        return x.data
    else:
        return x


# Note: Fixed the typo in the function name from 'maybse_custom_array_tree' to 'maybe_custom_array_tree'
def maybe_custom_array_tree(x):
    """
    Apply maybe_custom_array recursively to all elements in a nested structure.

    This function traverses a potentially nested data structure (tree) and applies
    maybe_custom_array to each element. CustomArray instances are treated as leaves
    during the traversal. CustomArray is lazily imported to avoid circular dependencies.

    Parameters
    ----------
    x : Any
        The input structure which may contain CustomArray instances.

    Returns
    -------
    Any
        A new structure with the same shape as x, where each CustomArray has been
        replaced with its underlying value.

    Examples
    --------
    >>> from saiunit.custom_array import CustomArray
    >>> import jax.numpy as jnp
    >>> # Create a nested structure with CustomArray instances
    >>> data = {
    ...     'a': 1,
    ...     'b': CustomArray(2),
    ...     'c': [3, CustomArray(4), jnp.array([5, CustomArray(6)])]
    ... }
    >>> result = maybe_custom_array_tree(data)
    >>> result['a']
    1
    >>> result['b']
    2
    >>> result['c'][1]
    4
    >>> result['c'][2][1]
    6
    """
    global CustomArray
    if CustomArray is None:
        from saiunit.custom_array import CustomArray
    return jax.tree.map(maybe_custom_array, x, is_leaf=lambda a: isinstance(a, CustomArray))
