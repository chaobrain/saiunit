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
    global CustomArray
    if CustomArray is None:
        from saiunit.custom_array import CustomArray
    if isinstance(x, CustomArray):
        return x.value
    else:
        return x


def maybse_custom_array_tree(x):
    global CustomArray
    if CustomArray is None:
        from saiunit.custom_array import CustomArray
    return jax.tree.map(maybe_custom_array, x, is_leaf=lambda x: isinstance(x, CustomArray))
