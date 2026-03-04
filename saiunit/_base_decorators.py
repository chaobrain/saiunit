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

"""Decorators for dimension and unit checking/assignment."""

from collections.abc import Callable
from functools import partial, wraps
from typing import cast

import jax

from ._base_dimension import (
    DIMENSIONLESS,
    DimensionMismatchError,
    UnitMismatchError,
    get_dim_for_display,
)
from ._base_getters import (
    get_dim,
    get_unit,
    has_same_unit,
    have_same_dim,
)
from ._base_unit import UNITLESS, Unit
from ._misc import set_module_as


__all__ = [
    'check_units',
    'check_dims',
    'assign_units',
]


def _is_quantity(x):
    from ._base_quantity import Quantity
    return isinstance(x, Quantity)


@set_module_as('saiunit')
def check_dims(**au):
    """
    Decorator to check dimensions of arguments passed to a function

    Examples
    --------
    >>> from saiunit import *
    >>> @check_dims(I=amp.dim, R=ohm.dim, wibble=metre.dim, result=volt.dim)
    ... def getvoltage(I, R, **k):
    ...     return I*R

    You don't have to check the units of every variable in the function, and
    you can define what the units should be for variables that aren't
    explicitly named in the definition of the function. For example, the code
    above checks that the variable wibble should be a length, so writing

    >>> getvoltage(1*amp, 1*ohm, wibble=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function "getvoltage" variable "wibble" has wrong dimensions, dimensions were (1) (m)

    fails, but

    >>> getvoltage(1*amp, 1*ohm, wibble=1*metre)
    1. * volt

    By using the special name ``result``, you can check the return value of the
    function.

    You can also use ``1`` or ``bool`` as a special value to check for a
    unitless number or a boolean value, respectively:

    >>> @check_units(value=1, absolute=bool, result=bool)
    ... def is_high(value, absolute=False):
    ...     if absolute:
    ...         return abs(value) >= 5
    ...     else:
    ...         return value >= 5

    This will then again raise an error if the argument if not of the expected
    type:

    >>> is_high(7)
    True
    >>> is_high(-7, True)
    True
    >>> is_high(3, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: Function "is_high" expected a boolean value for argument "absolute" but got 4.

    If the return unit depends on the unit of an argument, you can also pass
    a function that takes the units of all the arguments as its inputs (in the
    order specified in the function header):

    >>> @check_units(result=lambda d: d**2)
    ... def square(value):
    ...     return value**2

    If several arguments take arbitrary units but they have to be
    consistent among each other, you can state the name of another argument as
    a string to state that it uses the same unit as that argument.

    >>> @check_units(summand_1=None, summand_2='summand_1')
    ... def multiply_sum(multiplicand, summand_1, summand_2):
    ...     "Calculates multiplicand*(summand_1 + summand_2)"
    ...     return multiplicand*(summand_1 + summand_2)
    >>> multiply_sum(3, 4*mV, 5*mV)
    27. * mvolt
    >>> multiply_sum(3*nA, 4*mV, 5*mV)
    27. * pwatt
    >>> multiply_sum(3*nA, 4*mV, 5*nA)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function 'multiply_sum' expected the same arguments for arguments 'summand_1', 'summand_2', but argument 'summand_1' has unit V, while argument 'summand_2' has unit A.

    Raises
    ------

    DimensionMismatchError
        In case the input arguments or the return value do not have the
        expected dimensions.
    TypeError
        If an input argument or return value was expected to be a boolean but
        is not.

    Notes
    -----
    This decorator will destroy the signature of the original function, and
    replace it with the signature ``(*args, **kwds)``. Other decorators will
    do the same thing, and this decorator critically needs to know the signature
    of the function it is acting on, so it is important that it is the first
    decorator to act on a function. It cannot be used in combination with
    another decorator that also needs to know the signature of the function.

    Note that the ``bool`` type is "strict", i.e. it expects a proper
    boolean value and does not accept 0 or 1. This is not the case the other
    way round, declaring an argument or return value as "1" *does* allow for a
    ``True`` or ``False`` value.
    """
    from ._base_quantity import Quantity

    def do_check_units(f):
        @wraps(f)
        def new_f(*args, **kwds):
            newkeyset = kwds.copy()
            arg_names = f.__code__.co_varnames[0: f.__code__.co_argcount]
            for n, v in zip(arg_names, args[0: f.__code__.co_argcount]):
                if (
                    not isinstance(v, (Quantity, str, bool))
                    and v is not None
                    and n in au
                ):
                    try:
                        # allow e.g. to pass a Python list of values
                        v = Quantity(v)
                    except TypeError:
                        if have_same_dim(au[n], 1):
                            raise TypeError(f"Argument {n} is not a unitless value/array.")
                        else:
                            raise TypeError(
                                f"Argument '{n}' is not a array, "
                                "expected a array with dimensions "
                                f"{au[n]}"
                            )
                newkeyset[n] = v

            for k in newkeyset:
                # string variables are allowed to pass, the presumption is they
                # name another variable. None is also allowed, useful for
                # default parameters
                if (
                    k in au
                    and not isinstance(newkeyset[k], str)
                    and not newkeyset[k] is None
                    and not au[k] is None
                ):
                    if au[k] == bool:
                        if not isinstance(newkeyset[k], bool):
                            value = newkeyset[k]
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected a boolean value "
                                f"for argument '{k}' but got "
                                f"'{value}'"
                            )
                            raise TypeError(error_message)
                    elif isinstance(au[k], str):
                        if not au[k] in newkeyset:
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected its argument to have the "
                                f"same units as argument '{k}', but "
                                "there is no argument of that name"
                            )
                            raise TypeError(error_message)
                        if not have_same_dim(newkeyset[k], newkeyset[au[k]]):
                            d1 = get_dim(newkeyset[k])
                            d2 = get_dim(newkeyset[au[k]])
                            error_message = (
                                f"Function '{f.__name__}' expected "
                                f"the argument '{k}' to have the same "
                                f"dimensions as argument '{au[k]}', but "
                                f"argument '{k}' has "
                                f"unit {get_dim_for_display(d1)}, "
                                f"while argument '{au[k]}' "
                                f"has dimension {get_dim_for_display(d2)}."
                            )
                            raise DimensionMismatchError(error_message)
                    elif not have_same_dim(newkeyset[k], au[k]):
                        unit = repr(au[k])
                        value = newkeyset[k]
                        error_message = (
                            f"Function '{f.__name__}' "
                            "expected a array with dimension "
                            f"{unit} for argument '{k}' but got "
                            f"'{value}'"
                        )
                        raise DimensionMismatchError(
                            error_message,
                            get_dim(newkeyset[k])
                        )

            result = f(*args, **kwds)
            if "result" in au:
                if isinstance(au["result"], Callable) and au["result"] != bool:
                    expected_result = au["result"](*[get_dim(a) for a in args])
                else:
                    expected_result = au["result"]

                if (
                    jax.tree.structure(expected_result, is_leaf=_is_quantity)
                    !=
                    jax.tree.structure(result, is_leaf=_is_quantity)
                ):
                    raise TypeError(
                        f"Expected a return value of type {expected_result} but got {result}"
                    )

                jax.tree.map(
                    partial(_check_dim, f), result, expected_result,
                    is_leaf=_is_quantity
                )
            return result

        new_f._orig_func = f
        # store the information in the function, necessary when using the
        # function in expressions or equations
        if hasattr(f, "_orig_arg_names"):
            arg_names = f._orig_arg_names
        else:
            arg_names = f.__code__.co_varnames[: f.__code__.co_argcount]
        new_f._arg_names = arg_names
        new_f._arg_units = [au.get(name, None) for name in arg_names]
        return_unit = au.get("result", None)
        if return_unit is None:
            new_f._return_unit = None
        else:
            new_f._return_unit = return_unit
        if return_unit == bool:
            new_f._returns_bool = True
        else:
            new_f._returns_bool = False
        new_f._orig_arg_names = arg_names

        # copy any annotation attributes
        if hasattr(f, "_annotation_attributes"):
            for attrname in f._annotation_attributes:
                setattr(new_f, attrname, getattr(f, attrname))
        new_f._annotation_attributes = getattr(f, "_annotation_attributes", []) + [
            "_arg_units",
            "_arg_names",
            "_return_unit",
            "_orig_func",
            "_returns_bool",
        ]
        return new_f

    return do_check_units


def _check_dim(f, val, dim):
    dim = DIMENSIONLESS if dim is None else dim
    if not have_same_dim(val, dim):
        unit = get_dim_for_display(dim)
        error_message = (
            "The return value of function "
            f"'{f.__name__}' was expected to have "
            f"dimension {unit} but was "
            f"'{val}'"
        )
        raise DimensionMismatchError(error_message, get_dim(val))


@set_module_as('saiunit')
def check_units(**au):
    """
    Decorator to check units of arguments passed to a function

    Examples
    --------
    >>> from saiunit import *
    >>> @check_units(I=amp, R=ohm, wibble=metre, result=volt)
    ... def getvoltage(I, R, **k):
    ...     return I*R

    You don't have to check the units of every variable in the function, and
    you can define what the units should be for variables that aren't
    explicitly named in the definition of the function. For example, the code
    above checks that the variable wibble should be a length, so writing

    >>> getvoltage(1*amp, 1*ohm, wibble=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function "getvoltage" variable "wibble" has wrong dimensions, dimensions were (1) (m)

    fails, but

    >>> getvoltage(1*amp, 1*ohm, wibble=1*metre)
    1. * volt

    By using the special name ``result``, you can check the return value of the
    function.

    You can also use ``1`` or ``bool`` as a special value to check for a
    unitless number or a boolean value, respectively:

    >>> @check_units(value=1, absolute=bool, result=bool)
    ... def is_high(value, absolute=False):
    ...     if absolute:
    ...         return abs(value) >= 5
    ...     else:
    ...         return value >= 5

    This will then again raise an error if the argument if not of the expected
    type:

    >>> is_high(7)
    True
    >>> is_high(-7, True)
    True
    >>> is_high(3, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: Function "is_high" expected a boolean value for argument "absolute" but got 4.

    If the return unit depends on the unit of an argument, you can also pass
    a function that takes the units of all the arguments as its inputs (in the
    order specified in the function header):

    >>> @check_units(result=lambda d: d**2)
    ... def square(value):
    ...     return value**2

    If several arguments take arbitrary units but they have to be
    consistent among each other, you can state the name of another argument as
    a string to state that it uses the same unit as that argument.

    >>> @check_units(summand_1=None, summand_2='summand_1')
    ... def multiply_sum(multiplicand, summand_1, summand_2):
    ...     "Calculates multiplicand*(summand_1 + summand_2)"
    ...     return multiplicand*(summand_1 + summand_2)
    >>> multiply_sum(3, 4*mV, 5*mV)
    27. * mvolt
    >>> multiply_sum(3*nA, 4*mV, 5*mV)
    27. * pwatt
    >>> multiply_sum(3*nA, 4*mV, 5*nA)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function 'multiply_sum' expected the same arguments for arguments 'summand_1', 'summand_2', but argument 'summand_1' has unit V, while argument 'summand_2' has unit A.

    Raises
    ------

    DimensionMismatchError
        In case the input arguments or the return value do not have the
        expected dimensions.
    TypeError
        If an input argument or return value was expected to be a boolean but
        is not.

    Notes
    -----
    This decorator will destroy the signature of the original function, and
    replace it with the signature ``(*args, **kwds)``. Other decorators will
    do the same thing, and this decorator critically needs to know the signature
    of the function it is acting on, so it is important that it is the first
    decorator to act on a function. It cannot be used in combination with
    another decorator that also needs to know the signature of the function.

    Note that the ``bool`` type is "strict", i.e. it expects a proper
    boolean value and does not accept 0 or 1. This is not the case the other
    way round, declaring an argument or return value as "1" *does* allow for a
    ``True`` or ``False`` value.
    """
    from ._base_quantity import Quantity

    def do_check_units(f):
        @wraps(f)
        def new_f(*args, **kwds):
            newkeyset = kwds.copy()
            arg_names = f.__code__.co_varnames[0: f.__code__.co_argcount]
            for n, v in zip(arg_names, args[0: f.__code__.co_argcount]):
                if (
                    not isinstance(v, (Quantity, str, bool))
                    and v is not None
                    and n in au
                ):
                    try:
                        # allow e.g. to pass a Python list of values
                        v = Quantity(v)
                    except TypeError:
                        if have_same_dim(au[n], 1):
                            raise TypeError(f"Argument {n} is not a unitless value/array.")
                        else:
                            raise TypeError(
                                f"Argument '{n}' is not a array, "
                                "expected a array with dimensions "
                                f"{au[n]}"
                            )
                newkeyset[n] = v

            for k in newkeyset:
                # string variables are allowed to pass, the presumption is they
                # name another variable. None is also allowed, useful for
                # default parameters
                if (
                    k in au
                    and not isinstance(newkeyset[k], str)
                    and not newkeyset[k] is None
                    and not au[k] is None
                ):
                    if au[k] == bool:
                        if not isinstance(newkeyset[k], bool):
                            value = newkeyset[k]
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected a boolean value "
                                f"for argument '{k}' but got "
                                f"'{value}'"
                            )
                            raise TypeError(error_message)
                    elif isinstance(au[k], str):
                        if not au[k] in newkeyset:
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected its argument to have the "
                                f"same units as argument '{k}', but "
                                "there is no argument of that name"
                            )
                            raise TypeError(error_message)
                        if not has_same_unit(newkeyset[k], newkeyset[au[k]]):
                            d1 = get_unit(newkeyset[k])
                            d2 = get_unit(newkeyset[au[k]])
                            error_message = (
                                f"Function '{f.__name__}' expected "
                                f"the argument '{k}' to have the same "
                                f"units as argument '{au[k]}', but "
                                f"argument '{k}' has "
                                f"unit {d1}, "
                                f"while argument '{au[k]}' "
                                f"has unit {d2}."
                            )
                            raise UnitMismatchError(error_message)
                    elif not has_same_unit(newkeyset[k], au[k]):
                        unit = repr(au[k])
                        value = newkeyset[k]
                        error_message = (
                            f"Function '{f.__name__}' "
                            "expected a array with unit "
                            f"{unit} for argument '{k}' but got "
                            f"'{value}'"
                        )
                        raise UnitMismatchError(error_message, get_unit(newkeyset[k]))

            result = f(*args, **kwds)
            if "result" in au:
                if isinstance(au["result"], Callable) and au["result"] != bool:
                    expected_result = au["result"](*[get_dim(a) for a in args])
                else:
                    expected_result = au["result"]

                if (
                    jax.tree.structure(expected_result, is_leaf=_is_quantity)
                    !=
                    jax.tree.structure(result, is_leaf=_is_quantity)
                ):
                    raise TypeError(
                        f"Expected a return value of type {expected_result} but got {result}"
                    )

                jax.tree.map(
                    partial(_check_unit, f), result, expected_result,
                    is_leaf=_is_quantity
                )
            return result

        new_f._orig_func = f
        # store the information in the function, necessary when using the
        # function in expressions or equations
        if hasattr(f, "_orig_arg_names"):
            arg_names = f._orig_arg_names
        else:
            arg_names = f.__code__.co_varnames[: f.__code__.co_argcount]
        new_f._arg_names = arg_names
        new_f._arg_units = [au.get(name, None) for name in arg_names]
        return_unit = au.get("result", None)
        if return_unit is None:
            new_f._return_unit = None
        else:
            new_f._return_unit = return_unit
        if return_unit == bool:
            new_f._returns_bool = True
        else:
            new_f._returns_bool = False
        new_f._orig_arg_names = arg_names

        # copy any annotation attributes
        if hasattr(f, "_annotation_attributes"):
            for attrname in f._annotation_attributes:
                setattr(new_f, attrname, getattr(f, attrname))
        new_f._annotation_attributes = getattr(f, "_annotation_attributes", []) + [
            "_arg_units",
            "_arg_names",
            "_return_unit",
            "_orig_func",
            "_returns_bool",
        ]
        return new_f

    return do_check_units


class CallableAssignUnit(Callable):
    without_result_units = Callable

    def __call__(self, *args, **kwargs):
        pass


class Missing:
    pass


missing = Missing()


@set_module_as('saiunit')
def assign_units(f: Callable = missing, **au) -> CallableAssignUnit | Callable[[Callable], CallableAssignUnit]:
    """
    Decorator to transform units of arguments passed to a function and optionally assign units to the return value.

    This decorator performs two main functions:
    1. Removes units from input arguments based on specified expected units
    2. Optionally assigns units to the return value if 'result' is specified

    Parameters
    ----------
    f : Callable, optional
        The function to be decorated. If missing, returns a partial decorator.
    **au : dict
        Keyword arguments specifying expected units for function parameters.
        Use parameter names as keys and expected units as values.
        Special key 'result' can be used to specify return value units.

    Returns
    -------
    CallableAssignUnit
        The decorated function with unit transformation capabilities.

    Examples
    --------
    Basic usage to transform input units:
    >>> from saiunit import *
    >>> @assign_units(I=amp, R=ohm)
    ... def getvoltage(I, R):
    ...     return I*R

    You can specify units for kwargs:
    >>> @assign_units(wibble=metre)
    ... def func(wibble=None):
    ...     return wibble

    To specify return value units:
    >>> @assign_units(I=amp, R=ohm, result=volt)
    ... def getvoltage(I, R):
    ...     return I*R

    The return units can be dynamic based on input units:
    >>> @assign_units(result=lambda d: d**2)
    ... def square(value):
    ...     return value**2

    The decorated function has a 'without_result_units' attribute that
    returns the raw result without unit assignment:
    >>> func = assign_units(result=volt)(lambda x: x)
    >>> func(3*mV).without_result_units()
    0.003

    Notes
    -----
    1. The decorator checks that input arguments have compatible dimensions
       with the specified units before removing them.
    2. When 'result' is specified, the return value will be assigned the given units.
    3. The 'without_result_units' attribute provides access to the undecorated version
       that skips the return value unit assignment step.
    """
    from ._base_quantity import Quantity

    if f is missing:
        return partial(assign_units, **au)

    @wraps(f)
    def new_f(*args, **kwds):
        arg_names = f.__code__.co_varnames[0: f.__code__.co_argcount]
        newkeyset = kwds.copy()
        for n, v in zip(arg_names, args[0: f.__code__.co_argcount]):
            newkeyset[n] = v
        for n, v in tuple(newkeyset.items()):
            if n in au and v is not None:
                specific_unit = au[n]

                if (
                    jax.tree.structure(specific_unit, is_leaf=_is_quantity)
                    !=
                    jax.tree.structure(v, is_leaf=_is_quantity)
                ):
                    raise TypeError(
                        f"For argument '{n}', we expect the input type "
                        f"with the structure like {specific_unit}, "
                        f"but we got {v}"
                    )

                v = jax.tree.map(
                    partial(_remove_unit, f.__name__, n),
                    specific_unit,
                    v,
                    is_leaf=_is_quantity
                )
            newkeyset[n] = v

        result = f(**newkeyset)
        if "result" in au:
            if isinstance(au["result"], Callable) and au["result"] != bool:
                expected_result = au["result"](*[get_unit(a) for a in args])
            else:
                expected_result = au["result"]

            expected_pytree = jax.tree.structure(
                expected_result,
                is_leaf=lambda x: isinstance(x, Quantity) or x is None
            )
            result_pytree = jax.tree.structure(result, is_leaf=lambda x: isinstance(x, Quantity) or x is None)
            if (
                expected_pytree
                !=
                result_pytree
            ):
                raise TypeError(
                    f"Expected a return value of pytree {expected_pytree} with type {expected_result}, "
                    f"but got the pytree {result_pytree} and the value {result}"
                )

            result = jax.tree.map(
                partial(_assign_unit, f),
                result,
                expected_result,
                is_leaf=lambda x: isinstance(x, Quantity) or x is None
            )
        return result

    def without_result_units(*args, **kwds):
        arg_names = f.__code__.co_varnames[0: f.__code__.co_argcount]
        newkeyset = kwds.copy()
        for n, v in zip(arg_names, args[0: f.__code__.co_argcount]):
            newkeyset[n] = v
        for n, v in tuple(newkeyset.items()):
            if n in au and v is not None:
                specific_unit = au[n]

                if (
                    jax.tree.structure(specific_unit, is_leaf=_is_quantity)
                    !=
                    jax.tree.structure(v, is_leaf=_is_quantity)
                ):
                    raise TypeError(
                        f"For argument '{n}', we expect the input type {specific_unit} but got {v}"
                    )

                v = jax.tree.map(
                    partial(_remove_unit, f.__name__, n),
                    specific_unit,
                    v,
                    is_leaf=_is_quantity
                )
            newkeyset[n] = v

        result = f(**newkeyset)
        return result

    new_f.without_result_units = without_result_units

    return cast(CallableAssignUnit, new_f)


def _remove_unit(fname, n, unit, v):
    from ._base_quantity import Quantity

    if unit is None:
        return v

    # if the specific unit is a boolean, just check and return
    elif unit is bool:
        if isinstance(v, bool):
            return v
        else:
            raise TypeError(
                f"Function '{fname}' expected a boolean "
                f"value for argument '{n}' but got '{v}'"
            )

    elif isinstance(unit, Unit):
        if isinstance(v, Quantity):
            v = v.to_decimal(unit)
            return v
        else:
            raise TypeError(
                f"Function '{fname}' expected a Quantity "
                f"object for argument '{n}' but got '{v}'"
            )

    elif unit == 1:
        if isinstance(v, Quantity):
            raise TypeError(
                f"Function '{fname}' expected a Number object for argument '{n}' but got '{v}'"
            )
        return v

    else:
        raise TypeError(
            f"Function '{fname}' expected a target unit object or"
            f" a Number, boolean object for checking, but got '{unit}'"
        )


def _check_unit(f, val, unit):
    unit = UNITLESS if unit is None else unit
    if not has_same_unit(val, unit):
        raise UnitMismatchError(
            f"The return value of function '{f.__name__}' was expected to have "
            f"unit {unit} but got unit {get_unit(val)} (value: {val!r})",
            unit, get_unit(val),
        )


def _assign_unit(f, val, unit):
    from ._base_quantity import Quantity

    if unit is None or unit is bool:
        return val
    return Quantity(val, unit=unit)
