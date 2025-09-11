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

import collections
import functools
import itertools
import operator
from collections import OrderedDict
from typing import Set, Tuple, List, Dict, Union, Callable, Optional, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum

from saiunit._compatible_import import safe_map, unzip2
from ._einops_parsing import ParsedExpression, _ellipsis, AnonymousAxis, EinopsError
from ._fun_array_creation import asarray, zeros_like
from ._fun_keep_unit import reshape, transpose, expand_dims, tile, where, squeeze
from ._misc import shape
from .._base import Quantity, UNITLESS
from .._misc import set_module_as, maybe_custom_array, maybe_custom_array_tree

T = TypeVar('T')

__all__ = [
    'einreduce',
    'einrearrange',
    'einrepeat',
    'einshape',
    'einsum',
]

ReductionCallable = Callable[[jax.typing.ArrayLike, Tuple[int, ...]], jax.typing.ArrayLike]
Reduction = Union[str, ReductionCallable]

_reductions = ("min", "max", "sum", "mean", "prod", "any", "all")

# magic integers are required to stay within
# traceable subset of language
_unknown_axis_length = -999999
_expected_axis_length = -99999


def is_float_type(x: jax.typing.ArrayLike):
    return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")


def add_axis(x: jax.typing.ArrayLike, new_position: int):
    return expand_dims(asarray(x), new_position)


def add_axes(x: jax.typing.ArrayLike, n_axes, pos2len):
    repeats = [1] * n_axes
    for axis_position, axis_length in pos2len.items():
        x = add_axis(x, axis_position)
        repeats[axis_position] = axis_length
    return tile(x, repeats)


def _product(sequence: List[int]) -> int:
    """minimalistic product that works both with numbers and symbols. Supports empty lists"""
    result = 1
    for element in sequence:
        result *= element
    return result


def _reduce_axes(tensor, reduction_type: Reduction, reduced_axes: List[int]):
    if callable(reduction_type):
        # custom callable
        return reduction_type(tensor, tuple(reduced_axes))
    else:
        # one of built-in operations
        assert reduction_type in _reductions
        if reduction_type == "mean":
            if not is_float_type(tensor):
                raise NotImplementedError("reduce_mean is not available for non-floating tensors")
        return __reduce(tensor, reduction_type, tuple(reduced_axes))


def __reduce(x: jax.typing.ArrayLike, operation: str, reduced_axes):
    if operation == "min":
        return x.min(axis=reduced_axes)
    elif operation == "max":
        return x.max(axis=reduced_axes)
    elif operation == "sum":
        return x.sum(axis=reduced_axes)
    elif operation == "mean":
        return x.mean(axis=reduced_axes)
    elif operation == "prod":
        return x.prod(axis=reduced_axes)
    elif operation == "any":
        return x.any(axis=reduced_axes)
    elif operation == "all":
        return x.all(axis=reduced_axes)
    else:
        raise NotImplementedError("Unknown reduction ", operation)


def _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes):
    # 'collapses' neighboring axes if those participate in the result pattern in the same order
    # TODO add support for added_axes
    assert len(axes_reordering) + len(reduced_axes) == len(init_shapes)
    # joining consecutive axes that will be reduced
    # possibly we can skip this if all backends can optimize this (not sure)
    reduced_axes = tuple(sorted(reduced_axes))
    for i in range(len(reduced_axes) - 1)[::-1]:
        if reduced_axes[i] + 1 == reduced_axes[i + 1]:
            removed_axis = reduced_axes[i + 1]
            removed_length = init_shapes[removed_axis]
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            reduced_axes = reduced_axes[: i + 1] + tuple(axis - 1 for axis in reduced_axes[i + 2:])

    # removing axes that are moved together during reshape
    def build_mapping():
        init_to_final = {}
        for axis in range(len(init_shapes)):
            if axis in reduced_axes:
                init_to_final[axis] = None
            else:
                after_reduction = sum(x is not None for x in init_to_final.values())
                init_to_final[axis] = list(axes_reordering).index(after_reduction)
        return init_to_final

    init_axis_to_final_axis = build_mapping()

    for init_axis in range(len(init_shapes) - 1)[::-1]:
        if init_axis_to_final_axis[init_axis] is None:
            continue
        if init_axis_to_final_axis[init_axis + 1] is None:
            continue
        if init_axis_to_final_axis[init_axis] + 1 == init_axis_to_final_axis[init_axis + 1]:
            removed_axis = init_axis + 1
            removed_length = init_shapes[removed_axis]
            removed_axis_after_reduction = sum(x not in reduced_axes for x in range(removed_axis))

            reduced_axes = tuple(axis if axis < removed_axis else axis - 1 for axis in reduced_axes)
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            old_reordering = axes_reordering
            axes_reordering = []
            for axis in old_reordering:
                if axis == removed_axis_after_reduction:
                    pass
                elif axis < removed_axis_after_reduction:
                    axes_reordering.append(axis)
                else:
                    axes_reordering.append(axis - 1)
            init_axis_to_final_axis = build_mapping()

    return init_shapes, reduced_axes, axes_reordering, final_shapes


CookedRecipe = Tuple[Optional[List[int]], Optional[List[int]], List[int], Dict[int, int], Optional[List[int]], int]

# Actual type is tuple[tuple[str, int], ...]
# However torch.jit.script does not "understand" the correct type,
# and torch_specific will use list version.
HashableAxesLengths = Tuple[Tuple[str, int], ...]
FakeHashableAxesLengths = List[Tuple[str, int]]


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    # structure is non-mutable. In future, this can be non-mutable dataclass (python 3.7+)
    # update: pytorch 2.0 torch.jit.script seems to have problems with dataclasses unless they were explicitly provided

    def __init__(
        self,
        # list of sizes (or just sizes) for elementary axes as they appear in left expression.
        # this is what (after computing unknown parts) will be a shape after first transposition.
        # This does not include any ellipsis dimensions.
        elementary_axes_lengths: List[int],
        # if additional axes are provided, they should be set in prev array
        # This shows mapping from name to position
        axis_name2elementary_axis: Dict[str, int],
        # each dimension in input can help to reconstruct length of one elementary axis
        # or verify one of dimensions. Each element points to element of elementary_axes_lengths.
        input_composition_known_unknown: List[Tuple[List[int], List[int]]],
        # permutation applied to elementary axes, if ellipsis is absent
        axes_permutation: List[int],
        # permutation puts reduced axes in the end, we only need to know the first position.
        first_reduced_axis: int,
        # at which positions which of elementary axes should appear. Axis position -> axis index.
        added_axes: Dict[int, int],
        # ids of axes as they appear in result, again pointers to elementary_axes_lengths,
        # only used to infer result dimensions
        output_composite_axes: List[List[int]],
    ):
        self.elementary_axes_lengths: List[int] = elementary_axes_lengths
        self.axis_name2elementary_axis: Dict[str, int] = axis_name2elementary_axis
        self.input_composition_known_unknown: List[Tuple[List[int], List[int]]] = input_composition_known_unknown
        self.axes_permutation: List[int] = axes_permutation

        self.first_reduced_axis: int = first_reduced_axis
        self.added_axes: Dict[int, int] = added_axes
        self.output_composite_axes: List[List[int]] = output_composite_axes


def _reconstruct_from_shape_uncached(
    self: TransformRecipe,
    shape: List[int],
    axes_dims: FakeHashableAxesLengths
) -> CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, theano) and UnknownSize (tf, previously mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    # magic number
    need_init_reshape = False

    # last axis is allocated for collapsed ellipsis
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    for axis, dim in axes_dims:
        axes_lengths[self.axis_name2elementary_axis[axis]] = dim

    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composition_known_unknown):
        length = shape[input_axis]
        if len(known_axes) == 0 and len(unknown_axes) == 1:
            # shortcut for the most common case
            axes_lengths[unknown_axes[0]] = length
            continue

        known_product = 1
        for axis in known_axes:
            known_product *= axes_lengths[axis]

        if len(unknown_axes) == 0:
            if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                raise EinopsError(f"Shape mismatch, {length} != {known_product}")
        else:
            # assert len(unknown_axes) == 1, 'this is enforced when recipe is created, so commented out'
            if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                raise EinopsError(f"Shape mismatch, can't divide axis of length {length} in chunks of {known_product}")

            unknown_axis = unknown_axes[0]
            inferred_length: int = length // known_product
            axes_lengths[unknown_axis] = inferred_length

        if len(known_axes) + len(unknown_axes) != 1:
            need_init_reshape = True

    # at this point all axes_lengths are computed (either have values or variables, but not Nones)

    # elementary axes are ordered as they appear in input, then all added axes
    init_shapes: Optional[List[int]] = axes_lengths[: len(self.axes_permutation)] if need_init_reshape else None

    need_final_reshape = False
    final_shapes: List[int] = []
    for grouping in self.output_composite_axes:
        lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
        final_shapes.append(_product(lengths))
        if len(lengths) != 1:
            need_final_reshape = True

    added_axes: Dict[int, int] = {
        pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()
    }

    # this list can be empty
    reduced_axes = list(range(self.first_reduced_axis, len(self.axes_permutation)))

    n_axes_after_adding_axes = len(added_axes) + len(self.axes_permutation)

    axes_reordering: Optional[List[int]] = self.axes_permutation
    if self.axes_permutation == list(range(len(self.axes_permutation))):
        axes_reordering = None

    _final_shapes = final_shapes if need_final_reshape else None
    return init_shapes, axes_reordering, reduced_axes, added_axes, _final_shapes, n_axes_after_adding_axes


_reconstruct_from_shape = functools.lru_cache(1024)(_reconstruct_from_shape_uncached)


def _apply_recipe(
    recipe: TransformRecipe,
    x: jax.typing.ArrayLike | Quantity,
    reduction_type: Reduction,
    axes_lengths: HashableAxesLengths
) -> jax.typing.ArrayLike | Quantity:
    # this method implements actual work for all backends for 3 operations
    try:
        init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added = (
            _reconstruct_from_shape(recipe, _get_shape(x), axes_lengths))
    except TypeError:
        # shape or one of passed axes lengths is not hashable (i.e. they are symbols)
        _result = _reconstruct_from_shape_uncached(recipe, _get_shape(x), axes_lengths)
        (init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added) = _result
    if init_shapes is not None:
        x = reshape(x, init_shapes)
    if axes_reordering is not None:
        x = transpose(asarray(x), axes_reordering)
    if len(reduced_axes) > 0:
        x = _reduce_axes(x, reduction_type=reduction_type, reduced_axes=reduced_axes)
    if len(added_axes) > 0:
        x = add_axes(x, n_axes=n_axes_w_added, pos2len=added_axes)
    if final_shapes is not None:
        x = reshape(asarray(x), final_shapes)
    return x


@functools.lru_cache(256)
def _prepare_transformation_recipe(
    pattern: str,
    operation: Reduction,
    axes_names: Tuple[str, ...],
    ndim: int,
) -> TransformRecipe:
    """Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)

    # checking that axes are in agreement - new axes appear only in repeat, while disappear only in reduction
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError("Ellipsis found in right side, but not left side of a pattern {}".format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError("Ellipsis inside parenthesis in the left side is not allowed: {}".format(pattern))
    if operation == "rearrange":
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError("Non-unitary anonymous axes are not supported in rearrange (exception is length 1)")
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError("Identifiers only on one side of expression (should be on both): {}".format(difference))
    elif operation == "repeat":
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError("Unexpected identifiers on the left side of repeat: {}".format(difference))
        axes_without_size = set.difference(
            {ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)},
            {*left.identifiers, *axes_names},
        )
        if len(axes_without_size) > 0:
            raise EinopsError("Specify sizes for new axes in repeat: {}".format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise EinopsError("Unexpected identifiers on the right side of reduce {}: {}".format(operation, difference))
    else:
        raise EinopsError("Unknown reduction {}. Expect one of {}.".format(operation, _reductions))

    if left.has_ellipsis:
        n_other_dims = len(left.composition) - 1
        if ndim < n_other_dims:
            raise EinopsError(f"Wrong shape: expected >={n_other_dims} dims. Received {ndim}-dim tensor.")
        ellipsis_ndim = ndim - n_other_dims
        ell_axes = [_ellipsis + str(i) for i in range(ellipsis_ndim)]
        left_composition = []
        for composite_axis in left.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    left_composition.append([axis])
            else:
                left_composition.append(composite_axis)

        rght_composition = []
        for composite_axis in rght.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    rght_composition.append([axis])
            else:
                group = []
                for axis in composite_axis:
                    if axis == _ellipsis:
                        group.extend(ell_axes)
                    else:
                        group.append(axis)
                rght_composition.append(group)

        left.identifiers.update(ell_axes)
        left.identifiers.remove(_ellipsis)
        if rght.has_ellipsis:
            rght.identifiers.update(ell_axes)
            rght.identifiers.remove(_ellipsis)
    else:
        if ndim != len(left.composition):
            raise EinopsError(f"Wrong shape: expected {len(left.composition)} dims. Received {ndim}-dim tensor.")
        left_composition = left.composition
        rght_composition = rght.composition

    # parsing all dimensions to find out lengths
    axis_name2known_length: Dict[Union[str, AnonymousAxis], int] = OrderedDict()
    for composite_axis in left_composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length

    # axis_ids_after_first_reshape = range(len(axis_name2known_length)) at this point

    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)

    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}

    # axes provided as kwargs
    for elementary_axis in axes_names:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise EinopsError("Invalid name for an axis", elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError("Axis {} is not used in transform".format(elementary_axis))
        axis_name2known_length[elementary_axis] = _expected_axis_length

    input_axes_known_unknown = []
    # some shapes are inferred later - all information is prepared for faster inference
    for i, composite_axis in enumerate(left_composition):
        known: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise EinopsError("Could not infer sizes for {}".format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(
            ([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown])
        )

    axis_position_after_reduction: Dict[str, int] = {}
    for axis_name in itertools.chain(*left_composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)

    result_axes_grouping: List[List[int]] = [
        [axis_name2position[axis] for axis in composite_axis] for i, composite_axis in enumerate(rght_composition)
    ]

    ordered_axis_left = list(itertools.chain(*left_composition))
    ordered_axis_rght = list(itertools.chain(*rght_composition))
    reduced_axes = [axis for axis in ordered_axis_left if axis not in rght.identifiers]
    order_after_transposition = [axis for axis in ordered_axis_rght if axis in left.identifiers] + reduced_axes
    axes_permutation = [ordered_axis_left.index(axis) for axis in order_after_transposition]
    added_axes = {
        i: axis_name2position[axis_name]
        for i, axis_name in enumerate(ordered_axis_rght)
        if axis_name not in left.identifiers
    }

    first_reduced_axis = len(order_after_transposition) - len(reduced_axes)

    return TransformRecipe(
        elementary_axes_lengths=list(axis_name2known_length.values()),
        axis_name2elementary_axis={axis: axis_name2position[axis] for axis in axes_names},
        input_composition_known_unknown=input_axes_known_unknown,
        axes_permutation=axes_permutation,
        first_reduced_axis=first_reduced_axis,
        added_axes=added_axes,
        output_composite_axes=result_axes_grouping,
    )


def _prepare_recipes_for_all_dims(
    pattern: str, operation: Reduction, axes_names: Tuple[str, ...]
) -> Dict[int, TransformRecipe]:
    """
    Internal function, used in layers.
    Layer makes all recipe creation when it is initialized, thus to keep recipes simple we pre-compute for all dims
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    dims = [len(left.composition)]
    if left.has_ellipsis:
        dims = [len(left.composition) - 1 + ellipsis_dims for ellipsis_dims in range(8)]
    return {ndim: _prepare_transformation_recipe(pattern, operation, axes_names, ndim=ndim) for ndim in dims}


def _get_shape(x) -> Tuple[int, ...]:
    if isinstance(x, Quantity):
        shape = x.shape
    else:
        shape = asarray(x).shape
    return shape


@set_module_as('brainstate.math')
def einreduce(
    x: Union[jax.typing.ArrayLike, Quantity, Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    pattern: str,
    reduction: Reduction,
    **axes_lengths: int
) -> jax.typing.ArrayLike | Quantity:
    """
    ``einreduce`` provides combination of reordering and reduction using reader-friendly notation.

    Examples for reduce operation:

    ```python
    >>> import brainstate as bst
    >>> x = bst.random.randn(100, 32, 64)

    # perform max-reduction on the first axis
    >>> y = einreduce(x, 't b c -> b c', 'max')

    # same as previous, but with clearer axes meaning
    >>> y = einreduce(x, 'time batch channel -> batch channel', 'max')

    >>> x = bst.random.randn(10, 20, 30, 40)

    # 2d max-pooling with kernel size = 2 * 2 for image processing
    >>> y1 = einreduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)

    # if one wants to go back to the original height and width, depth-to-space trick can be applied
    >>> y2 = einrearrange(y1, 'b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)', h2=2, w2=2)
    >>> assert einshape(x, 'b _ h w') == einshape(y2, 'b _ h w')

    # Adaptive 2d max-pooling to 3 * 4 grid
    >>> einreduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=3, w1=4).shape
    (10, 20, 3, 4)

    # Global average pooling
    >>> einreduce(x, 'b c h w -> b c', 'mean').shape
    (10, 20)

    # Subtracting mean over batch for each channel
    >>> y = x - einreduce(x, 'b c h w -> () c () ()', 'mean')

    # Subtracting per-image mean for each channel
    >>> y = x - einreduce(x, 'b c h w -> b c () ()', 'mean')

    ```

    Parameters
    ----------
    x: array_like, Quantity
      Array of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
      list of tensors is also accepted, those should be of the same type and shape
    pattern: string
     reduction pattern
    reduction:
      One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
      alternatively, a callable f(tensor, reduced_axes) -> tensor can be provided.
      This allows using various reductions, examples: np.max, tf.reduce_logsumexp, torch.var, etc.
    axes_lengths:
      any additional specifications for dimensions

    Returns
    -------
    out:  tensor of the same type as input
    """
    x = maybe_custom_array_tree(x)
    shape = _get_shape(x)
    try:
        hashable_axes_lengths = tuple(axes_lengths.items())
        recipe = _prepare_transformation_recipe(pattern, reduction, axes_names=tuple(axes_lengths), ndim=len(shape))
        return _apply_recipe(recipe, x, reduction_type=reduction, axes_lengths=hashable_axes_lengths)
    except EinopsError as e:
        message = ' Error while processing {}-reduction pattern "{}".'.format(reduction, pattern)
        if not isinstance(x, list):
            message += "\n Input tensor shape: {}. ".format(shape)
        else:
            message += "\n Input is list. "
        message += "Additional info: {}.".format(axes_lengths)
        raise EinopsError(message + "\n {}".format(e))


@set_module_as('brainstate.math')
def einrearrange(
    x: Union[jax.typing.ArrayLike, Quantity, Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    pattern: str,
    **axes_lengths
) -> jax.typing.ArrayLike | Quantity:
    """
    ``einrearrange`` is a reader-friendly smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    Examples for rearrange operation:

    ```python
    # suppose we have a set of 32 images in "h w c" format (height-width-channel)
    >>> import brainstate as bst
    >>> images = [bst.random.randn(30, 40, 3) for _ in range(32)]

    # stack along first (batch) axis, output is a single array
    >>> einrearrange(images, 'b h w c -> b h w c').shape
    (32, 30, 40, 3)

    # concatenate images along height (vertical axis), 960 = 32 * 30
    >>> einrearrange(images, 'b h w c -> (b h) w c').shape
    (960, 40, 3)

    # concatenated images along horizontal axis, 1280 = 32 * 40
    >>> einrearrange(images, 'b h w c -> h (b w) c').shape
    (30, 1280, 3)

    # reordered axes to "b c h w" format for deep learning
    >>> einrearrange(images, 'b h w c -> b c h w').shape
    (32, 3, 30, 40)

    # flattened each image into a vector, 3600 = 30 * 40 * 3
    >>> einrearrange(images, 'b h w c -> b (c h w)').shape
    (32, 3600)

    # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
    >>> einrearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
    (128, 15, 20, 3)

    # space-to-depth operation
    >>> einrearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
    (32, 15, 20, 12)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        x: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
                list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    return einreduce(x, pattern, reduction="rearrange", **axes_lengths)


@set_module_as('brainstate.math')
def einrepeat(
    x: Union[jax.typing.ArrayLike, Quantity, Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    pattern: str,
    **axes_lengths
) -> jax.typing.ArrayLike | Quantity:
    """
    ``einrepeat`` allows reordering elements and repeating them in arbitrary combinations.
    This operation includes functionality of repeat, tile, broadcast functions.

    Examples for repeat operation:

    ```python
    # a grayscale image (of shape height x width)
    >>> import brainstate as bst
    >>> image = bst.random.randn(30, 40)

    # change it to RGB format by repeating in each channel
    >>> einrepeat(image, 'h w -> h w c', c=3).shape
    (30, 40, 3)

    # repeat image 2 times along height (vertical axis)
    >>> einrepeat(image, 'h w -> (repeat h) w', repeat=2).shape
    (60, 40)

    # repeat image 2 time along height and 3 times along width
    >>> einrepeat(image, 'h w -> (h2 h) (w3 w)', h2=2, w3=3).shape
    (60, 120)

    # convert each pixel to a small square 2x2. Upsample image by 2x
    >>> einrepeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (60, 80)

    # pixelate image first by downsampling by 2x, then upsampling
    >>> downsampled = einreduce(image, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
    >>> einrepeat(downsampled, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (30, 40)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        x: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
            list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        Tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    return einreduce(x, pattern, reduction="repeat", **axes_lengths)


@set_module_as('brainstate.math')
def einshape(
    x: jax.typing.ArrayLike | Quantity,
    pattern: str
) -> dict:
    """
    Parse a tensor shape to dictionary mapping axes names to their lengths.

    ```python
    >>> import jax.numpy as jnp
    # Use underscore to skip the dimension in parsing.
    >>> x = jnp.zeros([2, 3, 5, 7])
    >>> einshape(x, 'batch _ h w')
    {'batch': 2, 'h': 5, 'w': 7}

    # `parse_shape` output can be used to specify axes_lengths for other operations:
    >>> y = jnp.zeros([700])
    >>> einrearrange(y, '(b c h w) -> b c h w', **einshape(x, 'b _ h w')).shape
    (2, 10, 5, 7)

    ```

    For symbolic frameworks may return symbols, not integers.

    Parameters:
        x: tensor of any supported framework
        pattern: str, space separated names for axes, underscore means skip axis

    Returns:
        dict, maps axes names to their lengths
    """
    x = maybe_custom_array_tree(x)
    _shape = _get_shape(x)
    exp = ParsedExpression(pattern, allow_underscore=True)
    if exp.has_composed_axes():
        raise RuntimeError(f"Can't parse shape with composite axes: {pattern} {_shape}")
    if len(_shape) != len(exp.composition):
        if exp.has_ellipsis:
            if len(_shape) < len(exp.composition) - 1:
                raise RuntimeError(f"Can't parse shape with this number of dimensions: {pattern} {_shape}")
        else:
            raise RuntimeError(f"Can't parse shape with different number of dimensions: {pattern} {_shape}")
    if exp.has_ellipsis:
        ellipsis_idx = exp.composition.index(_ellipsis)
        composition = (
            exp.composition[:ellipsis_idx]
            + ["_"] * (len(_shape) - len(exp.composition) + 1)
            + exp.composition[ellipsis_idx + 1:]
        )
    else:
        composition = exp.composition
    result = {}
    for (axis_name,), axis_length in zip(composition, _shape):  # type: ignore
        if axis_name != "_":
            result[axis_name] = axis_length
    return result


# _enumerate_directions is not exposed in the public API
def _enumerate_directions(x):
    """
    For an n-dimensional tensor, returns tensors to enumerate each axis.
    ```python
    x = np.zeros([2, 3, 4]) # or any other tensor
    i, j, k = _enumerate_directions(x)
    result = i + 2*j + 3*k
    ```

    `result[i, j, k] = i + 2j + 3k`, and also has the same shape as result
    Works very similarly to numpy.ogrid (open indexing grid)
    """
    shape = _get_shape(x)
    result = []
    for axis_id, axis_length in enumerate(shape):
        shape = [1] * len(shape)
        shape[axis_id] = axis_length
        result.append(jnp.reshape(jnp.arange(0, axis_length), shape))
    return result


def _removechars(s, chars):
    return s.translate(str.maketrans(dict.fromkeys(chars)))


def _partition_list(bs: Sequence[bool], l: Sequence[T]) -> tuple[list[T], list[T]]:
    assert len(bs) == len(l)
    lists = [], []  # type: ignore
    for b, x in zip(bs, l):
        lists[b].append(x)
    return lists


def _sum(x, axes, preferred_element_type):
    if jax.dtypes.result_type(x, preferred_element_type) != x.dtype:
        x = x.astype(preferred_element_type)
    unit = UNITLESS
    if isinstance(x, Quantity):
        unit = x.unit
        x = x.mantissa
    x = jax.lax.reduce(x, np.array(0, x.dtype), jax.lax.add if x.dtype != jnp.bool_ else jax.lax.bitwise_or, axes)
    return x if unit.is_unitless else Quantity(x, unit=unit)


def _sum_uniques(operand, names, uniques, preferred_element_type):
    if uniques:
        axes = [names.index(name) for name in uniques]
        operand = _sum(operand, axes, preferred_element_type)
        names = _removechars(names, uniques)
    return operand, names


def _dot_general(
    lhs: jax.typing.ArrayLike | Quantity,
    rhs: jax.typing.ArrayLike | Quantity,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None
) -> jax.Array | Quantity:
    unit = UNITLESS
    if isinstance(lhs, Quantity):
        unit = unit * lhs.unit
        lhs = lhs.mantissa
    if isinstance(rhs, Quantity):
        unit = unit * rhs.unit
        rhs = rhs.mantissa
    r = jax.lax.dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type
    )
    if unit.is_unitless:
        return r
    else:
        return Quantity(r, unit=unit)


def _delta(dtype: jax.typing.DTypeLike, shape_, axes: Sequence[int]) -> jax.Array:
    """This utility function exists for creating Kronecker delta arrays."""
    axes = safe_map(int, axes)
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    base_shape = tuple(np.take(shape_, axes))
    iotas = [jax.lax.broadcasted_iota(np.uint32, base_shape, i)
             for i in range(len(base_shape))]
    eyes = [jax.lax.eq(i1, i2) for i1, i2 in zip(iotas[:-1], iotas[1:])]
    result = jax.lax.convert_element_type(functools.reduce(operator.and_, eyes), new_dtype=dtype)
    return jax.lax.broadcast_in_dim(result, shape_, axes)


def _sum_repeats(operand, names, counts, keep_names, preferred_element_type):
    for name, count in counts.items():
        if count > 1:
            axes = [i for i, n in enumerate(names) if n == name]
            eye = _delta(np.dtype('bool'), operand.shape, axes)
            operand = where(eye, operand, zeros_like(operand))
            if name not in keep_names:
                operand = _sum(operand, axes, preferred_element_type)
                names = names.replace(name, '')
            else:
                operand = _sum(operand, axes[:-1], preferred_element_type)
                names = names.replace(name, '', count - 1)
    return operand, names


def _filter_singleton_dims(operand, names, other_shape, other_names):
    keep = [not (operand.shape[i] == 1) or j == -1 or (other_shape[j] == 1)
            for i, j in enumerate(map(other_names.find, names))]
    sqez_axes, keep_axes = _partition_list(keep, list(range(operand.ndim)))
    return squeeze(operand, sqez_axes), "".join(names[i] for i in keep_axes)


def _einsum(
    operands: Sequence,
    contractions: Sequence[tuple[tuple[int, ...], frozenset[str], str]],
    precision,
    preferred_element_type,
):
    operands = list(map(asarray, operands))
    if preferred_element_type is None:
        preferred_element_type = jax.dtypes.result_type(*operands)

    for operand_indices, contracted_names_set, einstr in contractions:
        contracted_names = sorted(contracted_names_set)
        input_str, result_names = einstr.split('->')
        input_names = input_str.split(',')

        # switch on the number of operands to be processed in this loop iteration.
        # every case here sets 'operand' and 'names'.
        if len(operand_indices) == 1:
            operand = operands.pop(operand_indices[0])
            names, = input_names
            counts = collections.Counter(names)

            # sum out unique contracted indices with a single reduce-sum
            uniques = [name for name in contracted_names if counts[name] == 1]
            operand, names = _sum_uniques(operand, names, uniques, preferred_element_type)

            # for every repeated index, do a contraction against an identity matrix
            operand, names = _sum_repeats(operand, names, counts, result_names, preferred_element_type)

        elif len(operand_indices) == 2:
            lhs, rhs = map(operands.pop, operand_indices)
            lhs_names, rhs_names = input_names

            # handle cases where one side of a contracting or batch dimension is 1
            # but its counterpart is not.
            lhs, lhs_names = _filter_singleton_dims(lhs, lhs_names, shape(rhs), rhs_names)
            rhs, rhs_names = _filter_singleton_dims(rhs, rhs_names, shape(lhs), lhs_names)

            lhs_counts = collections.Counter(lhs_names)
            rhs_counts = collections.Counter(rhs_names)

            # sum out unique contracted indices in lhs and rhs
            lhs_uniques = [name for name in contracted_names
                           if lhs_counts[name] == 1 and rhs_counts[name] == 0]
            lhs, lhs_names = _sum_uniques(lhs, lhs_names, lhs_uniques, preferred_element_type)

            rhs_uniques = [name for name in contracted_names
                           if rhs_counts[name] == 1 and lhs_counts[name] == 0]
            rhs, rhs_names = _sum_uniques(rhs, rhs_names, rhs_uniques, preferred_element_type)

            # for every repeated index, contract against an identity matrix
            lhs, lhs_names = _sum_repeats(lhs, lhs_names, lhs_counts,
                                          result_names + rhs_names,
                                          preferred_element_type)
            rhs, rhs_names = _sum_repeats(rhs, rhs_names, rhs_counts,
                                          result_names + lhs_names,
                                          preferred_element_type)

            lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
            contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
            lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
            batch_names = [x for x in result_names if x in lhs_and_rhs_names]

            lhs_batch, rhs_batch = unzip2((lhs_names.find(n), rhs_names.find(n)) for n in batch_names)

            # NOTE: this can fail non-deterministically in python3, maybe
            # due to opt_einsum
            assert jax.config.jax_dynamic_shapes or all(
                name in lhs_names and name in rhs_names and
                lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
                for name in contracted_names), \
                ("Incompatible reduction dimensions: "
                 f"lhs.shape={lhs.shape} lhs_names={lhs_names} "
                 f"rhs.shape={rhs.shape} rhs_names={rhs_names}")

            # contract using dot_general
            batch_names_str = ''.join(batch_names)
            lhs_cont, rhs_cont = unzip2((lhs_names.index(n), rhs_names.index(n)) for n in contracted_names)
            deleted_names = batch_names_str + ''.join(contracted_names)
            remaining_lhs_names = _removechars(lhs_names, deleted_names)
            remaining_rhs_names = _removechars(rhs_names, deleted_names)
            # Try both orders of lhs and rhs, in the hope that one of them means we
            # don't need an explicit transpose. opt_einsum likes to contract from
            # right to left, so we expect (rhs,lhs) to have the best chance of not
            # needing a transpose.
            names = batch_names_str + remaining_rhs_names + remaining_lhs_names
            if names == result_names:
                dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
                operand = _dot_general(rhs, lhs, dimension_numbers, precision,
                                       preferred_element_type=preferred_element_type)
            else:
                names = batch_names_str + remaining_lhs_names + remaining_rhs_names
                dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
                operand = _dot_general(lhs, rhs, dimension_numbers, precision,
                                       preferred_element_type=preferred_element_type)
        else:
            raise NotImplementedError  # if this is actually reachable, open an issue!

        # the resulting 'operand' with axis labels 'names' should be a permutation
        # of the desired result
        assert len(names) == len(result_names) == len(set(names))
        assert set(names) == set(result_names)
        if names != result_names:
            perm = tuple(names.index(name) for name in result_names)
            operand = transpose(operand, perm)
        operands.append(operand)  # used in next iteration

    ret = operands[0].mantissa if isinstance(operands[0], Quantity) else operands[0]
    ret = jax.lax.convert_element_type(ret, preferred_element_type)
    if isinstance(operands[0], Quantity):
        return Quantity(ret, unit=operands[0].unit)
    return ret


class Unoptimized(opt_einsum.paths.PathOptimizer):
    """Unoptimized path for einsum."""

    def __call__(self, inputs, *args, **kwargs):
        return [(0, 1)] * (len(inputs) - 1)


def _default_poly_einsum_handler(*operands, **kwargs):
    dummy = collections.namedtuple('dummy', ['shape', 'dtype'])
    dummies = [dummy(tuple(d if type(d) is int else 8 for d in x.shape), x.dtype)
               if hasattr(x, 'dtype') else x for x in operands]
    mapping = {id(d): i for i, d in enumerate(dummies)}
    out_dummies, contractions = opt_einsum.contract_path(*dummies, **kwargs)
    contract_operands = [operands[mapping[id(d)]] for d in out_dummies]
    return contract_operands, contractions


@set_module_as('saiunit.math')
def einsum(
    subscripts,
    /,
    *operands,
    optimize: str | bool | list[tuple[int, ...]] = "optimal",
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
    """Einstein summation for arrays and quantities.

    ``einsum`` is a powerful and generic API for computing various reductions,
    inner products, outer products, axis reorderings, and combinations thereof
    across one or more input arrays. It has a somewhat complicated overloaded API;
    the arguments below reflect the most common calling convention. The Examples
    section below demonstrates some of the alternative calling conventions.

    Args:
      subscripts: string containing axes names separated by commas.
      *operands: sequence of one or more arrays / quantities corresponding to the subscripts.
      optimize: specify how to optimize the order of computation. In JAX this defaults
        to ``"optimal"`` which produces optimized expressions via the opt_einsum_
        package. Other options are ``True`` (same as ``"optimal"``), ``False``
        (unoptimized), or any string supported by ``opt_einsum``, which
        includes ``"auto"``, ``"greedy"``, ``"eager"``, and others.
      precision: either ``None`` (default), which means the default precision for
        the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
        ``Precision.HIGH`` or ``Precision.HIGHEST``).
      preferred_element_type: either ``None`` (default), which means the default
        accumulation type for the input types, or a datatype, indicating to
        accumulate results to and return a result with that datatype.

    Returns:
      array containing the result of the einstein summation.

    Examples:
      The mechanics of ``einsum`` are perhaps best demonstrated by example. Here we
      show how to use ``einsum`` to compute a number of quantities from one or more
      arrays. For more discussion and examples of ``einsum``, see the documentation
      of :func:`numpy.einsum`.


      >>> import saiunit as bu
      >>> M = bu.math.arange(16).reshape(4, 4) * bu.ohm
      >>> x = bu.math.arange(4) * bu.mA
      >>> y = bu.math.array([5, 4, 3, 2]) * bu.mV

      **Vector product**

      >>> bu.math.einsum('i,i', x, y)
      16.0 uW
      >>> bu.math.vecdot(x, y)
      16.0 uW

      Here are some alternative ``einsum`` calling conventions to compute the same
      result:

      >>> bu.math.einsum('i,i->', x, y)  # explicit form
      16.0 uW
      >>> bu.math.einsum(x, (0,), y, (0,))  # implicit form via indices
      16.0 uW
      >>> bu.math.einsum(x, (0,), y, (0,), ())  # explicit form via indices
      16.0 uW

      **Matrix product**

      >>> bu.math.einsum('ij,j->i', M, x)  # explicit form
      ArrayImpl([14., 38., 62., 86.], dtype=float32) * mvolt
      >>> bu.math.matmul(M, x)
      Array([14, 38, 62, 86], dtype=float32) * mvolt

      Here are some alternative ``einsum`` calling conventions to compute the same
      result:

      >>> bu.math.einsum('ij,j', M, x) # implicit form
      Array([14, 38, 62, 86], dtype=float32) * mvolt
      >>> bu.math.einsum(M, (0, 1), x, (1,), (0,)) # explicit form via indices
      Array([14, 38, 62, 86], dtype=float32)
      >>> bu.math.einsum(M, (0, 1), x, (1,))  # implicit form via indices
      Array([14, 38, 62, 86], dtype=float32) * mvolt

      **Outer product**

      >>> bu.math.einsum("i,j->ij", x, y)
      Array([[ 0,  0,  0,  0],
             [ 5,  4,  3,  2],
             [10,  8,  6,  4],
             [15, 12,  9,  6]], dtype=float32) * uwatt
      >>> bu.math.outer(x, y)
      Array([[ 0,  0,  0,  0],
             [ 5,  4,  3,  2],
             [10,  8,  6,  4],
             [15, 12,  9,  6]], dtype=float32) * uwatt

      Some other ways of computing outer products:

      >>> bu.math.einsum("i,j", x, y)  # implicit form
      Array([[ 0,  0,  0,  0],
             [ 5,  4,  3,  2],
             [10,  8,  6,  4],
             [15, 12,  9,  6]], dtype=float32) * uwatt
      >>> bu.math.einsum(x, (0,), y, (1,), (0, 1))  # explicit form via indices
      Array([[ 0,  0,  0,  0],
             [ 5,  4,  3,  2],
             [10,  8,  6,  4],
             [15, 12,  9,  6]], dtype=float32) * uwatt
      >>> bu.math.einsum(x, (0,), y, (1,))  # implicit form via indices
      Array([[ 0,  0,  0,  0],
             [ 5,  4,  3,  2],
             [10,  8,  6,  4],
             [15, 12,  9,  6]], dtype=float32) * uwatt

      **1D array sum**

      >>> bu.math.einsum("i->", x)  # requires explicit form
      Array(6, dtype=float32) * mA
      >>> bu.math.einsum(x, (0,), ())  # explicit form via indices
      Array(6, dtype=float32) * mA
      >>> bu.math.sum(x)
      Array(6, dtype=float32) * mA

      **Sum along an axis**

      >>> bu.math.einsum("...j->...", M)  # requires explicit form
      Array([ 6, 22, 38, 54], dtype=float32) * ohm
      >>> bu.math.einsum(M, (..., 0), (...,))  # explicit form via indices
      Array([ 6, 22, 38, 54], dtype=float32) * ohm
      >>> M.sum(-1)
      Array([ 6, 22, 38, 54], dtype=float32) * ohm

      **Matrix transpose**

      >>> y = bu.math.array([[1, 2, 3],
      ...                    [4, 5, 6]]) * bu.mV
      >>> bu.math.einsum("ij->ji", y)  # explicit form
      Array([[1, 4],
             [2, 5],
             [3, 6]], dtype=float32) * mV
      >>> bu.math.einsum("ji", y)  # implicit form
      Array([[1, 4],
             [2, 5],
             [3, 6]], dtype=float32) * mV
      >>> bu.math.einsum(y, (1, 0))  # implicit form via indices
      Array([[1, 4],
             [2, 5],
             [3, 6]], dtype=float32) * mV
      >>> bu.math.einsum(y, (0, 1), (1, 0))  # explicit form via indices
      Array([[1, 4],
             [2, 5],
             [3, 6]], dtype=float32) * mV
      >>> bu.math.transpose(y)
      Array([[1, 4],
             [2, 5],
             [3, 6]], dtype=float32) * mV

      **Matrix diagonal**

      >>> bu.math.einsum("ii->i", M)
      Array([ 0,  5, 10, 15], dtype=float32) * ohm
      >>> bu.math.diagonal(M)
      Array([ 0,  5, 10, 15], dtype=float32) * ohm

      **Matrix trace**

      >>> bu.math.einsum("ii", M)
      Array(30, dtype=float32) * ohm
      >>> bu.math.trace(M)
      Array(30, dtype=float32) * ohm

      **Tensor products**

      >>> x = bu.math.arange(30).reshape(2, 3, 5) * bu.mA
      >>> y = bu.math.arange(60).reshape(3, 4, 5) * bu.ohm
      >>> bu.math.einsum('ijk,jlk->il', x, y)  # explicit form
      Array([[ 3340,  3865,  4390,  4915],
             [ 8290,  9940, 11590, 13240]], dtype=float32) * volt
      >>> bu.math.tensordot(x, y, axes=((1, 2), (0, 2)))
      Array([[ 3340,  3865,  4390,  4915],
             [ 8290,  9940, 11590, 13240]], dtype=float32) * volt
      >>> bu.math.einsum('ijk,jlk', x, y)  # implicit form
      Array([[ 3340,  3865,  4390,  4915],
             [ 8290,  9940, 11590, 13240]], dtype=float32) * volt
      >>> bu.math.einsum(x, (0, 1, 2), y, (1, 3, 2), (0, 3))  # explicit form via indices
      Array([[ 3340,  3865,  4390,  4915],
             [ 8290,  9940, 11590, 13240]], dtype=float32) * volt
      >>> bu.math.einsum(x, (0, 1, 2), y, (1, 3, 2))  # implicit form via indices
      Array([[ 3340,  3865,  4390,  4915],
             [ 8290,  9940, 11590, 13240]], dtype=float32) * volt

      **Chained dot products**

      >>> w = bu.math.arange(5, 9).reshape(2, 2) * bu.mA
      >>> x = bu.math.arange(6).reshape(2, 3) * bu.ohm
      >>> y = bu.math.arange(-2, 4).reshape(3, 2) * bu.mV
      >>> z = bu.math.array([[2, 4, 6], [3, 5, 7]]) * bu.mA
      >>> bu.math.einsum('ij,jk,kl,lm->im', w, x, y, z)
      Array([[ 481,  831, 1181],
             [ 651, 1125, 1599]], dtype=float32) * metre ** 4 * kilogram ** 2 * second ** -6 * amp ** -1
      >>> bu.math.einsum(w, (0, 1), x, (1, 2), y, (2, 3), z, (3, 4))  # implicit, via indices
      Array([[ 481,  831, 1181],
             [ 651, 1125, 1599]], dtype=float32) * metre ** 4 * kilogram ** 2 * second ** -6 * amp ** -1
      >>> w @ x @ y @ z  # direct chain of matmuls
      Array([[ 481,  831, 1181],
             [ 651, 1125, 1599]], dtype=float32) * metre ** 4 * kilogram ** 2 * second ** -6 * amp ** -1
      >>> bu.math.multi_dot([w, x, y, z])
      Array([[ 481,  831, 1181],
             [ 651, 1125, 1599]], dtype=float32) * metre ** 4 * kilogram ** 2 * second ** -6 * amp ** -1

    .. _opt_einsum: https://github.com/dgasmith/opt_einsum
    """
    # operands = jax.tree.map(
    #     lambda x: x.factorless() if isinstance(x, Quantity) else x,
    #     operands,
    #     is_leaf=lambda x: isinstance(x, Quantity)
    # )
    operands = maybe_custom_array_tree(operands)
    operands = (subscripts, *operands)
    spec = operands[0] if isinstance(operands[0], str) else None
    path_type = 'optimal' if optimize is True else Unoptimized() if optimize is False else optimize

    # Allow handling of shape polymorphism
    non_constant_dim_types = {
        type(d) for op in operands if not isinstance(op, str)
        for d in np.shape(op) if not jax.core.is_constant_dim(d)
    }
    if not non_constant_dim_types:
        contract_path = opt_einsum.contract_path
    else:
        # ty = next(iter(non_constant_dim_types))
        # contract_path = _poly_einsum_handlers.get(ty, _default_poly_einsum_handler)
        contract_path = _default_poly_einsum_handler
    # using einsum_call=True here is an internal api for opt_einsum... sorry
    operands, contractions = contract_path(*operands, einsum_call=True, use_blas=True, optimize=path_type)
    contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)

    f_einsum = jax.jit(_einsum, static_argnums=(1, 2, 3), inline=True)
    if spec is not None:
        f_einsum = jax.named_call(f_einsum, name=spec)
    return f_einsum(operands,
                    contractions,
                    precision,
                    preferred_element_type)
