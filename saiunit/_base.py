# ruff: noqa: F401, F403
# Backward-compatibility shim: re-exports everything from the split modules.
# External packages (e.g. brainunit) may still do `from saiunit._base import *`.

from ._base_decorators import *
from ._base_dimension import *
from ._base_getters import *
from ._base_quantity import *
from ._base_unit import *

# Private symbols that external code may reference directly
from ._base_decorators import (
    CallableAssignUnit,
    Missing,
    _assign_unit,
    _check_dim,
    _check_unit,
    _is_quantity,
    _remove_unit,
    missing,
)
from ._base_dimension import (
    _dim2index,
    _dimension_cache,
    _iclass_label,
    _ilabel,
    _is_tracer,
    get_dim_for_display,
)
from ._base_getters import (
    _assert_not_quantity,
    _short_str,
    _to_quantity,
    array_with_unit,
    change_printoption,
    has_same_unit,
    have_same_dim,
    is_scalar_type,
    unit_scale_align_to_first,
)
from ._base_quantity import (
    PyTree,
    StaticScalar,
    _IndexUpdateHelper,
    _IndexUpdateRef,
    _all_slice,
    _check_units_and_collect_values,
    _element_not_quantity,
    _process_list_with_units,
    _quantity_with_unit,
    _replace_with_array,
    _wrap_function_change_unit,
    _wrap_function_keep_unit,
    _wrap_function_remove_unit,
    _zoom_values_with_units,
    compat_with_equinox,
    compatible_with_equinox,
)
from ._base_unit import (
    _ambiguous_keys,
    _assert_same_base,
    _find_a_name,
    _find_standard_unit,
    _fmt_exp,
    _format_display_parts,
    _get_display_parts,
    _merge_display_parts,
    _normalise_display_parts,
    _select_preferred_standard_unit,
    _siprefixes,
    _standard_unit_preference_score,
    _standard_units,
    _to_unit,
    add_standard_unit,
)

__all__ = [
    'Dimension',
    'Unit',
    'Quantity',
    'DimensionMismatchError',
    'UnitMismatchError',
    'DIMENSIONLESS',
    'UNITLESS',
    'is_dimensionless',
    'is_unitless',
    'get_dim',
    'get_unit',
    'get_mantissa',
    'get_magnitude',
    'display_in_unit',
    'split_mantissa_unit',
    'maybe_decimal',
    'check_dims',
    'check_units',
    'assign_units',
    'fail_for_dimension_mismatch',
    'fail_for_unit_mismatch',
    'assert_quantity',
    'get_or_create_dimension',
]
