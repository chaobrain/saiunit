import jax.numpy as jnp
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

lax_array_manipulation = [
    'slice', 'dynamic_slice', 'dynamic_update_slice', 'gather',
    'index_take', 'slice_in_dim', 'index_in_dim', 'dynamic_slice_ind_dim', 'dynamic_index_in_dim',
    'dynamic_update_slice_in_dim', 'dynamic_update_index_in_dim',
    'sort', 'sort_key_val',
    ]

lax_keep_unit_unary = [
    'neg',
    'cummax', 'cummin', 'cumsum',
    'scatter', 'scatter_add', 'scatter_sub', 'scatter_mul', 'scatter_min', 'scatter_max', 'scatter_apply',
]

lax_keep_unit_binary = [
    'sub', 'complex', 'pad',
]
lax_keep_unit_nary = [
    'clamp',
]

lax_type_conversion = [
    'convert_element_type', 'bitcast_convert_type',
]

lax_keep_unit_return_Quantity_index = [
    'approx_max_k', 'approx_min_k', 'top_k',
]

lax_broadcasting_arrays = [
    'broadcast', 'broadcast_in_dim', 'broadcast_to_rank',
]
