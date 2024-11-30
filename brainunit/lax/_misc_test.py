import jax.numpy as jnp
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

lax_misc = [
    'after_all', 'reduce', 'reduce_precision',

    # getting attribute funcs
    'broadcast_shapes',
]