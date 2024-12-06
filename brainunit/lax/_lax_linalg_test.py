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


lax_linear_algebra_keep_unit_unary = [
    'eigh',
]

lax_linear_algebra_change_unit_unary = [
    'cholesky',
]

lax_linear_algebra_keep_unit_unary_return_2 = [
    'eigh', 'hessenberg', 'qr',
]

lax_linear_algebra_keep_unit_unary_return_3 = [
    'eig', 'lu',
]

lax_linear_algebra_qdwh = [
    'qdwh',
]

lax_linear_algebra_schur = [
    'schur',
]

lax_linear_algebra_svd = [
    'svd',
]

lax_linear_algebra_tridiagonal = [
    'tridiagonal',
]

lax_linear_algebra_binary = [
    'householder_product', 'triangular_solve',
]

lax_linear_algebra_nary = [
    'tridiagonal_solve',
]
