# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import os
import sys


def make(root_dir):
    sys.path.insert(0, os.path.abspath(root_dir))

    import saiunit

    # base directory
    root_dir = os.path.abspath(root_dir)
    brainunit_dir = os.path.abspath(os.path.join(root_dir, 'brainunit'))
    os.makedirs(brainunit_dir, exist_ok=True)

    # def create_brainunit_setup():
    # pyproject
    with open(os.path.join(brainunit_dir, 'pyproject.toml.template'), 'r') as f:
        pyproject = f.read()
    with open(os.path.join(brainunit_dir, 'pyproject.toml'), 'w') as f:
        f.write(pyproject.replace('saiunit==', f'saiunit=={saiunit.__version__}'))

    # setup
    with open(os.path.join(brainunit_dir, 'setup.py.template'), 'r') as f:
        setup = f.read()
    with open(os.path.join(brainunit_dir, 'setup.py'), 'w') as f:
        f.write(setup.replace('saiunit==', f'saiunit=={saiunit.__version__}'))


if __name__ == '__main__':
    make('.')
