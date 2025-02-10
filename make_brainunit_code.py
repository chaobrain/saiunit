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

import glob
import os
import shutil
import sys


def make(root_dir):
    sys.path.insert(0, os.path.abspath(root_dir))

    import saiunit

    # base directory
    root_dir = os.path.abspath(root_dir)
    brainunit_dir = os.path.abspath(os.path.join(root_dir, 'brainunit'))
    os.makedirs(brainunit_dir, exist_ok=True)
    saiunit_dir = os.path.abspath(os.path.join(root_dir, './'))
    # print('root_dir = ', root_dir)
    # print('brainunit_dir = ', brainunit_dir)
    # print('saiunit_dir = ', saiunit_dir)
    py_files = glob.glob(os.path.join(saiunit_dir, 'saiunit', '**', '*.py'), recursive=True)

    # def create_brainunit_package():
    # read template
    with open(os.path.join(brainunit_dir, 'pyfile.template'), 'r') as f:
        template = f.read()

    with open(os.path.join(brainunit_dir, 'pyfile.template'), 'r') as f:
        pyfile = f.read()

    # create '__init__.py' file
    for file in py_files:

        su_path, filename = os.path.split(file.replace(saiunit_dir, '.'))
        su_path = su_path.replace('\\', '/')
        bu_path = su_path.replace('saiunit', 'brainunit')
        filename = filename.replace('\\', '/')
        os.makedirs(os.path.join(brainunit_dir, f'{bu_path}'), exist_ok=True)
        if filename == '__init__.py':
            # print(bu_path, filename)
            shutil.copyfile(file, os.path.join(brainunit_dir, f'{bu_path}/{filename}'))

        else:
            filename = filename.split(".")[0]
            if filename.endswith('_test'):
                continue
            bu_module = f'{su_path}.{filename}'
            bu_module = bu_module.replace('./', '').replace('/', '.')
            with open(os.path.join(brainunit_dir, f'{bu_path}/{filename}.py'), 'w') as f:
                f.write(pyfile.format(path=bu_module))

    # # create 'a.py' file
    # for path, names in paths.items():
    #     if len(path.split('.')) == 1:
    #         continue
    #     bu_path = path.replace('.', '/').replace('saiunit', 'brainunit')
    #     if os.path.isdir(os.path.join(brainunit_dir, bu_path)):
    #         continue
    #
    #     names = [f'{name} as {name}' for name in names]
    #     filename = os.path.join(brainunit_dir, f'{bu_path}.py')
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     with open(filename, 'w') as file:
    #         file.write(template.format(path=path, functions=",\n    ".join(names)))

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
