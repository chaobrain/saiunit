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


def make(root_dir):
    # base directory
    root_dir = os.path.abspath(root_dir)
    saiunit_dir = os.path.abspath(os.path.join(root_dir, '../'))

    # Define the source and destination directories
    source_dir = os.path.abspath(os.path.join(root_dir, './docs/'))
    destination_dir = os.path.abspath(os.path.join(root_dir, './docs/'))

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Define the file extensions to process
    file_extensions = ['.md', '.rst', '.ipynb']

    # def create_brainunit_doc():
    # Iterate over the files in the source directory
    for ext in file_extensions:
        # print(source_dir)
        # print(os.path.exists(source_dir))
        # print(glob.glob(os.path.join(source_dir, '**', f'*{ext}'), recursive=True))
        for file_path in glob.glob(os.path.join(source_dir, '**', f'*{ext}'), recursive=True):
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Replace occurrences of 'saiunit' with 'brainunit'
            modified_content = content.replace('saiunit', 'brainunit')

            # Define the destination file path
            relative_path = os.path.relpath(file_path, source_dir)
            destination_path = os.path.join(destination_dir, relative_path)

            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            # Write the modified content to the destination file
            with open(destination_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)

            print(destination_path)

