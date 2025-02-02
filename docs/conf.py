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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# a_list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import shutil
import sys

package = os.environ.get('TARGET', 'saiunit')

sys.path.insert(0, os.path.abspath(os.path.curdir))
sys.path.insert(0, os.path.abspath('../'))

if package == 'brainunit':
    import make_brainunit_doc
    import make_brainunit_code

    print(make_brainunit_doc)
    print(make_brainunit_code)

    sys.path.insert(0, os.path.abspath('../brainunit'))

import auto_generater

auto_generater.main(package)

os.makedirs('apis/', exist_ok=True)
changelogs = [
    ('../changelog.md', 'apis/changelog.md'),
]
for source, dest in changelogs:
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copyfile(source, dest)

# -- Project information -----------------------------------------------------

project = package
copyright = f'2025, {package}'
author = 'BDP Ecosystem'

# The full version, including alpha/beta/rc tags
import saiunit
release = saiunit.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_nb',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_thebe',
    'sphinx_design'
]
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']

# source_suffix = '.rst'
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "sphinx_book_theme"
html_logo = f"_static/{package}.png"
html_title = f"{package}"
html_copy_source = True
html_sourcelink_suffix = ""
html_favicon = f"_static/{package}.png"
html_last_updated_fmt = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
jupyter_execute_notebooks = "off"
thebe_config = {
    "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
    "repository_branch": "master",
}

html_theme_options = {
    'show_toc_level': 2,
}

# -- Options for myst ----------------------------------------------
# Notebook cell execution timeout; defaults to 30.
autodoc_default_options = {
    'exclude-members': '....,default_rng',
}
