# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../AIDE/'))
import sphinx_rtd_theme

project = 'AIDE'
copyright = '2024, Gonzalez-Calabuig, M., Cortés-Andrés, J., Williams, T., Zhang, M., Pellicer-Valero, O.J., Fernández-Torres, M.Á., Camps-Valls, G.'
author = 'Gonzalez-Calabuig, M., Cortés-Andrés, J., Williams, T., Zhang, M., Pellicer-Valero, O.J., Fernández-Torres, M.Á., Camps-Valls, G.'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme', 
    'python_docs_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'python_docs_theme'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = ['custom.css',]
