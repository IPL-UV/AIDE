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
copyright = '2023, Cortés-Andrés, J., Gonzalez-Calabuig, M., Zhang, M., Williams, T., Pellicer-Valero, O. J., Fernández-Torres, M.-Á., and Camps-Valls, G.'
author = 'Cortés-Andrés, J., Gonzalez-Calabuig, M., Zhang, M., Williams, T., Pellicer-Valero, O. J., Fernández-Torres, M.-Á., and Camps-Valls, G.'
release = '0.0.1'

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
html_theme = 'python_docs_theme'
html_static_path = [] #['_static']
