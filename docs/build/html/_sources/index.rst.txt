.. AIDE documentation master file, created by
   sphinx-quickstart on Tue May 23 19:48:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AIDE's Documentation!
================================

Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project.

About AIDE
^^^^^^^^^^
Artificial Intelligence for Disentangling Extremes (AIDE) toolbox that allows for tackling generic problems of detection and impact assessment of 
events such as tropical cyclones and severe convective storms, heat waves, and droughts, as well as persistent winter extremes, among others. 
The open-source toolbox integrates advanced ML models, ranging in complexity, assumptions, and sophistication, and can yield spatio-temporal explicit
output maps with probabilistic heatmap estimates. We included supervised and unsupervised algorithms, deterministic and probabilistic, convolutional
and recurrent neural networks, and detection methods based on density estimation. The toolbox is intended for scientists, engineers, and students 
with basic knowledge of extreme events detection, outlier detection techniques, and Deep Learning (DL), as well as Python programming with basic 
packages (Numpy, Scikit-learn, Matplotlib) and DL packages (PyTorch, PyTorch Lightning).

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started:

   usage
   examples
   data
   config

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation:

   api_ref

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials:

   detection
   impacts


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`