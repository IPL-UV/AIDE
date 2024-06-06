.. _section-identifier_data:
.. role:: raw-html(raw)
   :format: html

Database 
========

The user must provide a dataset (temporal, spatial, or spatio-temporal) and describe an experimental setup to adapt the toolbox for their research. The preprocessing must implemented within the dataset class, 
for which we provide a template based on a PyTorch wrapper, as well as several examples in the Tutorials section. Below is a description of the dataset class.

The initialization function receives the following parameters:

    - :raw-html:`<font color="#008000">config</font>`: This parameter contains a dictionary with the :raw-html:`<font color="#CC0033">data</font>` section of the configuration (yaml) file. See more details at Section 2 of :ref:`section-identifier_config`. 
    - :raw-html:`<font color="#008000">period</font>`: Parameter to specify the data split. Usage: The toolbox internally calls the Dataset class three times, the value of period will change from :raw-html:`<font color="#CC0033">train</font>` to :raw-html:`<font color="#CC0033">val</font>` and, finally, to :raw-html:`<font color="#CC0033">test</font>`. Use this fixed value parameter to specify the data to be loaded per iteration.    


Pyod Database Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, config, period='train'):
        # initialize the class and perform all the preprocessing

        def __getallitems__(self, index):
        # return the data and labels


Pytorch Database Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, config, period='train'):
        # initialize the class and perform all the preprocessing

        def __getitem__(self, index):
        # return the data and labels

        def __len__(self):
        # return the size of the dataset
