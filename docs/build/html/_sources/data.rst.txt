Database Script
=================

The toolbox needs data, and that data needs to be in a specific format. To define this format, the user must provide a dataset 
(temporal, spatial, or spatio-temporal) and describe an experimental setup. The preprocessing must implemented within the dataset class, 
for which we provide a template based on a PyTorch wrapper, as well as several examples. Below is a description of the dataset class.


Pyod Database Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root, split, transform=None, target_transform=None, download=False):
        # initialize the class and perform all the preprocessing

        def __getallitems__(self, index):
        # return the data and labels


Pytorch Database Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root, split, transform=None, target_transform=None, download=False):
        # initialize the class and perform all the preprocessing

        def __getitem__(self, index):
        # return the data and labels

        def __len__(self):
        # return the size of the dataset
