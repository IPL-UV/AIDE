.. _section-identifier_config:
.. role:: raw-html(raw)
   :format: html

Configuration File
==================

The communication between the user and the toolbox is performed through a configuration file containing a list of tunable system parameters.
This file is implemented in YAML, a simple and concise language that maps easily into native data structures. 
Its comprehensibility makes it accessible to developers and non-developers and facilitates tracking experiment changes over time. 

1. Defining the Task and Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    name: 'AIDE'
    # Addressed task, choices: Classification, OutlierDetection, ImpactAssessment
    task: ...
    # Use a previously saved model to skip the train phase (True/False)
    from_scratch: ...
    # Path to the best model, required if from_scratch: False
    best_run_path: ''
    # Directory to save model outputs and results
    save_path: "experiments/"

2. Defining the Dataset
~~~~~~~~~~~~~~~~~~~~~~~

Pointer to the Dataset class. This section can be customized by adding more variables. See Section :ref:`section-identifier_data` for more details on how to create your own Dataset class.

.. code-block:: yaml

    # Database and DataLoader definition
    data:
        name: ... # Dataset class name 
        data_dim: ...  # Data dimension
        input_size: ...  # Number of features
        features:  # Name of the features of the database
        features_selected: ...  # Features selected from the whole set of features
        num_classes: ...  # Number of categories in the database (drought, non-drought, e.g.)
        lon_slice_test: ...  # If visualization 2D enabled, min/max longitude coordinates (test)
        lat_slice_test: ...  # If visualization 2D enabled, min/max latitude coordinates (test)

3. Defining the Model
~~~~~~~~~~~~~~~~~~~~~

To specify the architecture to train, use the parameter :raw-html:`<font color="#008000">type</font>`. This can be a user-defined model or a model available in the toolbox (see Section :ref:`section-identifier_models` for more details). 

.. code-block:: yaml

    # Architecture definition
    arch:
        # Select a user-defined model (true/false)
        user_defined: ...
        # Type of architecture to be used (e.g., 'UNET')
        type: ...
        # Parameters to configure the architecture
        params:
            param_1: ...
        # Model input dimension (1: 1D, 2: 2D)
        input_model_dim: ...
        # Model output dimension (1: 1D, 2: 2D)
        output_model_dim: ...

4. Defining the Training
~~~~~~~~~~~~~~~~~~~~~~~~

This part of the configuration file allows for specifying the parameters of:

    - :raw-html:`<font color="#008000">loss</font>` function: Can be either custom or from a Python package. To choose from a Python package, set :raw-html:`<font color="#008000">user_defined</font>`: :raw-html:`<font color="#CC0033">False</font>` and specify loss name and package (e.g. :raw-html:`<font color="#008000">type</font>`: :raw-html:`<font color="#CC0033">'sigmoid_focal_loss'</font>` and :raw-html:`<font color="#008000">package</font>`: :raw-html:`<font color="#CC0033">'torchvision.ops' </font>`). For custom losses, see section Custom Loss in :ref:`section-identifier_advanced`
    - :raw-html:`<font color="#008000">optimizer</font>`: Defines the parameters to initialize the optimizer. :raw-html:`<font color="#008000">type</font>` can be any of `torch.optim <https://pytorch.org/docs/stable/optim.html>`_.
    - :raw-html:`<font color="#008000">trainer</font>`: Defines the parameters to initialize the Pytorch Lightning trainer. 
    - :raw-html:`<font color="#008000">dataloader</font>`: Defines the number of workers for the Pytorch Lightning dataloader. 

.. code-block:: yaml

    # Definition of the training stage
    implementation:
        # Loss function
        loss:
            user_defined: ... # Select user-defined model (true/false)
            type: ...  # Python class name
            package: ...  # Python package, none for user defined
            activation:
                type: ... # Activation before computing the loss function
            masked: ...  # Use masks to compute loss
            # Parameters for the loss function
            params:
                reduction: 'none'
                param_1: ...
        # Definition of the optimizer
        optimizer:
            type: ...  # Optimizer type
            lr: ...  # Learning rate
            weight_decay: ...  # Weight decay
            gclip_value: ...  # Gradient clipping values
        # Definition of PyTorch trainer
        trainer:
            accelerator: ...  # Choices: gpu/cpu
            devices: ... # 
            epochs: ...  # Number of epochs
            batch_size: ...  # Batch size
            monitor:  # Metric to be monitored during training
                split: ...  # Choices: train/val/test
                metric: ...  # Either loss or a metric's name to monitor for early stopping and checkpoints
            monitor_mode: ...  # Monitor mode (increase or decrease monitored metric value)
            early_stop: ...  # Number of steps to perform early stopping
        # Definition of PyTorch data loader
        data_loader:
            num_workers: ... # Number of CPUs to read the data in parallel

5. Defining the Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The toolbox provides several modules for evaluation at inference: metrics, visualizations, characterization and XAI. The metrics module will always be run while the other can be (de)activated. For more details of the capabilities of each module, please refeer to Section :ref:`section-identifier_evaluations`.

.. code-block:: yaml

    # Types of chosen evaluations, choices: Visualization, Characterization, XAI
    evaluation:
        metrics:
            Metric_1: {param_1: ...}  # Metric for evaluation, from torchmetrics. Metric_1 has to be the name of the metric as in torchmetrics docs
        
        visualization:
            activate: ... # Choices: True/False
            params:
                param_1: ...

        characterization:
            activate: ... # Choices: True/False
            params:
                param_1: ...

        xai:
            activate: ... # Choices: True/False
            params:
                param_1: ...

