Configuration File
==================

The communication between the user and the toolbox is performed through a configuration file containing a list of tunable system parameters.
This file is implemented in YAML, a simple and concise language that maps easily into native data structures. 
Its comprehensibility makes it accessible to developers and non-developers and facilitates tracking experiment changes over time. 

1. Defining the Task and Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    name: 'AIDE'
    # Addressed task, choices: Classification, OutlierDetection
    task: 'Classification'
    # Use a previously saved model to skip the train phase
    from_scratch: ...
    # Path to the best model, required if from_scratch: false
    best_run_path: ''
    # Directory to save model outputs and results
    save_path: "experiments/"

2. Defining the Dataset
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # Database and DataLoader definition
    data:
        name: ...
        data_dim: ...  # Data dimension
        input_size: ...  # Number of features
        features:  # Name of the features of the database
        features_selected: ...  # Features selected from the whole set of features
        num_classes: ...  # Number of categories in the database (drought, non-drought, e.g.)
        time_aggregation: ...  # Time aggregation for visualization purposes (true/false)
        lon_slice_test: ...  # If visualization 2D enabled, min/max longitude coordinates (test)
        lat_slice_test: ...  # If visualization 2D enabled, min/max latitude coordinates (test)

3. Defining the Model
~~~~~~~~~~~~~~~~~~~~~

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

.. code-block:: yaml

    # Definition of the training stage
    implementation:
        # Loss function
        loss:
            # Select a user-defined loss (true/false)
            user_defined: ...
            type: ...  # 'binary_cross_entropy' # Type
            package: ...  # 'torch.nn.functional' #
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
            num_gpus: ...  # Number of GPUs to be used
            epochs: ...  # Number of epochs
            batch_size: ...  # Batch size
            monitor:  # Metric to be monitored during training
                split: ...  # Choices: train/val/test
                metric: ...  # Either loss or a metric's name to monitor for early stopping and checkpoints
            monitor_mode: ...  # Monitor mode (increase or decrease monitored metric value)
            early_stop: ...  # Number of steps to perform early stopping
            save_dir: "experiments/"  # Directory to save model outputs and results
        # Definition of PyTorch data loader
        data_loader:
            num_workers: ...

5. Defining the Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # Types of chosen evaluations, choices: Visualization, Characterization, XAI
    evaluation:
        metrics:
            Metric_1: {param_1: ...}  # Metric for evaluation, from torchmetrics.
                                     # Metric_1 has to be the name of the metric as in torchmetrics docs
        # Perform visualization, characterization, and XAI (true/false)
        visualization:
            activate: true
            params:
                param_1: ...

        characterization:
            activate: true
            params:
                param_1: ...

        xai:
            activate: true
            params:
                param_1: ...

