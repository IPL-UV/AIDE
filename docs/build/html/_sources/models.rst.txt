.. _section-identifier_models:



Available models
==================

    To keep the toolbox updated, we compile a compendium of available models that drews from several widely recognised, externally developed, and highly active libraries: 
    
PyOD
~~~~~

    For Outlier Detection, the `Python Outlier Detection <https://github.com/yzhao062/pyod>`_ (PyOD)  library was selected as the one best fitted to provide tested models and successfully curated products for multivariate data. PyOD includes individual, ensembles, and combinations of detection algorithms, as well as shallow and specific deep neural network methods.

    To use PyOD, specify the following fields on the configuration file:

    .. code-block:: yaml

        arch:
            user_defined: False
            type: Model_class_name (String) (eg. knn.KNN)
            args:
                ...
            input_model_dim: ...
            output_model_dim: ...
            step_samples_train: ...
            step_samples_evaluation: ...
    
SMP
~~~~

    The user can choose a wide range of state-of-the-art models from the `Segmentation Models Pytorch <https://github.com/qubvel/segmentation_models.pytorch>`_ (SMP) to address extreme event detection as a binary segmentation task. This library extends the powerful image labelling library of `PyTorch Image Models <https://github.com/huggingface/pytorch-image-models>`_  (timm). 
    
    The SMP library adds decoder structures (based on popular encoder-decoder architectures) to the models found in timm, and others, to be applied for image segmentation.

    To use SMP, specify the following fields on the configuration file:

    .. code-block:: yaml

        arch:
            user_defined: False 
            type: Model_architecture (String) (eg. Unet)
            params:
                encoder_name: Model_encoder (String) (eg. resnet34)
                ...
            input_model_dim: ...
            output_model_dim: ...

TSAI
~~~~~

    We incorporate the deep learning library for `Time Series and Sequences <https://github.com/timeseriesAI/tsai>`_ (TSAI) for handling 1D time series for the classification or impact assessment tasks.

    To use TSAI, specify the following fields on the configuration file:

    .. code-block:: yaml

        arch:
            user_defined: False
            type: tsai_models.Model_name (String) (eg. tsai_models.LSTM)
            params:
                out_len: ...
                ...
            input_model_dim: ...
            output_model_dim: ...

Gaussian Process Layer
~~~~~~~~~~~~~~~~~~~~~~~

    The AIDE toolbox provides the opportunity to implement advanced Deep Gaussian Process-based regression algorithms by integrating `GPyTorch <https://gpytorch.ai/>`_ . This is achieved by harmonizing Deep Learning (DL) models with a Gaussian Process (GP) as final layer. 

    To add a GP layer on top of your DL model, specify the fields below in the configuration file. Further information about the parameters and available options can be found in the `GPyTorch documentation <https://docs.gpytorch.ai/en/stable/>`_ . 

    .. code-block:: yaml

        loss: 
            user_defined: false 
            type: 'VariationalELBO'
            package: 'gpytorch.mlls'
            activation: 
                type: 'ApproximateGP' 
                input_size: ...
                likelihood: 
                    GaussianLikelihood: {}
                settings:
                    train:
                        num_likelihood_samples: ...
                    val:
                        num_likelihood_samples: ...
                    test:
                        num_likelihood_samples: ...
                params: 
                    variational_strategy:
                        GridInterpolationVariationalStrategy: 
                            grid_size: ...
                            grid_bounds: ...
                            variational_distribution:
                                CholeskyVariationalDistribution: {num_inducing_points: ...}
                            base: false 
                    mean: 
                        ConstantMean: {}
                    covar: 
                        RBFKernel: 
                            params: {}
                            nested: 
                                ScaleKernel:
                                    params: {} 
                                    nested: false   
                    scale_to_bounds: ...
                    output_distribution: 
                        MultivariateNormal: {}              
            masked: ...