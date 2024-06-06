
.. _section-identifier_advanced:

.. role:: raw-html(raw)
   :format: html

Advanced features
==================

Debug option
~~~~~~~~~~~~~

    The toolbox includes a debug option which reduces the training to a single number of epochs. To enable this feature, set the parameter :raw-html:`<font color="#008000">debug</font>` to :raw-html:`<font color="#CC0033">True</font>` or to a :raw-html:`<font color="#CC0033">number</font>` in the configuration file. When this option is set, the training is reduced to 1 epoch. If set to a number, the test and inference is reduced to that number of samples.

    .. code-block:: yaml

        debug: True

    The debug option is recommended to use when the model is not performing as expected. 

Tensorboard
~~~~~~~~~~~

One feature that deserves special mention for its usefulness, mainly during the training
stage, is the AIDE toolbox-TensorBoard compatibility. `TensorBoard <https://www.tensorflow.org/tensorboard?hl=es-419>`_ is an invaluable tool for training DL architectures because it provides real-time insights into model performance, allowing users to visualise and monitor training metrics,
analyse model architectures, and identify performance bottlenecks. We show an example of the TensorBoard interface in the following figure during training,
to monitor the loss on the training and validation subsets of the model.

.. image:: _static/images/tensorboard.png
  :width: 800
  :align: center
  :alt: Tensorboard training
    
Custom models
~~~~~~~~~~~~~~

    Apart from the predefined models, the toolbox includes a user-defined option, i.e., it offers users the possibility of developing and including their own models. 

    The implementation in python should be included in the :raw-html:`<font color="#CC0033">AIDE</font>` > :raw-html:`<font color="#CC0033">user_defined</font>` > :raw-html:`<font color="#CC0033">models</font>` > :raw-html:`<font color="#CC0033">user_defined.py</font>` as a Pytorch :raw-html:`<font color="#CC0033">nn.Module</font>` class. 
    
    To use user-defined model, specify the following fields on the configuration file:

    .. code-block:: yaml

        arch:
            user_defined: True
            type: Model_class_name (String)
            params:
                ...
            input_model_dim: ...
            output_model_dim: ...

Custom Losses
~~~~~~~~~~~~~~
    
    To further tailor the model's training, the toolbox includes the possibility to ingrate user-defined loss functions. To incorporate your loss, create a file with your Python class :raw-html:`<font color="#CC0033">nn.Module</font>` at :raw-html:`<font color="#CC0033">AIDE</font>` > :raw-html:`<font color="#CC0033">user_defined</font>` > :raw-html:`<font color="#CC0033">losses</font>` > :raw-html:`<font color="#CC0033">your_loss_name.py</font>` and import the class at the :raw-html:`<font color="#CC0033">__init_.py</font>` file of that same folder. 
    
    Then, use the loss section on the configuration file to choose the new loss. The parameter :raw-html:`<font color="#008000">type</font>` has to match with the name of the Python class. The parameter :raw-html:`<font color="#008000">package</font>` has to be :raw-html:`<font color="#CC0033">'none'</font>` to perform a local search of the loss. 
    
    .. code-block:: yaml

        loss: 
            user_defined: True
            type: Loss_class_name (String)
            package: 'none'  
            activation: 
                type: ... 
            masked: ... (Options: True/False)
            params:
                ...

Custom Evaluation
~~~~~~~~~~~~~~~~~~

    At the final stage of developing your model, you may require alternative evaluations of your model. In this case, the toolbox provides an empty class where you can implement any complementary analysis of results. 
    This class can be found at :raw-html:`<font color="#CC0033">AIDE</font>` > :raw-html:`<font color="#CC0033">evaluators</font>` > :raw-html:`<font color="#CC0033">custom</font>` > :raw-html:`<font color="#CC0033">customEvaluator.py</font>`. 
    It will receive the variable :raw-html:`<font color="#008000">inference_outputs</font>`, a dictionary with the outputs of the model (x), the ground-truth (labels) and masks. 

    .. code-block:: python

        class CustomEvaluator():

            def __init__(self, config, model, dataloader):
                
                self.config = config
                self.model = model
                self.test_loader = dataloader
            
            def evaluate(self, inference_outputs):
                """

                Include your code here

                """
    
    
    This new evaluation block can then be used through the configuration file as follows: 

    .. code-block:: yaml

        custom:
            activate: false
            params: ~ 

    The parameter :raw-html:`<font color="#008000">params</font>` allows for the definition of any extra variables that you may want to define from outside the toolbox. 