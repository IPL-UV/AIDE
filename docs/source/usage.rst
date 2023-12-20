Quick Start
=====

To install the AIDE toolbox, follow the steps here below:

.. code-block:: console

   # 1) Create an empty pip environment
   python3 -m venv ./aide_env 


   # 2) Activate environment
   source ./aide_env/bin/activate


   # 3) Install dependencies
   pip install -r requirements.txt install libs

There are a number of user preparation which needs to be done before using the AIDE toolbox:
 - Provide the data to be used (e.g. download DroughtED data from https://doi.org/10.5281/zenodo.4284815)
 - Define the database script to be used (e.g. use existing DroughtED, EDSL, etc. or create from the template)
 - Define task (Outlier Detection, Detection, Impact Assement) and configure your experiment in the config file.

To run the toolbox, once the above steps are done, simply call the main.py script with your config file as argument:

.. code-block:: console

   # 4) Run main.py of AIDE using a config file. Some examples:

   # DroughtED database and K-Nearest Neighbors (KNN) model (from PyOD) 
   python main.py --config=/configs/config_DroughtED_OutlierDetection.yaml

   # DroughtED database and LSTM-based architecture (user-defined) 
   python main.py --config=/configs/config_DroughtED_DeepLearning.yaml
