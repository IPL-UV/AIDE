Usage
=====

.. _installation:

Installation
------------

To use the AIDE toolbox, follow the steps here below:

.. code-block:: console

   # 1) Create an empty pip environment
   python3 -m venv ./aide_env 


   # 2) Activate environment
   source ./aide_env/bin/activate


   # 3) Install dependencies
   pip install -r requirements.txt install libs


   # 4) Run main.py of AIDE using a config file. Some examples:

   # DroughtED database and K-Nearest Neighbors (KNN) model (from PyOD) 
   python main.py --config=/configs/config_DroughtED_OutlierDetection.yaml

   # DroughtED database and LSTM-based architecture (user-defined) 
   python main.py --config=/configs/config_DroughtED_DeepLearning.yaml
