Usage
=====

.. _installation:

Installation
------------

To use the XAIDA4Detection toolbox, follow the steps here below:

.. code-block:: console

   # 1) Create an empty pip environment
   python3 -m venv ./xaida_env 


   # 2) Activate environment
   source ./xaida_env/bin/activate


   # 3) Install dependencies
   pip install -r requirements_xaida.txt install libs


   # 4) Run main.py of XAIDA4Detection using a config file. Some examples:

   # DroughtED database and K-Nearest Neighbors (KNN) model (from PyOD) 
   python main.py --config=/configs/config_DroughtED_PYOD.yaml

   # DroughtED database and LSTM-based architecture (user-defined) 
   python main.py --config=/configs/config_DroughtED_LSTM.yaml
