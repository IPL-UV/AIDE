# AIDE: Artificial Intelligence for Disentangling Extremes
Open source code for the detection, characterization and impact assessment of spatio-temporal extreme events

## Description
The AIDE toolbox consists of a full pipeline for the detection, characterization and impact assessment of extreme events using ML and computer vision tools. Its purpose is to provide an ML-based generic and flexible pipeline to detect, characterize and evaluate the impacts of extreme events based on spatio-temporal Earth and climate observational data. The pipeline consists of three different stages:

1) Data loading and pre-processing
2) ML architecture selection and training
3) Evaluation and visualization of results

## Usage and Documentation
```python
# 1) Create an empty pip environment
python3 -m venv ./aide_env 


# 2) Activate the environment
source ./aide_env/bin/activate


# 3) Install dependencies
pip install -r requirements.txt install libs


# 4) Run main.py of AIDE using a config file. Some examples:

# DroughtED database and K-Nearest Neighbors (KNN) model (from PyOD) 
python main.py --config=./configs/config_DroughtED_OutlierDetection.yaml

# DroughtED database and LSTM-based architecture (user-defined) 
python main.py --config=./configs/config_DroughtED_DeepLearning.yaml
```

Documentation can be found on [**Read the Docs**](http://aidextremes.readthedocs.org/en/latest), as well as in the [`docs/`](https://github.com/IPL-UV/AIDE/tree/main/docs) on the toolbox source.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Citation
If you use this code for your research, please cite **The AIDE Toolbox: AI for Disentangling Extreme Events**:

Gonzalez-Calabuig, M., Cortés-Andrés, J., Williams, T., Zhang, M., Pellicer-Valero, O.J., Fernández-Torres, M.Á., Camps-Valls, G.: The AIDE Toolbox: AI for Disentangling Extreme Events. IEEE Geoscience and Remote Sensing Magazine 12(3), 1–8 (2024). https://doi.org/10.1109/MGRS.2024.3382544

## Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement 101003469.

[<img src="AIDE/tutorials/imgs/xaida_logo.png" width="130" />](AIDE/tutorials/imgs/xaida_logo.png)

## License
[MIT](https://choosealicense.com/licenses/mit/)
