Code for processing data can be hard to maintain and version, therefore pytorch decouples the data loader from the rest of the code. This modularity helps maintain a manageable code. 

PyTorch provides: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow you to use preloaded datasets as well as your own data. 

`Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.

The task is to detect drought in Syria using two hydrometeorological variables as input taken from the [Earth System Data Lab](https://www.earthsystemdatalab.net/index.php/documentation/data-sets/). This includes 2m air temperature from the [ERA5 database](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5), and root moisture from the [GLEAM](https://www.gleam.eu/). Both of these variables are stored as zarr files which contain coordinates and time measurements.

The ground truth data comes from the [Geocoded Disasters (GDIS) Dataset](https://sedac.ciesin.columbia.edu/data/set/pend-gdis-1960-2018) from NASA’s Socioeconomic Data and Applications Center (SEDAC). This data consists of masks, coordinates and time.
