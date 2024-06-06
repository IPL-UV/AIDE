#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from datetime import datetime
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch
import torch.nn.functional as F
import dask
dask.config.set(scheduler='synchronous')

class DroughtED(torch.utils.data.Dataset):
    """This class is used to load the DroughtED dataset using PyTorch's Dataset class.
    
    :param config: The configuration file
    :type config: dict
    :param period: The period of the dataset to load, defaults to 'train'
    :type period: str, optional
    :param window_size: The window size of the dataset, defaults to 12
    :type window_size: int, optional
    :param features_selected: The features to be used, defaults to ['NDVI', 'precipitation', 'temperature']
    :type features_selected: list, optional
    :param num_classes: The number of classes, defaults to 6
    :type num_classes: int, optional
    :param train_slice: The time period of the training set, defaults to {'start': 1982, 'end': 2006}
    :type train_slice: dict, optional
    :param val_slice: The time period of the validation set, defaults to {'start': 2007, 'end': 2011}
    :type val_slice: dict, optional
    :param test_slice: The time period of the test set, defaults to {'start': 2012, 'end': 2016}
    :type test_slice: dict, optional
    :param id2class: The dictionary of class IDs, defaults to {'None': 0, 'D0': 1, 'D1': 2, 'D2': 3, 'D3': 4, 'D4': 5}
    :type id2class: dict, optional
    :param weights: The weights of each class, defaults to [100-ones_percentage, ones_percentage]
    :type weights: list, optional
    :param dfs: The dataframe of the dataset, defaults to None
    :type dfs: pandas.core.frame.DataFrame, optional
    :param X: The input data, defaults to None
    :type X: numpy.ndarray, optional
    :param y: The output data, defaults to None
    :type y: numpy.ndarray, optional
    :param class_bound: The boundary of the classes, defaults to 2
    :type class_bound: int, optional
    :param start: The start date of the dataset, defaults to None
    :type start: datetime, optional
    :param end: The end date of the dataset, defaults to None
    :type end: datetime, optional
    :param dates: The dates of the dataset, defaults to None
    :type dates: pandas.core.indexes.datetimes.DatetimeIndex, optional
    :param ones_percentage: The percentage of the positive class in the binarise function, defaults to None
    :type ones_percentage: float, optional
    """
    
    def __init__(self, config, period = 'train'):
        """Processing of the dataset.
        :param config: The configuration file
        :type config: dict
        :param period: The period of the dataset to load, defaults to 'train'
        :type period: str, optional
        """
        self.config = config['data'] # configuration file
        self.period = period # train/val/test
        self.window_size = self.config['window_size']
        self.features_selected = self.config['features_selected']
        self.num_classes = self.config['num_classes']
        self.train_slice = self.config['train_slice']
        self.val_slice = self.config['val_slice']
        self.test_slice = self.config['test_slice']
        
        # Class IDs
        self.id2class = {
            'None': 0,
            'D0': 1,
            'D1': 2,
            'D2': 3,
            'D3': 4,
            'D4': 5,
        }

        self.scaler_dict = {}
        self.scaler_dict_past = {}

        # Read database filenames
        self.dfs = self.read_database_files()
        if self.num_classes == 2:
            self.binarize_data() 
       
        # Data pre-processing
        data = self.loadXY(period = self.period)
        self.X, self.y = data[0], data[1]
   
        ones_percentage = np.count_nonzero(self.y)*100/len(self.X) 
        self.weights = [100-ones_percentage, ones_percentage]
        print(self.weights)
    
    def read_database_files(self):
        """Read the database files.
        
        :return: The dataframe of the dataset
        :rtype: pandas.core.frame.DataFrame
        """
        files = {}

        for dirname, _, filenames in os.walk(os.path.join(self.config['root'], self.config['data_file'])):
            for filename in filenames:
                if 'val' in filename: 
                    files['val'] = os.path.join(dirname, filename)
                if 'test' in filename: 
                    files['test'] = os.path.join(dirname, filename)

        df_list = []
        for split in ['val', 'test']:
            df = pd.read_csv(files[split]).set_index(['fips', 'date'])
            df_list.append(df)

        dfs = pd.concat(df_list, axis=0, ignore_index=False)
        self.dfs = self.select_slice(dfs)
        return self.dfs 

    def select_slice(self, dfs):
        """Select the time period of the dataset.
        
        :param dfs: The dataframe of the dataset
        :type dfs: pandas.core.frame.DataFrame
        :return: Slice of the dataframe
        :rtype: pandas.core.frame.DataFrame
        """
        dates = dfs.index.get_level_values(1)
        slice = getattr(self, self.period+'_slice')
        start = slice['start']
        end = slice['end']
        dfs = dfs[(dates > start) & (dates < end)].copy()
        return dfs

    def binarize_data(self):
        """Binaries the data based off a threshold set by the class_bound parameter.

        :param class_bound: The boundary of the classes, defaults to 2
        :type class_bound: int, optional
        return: The binarised data
        :rtype: pandas.core.frame.DataFrame
        """
        class_bound = self.config['class_bound']
        self.dfs.loc[self.dfs["score"] < class_bound] = 0
        self.dfs.loc[self.dfs["score"] >= class_bound] = 1
                                    
    def loadXY(self,
        period='train', # data period to load
        random_state=42, # keep this at 42
        normalize=True, # standardize data
    ):
        """Load the data and split it into X and y, and a conditional normalisation statement.
        
        :param period: The period of the dataset to load, defaults to 'train'
        :type period: str, optional
        :return: The input and output data
        :rtype: numpy.ndarray, numpy.ndarray
        """
        ## Initialize random state
        if random_state is not None:
            np.random.seed(random_state)

        ## Get column's name for the meteorological indicatos
        time_data_cols = sorted([c for c in self.dfs.columns if c not in ["fips", "date", "score"]])
        time_data_cols = [time_data_cols[int(i)] for i in self.features_selected]

        ## Filter all data point that do not have a score defined (NaN value)
        score_df = self.dfs.dropna(subset=["score"])

        ## Create variables to store the data
        max_buffer = 7
        X_time = np.empty((len(self.dfs) // self.window_size , self.window_size, len(time_data_cols)))
        y_target = np.empty((len(self.dfs) // self.window_size , 1))
        
        count = 0
        ## Iteration over the fips
        # print(self.dfs.index.get_level_values(0))
        for fips in tqdm(score_df.index.get_level_values(0).unique()):
        # for fips in tqdm(score_df.index.get_level_values(0).unique().values[:100]):

            ## Select randomly where to start sampling
            if random_state is not None and self.window_size != 1:
                start_i = np.random.randint(1, self.window_size)
            else:
                start_i = 1
            
            ## Get all samples with the fips ID that we are evaluating 
            fips_df = self.dfs[(self.dfs.index.get_level_values(0) == fips)]
            X = fips_df[time_data_cols].values
            y = fips_df["score"].values

            for idx, i in enumerate(range(start_i, len(y) - (self.window_size + max_buffer), self.window_size)):
                ## Save window of samples
                X_time[count, :, : len(time_data_cols)] = X[i : i + self.window_size]
                
                ## 
                temp_y = y[i + self.window_size : i + self.window_size + max_buffer]
                y_target[count] = int(np.around(np.array(temp_y[~np.isnan(temp_y)][0])))

                count += 1

        print(f"loaded {count} samples")

        # Normalize the data
        if normalize:
            X_time = self.normalize(X_time)
        data = [X_time[:count], y_target[:count]]


        return tuple(data)
    
    def interpolate_nans(self, padata, pkind='linear'):
        """Interpolate over nans in a 1D array.
        
        :param padata: 1D array with nans to interpolate over
        :type padata: numpy.ndarray
        :param pkind: Interpolation method, defaults to 'linear'
        :type pkind: str, optional
        :return: Interpolated array
        :rtype: numpy.ndarray
        
        see: https://stackoverflow.com/a/53050216/2167159
        """
        aindexes = np.arange(padata.shape[0])
        agood_indexes, = np.where(np.isfinite(padata))
        f = interp1d(agood_indexes
               , padata[agood_indexes]
               , bounds_error=False
               , copy=False
               , fill_value="extrapolate"
               , kind=pkind)
        return f(aindexes)
                                                                                
            
    def normalize(self, X_time):
        """Normalise the data using a standard scaler.
        
        :param X_time: The input data
        :type X_time: numpy.ndarray
        :return: The normalised data
        :rtype: numpy.ndarray
        """
        X_time_train = self.loadXY(period = 'train', normalize=False)
        
        X_time_train = X_time_train[0]
        for index in tqdm(range(X_time.shape[-1])):
            # Fit data    
            # self.scaler_dict[index] = RobustScaler().fit(X_time_train[:, :, index].reshape(-1, 1))
            self.scaler_dict[index] = StandardScaler().fit(X_time_train[:, :, index].reshape(-1, 1))
            X_time[:, :, index] = (
                self.scaler_dict[index]
                .transform(X_time[:, :, index].reshape(-1, 1))
                .reshape(-1, X_time.shape[-2])
            )
        X_time = np.clip(X_time, a_min=-3., a_max=3.) / 3.
        index = 0
        return X_time                                          
                                                    
    def __getitem__(self, index):
        """Return item from dataset to pass to the dataLoader.
        
        :param index: The index of the item to return
        :type index: int
        :return: The item
        :rtype: dict
        """  
        if self.num_classes == 2:
            return {'x': torch.Tensor(self.X[index, :]).permute(1,0) , 'labels': torch.Tensor(self.y[index])}
        else:
            return {'x': torch.Tensor(self.X[index, :]), 
                'labels': F.one_hot(torch.Tensor(self.y[index]).squeeze().long(),num_classes=self.num_classes)}

    def __getallitems__(self):
        """Return all items from dataset to pass to the dataLoader.
        
        :return: All items in the dataset
        :rtype: dict
        """
        return {'x': np.transpose(np.squeeze(self.X), (1,0)), 'labels': np.squeeze(self.y)}
        
    def __len__(self):
        """Return the length of the dataset.
        
        :return: The length of the dataset (number of samples)
        :rtype: int
        """
        return len(self.X)