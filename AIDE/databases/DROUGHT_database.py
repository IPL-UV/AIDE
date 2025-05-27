#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import xarray as xr
import pandas as pd
import torch
import torch.utils.data 
eps = 1e-7

import dask
dask.config.set(scheduler='synchronous')

# solve the multi-process error
import matplotlib.pyplot as plt
#plt.switch_backend('agg')


class DROUGHT(torch.utils.data.Dataset):
    """This class is used to load the DROUGHT dataset using PyTorch's Dataset class.

    """
    def __init__(self, config, period='train', precision=32):
        """Initialization of the dataset.

        :param config: The configuration file.
        :type config: Dict
        :param period: The period of the dataset to load, defaults to 'train'.
        :type period: Str, optional
        :param precision: The precision of float variables, defaults to 64.
        :type precision: Int, optional
        """
        self.task = config['task']
        self.total_config = config
        self.config = config['data']
        self.period = period
        if self.period == 'train':
            self.mode = self.period
        else:
            self.mode = 'test'
        self.precision = precision
        
        print("Constructing DROUGHT {}...".format(self.period))
        
        # ESDC data
        data = []
        data_path= os.path.join(self.config['root'], self.config['data_file'])
        assert os.path.exists(data_path), \
            f'Data path {data_path} does not exist, check "root" and "data_file" in the configuration file'

        self.dataset = xr.open_dataset(data_path)
        # Features
        self.data = self.dataset[np.array(self.config['features'])[self.config['features_selected']]]
        # Temporal slice
        start, end = eval(self.config['time_slice'])
        self.data = self.data.sel(time=slice(start,end))
        self.data_dim = int(self.config['data_dim'])

        # Labels for droughts
        
        if self.task == "Classification" or self.task == "OutlierDetection":
            self.labels = xr.open_dataset(os.path.join(self.config['root'], self.config['labels_file']))
        elif self.task == "ImpactAssessment":
            self.labels = xr.open_dataset(data_path)
        # Temporal slice
        start, end = eval(self.config['time_slice'])
        self.labels = self.labels.sel(time=slice(start,end)) 

        ####################################################################
        if self.task == 'Classification':
            start, end = eval("'1-January-2003','31-December-2018'")
        elif self.task == 'ImpactAssessment':
            start, end = eval("'1-January-2003','31-December-2009'")
        self.dataset = self.dataset.sel(time=slice(start,end))     
        self.dataset_labels = (self.labels['mask_EMDAT'].sel(time=slice(start,end)).copy(deep = True) > 0)      
        
        # Train/Validation/Test sets
        if self.period == 'train':
            data_aux = []
            labels_aux = []
            for interval in np.arange(np.size(self.config['train_slice']['exp'+str(self.config['nexp'])])):
                start, end = eval(self.config['train_slice']['exp'+str(self.config['nexp'])][interval])
                data_aux.append(self.data.sel(time=slice(start,end)))
                labels_aux.append(self.labels.sel(time=slice(start,end)))
            self.data = xr.concat(data_aux,dim='time').chunk({'time': 1})
            self.labels = xr.concat(labels_aux,dim='time').chunk({'time': 1})
        elif self.period == 'val':
            data_aux = []
            labels_aux = []
            for interval in np.arange(np.size(self.config['val_slice']['exp'+str(self.config['nexp'])])):
                start, end = eval(self.config['val_slice']['exp'+str(self.config['nexp'])][interval])
                data_aux.append(self.data.sel(time=slice(start,end)))
                labels_aux.append(self.labels.sel(time=slice(start,end)))
            self.data = xr.concat(data_aux,dim='time').chunk({'time': 1})
            self.labels = xr.concat(labels_aux,dim='time').chunk({'time': 1})
        elif self.period == 'test':
            data_aux = []
            labels_aux = []
            for interval in np.arange(np.size(self.config['test_slice']['exp'+str(self.config['nexp'])])):
                start, end = eval(self.config['test_slice']['exp'+str(self.config['nexp'])][interval])
                data_aux.append(self.data.sel(time=slice(start,end)))
                labels_aux.append(self.labels.sel(time=slice(start,end)))
            self.data = xr.concat(data_aux,dim='time').chunk({'time': 1})
            self.labels = xr.concat(labels_aux,dim='time').chunk({'time': 1})
        
        # Standardization based on climatology
        self.climatology_clipping = self.config['climatology_clipping']

        if self.task == 'ImpactAssessment':
            features = self.config['features'].copy()
            features.append(self.config['target_feature'])
            features_idx = self.config['features_selected'].copy()
            features_idx.append(len(self.config['features_selected']))
            list_of_variables = np.array(features)[features_idx]
        else:
            list_of_variables = np.array(self.config['features'])[self.config['features_selected']]
        

        if self.mode == 'train' \
            and not (os.path.isdir(self.config['climatology_mean_root']+'_'+'exp'+str(self.config['nexp'])) \
                     and os.path.isdir(self.config['climatology_std_root']+'_'+'exp'+str(self.config['nexp']))):  
            # # Climatology
            self._compute_climatology(list_of_variables)
            self.data = self._apply_climatology(self.data, np.array(self.config['features'])[self.config['features_selected']])
            if self.task == 'ImpactAssessment':
                self.labels = self._apply_climatology(self.labels, self.config['target_feature'])
        
        else:
            # Climatology
            self.climatology_mean = xr.open_mfdataset(self.config['climatology_mean_root']+'_'+'exp'+str(self.config['nexp'])+'/*.zarr', engine='zarr')
            self.climatology_std = xr.open_mfdataset(self.config['climatology_std_root']+'_'+'exp'+str(self.config['nexp'])+'/*.zarr', engine='zarr')

            # Climatology            
            self.data = self._apply_climatology(self.data, np.array(self.config['features'])[self.config['features_selected']])
            if self.task == 'ImpactAssessment':
                self.labels = self._apply_climatology(self.labels, self.config['target_feature'])      

        #self.data = self.data.where(abs(self.data) < 10)

        # Spatial slice
        start, end = eval(self.config['lat_slice_'+self.period])
        self.data = self.data.sel(lat=slice(start,end))
        start, end = eval(self.config['lon_slice_'+self.period])
        self.data = self.data.sel(lon=slice(start,end))
        # Spatial slice
        start, end = eval(self.config['lat_slice_'+self.period])
        self.labels = self.labels.sel(lat=slice(start,end))
        start, end = eval(self.config['lon_slice_'+self.period])
        self.labels = self.labels.sel(lon=slice(start,end))

        if self.task == 'Classification' or self.task == 'OutlierDetection':
            self.event_ids, event_inverse_values = np.unique(self.labels['mask_EMDAT'], return_inverse=True)
            self.labels['mask_EMDAT'].values = np.reshape(event_inverse_values, np.shape(self.labels['mask_EMDAT'])).astype('int')
            self.event_ids = np.arange(1,np.size(self.event_ids))
            self.event_locations = [np.max(self.labels['mask_EMDAT']==event_id,axis=2).values for event_id in self.event_ids]
            self.classes= ['None', 'Drought'] 

        # Probabilistic sampling, timestep level
        if self.period == 'train':
            #if self.task == "Classification" or self.task == "OutlierDetection":
            self.block_lat, self.block_lon, self.block_time = np.meshgrid(np.arange(len(self.data.lat.values)),
                                                                              np.arange(len(self.data.lon.values)),
                                                                              np.arange(len(self.data.time.values)),
                                                                              indexing='ij')
            '''elif self.task == "ImpactAssessment":
                self.block_lat, self.block_lon = np.meshgrid(np.arange(len(self.data.lat.values)),
                                                             np.arange(len(self.data.lon.values)),
                                                             indexing='ij')'''

            '''lat_degrees_per_pixel = np.abs(self.data.lat.values[1]-self.data.lat.values[0])
            lat_degrees_safeguard = lat_degrees_per_pixel*(eval(self.config['input_size_train'])[0]-1)
            
            lat_boundary_min = eval(self.config['lat_slice_val'])[0] + lat_degrees_safeguard
            lat_boundary_max = eval(self.config['lat_slice_val'])[1] - lat_degrees_safeguard
                
            lon_degrees_per_pixel = np.abs(self.data.lon.values[1]-self.data.lon.values[0])
            lon_degrees_safeguard = lon_degrees_per_pixel*(eval(self.config['input_size_train'])[1]-1)
            
            lon_boundary_min = eval(self.config['lon_slice_val'])[0] - lon_degrees_safeguard
            lon_boundary_max = eval(self.config['lon_slice_test'])[1] + lon_degrees_safeguard
            
            self.block_train = np.logical_not(np.logical_and(np.logical_and(self.data.lat.values[self.block_lat]<lat_boundary_min,
                                                                            self.data.lat.values[self.block_lat]>lat_boundary_max),
                                                             np.logical_and(self.data.lon.values[self.block_lon]>lon_boundary_min,
                                                                            self.data.lon.values[self.block_lon]<lon_boundary_max)))'''
            if self.data_dim > 1:
                self.block_train = np.zeros((self.block_lat.shape))
                if self.task == 'Classification':
                    self.block_train[0, 0:80] = 1
                elif self.task == 'ImpactAssessment':
                    self.block_train[0:64, 0:48] = 1
                    self.block_train[48:64, 48:128] = 1
            else:
                self.block_train = np.ones((self.block_lat.shape))
                self.block_train = self.block_train * self.labels['mask_EMDAT'].any(dim='time', keepdims=True).values
            
            #plt.imshow(self.block_train[:,:,0]), plt.show()# > IMPACT ASSESSMENT
            #plt.imshow(self.block_train[:,:,0]),plt.colorbar(), plt.show()# > DETECTION
            self.block_lat = self.block_lat.reshape(-1).astype('int')
            self.block_lon = self.block_lon.reshape(-1).astype('int')
            self.block_train = self.block_train.reshape(-1).astype('int')
            self.block_idxs = np.arange(len(self.block_lat))
            self.block_idxs = self.block_idxs[self.block_train==1]
            
            # self.num_total_samples_train = len(self.block_idxs)
            self.num_total_samples_train = self.config['n_samples']            
            self.block_idxs = np.random.choice(self.block_idxs, self.num_total_samples_train, replace=False)

            self.block_lat = self.block_lat[self.block_idxs]
            self.block_lon = self.block_lon[self.block_idxs]
            if self.task == "Classification" or self.task == "OutlierDetection":
                self.block_time = self.block_time.reshape(-1).astype('int')
                self.block_time = self.block_time[self.block_idxs]

    def _compute_climatology(self, list_of_variables):
        """Compute annual cycle, mean and standard deviation for data normalization.

        """
        print("Computing statistics for standardization based on climatology...")        
        os.makedirs(self.config['climatology_mean_root']+'_'+'exp'+str(self.config['nexp']), exist_ok=True)
        for feature in list_of_variables:
            #climatology_mean_aux = self.dataset[feature].groupby("time.month").mean("time", skipna = True)
            climatology_mean_aux = self.dataset[feature].where(self.dataset_labels == 0).groupby("time.month").mean("time", skipna = True)
            climatology_mean_aux = climatology_mean_aux.to_dataset()
            climatology_mean_aux.to_zarr(self.config['climatology_mean_root']+'_'+'exp'+str(str(self.config['nexp']))+'/'+feature+'.zarr')
        self.climatology_mean = xr.open_mfdataset(self.config['climatology_mean_root']+'_'+'exp'+str(self.config['nexp'])+'/*.zarr', engine='zarr')
        os.mkdir(self.config['climatology_std_root']+'_'+'exp'+str(str(self.config['nexp'])))
        for feature in list_of_variables:
            #climatology_std_aux = self.dataset[feature].groupby("time.month").std("time", skipna = True)
            climatology_std_aux = self.dataset[feature].where(self.dataset_labels == 0).groupby("time.month").std("time", skipna = True)
            climatology_std_aux = climatology_std_aux.to_dataset()
            climatology_std_aux.to_zarr(self.config['climatology_std_root']+'_'+'exp'+str(self.config['nexp'])+'/'+feature+'.zarr')
        self.climatology_std = xr.open_mfdataset(self.config['climatology_std_root']+'_'+'exp'+str(self.config['nexp'])+'/*.zarr', engine='zarr')
    
    def _apply_climatology(self, variable, list_of_features):
        """Apply climatology for data normalization.

        """
        if self.climatology_clipping:
            return xr.apply_ufunc(
                self._clip_standardize,
                variable.groupby("time.month"),
                self.climatology_mean[list_of_features],
                self.climatology_std[list_of_features],
                dask="parallelized",
            )   
        else:
            return xr.apply_ufunc(
                self._standardize,
                variable.groupby("time.month"),
                self.climatology_mean[list_of_features],
                self.climatology_std[list_of_features],
                dask="parallelized",
            ) 
            
    @staticmethod
    def _clip_standardize(x, m, s):
        """Standardization and Clipping, output data strictly ranges from [-1.0, 1.0].

        :param x: Source data.
        :type x: Array
        :param m: Mean.
        :type m: Array
        :param s: Variance.
        :type s: Array

        :return: Processed data.
        :rtype: Array
        """
        return np.clip((x - m) / (3*s + eps), -1.0, 1.0)

    @staticmethod
    def _standardize(x, m, s):
        """Standardization, output data with zero mean and unit variance.

        :param x: Source data.
        :type x: Array
        :param m: Mean.
        :type m: Array
        :param s: Variance.
        :type s: Array

        :return: Processed data.
        :rtype: Array
        """
        return (x - m) / (s + eps)

    def __getitem__(self, index):
        """Return item from dataset to pass to the dataLoader.

        :param index: Index of item.
        :type index: Int

        :return: Item at index.
        :rtype: Dict
        """
        if self.period == 'train':
            lat = self.block_lat[index]
            lon = self.block_lon[index]
            index = self.block_time[index] if self.task == "Classification" else 0

        else:
            lat = 0
            lon = 0

            if self.config['data_dim'] == 1:

                original_index = index
                index = original_index - self.total_points*int(original_index / self.total_points)
                if self.period == 'val':
                    index = index * 10

                lat_grid, lon_grid= np.meshgrid(np.arange(len(self.data.lat.values)), np.arange(len(self.data.lon.values)), indexing='ij')
                lat_grid = lat_grid.flatten()
                lon_grid = lon_grid.flatten()
                lat = lat_grid[index]
                lon = lon_grid[index]
                index = int(original_index / self.total_points) * eval(self.config['input_size_'+self.period])[2]
            
        num_features = np.size(self.config['features_selected'])

        # Features
        features = np.zeros((num_features, eval(self.config['input_size_' + self.period])[2],
                             eval(self.config['input_size_' + self.period])[0],
                             eval(self.config['input_size_' + self.period])[1]))
        # Masks + Data mask
        masks = np.zeros((num_features, eval(self.config['input_size_' + self.period])[2],
                             eval(self.config['input_size_' + self.period])[0],
                             eval(self.config['input_size_' + self.period])[1]))
        
        for i in np.arange(num_features):
            # Features
            if self.task == "Classification":
                # Features
                feature = self.data[np.array(self.config['features'])[self.config['features_selected'][i]]] \
                    [index:index + eval(self.config['input_size_' + self.period])[2],
                          lat:lat + eval(self.config['input_size_' + self.period])[0],
                          lon:lon + eval(self.config['input_size_' + self.period])[1]]
            elif self.task == "ImpactAssessment":
                start, end = eval(self.config[self.period+'_slice']['exp'+str(self.config['nexp'])][0])
                end = pd.to_datetime(end) - pd.DateOffset(months=1)
                feature = self.data[np.array(self.config['features'])[
                    self.config['features_selected'][i]]].sel(time=slice(start,end))
                feature = feature[:, lat:lat + eval(self.config['input_size_' + self.period])[0],
                                     lon:lon + eval(self.config['input_size_' + self.period])[1]]
            feature = feature.transpose('time', 'lat', 'lon').values
            features[i, :np.shape(feature)[0], :np.shape(feature)[1], :np.shape(feature)[2]] = feature
            # Masks
            mask = np.logical_not(np.isnan(feature))
            masks[i, :np.shape(mask)[0], :np.shape(mask)[1], :np.shape(mask)[2]] = mask                  

        if self.task == "ImpactAssessment":
            masks = np.sum(masks,axis=(0,1),keepdims=True) > 0
            
        # Fill missing values with 0s
        features[np.isnan(features)] = 0

        x = features

        # Labels
        gt = np.zeros(
            (1, eval(self.config['input_size_' + self.period])[2] if self.task=="Classification" else 1,
                eval(self.config['input_size_' + self.period])[0],
                eval(self.config['input_size_' + self.period])[1]))
        if self.task == "Classification":
            item_date = self.data['time'][index:index + eval(self.config['input_size_' + self.mode])[2]].values
            gt_found = np.where(np.in1d(item_date, self.labels['time']))[0]
            gt_aux = self.labels['mask_EMDAT'].sel(lat=self.data.lat[lat:lat + eval(self.config['input_size_' + self.period])[0]],
                                                   lon=self.data.lon[lon:lon + eval(self.config['input_size_' + self.period])[1]],
                                                   time=item_date[gt_found]).transpose('time', 'lat', 'lon').values > 0
        elif self.task == "ImpactAssessment":
            start, end = eval(self.config[self.period+'_slice']['exp'+str(self.config['nexp'])][0])
            start = pd.to_datetime(end) - pd.DateOffset(months=1)
            gt_aux = self.labels[self.config['target_feature']].sel(lat=self.data.lat[lat:lat + eval(self.config['input_size_' + self.period])[0]],
                                                                lon=self.data.lon[lon:lon + eval(self.config['input_size_' + self.period])[1]],
                                                                time=slice(start,end))
            gt_aux = gt_aux.transpose('time', 'lat', 'lon').mean("time", keepdims=True).values
        gt[0][:np.shape(gt_aux)[0], :np.shape(gt_aux)[1], :np.shape(gt_aux)[2]] = gt_aux
        gt[np.isnan(gt)] = 0
            
        # Times
        time = np.datetime_as_string(self.data.time.isel(time=index).values, unit='D')

        if self.precision == 16:
            x = x.astype('float16')
            masks = masks.astype('float16')
            gt = gt.astype('float16')
        elif self.precision == 32:
            x = x.astype('float32')
            masks = masks.astype('float32')
            gt = gt.astype('float32')
        
        # To tensor
        x = torch.from_numpy(x)
        masks = torch.from_numpy(masks)
        gt = torch.from_numpy(gt)

        if self.total_config['arch']['input_model_dim'] == 1:
            x = torch.squeeze(torch.squeeze(x, 3), 2)
            masks = torch.squeeze(torch.squeeze(masks,3),2)
            gt = torch.squeeze(torch.squeeze(gt, 3),2)
                     
        return {'x': x.float(), 'masks': masks, 'labels': gt, 'time': time}
    
    def __getallitems__(self):
        """Return all items from the whole dataset.

        :return: All items.
        :rtype: Dict
        """
        num_features = np.size(self.config['features_selected'])

        # Features    
        features = np.zeros((num_features,np.size(self.data.time),
                             np.size(self.data.lat),np.size(self.data.lon)))
        # Masks + Data mask
        masks = np.zeros((num_features,np.size(self.data.time),
                             np.size(self.data.lat),np.size(self.data.lon)))
        
        for i in np.arange(num_features):
            # Features
            feature = self.data[np.array(self.config['features'])[self.config['features_selected'][i]]]                   
            feature = feature.transpose('time','lat','lon').values
            features[i,:np.shape(feature)[0],:,:] = feature
            # Masks
            mask = np.logical_not(np.isnan(feature))
            masks[i,:np.shape(mask)[0],:,:] = mask   
            
        # Fill missing values with 0s
        features[np.isnan(features)] = 0 
    
        x = features
        
        # Labels
        item_date = self.data['time'].values            
        gt_found = np.where(np.in1d(item_date,self.labels['time']))[0]
        gt = np.zeros((1,np.size(self.data.time),np.size(self.data.lat),np.size(self.data.lon)))
        gt[:,gt_found,:,:] = self.labels['mask_EMDAT'].sel(lat=self.data.lat,
                                                   lon=self.data.lon,
                                                   time=item_date[gt_found]).transpose('time','lat','lon').values > 0

        # Times 
        time = np.datetime_as_string(self.data.time.values, unit='D')
        
        if self.precision == 16:
            x = x.astype('float16')
            masks = masks.astype('float16')
            gt = gt.astype('float16')
        elif self.precision == 32:
            x = x.astype('float32')
            masks = masks.astype('float32')
            gt = gt.astype('float32')
        

        return {'x': x, 'masks': masks, 'labels': gt, 'time': time}

    def __len__(self):
        """Return length of dataset.

        :return: Length of dataset (number of frames).
        :rtype: Int
        """
        if self.period == 'train':
            return self.num_total_samples_train
        else:
            if self.task == "Classification" or self.task == "OutlierDetection":
                samples_length = eval(self.config['input_size_'+self.period])[2]
                if self.config['data_dim'] == 1:
                    
                    if self.period == 'val':
                        self.total_points = int((np.size(self.data['lat']) * np.size(self.data['lon'])) / 10 )
                    else:
                        self.total_points = np.size(self.data['lat']) * np.size(self.data['lon'])
                    
                    return int((self.total_points) * (np.size(self.data['time'])/self.total_config['arch']['params']['out_len']))
                else: 
                    return np.size(self.data['time']) - samples_length

            elif self.task == "ImpactAssessment":
                return 1