#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
from skimage.morphology import binary_closing, remove_small_holes, remove_small_objects, h_maxima, h_minima
from skimage.segmentation import expand_labels
from skimage.measure import label, euler_number, perimeter, regionprops
from itertools import product
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils import *

class ClassificationCharacterizer():
    """ This class is used for classification characterization.

    """
    def __init__(self, config):
        """Initialization of the classification characterizer.

        :param config: The configuration file.
        :type config: Dict
        """
        self.config = config
        self.num_classes = self.config['data']['num_classes']
        
        self.save_path = config['save_path'] + '/characterization_stats'
            
        self.min_dist = config['evaluation']['characterization']['params']['min_distance']
        self.connectivity = config['arch']['output_model_dim']
        
        self.expand = False
        if config['arch']['output_model_dim'] == 1:
            self.expand = True
            
        self.time_aggregate = False
        if 'time_aggregation' in config['evaluation']['characterization']['params']:
            self.time_aggregate = config['evaluation']['characterization']['params']['time_aggregation']

        self.__create_evaluator_dirs()
    
    def __create_evaluator_dirs(self):
        """ Create directories for saving characterization stats.

        """
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
            
    def characterize(self, inference_outputs, events):
        """ Evaluate model and visualize results.

        :param inference_outputs: Predicted outputs during the inference stage. .
        :type inference_outputs: List[Tensor]
        :param events: Events based on top of the probabilistic outputs.
        :type events: List
        """
        print('Starting characterization ...')
        
        self.test_outputs = np.array([x.numpy() for x in inference_outputs['outputs']])
        self.test_labels = np.array([x.numpy() for x in inference_outputs['labels']]) 
        self.test_event_names = np.array([x for x in inference_outputs['event_names']]) 
        self.num_samples = np.shape(self.test_outputs)[0]

        if self.config['arch']['output_model_dim'] == 3:
            self.test_outputs = self.test_outputs.transpose(0,2,1,3,4).reshape(-1,self.test_outputs.shape[1],
                                                                               self.test_outputs.shape[3],
                                                                               self.test_outputs.shape[4])


        if self.time_aggregate: 
            self.test_outputs = np.concatenate(self.test_outputs, axis = 0)
            self.test_outputs = np.expand_dims(self.test_outputs, axis = (0, 1))
            self.connectivity = np.minimum(self.connectivity+1, 3) 
            self.num_samples = 1
            
        if self.expand: 
            self.test_outputs = np.expand_dims(self.test_outputs, axis = (-1))
            
        if self.num_classes > 2:
            self.test_outputs = self.test_outputs[:,1:] # Remove background class

        self.charact_set_results(events)
    
    def charact_set_results(self, events):
        """ Computes characteristics for each sample, all classes and their corresponding events.
        The output file contains the stats for each sample divided according to the class of the events.

        :param events: Events based on top of the probabilistic outputs.
        :type events: List
        """
        for sample in range(self.num_samples):
            
            # Predefine variables
            id_classes = []
            id_events = []
            stats_dict = {'Extent (samples)': [], 'Location (centroid)': [], 
                          'Location (weighted centroid)': [], 
                          'Max prob': [], 'Mean prob': [], 'Min prob':[],
                          'Std prob': []}
            
            # Loop over the classes 
            for class_id in range(self.num_classes-1):            
                
                # Get ids
                sample_events = events[class_id][sample] # gets the element of the list, each element belongs to a class and has sizes of n_samples, size_sample
                events_ids = np.unique(sample_events)[1:] # starting from one to remove background (=0, "no event" class)
            
                # Get characteristics for each event in the sample
                for event_id in events_ids: 
                    
                    # Append the correspondence: class-event
                    id_classes.append('Class_' + str(class_id+1))
                    id_events.append('Event_id_' + str(event_id))
                    
                    # Append the geometric properties
                    stats_dict['Extent (samples)'].append(np.sum(sample_events == event_id))
                    
                    # Get the event locations
                    event = np.where(sample_events == event_id, 1, np.nan) 
                    event_props = regionprops((event==1).squeeze().astype('int'), intensity_image=self.test_outputs[sample, class_id].squeeze())[0]
                    
                    # Append the probabilities properties
                    stats_dict['Location (centroid)'].append(tuple([c for c_id, c in enumerate(event_props.centroid) if np.shape(event)[c_id]!=1]))
                    stats_dict['Location (weighted centroid)'].append(tuple([c for c_id, c in enumerate(event_props.centroid_weighted) if np.shape(event)[c_id]!=1]))
                    stats_dict['Max prob'].append(np.nanmax(self.test_outputs[sample, class_id] * event))
                    stats_dict['Mean prob'].append(np.nanmean(self.test_outputs[sample, class_id] * event))
                    stats_dict['Min prob'].append(np.nanmin(self.test_outputs[sample, class_id] * event))
                    stats_dict['Std prob'].append(np.nanstd(self.test_outputs[sample, class_id] * event))
             
            # create the dataframe
            df = pd.DataFrame(data = stats_dict, index = [np.array(id_classes), np.array(id_events)])
 
            # export DataFrame to text file
            file_name = self.save_path + '/'+str(self.test_event_names[sample])+'.txt'
            with open(file_name, 'a') as f:
                df_string = df.to_string(header = True, index = True)
                f.write(df_string)

        print('Saved characterization results in folder' + self.save_path)

            
    
