#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from utils import *

from .visualizers import *
from .characterizers import *
from .xai import * 

class PytorchEvaluator():
    """
    Evaluator generic class
    """
    def __init__(self, config, model, dataloader):
        self.debug = config.get('debug', False)
        self.config = config
        self.task = self.config['task']
        self.model = model
        self.loader = dataloader
        self.num_classes = self.config['data']['num_classes'] if self.task == 'Classification' else 1
        self.threshold = 0.5
    
    def evaluate(self, inference_outputs, subset):
        """
        Evaluate model
        """
        if self.task == 'Classification':
            if self.config['evaluation']['visualization']['activate']:
                assert self.config['task'] == 'Classification', f'Visualization not implemented for this task. Set evaluation > visualization > activate: false in config file.'
                visualizer = globals()['ClassificationVisualizer'+str(self.config['arch']['output_model_dim'])+'D'](self.config, self.model, self.loader[subset])
                visualizer.visualize(inference_outputs)
             
            if self.config['evaluation']['characterization']['activate']:
                assert self.config['task'] == 'Classification', f'Characterization not implemented for this task. Set evaluation > characterization > activate: false in config file.'
                characterizer = globals()['ClassificationCharacterizer'](self.config)
                self.events = self.define_events(inference_outputs, task='characterization')
                characterizer.characterize(inference_outputs, self.events)

        if self.config['evaluation']['xai']['activate']:
            xai = globals()['XAI'](self.config, self.model, self.loader[subset])
            if self.task == 'Classification' and self.config['evaluation']['xai']['params']['mask'] == 'events':
                self.events = self.define_events(inference_outputs, task='xai')
                events = np.expand_dims(self.events, axis=0) if np.shape(self.events)[0] > 1 else self.events
                events = np.vstack(events)
                events = np.moveaxis(events, 1, 0)
                events = np.squeeze(events, axis=-1) if np.shape(events)[-1] == 1 else events
                xai.xai(events = events)
            else:
                xai.xai()
       
        return inference_outputs
    
    def inference(self, subset):
        """
        Infere results
        """
        test_outputs = []
        test_labels = []
        test_time = []
        test_event_names = []

        self.model.eval()
        with torch.no_grad():
            for index, sample in enumerate((pbar := tqdm(self.loader[subset]))):
                pbar.set_description('Infering Dataloader')
                if not 'masks' in sample.keys():
                    sample['masks'] = torch.ones(sample['x'].shape)
                if not 'time' in sample.keys():
                    sample['time'] = index
                if not 'event_name' in sample.keys():
                    sample['event_name'] = index

                x, masks, labels = adapt_variables(self.config, sample['x'], sample['masks'], sample['labels'])
                output = self.model(x)

                if isinstance(output, tuple):
                    output = output[-1]

                output, labels, time, event_name = self.adapt_input(output, labels, sample['time'], sample['event_name'])
                
                test_outputs.append(output)
                test_labels.append(labels)
                test_time.append(time)
                test_event_names.append(event_name)
                if self.debug and index == int(self.debug-1): break
                        
            return {'outputs':test_outputs, 'labels':test_labels, 'time':test_time, 'event_names': test_event_names}
    
    def adapt_input(self, output, labels, time, event_name):
        """
        Adapt input size for evaluation purposes
        """
        if self.config['arch']['output_model_dim'] == 1:
            output = output[0]
            labels = labels[0]
        
        elif self.config['arch']['output_model_dim'] == 2:
            output = output[0]
            labels = labels[0,0]

        elif self.config['arch']['output_model_dim'] == 3:
            output = output[0,:,0]
            labels = labels[0,0,0]
        
        if not isinstance(time, int):
            time = time[0]

        if not isinstance(event_name, int):
            event_name = event_name[0]

        return output, labels, time, event_name
    
    def define_events(self, inference_outputs, task):
        """
        Identify events and get their ids (unique values)
        """
  
        if task == 'xai' and (self.config['evaluation']['characterization']['activate'] and \
           'events' in self.config['evaluation']['xai']['params']['mask']) and \
           (self.config['evaluation']['characterization']['params']['threshold'] == \
            self.config['evaluation']['xai']['params']['threshold']):
               
            return self.events
            
        else:
        
            test_outputs = np.array([x.numpy() for x in inference_outputs['outputs']])
            test_labels = np.array([x.numpy() for x in inference_outputs['labels']]) 
            num_samples = np.shape(test_outputs)[0]

            connectivity = self.config['arch']['output_model_dim']
            
            remove_scant = False
            expand = False
            if self.config['arch']['output_model_dim'] > 1:
                remove_scant = self.config['evaluation'][task]['params']['remove_scant']
            else: 
                expand = True
                
            self.time_aggregate = False
            if 'time_aggregation' in self.config['evaluation'][task]['params']:
                self.time_aggregate = self.config['evaluation'][task]['params']['time_aggregation']
            
            if self.time_aggregate: 
                test_outputs = np.concatenate(test_outputs, axis = 0)
                test_outputs = np.expand_dims(test_outputs, axis = 0)
                connectivity += 1 
                num_samples = 1
                
            if expand: 
                test_outputs = np.expand_dims(test_outputs, axis = (-1))
                
            all_events = []
            
            # Check to allow characterization
            assert np.prod(list(np.shape(test_outputs)[2:])) > 1, 'Output size no valid for characterization'
    
            # Same TH for all classes if unique
            if self.num_classes == 2:
                self.threshold = self.set_threshold(task=task)
                
            # Loop over the classes
            tmp_test_outputs = np.argmax(test_outputs, axis=1) if self.num_classes > 2 else (test_outputs > self.threshold) 
            for i in range(self.num_classes-1):
                
                # Samples of the corresponding class
                tmp_test_outputs_aux = np.where(tmp_test_outputs == i+1, 1, 0)
                # Preprocess the predicted labels to remove small events or small holes
                if remove_scant:
                    tmp_test_outputs_aux = self.remove_scant_labels(tmp_test_outputs_aux, connectivity, task)
                
                events = np.zeros_like(tmp_test_outputs_aux).astype(np.uint8) # shape = n_samples, size of the sample
                for j in range(num_samples):
    
                    # Get an index identifying events
                    # we consider as the same event all pixels that are connected
                    # individual pixels that can't be connected are events by themselves
                    events[j] = label(tmp_test_outputs_aux[j].astype(int), connectivity = connectivity)
                    events[j] = self.connect_events(events[j], task)
                    
                    # If an event has been identified (any True values existing)
                    if (events[j] == 1).any():
                        events[j] *= j + 1 # multiply by the index so each sample has a different id for the event. Take into acount the loop starts from zero 
                    
                all_events.append(events)

            return all_events

    def set_threshold(self, task):

        if task == 'characterization':
            threshold_metric = list(self.config['evaluation']['characterization']['params']['threshold'])[0]
            threshold_metric_params = list(self.config['evaluation']['characterization']['params']['threshold'].values())[0]
            threshold_lower_is_best = self.config['evaluation']['characterization']['params']['threshold_lower_is_best']

        elif task == 'xai':
            
            if (not self.config['evaluation']['characterization']['activate'] and \
                'events' in self.config['evaluation']['xai']['params']['mask']) or \
                (self.config['evaluation']['characterization']['activate'] and \
                 'events' in self.config['evaluation']['xai']['params']['mask'] and \
                self.config['evaluation']['characterization']['params']['threshold'] != \
                self.config['evaluation']['xai']['params']['threshold']):
                
                threshold_metric = list(self.config['evaluation']['characterization']['params']['threshold'])[0]
                threshold_metric_params = list(self.config['evaluation']['characterization']['params']['threshold'].values())[0]
                threshold_lower_is_best = self.config['evaluation']['xai']['params']['threshold_lower_is_best']
                    
            else:
                return self.threshold
        
        if threshold_metric != 'none':
            
            val_inference_outputs = self.inference('val')
            val_outputs = np.array([x.numpy() for x in val_inference_outputs['outputs']])
            val_labels = np.array([x.numpy() for x in val_inference_outputs['labels']]) 

            # Define the threshold and binarize
            # The optimal threshold should be different for each class but the same for all the samples
            self.threshold = determine_threshold(y_pred = val_outputs, 
                                                 y_true = val_labels,
                                                 metric_name = threshold_metric,
                                                 metric_params = threshold_metric_params,
                                                 lower_is_better = threshold_lower_is_best)
            
        else:
            self.threshold = 0.5
        
        return self.threshold
    
    def remove_scant_labels(self, tmp_test_outputs, connectivity):
        """
        Removes scant labels
        """
        
        print('Removing scant index labels')

        min_area_holes = self.config['evaluation'][task]['params']['min_area_holes'] 
        min_area_objects = self.config['evaluation'][task]['params']['min_area_objects'] 
                
        # 1) Remove small holes -> replace with ones 
        # (non drought areas surrounded by drought areas)
        # maximum area to be filled defined by the threshold
        transformed_labels = remove_small_holes(tmp_test_outputs, area_threshold = min_area_holes, 
                                                connectivity = connectivity)
        
        # 2) Remove small objects -> replace with zeros 
        # (drought areas surrounded by non-drought areas)
        # minimum area to retain the drought object defined by the threshold
        transformed_labels = remove_small_objects(transformed_labels, min_size = min_area_objects, 
                                                  connectivity = connectivity)
        
        return transformed_labels
    
    def connect_events(self, events, task):
        """
        events at less than min_dist are given the same label
        Min distance is spatial only
        """
        
        min_dist = self.config['evaluation'][task]['params']['min_distance']
        
        #print('Connecting events, distance (px):', min_dist)
        
        # Get the centroids 
        obj_properties = regionprops(events) # Labels with value 0 are ignored
        centroids = [np.array(obj_properties[i]['centroid']) for i in range(len(obj_properties))] 
        
        # Compute the Euler distance 
        distances = np.zeros((len(centroids), len(centroids)))
        for i, j in product(range(len(centroids)), range(len(centroids))):
            distances[i, j] = np.linalg.norm(centroids[i] - centroids[j]) 
        
        # The matrix is simetric so we retain the corresponding 
        # upper triangular matrix / removing also the main diagonal
        # k = 0 is the main diagonal, k < 0 is below it and k > 0 is above
        distances[np.triu(distances, k = 1) == 0] = np.nan
        id_objs1, id_objs2 = np.where(distances < min_dist)
        
        # +1 to avoid the zero as an index
        id_objs1 += 1
        id_objs2 += 1
        
        pool_ids = []
        connected_events = np.copy(events)
        for evID in range(len(distances)):
            
            if evID not in pool_ids: # check for repetitions    
                for near_evID in id_objs2[id_objs1 == evID]: # this checks for near events
                        
                    connected_events[events == near_evID] = evID
                    pool_ids.append(near_evID)
        
        return connected_events
                

        
        