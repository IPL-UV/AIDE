#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .visualizerAbstract import *


class ImpactAssessmentVisualizer2D(VisualizerAbstract):
    """
   Visualization for 2D outputs
    """

    def __init__(self, config, model, dataloader):
        """Initialization of the ClassificationVisualizer2D's parameters

        :param config: The configuration file
        :type config: dict
        :param model: PyTorch trained model for evaluation
        :type model: class: 'torch.nn.Module'
        :param dataloader: PyTorch data iterator
        :type dataloader: class: 'torch.utils.data.DataLoader'
        """
        super().__init__(config, model, dataloader)
        lon, lat = config['data']['lon_slice_test'].split(
            ','), config['data']['lat_slice_test'].split(',')
        self.coordinates = (float(lon[0]), float(
            lon[1]), float(lat[0]), float(lat[1]))

        self.config = config
        self.save_path = config['save_path']

        self.test_outputs = []
        self.test_labels = []
        self.test_masks = []
        self.test_times = []
        self.test_uncertainties = []
        self.__create_evaluator_dirs()

    def __create_evaluator_dirs(self):
        """ 
        Create directories to save figures for visualization
        """
        if not os.path.isdir(self.save_path+'/spatial_visualization'):
            os.mkdir(self.save_path+'/spatial_visualization')
    
    def per_sample_operations(self, outputs, labels, time_indexes, event_names, masks, uncertainties=None):
        """ Performs per sample plot of the extreme event detection maps and the variables' saving for the global operations

        :param outputs: Decision scores at the output of the model
        :type outputs: list
        :param labels: Samples' ground truth
        :type labels: list
        :param time_indexes: Time indexes
        :type time_indexes: list
        :param event_names: Event identifier
        :type event_names: list
        :param masks: Quality mask
        :type masks: list
        """
        for output, label, time, event_name, mask in (pbar := tqdm(zip(outputs, labels, time_indexes, event_names, masks), total=len(outputs))):
            pbar.set_description('Performing visualization')
            self.vmax = label.max()
            self.vmin = label.min()

            self.__save_global_results(output[0], mask, label, time)
            
        if uncertainties is not None:
            for uncertainty_upper, uncertainty_lower in zip(uncertainties['outputs_upper'], uncertainties['outputs_lower']):
                self.test_uncertainties.append(uncertainty_upper[0,0]/2)
                
    def __save_global_results(self,  output, mask, label, time):
        """ Prepare data for visualization

        :param output: Categorized model output
        :type output: tensor
        :param mask: Quality mask
        :type mask: tensor
        :param labels: Samples' ground truth
        :type labels: tensor
        :param time: Time indexes
        :type time: tensor
        """
        self.test_outputs.append(output)
        self.test_labels.append(label)
        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan
        self.test_masks.append(mask)
        self.test_times.append(time)
    
    def global_operations(self):
        for output, label, mask, time in zip(self.test_outputs, self.test_labels, self.test_masks, self.test_times):
            if "vmin" in self.config['evaluation']['visualization']['params']:
                self.vmin = self.config['evaluation']['visualization']['params']['vmin']
            if "vmax" in self.config['evaluation']['visualization']['params']:
                self.vmax = self.config['evaluation']['visualization']['params']['vmax']
            contourf_params = {'cmap':self.config['evaluation']['visualization']['params']['output_color_map'], 'levels': np.linspace(self.vmin,self.vmax,100)}
            misc.plot_spatial_signal(self.save_path+'/spatial_visualization/'+str(time)+'_output', output, torch.zeros(output.shape),
                                    mask, self.coordinates, contourf_params, 'Ouput Signal')
            contourf_params = {'cmap':self.config['evaluation']['visualization']['params']['gt_color_map'], 'levels':  np.linspace(self.vmin,self.vmax,100)}
            misc.plot_spatial_signal(self.save_path+'/spatial_visualization/'+str(time)+'_label', label, torch.zeros(label.shape),
                                    mask, self.coordinates, contourf_params, 'Ground-Truth Signal')
        if len(self.test_uncertainties) != 0:
            for output, uncertainty, time in zip(self.test_outputs, self.test_uncertainties, self.test_times):
                contourf_params = {'cmap':self.config['evaluation']['visualization']['params']['uncertainty_color_map'], 'levels':  np.linspace(uncertainty.min(),uncertainty.max(),100)}
                misc.plot_spatial_signal(self.save_path+'/spatial_visualization/'+str(time)+'_predictive_variance', uncertainty, torch.zeros(uncertainty.shape),
                                    mask, self.coordinates, contourf_params, 'Predictive variance')
                cv = uncertainty/output
                contourf_params = {'cmap':self.config['evaluation']['visualization']['params']['uncertainty_color_map'], 'levels':  np.linspace(cv.min(),cv.max(),100)}
                misc.plot_spatial_signal(self.save_path+'/spatial_visualization/'+str(time)+'_cv', cv, torch.zeros(uncertainty.shape),
                                    mask, self.coordinates, contourf_params, 'Coeficient of variation')
