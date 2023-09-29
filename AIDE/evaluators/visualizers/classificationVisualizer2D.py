#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .classificationVisualizerAbstract import *


class ClassificationVisualizer2D(ClassificationVisualizerAbstract):
    """
    Evaluator for 2D outputs
    """

    def __init__(self, config, model, dataloader):
        super().__init__(config, model, dataloader)
        lon, lat = config['data']['lon_slice_test'].split(
            ','), config['data']['lat_slice_test'].split(',')
        self.coordinates = (float(lon[0]), float(
            lon[1]), float(lat[0]), float(lat[1]))

        self.config = config
        self.save_path = config['save_path']
        self.num_classes = self.config['data']['num_classes']
        if self.num_classes == 2:
            self.num_classes = 1
            confusion_matrix_task = 'binary'
        else:
            confusion_matrix_task = 'multiclass'

        self.data_classes = [i for i in range(
            1, config['data']['num_classes'])]
        self.cm = ConfusionMatrix(confusion_matrix_task, num_classes=len(self.data_classes)+1, normalize='true')

        self.test_outputs = []
        self.test_labels = []
        self.temporal_plot_xticks_labels = []
        self.test_outputs_mean_when_event = dict.fromkeys(
            self.data_classes, [])
        self.test_outputs_mean_without_event = dict.fromkeys(
            self.data_classes, [])
        self.test_labels_when_event = dict.fromkeys(self.data_classes, [])

        self.__create_evaluator_dirs()

    def __create_evaluator_dirs(self):
        """ 
        Create directories to save figures for visualization
        """
        print(self.save_path+'/spatial_visualization')
        if not os.path.isdir(self.save_path+'/spatial_visualization'):
            os.mkdir(self.save_path+'/spatial_visualization')
    
    def per_sample_operations(self, outputs, labels, time_indexes, event_names, masks):
        """
        Performs per sample plot of the extreme event detection maps and the variables' saving for the global operations
        """
        for output, label, time, event_name, mask in (pbar := tqdm(zip(outputs, labels, time_indexes, event_names, masks), total=len(outputs))):
            pbar.set_description('Performing visualization')
            self.__plot_per_class_detection(output, time, mask)
            self.__plot_aggregated_detection(self.__categorize(output), time, mask)
            self.__save_global_results(output, mask, label, time)
    
    def __categorize(self,output):
        if self.num_classes == 1:
            return (output[0] > 0.5).int()
        else:
            return torch.max(output, dim=0)[1]

    def __save_global_results(self,  output, mask, labels, time):
        self.test_outputs.extend([x.item() for x in self.__categorize(output).flatten()])
        self.test_labels.extend([x.item() for x in labels.flatten().long()])
        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan
            
        if self.config['evaluation']['visualization']['params']['time_aggregation']:
            self.temporal_plot_xticks_labels.append(str(time))
            for num_class in self.data_classes:
                self.__save_data_to_plot_temporal(output[num_class-1], mask, labels, num_class)
                
    def __plot_per_class_detection(self, output, sample_name, mask):
        """
        Plot extreme event detection saliency map for each class and a given time step
        """
        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan
        for num_class in self.data_classes:
            label = self.test_loader.dataset.labels['mask'].where(self.test_loader.dataset.labels['mask'] != num_class, 0).any(dim='time').values
            misc.plot_spatial_signal(self.save_path+'/spatial_visualization/c'+str(num_class)+'_'+sample_name, output[num_class-1], 
                                     label, mask, self.coordinates)

    def __plot_aggregated_detection(self, output, sample_name, mask):
        """
        Plot extreme event detection categorical map for a given time step
        """
        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan
        label = self.test_loader.dataset.labels['mask'].any(
            dim='time').values.astype(int)
        misc.plot_spatial_aggregation(self.save_path+'/spatial_visualization/'+sample_name+'_aggregation',
                                      output, label, mask, self.coordinates, self.config['data']['num_classes'])

    def __save_data_to_plot_temporal(self, output, mask, labels, num_class):
        """ 
        Prepare data for visualization through time (aggregation mean)
        """
        label_mask = self.test_loader.dataset.labels['mask'].where(
            self.test_loader.dataset.labels['mask'] != num_class, 0).any(dim='time').values
        output = output*mask if mask!=None else output
        
        self.test_outputs_mean_when_event[num_class].append(torch.nanmean(output[label_mask == 1]))
        self.test_outputs_mean_without_event[num_class].append(torch.nanmean(output[label_mask == 0]))
        self.test_labels_when_event[num_class].append(int(labels[labels == num_class].any()))

    def global_operations(self):
        """
        Plot confusion matrix and visualize results through time
        """
        misc.plot_confusion_matrix(self.cm, torch.Tensor(
            self.test_outputs), torch.Tensor(self.test_labels).int(), self.save_path)
        if self.config['evaluation']['visualization']['params']['time_aggregation']:
            for num_class in self.data_classes:
                misc.plot_temporal_results(self.save_path+'/c'+str(num_class)+'_', self.temporal_plot_xticks_labels,
                                           self.test_labels_when_event[num_class], self.test_outputs_mean_when_event[num_class], self.test_outputs_mean_without_event[num_class])