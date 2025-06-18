#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .visualizerAbstract import *


class ClassificationVisualizer2D(VisualizerAbstract):
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
            self.__plot_per_class_detection(output, label, time, mask)
            self.__plot_aggregated_detection(self.__categorize(output), label, time, mask)
            self.__save_global_results(output, label, mask, time)
    
    def __categorize(self,output):
        """Categorize the model's output according to the number of classes

        :param output: Decision scores at the output of the model
        :type output: tensor
        :return: Categorized model output
        :rtype: tensor
        """
        if self.num_classes == 1:
            return (output[0] > 0.5).int()
        else:
            return torch.max(output, dim=0)[1]

    def __save_global_results(self, output, labels, mask, time):
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
        self.test_outputs.extend([x.item() for x in self.__categorize(output).flatten()])
        self.test_labels.extend([x.item() for x in labels.flatten().long()])
        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan
            
        if self.config['evaluation']['visualization']['params']['time_aggregation']:
            self.temporal_plot_xticks_labels.append(str(time))
            for num_class in self.data_classes:
                self.__save_data_to_plot_temporal(output[num_class-1], labels, mask, num_class)
                
    def __plot_per_class_detection(self, output, label, sample_name, mask):
        """Plot extreme event detection saliency map for each class and a given time step

        :param output: Decision scores at the output of the model
        :type output: tensor
        :param sample_name: Sample identifier
        :type sample_name: str or int
        :param mask: Quality mask
        :type mask: tensor
        """
        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan
        for num_class in self.data_classes:
            contourf_params = {'cmap':plt.cm.Reds, 'levels': np.linspace(0, 1.0, 11, endpoint=True), 'vmin':0, 'vmax':1}
            misc.plot_spatial_signal(self.save_path+'/spatial_visualization/c'+str(num_class)+'_'+sample_name, output[num_class-1], label, 
                                     mask, self.coordinates, contourf_params, 'Output Signal')

    def __plot_aggregated_detection(self, output, label, sample_name, mask):
        """Plot extreme event detection categorical map for a given time step

        :param output: Categorized model output
        :type output: tensor
        :param sample_name: Sample identifier
        :type sample_name: str
        :param mask: Quality mask
        :type mask: tensor
        """

        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan
        misc.plot_spatial_aggregation(self.save_path+'/spatial_visualization/'+sample_name+'_aggregation',
                                      output, label, mask, self.coordinates, self.config['data']['num_classes'])

    def __save_data_to_plot_temporal(self, output, labels, mask, num_class):
        """Prepare data for visualization through time (aggregation mean)

        :param output: Decision scores at the output of the model
        :type output: tensor
        :param labels: Samples' ground truth
        :type labels: tensor
        :param mask: Quality mask
        :type mask: tensor
        :param num_class: Number of classess in the classification
        :type num_class: int
        """
        output = output*mask if mask!=None else output
        if not hasattr(self, 'aggregated_output'):
            self.aggregated_output = dict.fromkeys(self.data_classes, [])
            self.aggregated_labels = dict.fromkeys(self.data_classes, [])
            self.event_area = dict.fromkeys(self.data_classes, torch.zeros(labels.shape, dtype=torch.bool))

        self.aggregated_output[num_class].append(output)
        self.aggregated_labels[num_class].append(int((labels == num_class).any()))
        self.event_area[num_class] = torch.logical_or(self.event_area[num_class], labels == num_class)

    def global_operations(self):
        """
        Plot confusion matrix and visualize results through time
        """
        misc.plot_confusion_matrix(self.cm, torch.Tensor(
            self.test_outputs), torch.Tensor(self.test_labels).int(), self.save_path)
        if self.config['evaluation']['visualization']['params']['time_aggregation']:
            for num_class in self.data_classes:
                event_area = (self.event_area[num_class] == num_class)

                for output in self.aggregated_output[num_class]:
                    self.test_outputs_mean_when_event[num_class].append(torch.nanmean(output[event_area == 1]))
                    self.test_outputs_mean_without_event[num_class].append(torch.nanmean(output[event_area == 0]))
                
                misc.plot_temporal_results(self.save_path+'/c'+str(num_class)+'_', self.temporal_plot_xticks_labels,
                                           self.aggregated_labels[num_class], self.test_outputs_mean_when_event[num_class], self.test_outputs_mean_without_event[num_class])