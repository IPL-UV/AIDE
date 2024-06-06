#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cycler import cycler

from .visualizerAbstract import *

class ClassificationVisualizer1D(VisualizerAbstract):
    """ 
    Visualization for 1D outputs
    """

    def __init__(self, config, model, dataloader):
        """Initialization of the ClassificationVisualizer1D's parameters

        :param config: The configuration file
        :type config: dict
        :param model: PyTorch trained model for evaluation
        :type model: class: 'torch.nn.Module'
        :param dataloader: PyTorch data iterator
        :type dataloader: class: 'torch.utils.data.DataLoader'
        """
        super().__init__(config, model, dataloader)
        self.save_path = config['save_path']

        self.num_classes = self.config['data']['num_classes']
        if self.num_classes == 2:
            self.num_classes = 1
            confusion_matrix_task = 'binary'
        else:
            confusion_matrix_task = 'multiclass'

        self.cm = ConfusionMatrix(task= confusion_matrix_task, num_classes = config['data']['num_classes']+1, normalize='true')

        self.test_outputs = []
        self.test_labels = []
        self.temporal_plot_xticks_labels = []

        if type(self.config['evaluation']['visualization']['params']['spatial_aggregation']) != bool:
            shape = eval(self.config['evaluation']['visualization']['params']['spatial_aggregation']) 
            shape = (self.num_classes, shape[0]*shape[1], shape[2])
            self.spatial_outputs = torch.zeros(shape)
            self.spatial_labels = torch.zeros(shape)
            self.spatial_masks = torch.ones(shape)

            lon, lat = config['data']['lon_slice_test'].split(','), config['data']['lat_slice_test'].split(',')
            self.coordinates = (float(lon[0]), float(lon[1]), float(lat[0]), float(lat[1]))

        self.__create_evaluator_dirs()
    
    def __create_evaluator_dirs(self):
        """ 
        Create directories to save figures for visualization
        """
        if not os.path.isdir(self.save_path+'/temporal_visualization'):
            os.mkdir(self.save_path+'/temporal_visualization')
        if type(self.config['evaluation']['visualization']['params']['spatial_aggregation']) != bool:
            if not os.path.isdir(self.save_path+'/spatial_visualization'):
                os.mkdir(self.save_path+'/spatial_visualization')
    
    def per_sample_operations(self, outputs, labels, time_indexes, event_names, masks):
        """Performs per sample plot of the extreme event detection maps and the variables' saving for the global operations

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
        self.step = 0
        for output, label, time, event_name, mask in (pbar := tqdm(zip(outputs, labels, time_indexes, event_names, masks), total=len(outputs))):
            pbar.set_description('Performing visualization')
            if self.num_classes != 1:
                self.__plot_per_class_detection(output, label, event_name)    
            else:
                self.__plot_binary_detection(output, label, event_name)
            
            self.__save_global_results(output, label, mask, time)
            self.step += 1
    
    def __plot_per_class_detection(self, output, labels, sample_name):
        """Plots the probability of the samples belonging to a certain class

        :param output: Decision scores at the output of the model
        :type output: tensor
        :param labels: Samples' ground truth
        :type labels: tensor
        :param sample_name: Sample identifier
        :type sample_name: str or int
        """
        fig = plt.figure(figsize=(14,12))
        ax = plt.subplot(111)
        plt.title('Temporal evolution')
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.plot(labels/torch.max(labels), c = cycle[np.unique(labels)[-1]], label = 'Event GT', linestyle='dashed', linewidth=3)
        ax.fill_between(np.arange(len(labels/torch.max(labels))), labels/torch.max(labels), 0, color=cycle[np.unique(labels)[-1]], alpha=0.4)
        plt.ylim(bottom=-0.1)
        class_names= self.test_loader.dataset.classes if hasattr(self.test_loader.dataset, 'classes') \
            else [str(c) for c in np.arange(self.num_classes)]
        for num_class in range(1, self.num_classes):
            ax.plot(output[num_class], label = 'P(t) class ' + class_names[num_class], color=cycle[num_class], linewidth=3)
        ax.set_ylabel('P(t)', fontsize=20)    
        ax.set_xlabel('Time', fontsize=20)  
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.save_path+'/temporal_visualization/'+str(sample_name)+'_per_class_temporal_evolution.png', dpi=300)
        plt.close()
    
    def __plot_binary_detection(self, output, labels, sample_name):
        """Plots the probability of the samples belonging to a certain class

        :param output: Decision scores at the output of the model
        :type output: tensor
        :param labels: Samples' ground truth
        :type labels: tensor
        :param sample_name: Sample identifier
        :type sample_name: str or int
        """
        fig = plt.figure(figsize=(14,12))
        ax = plt.subplot(111)
        plt.title('Temporal evolution')
        ax.plot(labels[0], label = 'Event GT', linestyle='dashed', linewidth=3, color='red')
        ax.plot(output[0], label = 'P(t)', linewidth=3, color='dodgerblue')
        plt.ylim(top=1.1, bottom=-0.1)
        ax.set_ylabel('P(t)', fontsize=20)    
        ax.set_xlabel('Time', fontsize=20)  
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.save_path+'/temporal_visualization/'+str(sample_name)+'_temporal_evolution.png', dpi=300)
        plt.close()
    
    def __plot_spatial_per_class_detection(self, output, label, sample_name, mask):
        """Plot extreme event detection saliency map for each class and a given time step

        :param output: Decision scores at the output of the model
        :type output: tensor
        :param sample_name: Sample identifier
        :type sample_name: str or int
        :param mask: Quality mask
        :type mask: tensor
        """

        for num_class in range(self.num_classes):
            contourf_params = {'cmap':plt.cm.Reds, 'levels': np.linspace(0, 1.0, 11, endpoint=True), 'vmin':0, 'vmax':1}
            misc.plot_spatial_signal(self.save_path+'/spatial_visualization/c'+str(num_class)+'_'+sample_name, output[num_class-1], label[num_class-1], 
                                     mask[num_class-1], self.coordinates, contourf_params, 'Output Signal')


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
        :param labels: Samples' ground truth
        :type labels: tensor
        :param time: Time indexes
        :type time: tensor
        """
        if mask!=None: 
            mask = torch.prod(torch.prod(mask, 1),0)
            mask[mask==0] = np.nan

        categorized_output = self.__categorize(output)
        self.test_outputs.extend([x.item() for x in categorized_output.flatten()])        
        self.test_labels.extend([x.item() for x in labels.flatten()])
        self.temporal_plot_xticks_labels.append(time)
        
        if type(self.config['evaluation']['visualization']['params']['spatial_aggregation']) != bool:
            for num_class in range(self.num_classes):
                self.spatial_outputs[num_class, self.step] = output
                self.spatial_labels[num_class, self.step] = labels
                self.spatial_masks[num_class, self.step] = mask
    
    def global_operations(self):
        """
        Plot confusion matrix and visualize results through time
        """
        misc.plot_confusion_matrix(self.cm , torch.Tensor(self.test_outputs).int(), torch.Tensor(self.test_labels).long(), self.save_path)

        if type(self.config['evaluation']['visualization']['params']['spatial_aggregation']) != bool:
            shape = (self.num_classes,) + eval(self.config['evaluation']['visualization']['params']['spatial_aggregation'])
            spatial_output = self.spatial_outputs.reshape(shape)
            spatial_labels = self.spatial_labels.reshape(shape)
            spatial_masks = self.spatial_masks.reshape(shape)

            for sample in range(shape[-1]):
                self.__plot_spatial_per_class_detection(spatial_output[:,:,:,sample], spatial_labels[:,:,:,sample], str(sample), spatial_masks[:,:,:,sample])
                
        if self.config['evaluation']['visualization']['params']['time_aggregation']:
            misc.plot_temporal_results(self.save_path+'/', self.temporal_plot_xticks_labels, self.test_labels, self.test_outputs)

