#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cycler import cycler

from .classificationVisualizerAbstract import *

class ClassificationVisualizer1D(ClassificationVisualizerAbstract):
    """
    Evaluator for 1D outputs
    """
    def __init__(self, config, model, dataloader):
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

        self.__create_evaluator_dirs()
    
    def __create_evaluator_dirs(self):
        """ 
        Create directories to save figures for visualization
        """
        if not os.path.isdir(self.save_path+'/temporal_visualization'):
            os.mkdir(self.save_path+'/temporal_visualization')
    
    def per_sample_operations(self, outputs, labels, time_indexes, event_names):
        """
        Performs per sample plot of the extreme event detection maps and the variables' saving for the global operations
        """
        for output, label, time, event_name in (pbar := tqdm(zip(outputs, labels, time_indexes, event_names), total=len(outputs))):
            pbar.set_description('Performing visualization')
            if self.num_classes != 1:
                self.__plot_per_class_detection(output, label, event_name)
            self.__save_global_results(self.__categorize(output), label, time)
    
    def __plot_per_class_detection(self, output, labels, sample_name):
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

    def __categorize(self,output):
        if self.num_classes == 1:
            return (output[0] > 0.5).int()
        else:
            return torch.max(output, dim=0)[1]

    def __save_global_results(self, output, labels, time):
        """ 
        Prepare data for visualization
        """
        self.test_outputs.extend([x.item() for x in output.flatten()])        
        self.test_labels.extend([x.item() for x in labels.flatten()])
        self.temporal_plot_xticks_labels.append(time)
    
    def global_operations(self):
        """
        Plot confusion matrix and visualize results through time
        """
        misc.plot_confusion_matrix(self.cm , torch.Tensor(self.test_outputs).int(), torch.Tensor(self.test_labels).long(), self.save_path)
        if self.config['evaluation']['visualization']['params']['time_aggregation']:
            misc.plot_temporal_results(self.save_path+'/', self.temporal_plot_xticks_labels, self.test_labels, self.test_outputs)

