#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

class OutlierDetectionVisualizer:
    """Class for visualization of results provided by Python Outlier Detection models
    """
    def __init__(self, config, models, dataset, step_samples=1):
        """__init__

        :param config: A dictionary containing all the configuration variables for the experiment.
        :type config: dict
        :param models: Models to be evaluated.
        :type models: PyodModel
        :param dataset: Dataset to evaluate the models.
        :type dataset: torch.utils.data.Dataset
        :param step_samples: Step to select 1 out of N samples for visualization purposes, defaults to 1
        :type step_samples: int, optional
        """
        self.config = config
        self.models = models
        self.dataset = dataset
        self.step_samples = step_samples
        self.arch_args = config['arch']['args'] 
        self.save_path = config['save_path']

        self.__print_models()
    
    def __print_models(self):
        """Print models' information.
        """
        for i, model in enumerate(self.models.keys()):
            print('Model', i + 1, model)

    def visualize(self):
        """Visualization of results.
        """
        x = self.dataset['x']
        labels = self.dataset['labels']

        # Adapt_variables
        x = x.reshape(x.shape[0],-1).transpose()
        labels = labels.reshape(-1)
        if 'masks' in self.dataset.keys():
            masks = self.dataset['masks']
            masks = masks.reshape(masks.shape[0],-1).transpose()
            masks = np.all(masks,axis=1)

            x = x[masks]
            labels = labels[masks]

        x = x[::self.step_samples]
        labels = labels[::self.step_samples]
        
        n_features = [x for x in range(len(self.config['data']['features_selected']))]

        vmin = -4
        vmax = 4

        for model_name, model in self.models.items():
        
            variable_pairs = list(product(n_features, n_features))
            n_vars = len(n_features)
            
            fig, ax = plt.subplots(n_vars, n_vars, figsize=(4 * n_vars, 4 * n_vars))
            ax = np.atleast_2d(ax)  # Ensures ax[i, j] works even for 1x1 case
        
            for pair in variable_pairs:
                i, j = pair
        
                axis = ax[i, j]
                
                if i == j:
                    # Diagonal: Histogram
                    axis.hist(x[:, i], bins=1000, density=True)
                    axis.set_xlabel(self.config['data']['features'][self.config['data']['features_selected'][i]])
                    axis.set_ylabel(self.config['data']['features'][self.config['data']['features_selected'][j]])
                else:
                    # Off-diagonal: Decision boundary and scatter
                    output = model.predict(x)
                    scores_pred = model.decision_function(x) * -1
                    threshold = np.percentile(scores_pred, 100 * self.arch_args['contamination'])
        
                    # Meshgrid for contour
                    xx, yy = np.meshgrid(np.linspace(vmin, vmax, 100),
                                         np.linspace(vmin, vmax, 100))
        
                    zero_column = np.zeros_like(xx.ravel())
                    xyzw = []
                    for k in range(n_vars):
                        if k == i:
                            xyzw.append(xx.ravel())
                        elif k == j:
                            xyzw.append(yy.ravel())
                        else:
                            xyzw.append(zero_column)
        
                    xyzw_pair = np.transpose(np.array(xyzw))
                    Z_pair = model.decision_function(xyzw_pair) * -1
                    Z_pair = Z_pair.reshape(xx.shape)
        
                    contour_lims = np.array([Z_pair.min(), threshold])
        
                    axis.contourf(xx, yy, Z_pair,
                                  levels=np.linspace(contour_lims.min(), contour_lims.max(), 7),
                                  cmap=plt.cm.Blues_r)
        
                    inliers = axis.scatter(x[:, i][labels == 0], x[:, j][labels == 0],
                                           c='white', s=20, edgecolor='k', label='true inliers')
                    outliers = axis.scatter(x[:, i][labels == 1], x[:, j][labels == 1],
                                            c='black', s=20, edgecolor='k', label='true outliers')
        
                    axis.set_xlabel(self.config['data']['features'][self.config['data']['features_selected'][i]])
                    axis.set_ylabel(self.config['data']['features'][self.config['data']['features_selected'][j]])
                    axis.set_xlim((vmin, vmax))
                    axis.set_ylim((vmin, vmax))
        
            # Suptitle
            fig.suptitle("Outlier Detection", fontsize=16, y=1.02)
        
            # Legend (only once, outside subplot loop)
            if n_vars > 1:
                fig.legend(
                    handles=[inliers, outliers],
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.01),
                    ncol=2,
                    prop=matplotlib.font_manager.FontProperties(size=10)
                )
        
            # Adjust layout: reserve space for suptitle
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
            # Save and show
            plt.savefig(f"{self.save_path}/outlier_detection_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.show()