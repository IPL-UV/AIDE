#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

class OutlierDetectionVisualizer:
    def __init__(self, config, models, dataset, step_samples=1):
        self.config = config
        self.models = models
        self.dataset = dataset
        self.step_samples = step_samples
        self.arch_args = config['arch']['args'] 
        self.save_path = config['save_path']

        self.__print_models()
    
    def __print_models(self):
        for i, model in enumerate(self.models.keys()):
            print('Model', i + 1, model)

    def visualize(self):
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

        for i, (model_name, model) in enumerate(self.models.items()):
                        
            variable_pairs = list(product(n_features, n_features))                

            fig, ax  = plt.subplots(len(n_features), len(n_features))
            for i, pair in enumerate(variable_pairs):

                if pair[0] == pair[1]:
                    try:
                        ax[pair[0], pair[1]].hist(x[:,pair[0]], bins=1000, density=True)
                        ax[pair[0], pair[1]].set_xlabel(self.config['data']['features'][self.config['data']['features_selected'][pair[0]]])
                        ax[pair[0], pair[1]].set_ylabel(self.config['data']['features'][self.config['data']['features_selected'][pair[1]]])
                    except:
                        ax.hist(x[:,pair[0]], bins=1000, density=True)
                        ax.set_xlabel(self.config['data']['features'][self.config['data']['features_selected'][pair[0]]])
                        ax.set_ylabel(self.config['data']['features'][self.config['data']['features_selected'][pair[1]]])
                    
                else:
                    output = model.predict(x)
                    n_errors = (output != labels).sum()

                    scores_pred = model.decision_function(x) * -1
                    threshold = np.percentile(scores_pred, 100 * self.arch_args['contamination'])
                        
                    xx, yy = np.meshgrid(np.linspace(vmin, vmax, 100), np.linspace(vmin, vmax, 100))

                    zero_colum = np.zeros_like(xx.ravel())
                    xyzw =  []
                    for element in range(len(n_features)):
                        if element == pair[0]:
                            xyzw.append(xx.ravel())
                        elif element == pair[1]:
                            xyzw.append(yy.ravel())
                        else:
                            xyzw.append(zero_colum)
                    xyzw_pair = np.transpose(np.array(xyzw))
                    Z_pair = model.decision_function(xyzw_pair) * -1
                    Z_pair = Z_pair.reshape(xx.shape)

                    countor_lims = np.array([Z_pair.min(), threshold])

                    ax[pair[0], pair[1]].contourf(xx, yy, Z_pair, levels=np.linspace(countor_lims.min(), countor_lims.max(), 7),
                                cmap=plt.cm.Blues_r)

                    inliers = ax[pair[0], pair[1]].scatter(x[:,pair[0]][labels==0], x[:,pair[1]][labels==0], c='white',
                                    s=20, edgecolor='k')
                    outliers = ax[pair[0], pair[1]].scatter( x[:,pair[0]][labels==1], x[:,pair[1]][labels==1], c='black',
                                    s=20, edgecolor='k')

                    ax[pair[0], pair[1]].set_xlabel(self.config['data']['features'][self.config['data']['features_selected'][pair[0]]])
                    ax[pair[0], pair[1]].set_ylabel(self.config['data']['features'][self.config['data']['features_selected'][pair[1]]])
                    ax[pair[0], pair[1]].set_xlim((vmin,vmax))
                    ax[pair[0], pair[1]].set_ylim((vmin,vmax))
                    plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
                    plt.suptitle("Outlier detection")
                    
            if len(n_features) > 1:
                fig.legend(
                            [inliers, outliers],
                            ['true inliers', 'true outliers'],
                            prop=matplotlib.font_manager.FontProperties(size=10),
                            loc='lower right', bbox_to_anchor=(1,-0.1))
            fig.tight_layout()
            plt.savefig(self.save_path+'/outlier_detection_'+model_name+'.png', dpi=300)
            plt.show()