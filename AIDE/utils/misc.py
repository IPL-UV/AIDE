#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator

mpl.rcParams['agg.path.chunksize'] = 10000

def plot_confusion_matrix(cm , outputs, labels, path):
    """Compute and plot confusion matrix for evaluation

    :param cm: Confusion matrix object
    :type cm: class: 'torchmetrics.ConfusionMatrix'
    :param outputs: Categorized model output
    :type outputs: tensor
    :param labels: Samples' ground truth 
    :type labels: tensor
    :param path: Path to save the generated figures
    :type path: str
    """

    plt.figure(figsize=(12,10))
    cm = cm(outputs, labels)
    sn.set(font_scale=1.4)
    sn.heatmap(np.round(cm,2), annot=True, annot_kws={"size": 18}, fmt="g", vmax=1)
    plt.xlabel('Predicted labels', fontsize=18)
    plt.ylabel('True labels', fontsize=18)
    plt.savefig(path+'/confusionMatrix.png', dpi=300)
    sn.set(font_scale=1) #Reset font scale

def plot_temporal_results(path, xticks_labels, labels, outputs_when_event, output_without_event=None):
    """Plot extreme event detection signals through time:
        - Ground truth extreme event signal
        - Extreme signal (Aggregated mean over region affected by the extreme event)
        - Non-extreme signal (Aggregated mean over region non-affected by the extreme event)

    :param path: Path to save the generated figures
    :type path: str
    :param xticks_labels: X axis' labels
    :type xticks_labels: list
    :param labels: Samples's ground truth
    :type labels: list
    :param outputs_when_event: Aggregated mean over region affected by the extreme event
    :type outputs_when_event: list
    :param output_without_event: Aggregated mean over region non-affected by the extreme event, defaults to None
    :type output_without_event: list, optional
    """

    fig = plt.figure(figsize=(14,12))
    ax = plt.subplot(111)
    plt.title('Temporal evolution')
    ax.plot(labels, c = 'tab:red', label = 'Event GT', linewidth=3)
    ax.plot(outputs_when_event, c = 'tab:purple', label = 'Event signal', linewidth=3)
    if output_without_event != None:
        ax.plot(output_without_event, c = 'tab:blue', label = 'Background signal', linewidth=3)
    ax.set_xticks(np.arange(len(xticks_labels))[::5])
    ax.set_xticklabels(xticks_labels[::5], rotation=60)    
    ax.set_ylabel('P(t)', fontsize=20)    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(path+'temporalEvolution.png', dpi=300)
    plt.close()


def plot_spatial_aggregation(path, output, label, mask, coordinates, num_classes):
    """Plot extreme event detection saliency map for a given time step

    :param path: Path to save the generated figures
    :type path: str
    :param output: Categorized model output 
    :type output:
    :param label: Samples' ground truth 
    :type label: 
    :param mask: Quality mask
    :type mask: 
    :param coordinates: Bounding box of the region of interest (lon_min, lon_max, lat_min, lat_max)
    :type coordinates: tuple
    :param num_classes: Number of classess in the classification
    :type num_classes: int
    """

    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    plt.title('Detection map')
    cmap = plt.cm.Reds
    boundaries = np.linspace(0, num_classes-1, num_classes+1)
    ticks = [x for x in range(num_classes+1)]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    im = ax.contourf(np.linspace(coordinates[0],coordinates[1],output.shape[1]),
                     np.linspace(coordinates[3],coordinates[2],output.shape[0])[::-1],
                     output*mask if mask!=None else output, transform=ccrs.PlateCarree(), cmap=plt.cm.Reds, vmin=0, vmax=num_classes, levels = boundaries, norm=norm)

    cb = plt.colorbar(im, ticks=ticks, ax=ax)    
    cb.ax.set_title('Class')
    ax.set_extent(coordinates)
    ax.coastlines(resolution='50m')
    if not torch.all(label == 0):
        line_c = ax.contour(label, extent=coordinates, colors=['black'], transform = ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.25, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    plt.savefig(path+'_spatialEvolution.png', dpi=300)
    plt.close()

def plot_spatial_signal(path, output, label, mask, coordinates, contourf_params, title):
    """Plot extreme event detection categorical map for a given time step

    :param path: Path to save the generated figures
    :type path: str
    :param output: Decision scores at the output of the model
    :type output: 
    :param label: Samples' ground truth 
    :type label: 
    :param mask: Quality mask
    :type mask: 
    :param coordinates: Bounding box of the region of interest (lon_min, lon_max, lat_min, lat_max)
    :type coordinates: tuple
    :param contourf_params: Dictionary containing customizable parameters for the contourf function
    :type contourf_params: dict 
    :param title: Title of the generated visualization
    :type title: str
    """

    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    plt.title(title)
    eps = 1e-7
    v = np.linspace(0, 1.0, 11, endpoint=True)
    im = ax.contourf(np.linspace(coordinates[0],coordinates[1],torch.squeeze(output).shape[1]),
                     np.linspace(coordinates[3],coordinates[2],torch.squeeze(output).shape[0])[::-1],
                     torch.squeeze(output*mask) if mask!=None else torch.squeeze(output),
                     v, transform=ccrs.PlateCarree(), **contourf_params)
    cb = plt.colorbar(im, ax=ax)#, ticks=v)
    cb.ax.set_title('P(t)')
    ax.set_extent(coordinates)
    ax.coastlines(resolution='50m')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.25, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    if not torch.all(label == 0):
        line_c = ax.contour(label, extent=coordinates, colors=['black'], transform = ccrs.PlateCarree())
    plt.savefig(path+'_spatialEvolution.png', dpi=300)
    plt.close()

    
