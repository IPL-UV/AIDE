.. _section-identifier_characterization:

.. role:: raw-html(raw)
   :format: html

Characterization
=================

For extreme events Classification, the toolbox provides a post-processing step of data characterization. Characterization is a methodology for summarising and describing the characteristics of a priori aggregated locations that constitute the events. When activated, this post-processing stage provides a .txt file with the following statistics:

    - Extent or the number of locations (pixels) covered by the event.
    - Centroid and weighted Centroid.
    - Maximum, minimum and mean probabilities provided by the model in the event region. 
    

Input Parameters
~~~~~~~~~~~~~~~~~~

To use this funcionality, use the following code snippet in your configuration file:

.. code-block:: yaml 

    characterization: 
        activate: True
        params:
            time_aggregation: ... 
            min_distance: ... 
            remove_scant: ... 
            min_area_holes: ... 
            min_area_objects: ... 
            threshold:
                Metric: {...}
            threshold_lower_is_best: ...


- :raw-html:`<font color="#008000">time_aggregation</font>`: Aggreggate through time, if test samples are taken chronologically ,type: bool
- :raw-html:`<font color="#008000"> min_distance</font>`: Maximum Euclidean distance between centroids to be connect, to deactivate it, set to zero (default), type: int
- :raw-html:`<font color="#008000">remove_scant</font>`: Remove scant labels by filling small holes and small objects, type: bool
- :raw-html:`<font color="#008000">min_area_holes</font>`: Size of the holes to remove when remove_scant set to True, type: int
- :raw-html:`<font color="#008000">min_area_objects</font>`: Size of the objects to remove when remove_scant set to True, type: int
- :raw-html:`<font color="#008000">threshold</font>` > :raw-html:`<font color="#008000">Metric</font>`: Metric to optimize the threshold, by default takes a value of 0.5, type: TorchMetric
- :raw-html:`<font color="#008000">threshold_lower_is_best</font>`: Sets the direction that means improvement, type: bool

Outputs
~~~~~~~~

Example of .txt file generated after executing the Characterization module:

.. image:: _static/images/characterization_1.png
  :width: 800
  :align: center
  :alt: 

.. image:: _static/images/characterization_2.png
  :width: 800
  :align: center
  :alt: 



