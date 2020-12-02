CR-Vision: Amazing Computer Vision Pipelines
==============================================

* `Documentation <https://cr-vision.carnotresearch.com/index.html>`_
* `GITHUB <https://github.com/carnotresearch/cr-vision>`_
* `Issue Tracker <https://github.com/carnotresearch/cr-vision/issues>`_
* `About Us <https://www.carnotresearch.com/>`_

Overview
----------

This package is an excellent collection of modules and 
scripts for day to day computer vision tasks. 

It builds on top of extensive ecosystem of image
processing, computer vision and deep learning libraries
available in Python including OpenCV, ImageIO, Pillow,
TensorFlow, Keras, Scikit-Image and Scikit-Video.
It also leverages other scientific computing and
machine learning packages.

It uses *Rx: Reactive Extensions for Python* to build
sophisticated, optimized, stream oriented and push based
computer vision pipelines. This is great for large 
computer vision applications involving a number of 
moving and sophisticated sub-systems.

Available functionality spans following areas:

* Basic image processing operations
* A variety of filters and effects
* Basic editing of images
* Image restoration (denoising, super-resolution)
* Face detection
* Template matching
* Pedestrian detection
* Image classification
* Traffic monitoring



Installing
--------------------

.. highlight:: shell

From PIP:: 

    python -m pip install cr-vision


Directly from GITHUB::

    python -m pip install git+https://github.com/carnotresearch/cr-vision.git


Working with the source code in development mode
-----------------------------------------------------


Clone the repository::

    git clone https://github.com/carnotresearch/cr-vision.git


Change into the code::

    cd cr-vision


Ensure that the dependencies are installed::

    python -m pip install -r requirements.txt


Install the package in development mode::

    python -m pip install -e .


Examples
-----------------


Explore the examples directory::

    cd examples/basic


Run an example::

    python ex_add_logo.py
