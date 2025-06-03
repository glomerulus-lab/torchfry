.. Fastfood documentation master file, created by
   sphinx-quickstart on Mon May 12 13:13:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Torched and Fried Documentation
===============================

`Torched and Fried`_ (``torchfry``) is a software package that implements kernel
approximation algorithms `Fastfood (Le, Sarlos, Smola, 2013)`_ 
and `Random Kitchen Sink (Rahimi, Recht, 2007)`_ as custom PyTorch layers for use in 
different networks. 

Our motivation for implementing the Fastfood algorithm with PyTorch is to create an easy
to use package that makes kernel methods scalable to high-dimensional datasets through
efficient random feature layers like that in 
`Deep Fried Convnets (Yang, Moczulski, Demil, et al., 2014)`_. 
Through testing, we have found training times of Fastfood networks to be faster than 
Random Kitchen Sink networks. The improvement is about 20% decrease in time. The parameter
count of Fastfood networks is much lower than those for Random Kitchen Sink networks, all 
while retaining the same image classification performance. 

.. _Torched and Fried: https://github.com/glomerulus-lab/torchfry
.. _Fastfood (Le, Sarlos, Smola, 2013): https://arxiv.org/abs/1408.3060
.. _Random Kitchen Sink (Rahimi, Recht, 2007): https://dl.acm.org/doi/10.5555/2981562.2981710
.. _Deep Fried Convnets (Yang, Moczulski, Demil, et al., 2014): https://arxiv.org/abs/1412.7149

.. toctree::
   :maxdepth: 1
   :caption: Table of Contents

   torchfry.networks
   torchfry.transforms
