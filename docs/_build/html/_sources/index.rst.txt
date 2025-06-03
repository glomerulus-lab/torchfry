.. Fastfood documentation master file, created by
   sphinx-quickstart on Mon May 12 13:13:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Torch & Fried Documentation
===========================

Torchfry is a software package that implements the `Fastfood`_ algorithm 
(Le, Sarlos, Smola, 2013) and `Random Kitchen Sink`_ algorithm (Rahimi, Recht, 2007) 
as custom PyTorch layers for use in different networks for more efficient kernel 
approximation. 

Our motivation for implementing the these algorithms with PyTorch is to create an easy
to use repository which can make kernel methods scalable to high-dimensional datasets. 
Through testing, we have found training times of Fastfood networks to be faster than 
Random Kitchen Sink networks. The parametercount of Fastfood networks is much lower than Random 
Kitchen Sink while retaining the same image classification performance. 

.. _Fastfood: https://arxiv.org/pdf/1408.3060
.. _Random Kitchen Sink: https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   FastFoodLayer
   RKSLayer
