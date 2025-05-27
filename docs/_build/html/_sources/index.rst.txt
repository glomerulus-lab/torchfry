.. Fastfood documentation master file, created by
   sphinx-quickstart on Mon May 12 13:13:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fastfood Documentation
======================

Fastfood Torch is a software package that implements the Fastfood algorithm 
(Le, Sarlos, Smola, 2013) and Random Kitchen Sink algorithm (Rahimi, Recht, 2007) 
as custom PyTorch layers for use in different networks for more efficient kernel 
approximation. 

Our motivation for implementing the Fastfood algorithm with PyTorch is to create an easy
to use repository which can make kernel methods scalable to high-dimensional datasets. 
Through testing, we have found training times of Fastfood networks to be faster than 
Random Kitchen Sink networks. The improvement is about 20% decrease in time. The parameter
count of Fastfood networks is much lower than Random Kitchen Sink while retaining the 
same image classification performance. 


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   FastFoodLayer
   RKSLayer
