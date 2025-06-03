<h1>
  Torched and Fried
</h1>

Torched and Fried is a software package that implements kernel approximation algorithms [Fastfood (Le, Sarlos, Smola, 2013)](https://arxiv.org/abs/1408.3060) and [Random Kitchen Sink (Rahimi, Recht, 2007)](https://dl.acm.org/doi/10.5555/2981562.2981710) into custom PyTorch layers. This allows for greater storage and computational efficiency in networks by reducing parameter counts with random feature layers, similarly to [Deep Fried Convnets (Yang, Moczulski, Demil, et al., 2014)](https://arxiv.org/abs/1412.7149). This package is distributed under the MIT License.

The performances of these layers are tested against each other in different networks with various hyperparameters, primarily Fashion MNIST and CIFAR-10 datasets.

<h2>Features</h2>

- **Transforms**: Custom PyTorch layers integrating Fastfood and RKS algorithms.
- **Networks**: Includes LeNet, MLP, and VGG network architectures that utilize the custom PyTorch layers to test on Fashion MNIST and CIFAR-10 datasets.
- **Performance Benchmarking**: Tools to compare the speed, accuracy, hyperparameter count, etc. of the networks, transforms, and low-level operations within the transforms.

<h2>Requirements</h2>

- **PyTorch** (tested with 2.5.1) 
- **SciPy** (tested with 1.14.1)
- **torchvision** (tested with 0.20.1)

**torchvision** is primarily used for the VGG model, along with files in the `tests` directory.

The following packages are needed for the files in the `tests` directory:
- **scikit-learn** (== 1.5.2)
- **scikit-learn-extra** (== 0.3.0)
- **matplotlib**
- **seaborn**

<h2>Installation</h2>

Navigate to the directory containing the project's `setup.py` file and run:
```
pip install .
```
This will install PyTorch (and its dependencies), SciPy, and the Torched and Fried module (named torchfry). 

The Fastfood layer can use different implementations of the Hadamard matrix transformation, each with different speeds. The test results included in this document utilize an implementation of the Fast Hadamard Transform (Dao, [2022](https://github.com/Dao-AILab/fast-hadamard-transform)), written in CUDA to leverage GPU parallelization with a PyTorch interface for easy use.

<h2>Usage</h2>

```
from torchfry.networks import LeNet, MLP, VGG
from torchfry.transforms import FastfoodLayer, RKSLayer
```

<h2>Speed</h2>

TODO: Compile more results with consistent and good labeling, create tables showing results across different networks, and brief discussion of results.

![](./tests/plots/ff_rks_l_lr.png)

---

![](./tests/plots/ff_rks_lu_scale.png)

---

This project includes modified and unmodified code from [**structured-nets**](https://github.com/HazyResearch/structured-nets) (originally licensed under the Apache License 2.0) and [**OnLearningTheKernel**](https://github.com/cs1160701/OnLearningTheKernel) (no original license).

See NOTICE for details.