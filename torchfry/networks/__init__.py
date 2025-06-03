# __init__.py
"""
Neural network implementations using the Fastfood and Random Kitchen Sink layers.
"""
from .LeNet import LeNet
from .MLP import MLP
from .VGG import VGG

__all__ = ['LeNet', 'MLP', 'VGG']