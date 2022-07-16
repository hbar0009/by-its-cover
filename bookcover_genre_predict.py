# imports
import os
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor


# Downloading the dataset
dataset_url = 'https://raw.githubusercontent.com/uchidalab/book-dataset/master/Task1/book30-listing-test.csv'
download_url(dataset_url, '.')