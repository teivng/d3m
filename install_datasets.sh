#!/bin/bash

echo "Installing the UCI Heart Disease dataset..."
mkdir -p data/uci_data
wget -O data/uci_data/uci_heart_torch.pt https://github.com/rgklab/detectron/raw/main/data/sample_data/uci_heart_torch.pt

echo "Installing the CIFAR-10.1 dataset..."
mkdir -p data/cifar10_data
wget -O data/cifar10_data/cifar10.1_v6_data.npy https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy
wget -O data/cifar10_data/cifar10.1_v6_labels.npy https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy