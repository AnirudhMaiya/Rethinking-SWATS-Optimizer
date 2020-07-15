# Rethinking-SWATS-Optimizer
Rethinking SWATS (Switches from Adam to SGD) Optimizer. Performing the switch Locally than Globally.

## Prerequisites

- PyTorch == 1.5.1

## Overview

Elementwise scaling of learning rate adopted by Adaptive Optimizers such as Adam, RMSProp etc often generalise poorly due to unstable and non-uniform learning rates at the end of training, although they scale well during the initial part of training. Hence SGD is the go-to for SOTA results since it generalizes better than adaptive methods.

SWATS is a method which switches from Adam to SGD when the difference between the bias corrected projected learning rate and the projected learning rate is less than a threshold ϵ.The projected learning rate is found by projecting the SGD's update onto Adam's update. The switch is global i.e. if one of the layers of the network switches to SGD, all the layers are switched to SGD. 

## Switching Locally than Globally

Switching all the layers to SGD just because a particular layer switched to SGD is something which has to be investigated. Why should a Dense or Batch Norm Layer impact Convolutional layer's training? So I went ahead and made the switch local i.e a layer's switch to SGD is independent of other layer switching to SGD i.e. at a given time some layers will be in SGD Phase while the rest of them are still using Adam.

## Experiment

To perform the analysis I use ResNet-18 on Cifar-10 from <a href = "https://github.com/kuangliu/pytorch-cifar">this repository</a>.<br>
Since paper's default value of ϵ (threshold to switch) wasn't working for me, I've reduced ϵ value from 10^-9 to 10^-5. Since ϵ value is left as a hyperparameter in the paper, I guess this is okay to do :)

## Setup

- batch-size = 128
- epochs = 200

1. SWATS (GLOBAL SWITCH)
    - initial step size (adam) 0.001
    - step size decay by a factor of 10 at 100 epochs.

2. New SWATS (LOCAL SWITCH)
    - step size decay by a factor of 10 at 75,150 epochs for layers which are in SGD Phase.
    - step size decay by a factor of 10 at 100 epochs for layers which are in Adam.

3. Adabound
    - step size decay by a factor of 10 at 100 epochs for AdaBound.

## Comparision

Since AdaBound is the only paper which I know which changes smoothly or in other words continuously transforms from Adam to SGD, rather than a hard switch like SWATS, it is fair to compare SWATS with AdaBound.

## Results

| Model   | Optimizer | Switch  | Test Acc.  |
| ------- | -------- | ------- |-----------|
| ResNet-18 | SWATS | Global (Vanilla) |  92.89 |
| ResNet-18 | SWATS | Local | 94.13 |
| ResNet-18 | AdaBound | NA | 93.0 |


## Switch Over Points(Steps) for Local Switch
| Layer | Steps | Estimated Learning Rate For SGD |
| ------- | -------- | ------- |
linear.weight | 29 | 0.015315 |
layer1.0.conv1.weight | 1011 | 0.149228 |
layer2.0.conv1.weight | 2354 | 0.673763 |
layer1.1.bn1.bias | 2597 | 0.204322 |
layer2.1.bn1.weight | 3230 | 0.416590 |
layer2.0.shortcut.0.weight | 3415 | 0.278827 |
layer1.0.bn1.bias | 3690 | 0.156899 |
bn1.bias | 4850 | 0.117717 |
layer1.1.bn2.bias |5574| 0.320231|
linear.bias |5645 | 0.015616|
layer3.0.bn2.weight |5744 | 0.420847|
layer2.1.bn1.bias |6897 | 0.378199|
layer3.0.bn1.bias |6996 | 0.599258|
layer2.0.shortcut.1.weight |7972| 0.276019|
conv1.weight| 7994 |0.042911|
layer1.0.conv2.weight| 8079| 0.649479|
layer1.1.bn2.weight| 10320 |0.157036|
layer2.0.bn2.bias| 11477 | 0.382424|
layer2.0.shortcut.1.bias |11477 |0.382424|
layer3.0.shortcut.0.weight |11729 |1.180122|
