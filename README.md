# Rethinking-SWATS-Optimiser
Rethinking SWATS (Switches from Adam to SGD) Optimiser. Switching Locally than Globally.

## Prerequisites

- PyTorch == 1.5.1

## Overview

Elementwise scaling of learning rate adopted by Adaptive Optimisers such as Adam, RMSProp etc often generalise poorly due to unstable and non-uniform learning rates at the end of training, although they scale well during the initial part of training. Hence SGD is the go-to for SOTA results since it generalizes better than adaptive methods.

SWATS is a method which switches from Adam to SGD when the difference between the bias corrected projected learning rate and the projected learning rate is less than a threshold ϵ. The switch is global i.e. if one of the layers of the network switches to SGD, all the layers are switched to SGD. 

## Switching Locally than Globally

Switching all the layers to SGD just because a particular layer switched to SGD is something which has to be investigated. Why should a Dense or Batch Norm Layer impact Convolutional layer's training? So I went ahead and made the switch local i.e a layer's switch to SGD is independent of other layer switching to SGD i.e. at a given time some layers will be in SGD Phase while the rest of them are still using Adam.

## Experiment

To perform the analysis I use ResNet-18 on Cifar-10 from <a href = "https://github.com/kuangliu/pytorch-cifar">this repository</a>.<br>
Since paper's default value of ϵ (threshold to switch) wasn't working for me, I've reduced ϵ value from 10^-9 to 10^-5. Since ϵ value is left as a hyperparameter in the paper, I guess this is okay to do :)

## Setup

- batch-size = 128
1. SWATS (GLOBAL SWITCH)
- initial step size for adam = 0.001
- 
- step size decay by a factor of 10 at 75,150 epochs for layers which are in SGD Phase.
- step size decay by a factor of 10 at 100 epochs for layers which are in Adam.
- step size decay by a factor of 10 at 100 epochs for AdaBound.
- epochs = 200

## Comparision

Since AdaBound is the only paper which I know which changes smoothly or in other words continuously transforms from Adam to SGD, rather than a hard switch like SWATS, it is fair to compare SWATS with AdaBound.

## Results

