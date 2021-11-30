#!/bin/bash

#########################################
# Seok-Ju Hahn (seokjuhahn@unist.ac.kr) #
#########################################

## CIFAR10 ##
# CIFAR10 experiment with IID setting - centralized evaluation
python3 server.py --dataset_name cifar10 --iid \
-N 100 -K 0.1 -R 500 -B 10 -E 5
--test_fraction 0

# CIFAR10 experiment with IID setting - federated evaluation
python3 server.py --dataset_name cifar10 --iid \
-N 100 -K 0.1 -R 500 -B 10 -E 5
--test_fraction 0.2

# CIFAR10 experiment with Pathological non-IID setting - centralized evaluation
python3 server.py --dataset_name cifar10 \
-N 100 -K 0.1 -R 500 -B 10 -E 5 -S 200
--test_fraction 0

# CIFAR10 experiment with Pathological non-IID setting - federated evaluation
python3 server.py --dataset_name cifar10 \
-N 100 -K 0.1 -R 500 -B 10 -E 5 -S 200
--test_fraction 0.2

