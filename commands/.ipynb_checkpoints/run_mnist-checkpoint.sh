#!/bin/bash

#########################################
# Seok-Ju Hahn (seokjuhahn@unist.ac.kr) #
#########################################

## MNIST ##
# MNIST experiment with IID setting - centralized evaluation
python3 server.py --dataset_name mnist --iid \
-N 100 -K 0.1 -R 500 -B 10 -E 5
--test_fraction 0

# MNIST experiment with IID setting - federated evaluation
python3 server.py --dataset_name mnist --iid \
-N 100 -K 0.1 -R 500 -B 10 -E 5
--test_fraction 0.2

# MNIST experiment with Pathological non-IID setting - centralized evaluation
python3 server.py --dataset_name mnist \
-N 100 -K 0.1 -R 500 -B 10 -E 5 -S 200
--test_fraction 0

# MNIST experiment with Pathological non-IID setting - federated evaluation
python3 server.py --dataset_name mnist \
-N 100 -K 0.1 -R 500 -B 10 -E 5 -S 200
--test_fraction 0.2

