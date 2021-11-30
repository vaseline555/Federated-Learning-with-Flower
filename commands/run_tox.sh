#!/bin/bash

#########################################
# Seok-Ju Hahn (seokjuhahn@unist.ac.kr) #
#########################################

## Tox21 ##
# Tox21 experiment with IID setting - centralized evaluation
python3 server.py --dataset_name tox21 --iid \
-N 5 -K 1.0 -R 50 -B 128 -E 5
--test_fraction 0

# Tox21 experiment with IID setting - federated evaluation
python3 server.py --dataset_name tox21 --iid \
-N 5 -K 1.0 -R 50 -B 128 -E 5
--test_fraction 0.2

# Tox21 experiment with Pathological non-IID setting - centralized evaluation
python3 server.py --dataset_name tox21 \
-N 5 -K 1.0 -R 50 -B 128 -E 5
--test_fraction 0

# Tox21 experiment with Pathological non-IID setting - federated evaluation
python3 server.py --dataset_name tox21 \
-N 5 -K 1.0 -R 50 -B 128 -E 5
--test_fraction 0

