# Simple Federated Learning with [Flower](http://flower.dev)
* By Seok-Ju Hahn (seokjuhahn@unist.ac.kr)
## Requirements
```
flwr>=0.17
torch>=1.10
torchvision>=0.9
PyTDC>=0.3
rdkit-pypi>=2021.9.2.1
scikit-learn>=1.0.1
```
or you can simply execute:
`pip install -r requirments.txt`
## Run (example)
```
python3 server.py --dataset_name mnist --iid \
-N 100 -K 0.1 -R 500 -B 10 -E 5 \
--test_fraction 0 \
--seed 951023
```
## Configurations
* `--dataset_name`: (required) which data to use for federated learning: {MNIST | CIFAR10 | TOX21}
* `-N` or `--num_clients`: total number of clients participating in federated learning
* `-K` or `--fraction`: fraction of participating clients at each round
* `-B` or `--batch_size`: batch size for client-side update/evaluation
* `-E` or `--num_epochs`: number of local epochs required for client-side update
* `-R` or `--num_rounds`: number of total rounds
* `-S` or `--num_shards`: number of resulting shards used for splitting dataset in non-IID setting (valid only if `--iid` argument is not passed)
* `--iid`: wheter to split data in an IID manner; if not used, the dataset is split into *pathological non-IID* setting propsed in [(McMahan et al., 2017)](https://arxiv.org/abs/1602.05629)
* `test_fraction`: fraction of testset for each client (set to zero if only centralized evluation (i.e. server-side evaluation)) is required)
* `--data_path`: path to store dataset
* `--seed`: random seed

## Datasets
### Tox21
**Dataset Description:**: Tox21 is a data challenge which contains qualitative toxicity measurements for 7,831 compounds on 12 different targets, such as nuclear receptors and stree response pathways.

**Task Description:** Binary classification. Given a drug SMILES string, predict the toxicity in a specific assay.

**Command**:  `sh commands/run_tox.sh`

**References:** [Tox21 Challenge.](https://www.frontiersin.org/research-topics/2954/tox21-challenge-to-build-predictive-models-of-nuclear-receptor-and-stress-response-pathways-as-media)


### MNIST
**Dataset Description:**: The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples with the size of 28x28.

**Task Description:** Multiclass classification. Given a gray-scale handwritten digit image, predict its label (0-9). 

**Command**:  `sh commands/run_mnist.sh`

**References:** [MNIST Database](http://yann.lecun.com/exdb/mnist/)


### CIFAR10
**Dataset Description:**: The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50,000 training images and 10,000 test images.

**Command**:  `sh commands/run_cifar.sh`

**Task Description:** Multiclass classification. Given a 3-channel image, predict its label (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 

**References:** [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
