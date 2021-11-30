import os
import sys
import torch
import torchvision
import flwr as fl
import numpy as np
from torch.utils.data import Dataset
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
from rdkit import Chem
import sklearn.metrics as metrics

def smiles_encoder(smiles, maxlen=386):
    SMILES_CHARS = [' ',
                  '#', '%', '(', ')', '+', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@',
                  'A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Y', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                  't', 'u', 'y']
    smi2index = dict((c, i) for i,c in enumerate(SMILES_CHARS))
    
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    one_hot = np.zeros((maxlen - 2, len(SMILES_CHARS)))
    for i, c in enumerate(smiles):
        one_hot[i, smi2index[c]] = 1
    one_hot = np.concatenate([
        np.zeros((1, len(SMILES_CHARS))),
        one_hot,
        np.zeros((1, len(SMILES_CHARS)))
    ])
    return one_hot

def npz_loader(path):
    sample = np.load(path)
    x, y = torch.from_numpy(sample['x']), torch.from_numpy(sample['y'])
    return (x, y)

def train(net, trainloader, epochs, device, flag):
    """Train the network."""
    # Define loss and optimizer
    if flag:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    
    # Train the network
    net.to(device)
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        num_examples_train = 0
        
        for data in trainloader:
            if flag:
                inputs, labels = data[0].float().to(device), data[1].float().to(device) 
            else:
                inputs, labels = data[0].float().to(device), data[1].long().to(device)
            num_examples_train += len(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if flag:
                weight = torch.tensor([0.1, 0.9]).to(device)
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                loss_class_weighted = loss * weight_
                loss = loss_class_weighted.mean()
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
        scheduler.step()
    return num_examples_train, running_loss

def test(net, testloader, device, flag):
    """Validate the network on the entire test set."""
    # Define loss and metrics
    if flag:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    correct, total = 0, 0
    loss = 0.0
    num_examples_test = 0
    
    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        if flag:
            y_true, y_pred = [], []
        for data in testloader:
            if flag:
                inputs, labels = data[0].float().to(device), data[1].float().to(device) 
            else:
                inputs, labels = data[0].float().to(device), data[1].long().to(device)
            num_examples_test += len(inputs)

            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            if flag:
                y_true.append(labels.detach().cpu())
                y_pred.append(torch.nn.functional.softmax(outputs.detach().cpu()))
            else:
                _, predicted = torch.max(outputs.data, 1) 
                correct += (predicted == labels).sum().item()
                accuracy = correct / total
        else:
            if flag:
                y_true, y_pred = torch.cat(y_true, 0).numpy(), torch.cat(y_pred, 0).numpy()
                fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
                metric = metrics.auc(fpr, tpr)
    return loss, num_examples_test, metric

# adapted from my code: https://github.com/vaseline555/Federated-Averaging-PyTorch/blob/main/src/utils.py
class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class CustomNumpyDataset(torch.utils.data.Dataset):
    """NumpyDataset with support of transforms."""
    def __init__(self, path, train):
        self.path = path
        if train:
            self.tensors = (
                torch.from_numpy(np.load(os.path.join(path, "X_train.npy"))), 
                torch.from_numpy(np.load(os.path.join(path, "y_train.npy")))
            )
            
        else:
            self.tensors = (
                torch.from_numpy(np.load(os.path.join(path, "X_test.npy"))), 
                torch.from_numpy(np.load(os.path.join(path, "y_test.npy")))
            )
        self.data = self.tensors[0].squeeze().float()
        self.targets = self.tensors[-1].float()
        
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        return x, y
    
    def __len__(self):
        return self.tensors[0].size(0)
    
def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    dataset_name = dataset_name.upper()
    
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        elif dataset_name in ["MNIST"]:
            transform = torchvision.transforms.ToTensor()
        else:
            # dataset not found exception
            error_message = f"...dataset \"{dataset_name}\" cannot be found in TorchVision Datasets!"
            raise AttributeError(error_message)
            
        # prepare raw training & test datasets
        training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
        
        # unsqueeze channel dimension for grayscale image datasets
        if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
            training_dataset.data.unsqueeze_(3)
        num_categories = np.unique(training_dataset.targets).shape[0]

        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()

        # split dataset according to iid flag
        if iid:
            # shuffle data
            shuffled_indices = torch.randperm(len(training_dataset))
            training_inputs = training_dataset.data[shuffled_indices]
            training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

            # partition data into num_clients
            split_size = len(training_dataset) // num_clients
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), split_size),
                    torch.split(torch.Tensor(training_labels), split_size)
                )
            )

            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset, transform=transform)
                for local_dataset in split_datasets
                ]
        else:
            # sort data by labels
            sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
            training_inputs = training_dataset.data[sorted_indices]
            training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

            # partition data into shards first
            shard_size = len(training_dataset) // num_shards #300
            shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
            shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

            # sort the list to conveniently assign samples to each clients from at least two classes
            shard_inputs_sorted, shard_labels_sorted = [], []
            for i in range(num_shards // num_categories):
                for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                    shard_inputs_sorted.append(shard_inputs[i + j])
                    shard_labels_sorted.append(shard_labels[i + j])

            # finalize local datasets by assigning shards to each client
            shards_per_clients = num_shards // num_clients
            local_datasets = [
                CustomTensorDataset(
                    (
                        torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                        torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                    ),
                    transform=transform
                ) 
                for i in range(0, len(shard_inputs_sorted), shards_per_clients)
            ]
            
    # get custom dataset outside torchvision.datasets
    else:
        if dataset_name in ["TOX21"]:
            if not os.path.exists(os.path.join(data_path, 'tox21')):
                os.makedirs(os.path.join(data_path, 'tox21'))
                label_list = retrieve_label_name_list('Tox21')
                data = Tox(name='Tox21', path=data_path, label_name=label_list[0])

                X_train, y_train = [], []
                for idx, sample in data.get_split()['train'].loc[:, ['Drug', 'Y']].iterrows():
                    X_train.append(smiles_encoder(sample.Drug))
                    y_train.append(sample.Y)
                for idx, sample in data.get_split()['valid'].loc[:, ['Drug', 'Y']].iterrows():
                    X_train.append(smiles_encoder(sample.Drug))
                    y_train.append(sample.Y)
                np.save(os.path.join(data_path, f'tox21/X_train'), np.array(X_train))
                np.save(os.path.join(data_path, f'tox21/y_train'), np.array(y_train))

                X_test, y_test = [], []
                for idx, sample in data.get_split()['test'].loc[:, ['Drug', 'Y']].iterrows():
                    X_test.append(smiles_encoder(sample.Drug))
                    y_test.append(sample.Y)
                np.save(os.path.join(data_path, f'tox21/X_test'), np.array(X_test))
                np.save(os.path.join(data_path, f'tox21/y_test'), np.array(y_test))
            else:
                print("Files already downloaded and verified")
        else:
            # dataset not found exception
            error_message = f"...dataset \"{dataset_name}\" is not supported!"
            raise AttributeError(error_message)
        
        # prepare raw training & test datasets
        training_dataset = CustomNumpyDataset(os.path.join(data_path, 'tox21'), train=True)
        test_dataset = CustomNumpyDataset(os.path.join(data_path, 'tox21'), train=False)
        
        # number of classes
        num_categories = np.unique(training_dataset.targets).shape[0]
        
        # split dataset according to iid flag
        if iid:
            # shuffle data
            shuffled_indices = torch.randperm(len(training_dataset))
            training_inputs = training_dataset.data[shuffled_indices]
            training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

            # partition data into num_clients
            split_size = len(training_dataset) // num_clients
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), split_size),
                    torch.split(torch.Tensor(training_labels), split_size)
                )
            )

            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset)
                for local_dataset in split_datasets
                ]
        else:
            # sort data by labels
            sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
            training_inputs = training_dataset.data[sorted_indices]
            training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

            # partition data into shards first
            shard_size = len(training_dataset) // num_shards #300
            shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
            shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

            # sort the list to conveniently assign samples to each clients from at least two classes
            shard_inputs_sorted, shard_labels_sorted = [], []
            for i in range(num_shards // num_categories):
                for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                    shard_inputs_sorted.append(shard_inputs[i + j])
                    shard_labels_sorted.append(shard_labels[i + j])

            # finalize local datasets by assigning shards to each client
            shards_per_clients = num_shards // num_clients
            local_datasets = [
                CustomTensorDataset(
                    (
                        torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                        torch.cat(shard_labels_sorted[i:i + shards_per_clients])
                    ),
                ) 
                for i in range(0, len(shard_inputs_sorted), shards_per_clients)
            ]
    return local_datasets, test_dataset
