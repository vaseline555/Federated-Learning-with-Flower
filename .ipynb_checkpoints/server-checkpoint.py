import sys
import random
import argparse
import torch
import flwr as fl
import numpy as np
from functools import partial
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split

import client
import models
import util


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def client_fn(cid, model, client_datasets, test_fraction, epochs, batch_size, device, center_eval=True, flag=False):
    model = model()

    train_size = int((1. - test_fraction) * len(client_datasets[int(cid)]))
    test_size = len(client_datasets[int(cid)]) - train_size
    
    train_dataset, test_dataset = random_split(client_datasets[int(cid)], [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if center_eval:
        test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return client.FlowerClient(cid, model, train_loader, test_loader, epochs, device, flag)

def get_parameters(model):
    return [values.cpu().numpy() for _, values in model.state_dict().items()]
    
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict)

def evaluate(weights, model, server_testset, device, flag):
    model = model()
    testloader = DataLoader(server_testset)
    set_parameters(model, weights)
    loss, _, metric = util.test(model, testloader, device, flag)
    sys.stdout.write(f"[SERVER] Evaluation Loss: {loss:.4f} | Metrics: {metric:.4f}" + "\r")
    return float(loss), {"metrics": float(metric)}


if __name__ == "__main__":
    # Parse configurations
    parser = argparse.ArgumentParser(description="Federated Learning Simulation based on Flower")
    parser.add_argument('--seed', type=int, default=5959, help='random seed')
    parser.add_argument('--data_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', required=True, help='which data to use for federated learning: {MNIST|CIFAR10|TOX21}')
    parser.add_argument('--num_clients', '-N', type=int, default=100, help='total number of clients to participate')
    parser.add_argument('--fraction', '-K', type=float, default=0.1, help='fraction of participating clients at each round')
    parser.add_argument('--batch_size', '-B', type=int, default=10, help='batch size for local update')
    parser.add_argument('--num_epochs', '-E', type=int, default=5, help='number of local epochs')
    parser.add_argument('--num_rounds', '-R', type=int, default=10, help='number of required rounds')
    parser.add_argument('--num_shards', '-S', type=int, default=200, help='number of resulting shards used for splitting dataset in pathological non-IID setting')
    parser.add_argument('--iid', action='store_true', help='wheter to split data in an IID manner')
    parser.add_argument('--test_fraction', type=float, default=0.0, help='fraction of test dataset at each client')
    args = parser.parse_args()
    
    # Greetings
    welcome_message = """
    ______       _                _           _   _                           _                                 
    |  ___|     | |              | |         | | | |                         (_)                              
    | |_ ___  __| | ___ _ __ __ _| |_ ___  __| | | |     ___  __ _ _ __ _ __  _ _ __   __ _  
    |  _/ _ \/ _` |/ _ | '__/ _` | __/ _ \/ _` | | |    / _ \/ _` | '__| '_ \| | '_ \ / _` | 
    | ||  __| (_| |  __| | | (_| | ||  __| (_| | | |___|  __| (_| | |  | | | | | | | | (_| | 
    \_| \___|\__,_|\___|_|  \__,_|\__\___|\__,_| \_____/\___|\__,_|_|  |_| |_|_|_| |_|\__, |   
                                                                                       __/ |                           
                                                                                      |___/  
                                                                                      
                               _ _   _        ______ _                        
                              (_| | | |      |  ____| |                       
                     __      ___| |_| |__    | |__  | | _____      _____ _ __ 
                     \ \ /\ / | | __| '_ \   |  __| | |/ _ \ \ /\ / / _ | '__|
                      \ V  V /| | |_| | | |  | |    | | (_) \ V  V |  __| |   
                       \_/\_/ |_|\__|_| |_|  |_|    |_|\___/ \_/\_/ \___|_|
                       
                       
    ----------------------------------------------------------------------------------------
    
    """
    done_by = """
                             .-') _        .-')   .-') _                    ('-.   
                            ( OO ) )      ( OO ).(  OO) )                 _(  OO)  
             ,--. ,--.  ,--./ ,--,',-.-')(_)---\_/     '._         ,-.-')(,------. 
             |  | |  |  |   \ |  |\|  |OO/    _ ||'--...__)        |  |OO)|  .---' 
             |  | | .-')|    \|  | |  |  \  :` `.'--.  .--'        |  |  \|  |     
             |  |_|( OO |  .     |/|  |(_/'..`''.)  |  |           |  |(_(|  '--.  
             |  | | `-' |  |\    |,|  |_..-._)   \  |  |          ,|  |_.'|  .--'  
            ('  '-'(_.-'|  | \   (_|  |  \       /  |  |         (_|  |   |  `---. 
             _`.-')-_ _ `-.-')`--' `--'   `-----'   ('-.   .-. .-')`--'   `------' 
                    ( (  OO) ( '.( OO )_                   ( OO ).-\  ( OO )               
                     \     .'_,--.   ,--.)       ,--.      / . --. /;-----.\               
                     ,`'--..._|   `.'   |        |  |.-')  | \-.  \ | .-.  |               
                     |  |  \  |         |        |  | OO .-'-'  |  || '-' /_)              
                     |  |   ' |  |'.'|  |        |  |`-' |\| |_.'  || .-. `.               
                     |  |   / |  |   |  |       (|  '---.' |  .-.  || |  \  |              
                     |  '--'  |  |   |  |        |      |  |  | |  || '--'  .-.            
                     `-------'`--'   `--'        `------'  `--' `--'`------'`-'
     
     - By: Seok-Ju Hahn (seokjuhahn@unist.ac.kr)
     - GitHub: vaseline555
    """
    print(welcome_message, done_by)
    
    # Show configurations
    print("\t<=== Configurations ===>\n")
    print(' '.join(f'\t  * {str(k).upper()}: {v}\n' for k, v in vars(args).items()))
    print("\t<======================>")
    
    # Set hyperparameters
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cpu")#
    center_eval = args.test_fraction == 0 

    # Load model
    if "mnist" in args.dataset_name.lower():
        model = models.MnistNet
        flag = False
    elif "cifar" in args.dataset_name.lower():
        model = models.CifarNet
        flag = False
    elif "tox" in args.dataset_name.lower():
        model = models.ToxNet
        flag = True
    else:
        raise NotImplementedError(f'[ERROR] ...dataset {args.dataset_name} is not supported!')
        sys.exit(0)
    
    # Load data
    client_dataests, server_testset = util.create_datasets(args.data_path, args.dataset_name, args.num_clients, args.num_shards, args.iid)

    # Pass parameters to the Strategy for server-side parameter initialization
    if center_eval:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.fraction,
            fraction_eval=args.fraction,
            min_fit_clients=int(args.num_clients * args.fraction),
            min_eval_clients=int(args.num_clients * args.fraction),
            min_available_clients=int(args.num_clients * args.fraction), 
            initial_parameters=fl.common.weights_to_parameters(get_parameters(model().to(device))),
            eval_fn=partial(evaluate,
                            model=model,
                            server_testset=server_testset,
                            device=device,
                            flag=flag)
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.fraction,
            fraction_eval=args.fraction,
            min_fit_clients=1,
            min_eval_clients=1,
            min_available_clients=1, 
            initial_parameters=fl.common.weights_to_parameters(get_parameters(model().to(device)))
        )
    
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=partial(client_fn,
                          model=model,
                          client_datasets=client_dataests,
                          test_fraction=args.test_fraction,
                          epochs=args.num_epochs,
                          batch_size=args.batch_size,
                          device=device,
                          flag=flag),
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        client_resources={"num_cpus": 1, "num_gpus": 1},
        strategy=strategy
    )
