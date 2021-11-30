import sys
import torch
import flwr as fl
from collections import OrderedDict

import util


class FlowerClient(fl.client.Client):
    def __init__(self, cid, model, train_loader, test_loader, epochs, device, flag):
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.flag = flag
        
    def get_weights(self):
        # Return model parameters as a list of NumPy ndarrays
        return [values.cpu().numpy() for _, values in self.model.state_dict().items()]
    
    def get_parameters(self):
        weights = self.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return fl.common.ParametersRes(parameters=parameters)
    
    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, ins):
        # Get weights
        weights = fl.common.parameters_to_weights(ins.parameters)
        
        # Set model parameters/weights
        self.set_parameters(weights)
        
        # Train model
        num_examples_train, loss = util.train(self.model, self.train_loader, epochs=self.epochs, device=self.device, flag=self.flag)
        sys.stdout.write(f"[CLIENT {self.cid.zfill(4)}] Training Loss: {loss:8.4f}" + "\r")
        # Return the refined weights and the number of examples used for training
        weights_prime = self.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        return fl.common.FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
        )
    
    def evaluate(self, ins):
        # Get weights
        weights = fl.common.parameters_to_weights(ins.parameters)
        
        # Set model parameters/weights
        self.set_parameters(weights)
        
        # Test model
        loss, num_examples_test, metric = util.test(self.model, self.test_loader, device=self.device, flag=self.flag)
        sys.stdout.write(f"[CLIENT {self.cid.zfill(4)}] Evaluation Loss: {loss:8.4f} | Metric: {metric:8.4f}" + "\r")
        return fl.common.EvaluateRes(
            loss=float(loss),
            num_examples=num_examples_test,
            metrics={"metrics": float(metric)}
        )
