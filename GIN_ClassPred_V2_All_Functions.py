import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric.utils as utils
import matplotlib as PL
from tqdm import tqdm
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt

import torch
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
import torch.nn as nn

def cox_ph_loss(risk_scores, survival_time, event, alpha=0.5, lambda_reg=1e-4):

    # Sort by survival times
    sorted_indices = torch.argsort(survival_time, descending=True)
    sorted_risk_scores = risk_scores[sorted_indices]
    sorted_event = event[sorted_indices]

    # Compute hazard ratio
    hazard_ratio = torch.exp(sorted_risk_scores)

    # Compute log risk
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))

    # Compute the uncensored likelihood
    uncensored_likelihood = sorted_risk_scores - log_risk

    # Select only uncensored (i.e., event occurred) and compute their contribution
    censored_likelihood = uncensored_likelihood * sorted_event

    # Negative log partial likelihood
    num_observed_events = torch.sum(sorted_event)
    neg_log_partial_likelihood = -torch.sum(censored_likelihood) / num_observed_events

    # L1 and L2 regularization terms
    l1_term = torch.norm(risk_scores, 1)
    l2_term = torch.norm(risk_scores, 2)

    # Final loss
    loss = neg_log_partial_likelihood + lambda_reg * ((1 - alpha) * l2_term + alpha * l1_term)

    return loss

# GIN
class GINModelBatchesExp(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels, num_classes):
        super(GINModelBatchesExp, self).__init__()

        # Expression GIN Conv Layer
        self.conv1_exp = GINConv(Linear(num_features_exp, hidden_channels), train_eps=True)
        self.conv2_exp = GINConv(Linear(hidden_channels, hidden_channels), train_eps=True)

        # Batch norm layer
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5) # Add dropout for regularization

        # Latent space
        self.merge = Linear(hidden_channels, hidden_channels)

        # Initiate weights
        torch.nn.init.xavier_uniform_(self.merge.weight.data)

        # MLP Prediction Class
        self.mlp_class = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Dropout(0.5), # Add dropout in the MLP as well
            torch.nn.Linear(hidden_channels, num_classes)
        )


    def forward(self, data):
        exp, edge_index = data.x, data.edge_index

        edge_index, _ = add_self_loops(edge_index, num_nodes=exp.size(0))

        x_exp = self.conv1_exp(exp, edge_index)
        x_exp = self.dropout(F.leaky_relu(self.bn1(x_exp), negative_slope=0.2))

        x_exp = self.conv2_exp(x_exp, edge_index)
        x_exp = self.dropout(F.leaky_relu(self.bn2(x_exp), negative_slope=0.2))

        x = self.merge(x_exp)
        x = F.leaky_relu(x, negative_slope=0.2)

        class_out = self.mlp_class(global_mean_pool(x, data.batch))

        return x, class_out
def RunEvaluationGINClass(graph, model):

  model.eval()
  latent_space = []
  class_out_logits = []
  class_out_list = []
  

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Running on:", device)

  model.to(device)
  i=1
  for data in tqdm(graph, desc="Eval"):
    
    i=i+1
    #print(i)
    latent, class_out = model(data.to(device))

    ## Latent space
    latent_space.append(latent.mean(dim=0, keepdim=True).detach().cpu().numpy())

    ## Status
    class_out_logits.append(class_out.detach().cpu().numpy())

    class_out_arg = torch.argmax(class_out, dim=1)
    class_out_list.append(class_out_arg.detach().cpu().numpy())



  return(np.concatenate(latent_space), np.concatenate(class_out_logits), np.concatenate(class_out_list))  
def RunTrainingGIN(graph, hidden_channels = 256, num_classes=11, epochs = 50,learning_rate = 0.001, batch_size=32):

  num_features_exp = graph[1].x.shape[1]

  model = GINModelBatchesExp(num_features_exp, hidden_channels, num_classes=num_classes)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  model.train()
  loader = DataLoader(graph, batch_size=batch_size, shuffle=True)

  criterion = torch.nn.CrossEntropyLoss()

  epoch_loss_list = []
  epoch_loss = 0

  #data = next(iter(loader))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Running on:", device)
  model = model.to(device)


  for epoch in tqdm(range(epochs), desc="Training"):
    for data in loader:
        optimizer.zero_grad()
        latent, class_out = model(data.to(device))

        #Class
        gt = data.Class.long()
        loss = criterion(class_out, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    epoch_loss_list.append(epoch_loss)

  import matplotlib.pyplot as plt
  plt.close()
  plt.scatter(range(len(epoch_loss_list)), epoch_loss_list)
  plt.show()
  plt.close()

  return(model)

#Linear
class LinearExp(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels, num_classes):
        super(LinearExp, self).__init__()

        # First Layer
        #self.merge = Linear(num_features_exp, hidden_channels)

        # MLP Prediction Class
        self.mlp_class = torch.nn.Sequential(
            torch.nn.Linear(num_features_exp, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5), 
            torch.nn.Linear(hidden_channels, num_classes)
        )


    def forward(self, data):
        exp = data.y
        class_out = self.mlp_class(exp)

        return class_out


def RunTrainingLinear(graph, hidden_channels = 256, num_classes=11, epochs = 50,learning_rate = 0.001, batch_size=32):

  num_features_exp = graph[1].y.shape[1]

  model = LinearExp(num_features_exp, hidden_channels, num_classes=num_classes)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  model.train()
  loader = DataLoader(graph, batch_size=batch_size, shuffle=True)

  criterion = torch.nn.CrossEntropyLoss()

  epoch_loss_list = []
  epoch_loss = 0

  #data = next(iter(loader))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Running on:", device)
  model = model.to(device)


  for epoch in tqdm(range(epochs), desc="Training"):
    for data in loader:
        optimizer.zero_grad()
        class_out = model(data.to(device))

        #Class
        gt = data.Class.long()
        loss = criterion(class_out, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    epoch_loss_list.append(epoch_loss)

  import matplotlib.pyplot as plt
  plt.close()
  plt.scatter(range(len(epoch_loss_list)), epoch_loss_list)
  plt.show()
  plt.close()

  return(model)





