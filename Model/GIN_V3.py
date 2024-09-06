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
    """
    Computes the Cox Proportional Hazards (PH) loss function.

    Parameters:
    risk_scores (torch.Tensor): Predicted risk scores for each sample.
    survival_time (torch.Tensor): Survival times for each sample.
    event (torch.Tensor): Binary indicator (0 or 1) representing whether an event occurred (1) or the data is censored (0).
    alpha (float, optional): Weighting parameter between L1 and L2 regularization. Default is 0.5.
    lambda_reg (float, optional): Regularization term for L1 and L2 penalties. Default is 1e-4.

    Returns:
    loss (torch.Tensor): The computed Cox PH loss.
    """
    
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


class GINModelBatchesExp(torch.nn.Module):
    """
    A Graph Isomorphism Network (GIN) model designed to process batches with expression data.

    Parameters:
    num_features_exp (int): Number of features in the expression data.
    hidden_channels (int): Number of hidden channels for the GIN model.
    num_classes (int): Number of output classes for prediction.

    Methods:
    forward(data): Forward pass of the model.
    """
    def __init__(self, num_features_exp, hidden_channels, num_classes):
        super(GINModelBatchesExp, self).__init__()

        # GIN Layers
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features_exp, hidden_channels), 
                torch.nn.ReLU(), 
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), 
                torch.nn.ReLU(), 
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)

        # MLP Prediction Head
        self.mlp_class = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        """
        Forward pass of the GIN model with expression data.

        Parameters:
        data (torch_geometric.data.Data): The input graph data containing node features and edges.

        Returns:
        latent (torch.Tensor): Latent representations of the nodes after GIN convolution.
        class_out (torch.Tensor): Output class predictions.
        """
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        latent = F.relu(x)

        # Class predictions
        class_out = self.mlp_class(latent)

        return latent, class_out


class LinearExp(torch.nn.Module):
    """
    A simple linear model designed for classification tasks with expression data.

    Parameters:
    num_features_exp (int): Number of features in the expression data.
    hidden_channels (int): Number of hidden channels for the MLP.
    num_classes (int): Number of output classes for prediction.

    Methods:
    forward(data): Forward pass of the model.
    """
    def __init__(self, num_features_exp, hidden_channels, num_classes):
        super(LinearExp, self).__init__()

        # MLP Prediction Class
        self.mlp_class = torch.nn.Sequential(
            torch.nn.Linear(num_features_exp, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        """
        Forward pass of the linear model with expression data.

        Parameters:
        data (torch_geometric.data.Data): The input data containing expression features.

        Returns:
        class_out (torch.Tensor): Output class predictions.
        """
        exp = data.y
        class_out = self.mlp_class(exp)

        return class_out


def RunTrainingLinear(graph, hidden_channels=256, num_classes=11, epochs=50, learning_rate=0.001, batch_size=32):
    """
    Trains a LinearExp model on the given graph dataset.

    Parameters:
    graph (list): A list of graph data objects for training.
    hidden_channels (int, optional): Number of hidden channels in the model. Default is 256.
    num_classes (int, optional): Number of output classes for prediction. Default is 11.
    epochs (int, optional): Number of training epochs. Default is 50.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
    batch_size (int, optional): Batch size for data loading. Default is 32.

    Returns:
    model (torch.nn.Module): The trained LinearExp model.
    """
    
    num_features_exp = graph[1].y.shape[1]
    model = LinearExp(num_features_exp, hidden_channels, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loader = DataLoader(graph, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss_list = []
    epoch_loss = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)
    model = model.to(device)

    for epoch in tqdm(range(epochs), desc="Training"):
        for data in loader:
            optimizer.zero_grad()
            class_out = model(data.to(device))

            # Class
            gt = data.Class.long()
            loss = criterion(class_out, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        epoch_loss_list.append(epoch_loss)

    # Plot loss
    plt.close()
    plt.scatter(range(len(epoch_loss_list)), epoch_loss_list)
    plt.show()
    plt.close()

    return model



