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
from torch_geometric.nn import global_mean_pool
import torch
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, global_mean_pool, LayerNorm
from torch.nn import Linear

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import torch.nn.functional as F


class GAN(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels, num_classes):
        super(GAN, self).__init__()

        # Attention GAT Conv Layers
        per_head_hidden_channels = hidden_channels // 5
        self.conv1_exp = GATConv(num_features_exp, per_head_hidden_channels, heads=5)
        self.conv2_exp = GATConv(per_head_hidden_channels * 5, per_head_hidden_channels, heads=5)


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
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    
    def forward(self, data):
        exp, edge_index = data.x, data.edge_index

        # GATConv layers require edge_index to be long type
        edge_index = edge_index.long()

        x_exp, attention_weights_1 = self.conv1_exp(exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        x_exp = self.dropout(self.bn1(x_exp))

        x_exp, attention_weights_2 = self.conv2_exp(x_exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        x_exp = self.dropout(self.bn2(x_exp))

        x = self.merge(x_exp)
        x = F.leaky_relu(x)

        class_out = self.mlp_class(global_mean_pool(x, data.batch))

        return x, class_out, attention_weights_1, attention_weights_2

def RunGAN1(graph,num_classes, hidden_channels = 255, epochs = 50, learning_rate = 0.001, batch_size=16, weight_decay=0.01):

  num_features_exp = graph[1].x.shape[1]

  model = GAN(num_features_exp, hidden_channels, num_classes)
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  model.train()
  loader = DataLoader(graph, batch_size=batch_size, shuffle=True, drop_last=True)

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
        latent, class_out, AT1, AT2 = model(data.to(device))

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
