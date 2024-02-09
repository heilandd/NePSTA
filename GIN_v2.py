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
from torch_geometric.nn import global_mean_pool
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Linear
import torch.optim as optim

class GINModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(Linear(num_features, hidden_channels), train_eps=True)
        self.conv2 = GINConv(Linear(hidden_channels, hidden_channels), train_eps=True)
        self.conv3 = GINConv(Linear(hidden_channels, num_classes), train_eps=True)
        self.fc = Linear(num_classes, num_features)  # Regression layer
        
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        ## Add self-loop 
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2) 
        central_node_mask = data.central_node_index==1  
        out = self.fc(x)[central_node_mask]  
        
        return x, out
class GINModelIntegrated(torch.nn.Module):
    def __init__(self, num_features_exp, num_features_cnv, hidden_channels, num_classes):
        super(GINModelIntegrated, self).__init__()
        self.conv1_exp = GINConv(Linear(num_features_exp, hidden_channels), train_eps=True)
        self.conv2_exp = GINConv(Linear(hidden_channels, hidden_channels), train_eps=True)
        
        self.conv1_cnv = GINConv(Linear(num_features_cnv, hidden_channels), train_eps=True)
        self.conv2_cnv = GINConv(Linear(hidden_channels, hidden_channels), train_eps=True)

        # Layer to merge the two different feature types
        self.merge = Linear(hidden_channels * 2, num_classes)

        # MLP for classification
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_classes, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        
        # Add embeddings for masked values 
        self.missing_embedding_x = torch.nn.Parameter(torch.randn(num_features_exp))
        self.missing_embedding_c = torch.nn.Parameter(torch.randn(num_features_cnv))

    def forward(self, data):
        exp, cnv, edge_index = data.x, data.c, data.edge_index
        
        # Add self-loop 
        edge_index, _ = add_self_loops(edge_index, num_nodes=exp.size(0))
        
        missing_indices_x = (data.x == 0)
        data.x[missing_indices_x] = self.missing_embedding_x
        
        missing_indices_c = (data.c == 0)
        data.c[missing_indices_c] = self.missing_embedding_c
        
        x_exp = self.conv1_exp(exp, edge_index)
        x_exp = F.leaky_relu(x_exp, negative_slope=0.2)
        x_exp = self.conv2_exp(x_exp, edge_index)
        x_exp = F.leaky_relu(x_exp, negative_slope=0.2)
        
        #print(x_exp.shape)


        x_cnv = self.conv1_cnv(cnv, edge_index)
        x_cnv = F.leaky_relu(x_cnv, negative_slope=0.2)
        x_cnv = self.conv2_cnv(x_cnv, edge_index)
        x_cnv = F.leaky_relu(x_cnv, negative_slope=0.2)
        #print(x_cnv.shape)


        x = torch.cat((x_exp, x_cnv), dim=1)
        x = self.merge(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        #print(x.shape)


        # Extract features of the central node
        central_node_features = x[data.central_node_index == 1]

        # Pass the central node's features to MLP for histological prediction
        histology_logits = self.mlp(central_node_features)
        #histology_out = F.softmax(histology_logits, dim=1)


        return x, histology_logits
class GINModelBatches(torch.nn.Module):
    def __init__(self, num_features_exp, num_features_cnv, hidden_channels, num_classes):
        super(GINModelBatches, self).__init__()
        self.conv1_exp = GINConv(Linear(num_features_exp, hidden_channels), train_eps=True)
        self.conv2_exp = GINConv(Linear(hidden_channels, hidden_channels), train_eps=True)
        
        self.conv1_cnv = GINConv(Linear(num_features_cnv, hidden_channels), train_eps=True)
        self.conv2_cnv = GINConv(Linear(hidden_channels, hidden_channels), train_eps=True)

        # Layer to merge the two different feature types
        self.merge = Linear(hidden_channels * 2, hidden_channels)

        # MLP for classification
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), 
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_classes)  
        )
        
        # Add embeddings for masked values 
        #self.missing_embedding_x = torch.nn.Parameter(torch.randn(num_features_exp))
        #self.missing_embedding_c = torch.nn.Parameter(torch.randn(num_features_cnv))

    def forward(self, data):
        exp, cnv, edge_index = data.x, data.c, data.edge_index
        
        # Add self-loop 
        edge_index, _ = add_self_loops(edge_index, num_nodes=exp.size(0))
 
        x_exp = self.conv1_exp(exp, edge_index)
        x_exp = F.leaky_relu(x_exp, negative_slope=0.2)
        x_exp = self.conv2_exp(x_exp, edge_index)
        x_exp = F.leaky_relu(x_exp, negative_slope=0.2)
        #print(x_exp.shape)


        x_cnv = self.conv1_cnv(cnv, edge_index)
        x_cnv = F.leaky_relu(x_cnv, negative_slope=0.2)
        x_cnv = self.conv2_cnv(x_cnv, edge_index)
        x_cnv = F.leaky_relu(x_cnv, negative_slope=0.2)
        #print(x_cnv.shape)


        x = torch.cat((x_exp, x_cnv), dim=1)
        x = self.merge(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        #print(x.shape)
        
        graph_representation = global_mean_pool(x, data.batch)
        print(graph_representation.shape)
        
        # Pass the central node's features to MLP for histological prediction
        graph_class_logits = self.mlp(x)
        graph_class_logits_out = global_mean_pool(graph_class_logits, data.batch)


        return graph_representation, graph_class_logits_out

def RunGINEmbeding(graph,num_features = 500, hidden_channels = 256, num_classes = 32, epochs = 50,learning_rate = 0.001):

  model = GINModel(num_features,hidden_channels, num_classes)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)


  def correlation_loss(pred, target):
    # Calculate mean-centered tensors
    pred_centered = pred - torch.mean(pred)
    target_centered = target - torch.mean(target)
    
    # Compute Pearson correlation coefficient
    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2))
    correlation = numerator / denominator
    
    # Convert correlation to a loss value (negative correlation)
    loss = 1.0 - correlation
    
    return loss

  model.train()
  loader = DataLoader(graph[1:100], batch_size=1, shuffle=F)


  epoch_loss_list = []
  epoch_loss = 0

  for epoch in tqdm(range(epochs), desc="Training"):
    for data in loader:
        optimizer.zero_grad()
        latent, out = model(data)
        
        #correlation = np.corrcoef(out.detach().numpy(), data.y.detach().numpy())[0, 1]
        #print("Correlation coefficient:", correlation)
        
        #loss = criterion(out, data.y) 
        loss = correlation_loss(out, data.y) 
        
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
def RunGINHistology(graph, hidden_channels = 256, num_histology_classes=6, epochs = 50,learning_rate = 0.001):
  
  #graph = r.initialized_obj
  num_features_exp = graph[1].x.shape[1]
  num_features_cnv = graph[1].c.shape[1]

  model = GINModelIntegrated(num_features_exp,num_features_cnv, hidden_channels,num_classes=num_histology_classes)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  model.train()
  loader = DataLoader(graph, batch_size=1, shuffle=F)

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
        
        latent, out = model(data.to(device))
        
        #correlation = np.corrcoef(out.detach().numpy(), data.y.detach().numpy())[0, 1]
        #print("Correlation coefficient:", correlation)
        
        loss = criterion(out, data.hc.long()) 
        #loss = correlation_loss(out, data.y) 
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    epoch_loss_list.append(epoch_loss)
  #import matplotlib.pyplot as plt
  #plt.close()
  #plt.scatter(range(len(epoch_loss_list)), epoch_loss_list)
  #plt.show()
  #plt.close()
  
  return(model)
def RunGINClass(graph, hidden_channels = 256, num_histology_classes=2, epochs = 50,learning_rate = 0.001):

  num_features_exp = graph[1].x.shape[1]
  num_features_cnv = graph[1].c.shape[1]

  model = GINModelIntegrated(num_features_exp,num_features_cnv, hidden_channels,num_classes=num_histology_classes)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  model.train()
  loader = DataLoader(graph, batch_size=1, shuffle=F)

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
        latent, out = model(data.to(device))
        
        #correlation = np.corrcoef(out.detach().numpy(), data.y.detach().numpy())[0, 1]
        #print("Correlation coefficient:", correlation)
        
        loss = criterion(out, data.cl.long()) 
        #loss = correlation_loss(out, data.y) 
        
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
def RunGINClassList(graphlist, hidden_channels = 256, num_histology_classes=10, epochs = 50,learning_rate = 0.001):

  graph = graphlist[1]
  num_features_exp = graph[1].x.shape[1]
  num_features_cnv = graph[1].c.shape[1]

  model = GINModelIntegrated(num_features_exp,num_features_cnv, hidden_channels,num_classes=num_histology_classes)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  model.train()
  criterion = torch.nn.CrossEntropyLoss()

  epoch_loss_list = []
  epoch_loss = 0

  #data = next(iter(loader))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Running on:", device)
  model = model.to(device)
  
  batches = len(graphlist)
  for i in range(batches):
    print("Batch Run:", i)
    loader = DataLoader(graphlist[i], batch_size=1, shuffle=F)
    for epoch in tqdm(range(epochs), desc="Training"):
      for data in loader:
        optimizer.zero_grad()
        latent, out = model(data.to(device))
        
        #correlation = np.corrcoef(out.detach().numpy(), data.y.detach().numpy())[0, 1]
        #print("Correlation coefficient:", correlation)
        
        loss = criterion(out, data.cl.long()) 
        #loss = correlation_loss(out, data.y) 
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
      epoch_loss /= len(loader)
      epoch_loss_list.append(epoch_loss)
    
    
  
  

  
  #import matplotlib.pyplot as plt
  #plt.close()
  #plt.scatter(range(len(epoch_loss_list)), epoch_loss_list)
  #plt.show()
  #plt.close()
  
  return(model)
def RunGINClassBatch(graph, hidden_channels = 256, num_histology_classes=10, epochs = 50,learning_rate = 0.001, batch_size=32):

  num_features_exp = graph[1].x.shape[1]
  num_features_cnv = graph[1].c.shape[1]

  model = GINModelBatches(num_features_exp,num_features_cnv, hidden_channels,num_classes=num_histology_classes)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  model.train()
  loader = DataLoader(graph, batch_size=batch_size, shuffle=F)

  criterion = torch.nn.CrossEntropyLoss()

  epoch_loss_list = []
  epoch_loss = 0

  data = next(iter(loader))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Running on:", device)
  model = model.to(device)
  
  
  for epoch in tqdm(range(epochs), desc="Training"):
    for data in loader:
        optimizer.zero_grad()
        latent, out = model(data.to(device))
        
        #correlation = np.corrcoef(out.detach().numpy(), data.y.detach().numpy())[0, 1]
        #print("Correlation coefficient:", correlation)
        
        gt = data.cl.long()
        #train = torch.argmax(out, dim=1)
        
        loss = criterion(out, gt) 
        #loss = correlation_loss(out, data.y) 
        
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
















