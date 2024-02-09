from tqdm import tqdm
from torch_geometric.data import Data
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

def reduceNN(graphs, hop):
    # Check if hop is valid for all subgraphs
    #if any(hop > subgraph.neighborhood_index.max() for subgraph in graphs):
    #    raise ValueError("Hop is larger than the max hop in some subgraphs")
      
    output_list = []
    for data in tqdm(graphs):
        if data.neighborhood_index.max()<hop:
          print("Exclude graph")
        else:
          # Filter nodes within the specified hop
          target_nodes = torch.where(data.neighborhood_index <= hop)[0]
          num_nodes = len(target_nodes)

          # Filter edges
          edge_mask = torch.tensor([src.item() in target_nodes and tgt.item() in target_nodes 
                                  for src, tgt in data.edge_index.t()])
          filtered_edge_index = data.edge_index[:, edge_mask]

          # Filter data.x
          filtered_x = data.x[target_nodes]
          central_node_index = data.central_node_index[target_nodes]

          # Create a new dataset with the filtered data
          new_data = Data(edge_index=filtered_edge_index, num_nodes=num_nodes, 
                          x=filtered_x, y=data.y, central_node_index=central_node_index,
                          neighborhood_index=data.neighborhood_index[target_nodes], Class=data.Class)
        
          output_list.append(new_data)

    return output_list
