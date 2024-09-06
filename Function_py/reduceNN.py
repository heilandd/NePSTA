from tqdm import tqdm
from torch_geometric.data import Data
import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch_geometric.utils as utils
import matplotlib as PL
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt

def reduceNN(graphs, hop):
    """
    Reduces each subgraph to include only nodes and edges within the specified hop distance from the central node.

    Parameters:
    graphs (list): A list of PyTorch Geometric Data objects, each representing a subgraph.
    hop (int): The maximum neighborhood distance (hop) to include in the reduced graph.

    Returns:
    output_list (list): A list of PyTorch Geometric Data objects with filtered nodes and edges based on the specified hop.
    """
    
    output_list = []
    
    for data in tqdm(graphs):
        # Check if the neighborhood index exceeds the specified hop
        if data.neighborhood_index.max() < hop:
            print("Exclude graph")
        else:
            # Filter nodes within the specified hop distance
            target_nodes = torch.where(data.neighborhood_index <= hop)[0]
            num_nodes = len(target_nodes)

            # Filter edges to retain only those between nodes within the hop distance
            edge_mask = torch.tensor([src.item() in target_nodes and tgt.item() in target_nodes 
                                      for src, tgt in data.edge_index.t()])
            filtered_edge_index = data.edge_index[:, edge_mask]

            # Filter node features (data.x) and central node index
            filtered_x = data.x[target_nodes]
            central_node_index = data.central_node_index[target_nodes]

            # Create a new graph dataset with the filtered data
            new_data = Data(
                edge_index=filtered_edge_index, 
                num_nodes=num_nodes, 
                x=filtered_x, 
                y=data.y, 
                central_node_index=central_node_index,
                neighborhood_index=data.neighborhood_index[target_nodes], 
                Class=data.Class
            )
            
            output_list.append(new_data)

    return output_list

