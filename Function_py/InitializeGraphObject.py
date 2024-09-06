import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib as PL
from tqdm import tqdm

import torch_geometric.utils as utils
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
import sklearn
from sklearn import preprocessing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data

def InitializeGraphObject(input_list, withOS=False, withClass=False, withStatus=False):
    """
    Initializes graph objects from the input list of subgraphs, converting the input data into PyTorch Geometric Data objects.

    Parameters:
    input_list (list): A list containing subgraph data where each item is a dictionary with 'edges', 'expression', 'CenterNode', 'neighborhood', 'clinical_data', and 'mask'.
    withOS (bool, optional): If True, includes Overall Survival (OS), Progression-Free Survival (PFS), and Event data in the graph object. Default is False.
    withClass (bool, optional): If True, includes class labels in the graph object. Default is False.
    withStatus (bool, optional): If True, includes status labels in the graph object. Default is False.

    Returns:
    output_list (list): A list of PyTorch Geometric Data objects with node and edge features, ready for graph-based learning models.
    """
    
    index = len(input_list)
    output_list = []
    
    for i in tqdm(range(index)):
        
        edges_df = input_list[i]["edges"]
        nodes_Exp = input_list[i]["expression"]
        central_node = input_list[i]["CenterNode"]
        neighborhood = input_list[i]["neighborhood"]
        graph_info = input_list[i]["clinical_data"]
        gene_mask = input_list[i]["mask"]
        
        # Node features: Gene expression
        node_features = np.asarray(nodes_Exp, dtype="float32")
        node_features_x = torch.as_tensor(node_features, dtype=torch.float)
        
        # Central node
        central_node_index = np.asarray(central_node, dtype="int8")
        central_node_index = torch.tensor(central_node_index, dtype=torch.long)
        
        # Neighborhood
        neighborhood_index = np.concatenate(neighborhood, dtype="int8")
        neighborhood_index = torch.tensor(neighborhood_index, dtype=torch.long)

        # Gene expression of the central node
        node_features_y = torch.as_tensor(node_features_x[central_node_index == 1, :], dtype=torch.float)
        
        # Edges
        nx_graph = nx.from_pandas_edgelist(edges_df, "from", "to")
        ptg_data = utils.from_networkx(nx_graph)
        
        # Assign node and graph features
        ptg_data.x = node_features_x
        ptg_data.y = node_features_y
        ptg_data.central_node_index = central_node_index
        ptg_data.neighborhood_index = neighborhood_index
        
        # Optional clinical data features
        if withOS:
            ptg_data.OS = torch.as_tensor(np.asarray(graph_info["OS"], dtype="float32"), dtype=torch.float)
            ptg_data.PFS = torch.as_tensor(np.asarray(graph_info["PFS"], dtype="float32"), dtype=torch.float)
            ptg_data.Event = torch.as_tensor(np.asarray(graph_info["Event"], dtype="int8"), dtype=torch.float)
        
        if withClass:
            ptg_data.Class = torch.as_tensor(np.asarray(graph_info["Class"], dtype="int8"), dtype=torch.float)
        
        if withStatus:
            ptg_data.Status = torch.as_tensor(np.asarray(graph_info["Status"], dtype="int8"), dtype=torch.float)
        
        output_list.append(ptg_data)
    
    return output_list



