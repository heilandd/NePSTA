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

#input_list = r.Subgraphs

def InitializeGraphObject(input_list, withOS=False, withClass=False, withStatus=False):
  
  
  index = len(input_list)
  output_list = []
  
  for i in tqdm(range(index)):
    
    edges_df = input_list[i]["edges"]
    #edges_attr = input_list[i][1]
    
    nodes_Exp = input_list[i]["expression"]
    central_node = input_list[i]["CenterNode"]
    neighborhood = input_list[i]["neighborhood"]
    graph_info = input_list[i]["clinical_data"]
    gene_mask = input_list[i]["mask"]
    
    
    
    ## Node features: Exp
    #nf = nodes_Exp.to_numpy()
    node_features = np.asarray(nodes_Exp, dtype="float32")
    node_features_x = torch.as_tensor(node_features, dtype=torch.float)
    
    
    ## Node features: Histology
    #nf = nodes_Hist.to_numpy()
    #node_features = np.asarray(nf[:,1], dtype="int8")
    #node_features.shape
    #node_features_z = torch.as_tensor(node_features, dtype=torch.float)
    
    
    ## Central node
    central_node_index = np.asarray(central_node, dtype="int8")
    central_node_index = torch.tensor(central_node_index, dtype=torch.long)
    
    ## neighborhood
    neighborhood_index = np.concatenate(neighborhood,dtype="int8")
    neighborhood_index = torch.tensor(neighborhood_index, dtype=torch.long)

    
    ## Gene expression of the central node
    node_features_y = torch.as_tensor(node_features_x[central_node_index==1,:], dtype=torch.float)
    ## Histology of the central node
    #node_features_hc = torch.as_tensor(node_features_z[central_node_index==1], dtype=torch.float)
  
    ## Edges  
    nx_graph = nx.from_pandas_edgelist(edges_df, "from", "to")
    ptg_data = utils.from_networkx(nx_graph)
    
    # Node Information
    ptg_data.x = node_features_x
    #ptg_data.h = node_features_z
    #ptg_data.hc = node_features_hc
    ptg_data.y = node_features_y
    ptg_data.central_node_index = central_node_index
    ptg_data.neighborhood_index = neighborhood_index
    
    
    # Subgraph Information
    if withOS==True:
      ptg_data.OS = torch.as_tensor(np.asarray(graph_info["OS"], dtype="float32"), dtype=torch.float)
      ptg_data.PFS = torch.as_tensor(np.asarray(graph_info["PFS"], dtype="float32"), dtype=torch.float)
      ptg_data.Event = torch.as_tensor(np.asarray(graph_info["Event"], dtype="int8"), dtype=torch.float)
    if withClass==True:
      ptg_data.Class = torch.as_tensor(np.asarray(graph_info["Class"], dtype="int8"), dtype=torch.float)
    if withStatus==True:
      ptg_data.Status = torch.as_tensor(np.asarray(graph_info["Status"], dtype="int8"), dtype=torch.float)
    
    
    output_list.append(ptg_data)
  
  return output_list






