import os
import torch
import numpy as np
from tqdm import tqdm

def runQC(graph, nr_nodes=15, export_index=True):
    """
    Performs quality control (QC) on a list of graph objects by filtering out subgraphs 
    that do not meet the minimum node count and neighborhood hop requirements.

    Parameters:
    graph (list): A list of PyTorch Geometric Data objects representing subgraphs.
    nr_nodes (int, optional): The minimum number of nodes required in a subgraph. Default is 15.
    export_index (bool, optional): If True, exports the index of the filtered subgraphs. Default is True.

    Returns:
    graph (list): A list of filtered PyTorch Geometric Data objects that meet the node and neighborhood criteria.
    """
    
    # Step 1: Remove subgraphs with fewer than the specified number of nodes
    nodes = []
    for i in tqdm(range(len(graph))):
        nodes.append(graph[i].num_nodes)
    
    nodes = np.hstack(nodes)
    index = np.where(nodes >= nr_nodes)[0]
    graph = [graph[i] for i in index]

    # Step 2: Remove subgraphs that do not have a maximum neighborhood index of 3 (3-hop)
    NN = []
    for i in tqdm(range(len(graph))):
        NN.append(graph[i].neighborhood_index.max().detach().cpu().numpy())
    
    samples = np.hstack(NN)
    index = np.where(samples == 3)[0]
    graph = [graph[i] for i in index]

    return graph
