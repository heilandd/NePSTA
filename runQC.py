import os
import torch
import numpy as np
from tqdm import tqdm

def runQC(graph,nr_nodes=15, export_index=True):
  ## Remove subgraphs with less then 3 hop
  nodes = []
  for i in tqdm(range(len(graph))):
    nodes.append(graph[i].num_nodes)
  
  nodes = np.hstack(nodes)
  index=np.where(nodes>=nr_nodes)[0]
  graph = [graph[i] for i in index]

  NN = []
  for i in tqdm(range(len(graph))):
    NN.append(graph[i].neighborhood_index.max().detach().cpu().numpy())
  
  samples = np.hstack(NN)
  index=np.where(samples==3)[0]
  graph = [graph[i] for i in index]
  
  
  
  return(graph)
