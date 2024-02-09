import numpy as np
import pandas as pd
import networkx as nx
import sklearn
from sklearn import preprocessing
import matplotlib as PL
from tqdm import tqdm

## Torch
import torch
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
import torch_geometric.utils as utils
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Linear
from sklearn.cluster import KMeans



def OptimalK(LS, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(LS)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(LS)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(LS)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (LS[i, 0] - curr_center[0]) ** 2 + (LS[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse
def RunEvaluationGIN(graph, model, plot=True):
  model.eval()
  
  #Get latent space
  
  latentspace = []
  expr = []

  for data in tqdm(graph, desc="Eval"):
    latent, out = model(data)
    latentspace.append(latent[data.central_node_index==1].detach().cpu().numpy())
    expr.append(out.detach().cpu().numpy())
    
  expr = np.concatenate(expr, axis=0)
  ls = np.concatenate(latentspace, axis=0)

  import umap
  import matplotlib.pyplot as plt
  reducer = umap.UMAP()
  embedding = reducer.fit_transform(ls)

  #Get optimal cluster
  k=OptimalK(embedding, 20)
  slopes = np.diff(k)
  elbow_point = np.argmax(slopes) + 1
  
  kmeans = KMeans(n_clusters = elbow_point).fit(ls)
  pred_clusters = kmeans.predict(ls)


  if(plot==True):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=pred_clusters)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the latent space', fontsize=12)
  
  export = [expr, ls, pred_clusters,embedding, plt]
  
  return(export)
def RunEvaluationGINHisto(graph, model):
  
  #model = r.model_GIN
  model.eval()
  

  hist_list = []
  #data = next(iter(loader))


  for data in tqdm(graph, desc="Eval"):
    out = model(data)[1]
    hist_list.append(out.detach().cpu().numpy())

  export = np.asarray(hist_list)
  
  return(np.concatenate(export, axis=0))
def RunEvaluationGINSurv(graph, model):
  model.eval()
  
  surv_list = []


  for data in tqdm(graph, desc="Eval"):
    out = model(data)
    surv = out.mean()
    surv_list.append(surv.detach().cpu().numpy())
  surv_list = np.asarray(surv_list)

  export = surv_list
  
  return(export)
def RunEvaluationGINClass(graph, model):
  
  #model = r.model_GIN
  model.eval()
  

  class_list = []
  #data = next(iter(loader))


  for data in tqdm(graph, desc="Eval"):
    out = model(data)[1]
    class_list.append(out.detach().cpu().numpy())

  export = np.asarray(class_list)
  
  return(np.concatenate(export, axis=0))
















