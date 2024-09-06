import numpy as np
import pandas as pd
import networkx as nx
import sklearn
from sklearn import preprocessing
import matplotlib as PL
from tqdm import tqdm

# Torch
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
    """
    Determines the optimal number of clusters using the elbow method.
    
    Parameters:
    LS (numpy.ndarray): The latent space (data points) to be clustered.
    kmax (int): The maximum number of clusters to evaluate.

    Returns:
    sse (list): Sum of squared errors (SSE) for each k value from 1 to kmax.
    """
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters=k).fit(LS)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(LS)
        curr_sse = 0
        
        # Calculate Euclidean distance of each point from its cluster center
        for i in range(len(LS)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (LS[i, 0] - curr_center[0]) ** 2 + (LS[i, 1] - curr_center[1]) ** 2
        
        sse.append(curr_sse)
    return sse

def RunEvaluationGIN(graph, model, plot=True):
    """
    Evaluates a GIN model on the given graph dataset and projects the latent space using UMAP.

    Parameters:
    graph (torch_geometric.data.DataLoader): The graph dataset.
    model (torch.nn.Module): The trained GIN model.
    plot (bool, optional): Whether to plot the UMAP projection. Default is True.

    Returns:
    export (list): A list containing expression data, latent space, predicted clusters, UMAP embedding, and plot.
    """
    model.eval()

    # Get latent space and expression outputs
    latentspace = []
    expr = []

    for data in tqdm(graph, desc="Eval"):
        latent, out = model(data)
        latentspace.append(latent[data.central_node_index == 1].detach().cpu().numpy())
        expr.append(out.detach().cpu().numpy())

    expr = np.concatenate(expr, axis=0)
    ls = np.concatenate(latentspace, axis=0)

    # UMAP projection
    import umap
    import matplotlib.pyplot as plt
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ls)

    # Get optimal cluster number using elbow method
    k = OptimalK(embedding, 20)
    slopes = np.diff(k)
    elbow_point = np.argmax(slopes) + 1

    # KMeans clustering
    kmeans = KMeans(n_clusters=elbow_point).fit(ls)
    pred_clusters = kmeans.predict(ls)

    if plot:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=pred_clusters)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the latent space', fontsize=12)

    export = [expr, ls, pred_clusters, embedding, plt]
    return export

def RunEvaluationGINHisto(graph, model):
    """
    Evaluates a GIN model on the given graph dataset for histogram prediction tasks.

    Parameters:
    graph (torch_geometric.data.DataLoader): The graph dataset.
    model (torch.nn.Module): The trained GIN model.

    Returns:
    export (numpy.ndarray): The histogram predictions for each graph in the dataset.
    """
    model.eval()

    hist_list = []

    for data in tqdm(graph, desc="Eval"):
        out = model(data)[1]
        hist_list.append(out.detach().cpu().numpy())

    export = np.asarray(hist_list)
    return np.concatenate(export, axis=0)

def RunEvaluationGINSurv(graph, model):
    """
    Evaluates a GIN model on the given graph dataset for survival prediction tasks.

    Parameters:
    graph (torch_geometric.data.DataLoader): The graph dataset.
    model (torch.nn.Module): The trained GIN model.

    Returns:
    export (numpy.ndarray): The survival predictions for each graph in the dataset.
    """
    model.eval()

    surv_list = []

    for data in tqdm(graph, desc="Eval"):
        out = model(data)
        surv = out.mean()
        surv_list.append(surv.detach().cpu().numpy())

    export = np.asarray(surv_list)
    return export

def RunEvaluationGINClass(graph, model):
    """
    Evaluates a GIN model on the given graph dataset for classification tasks.

    Parameters:
    graph (torch_geometric.data.DataLoader): The graph dataset.
    model (torch.nn.Module): The trained GIN model.

    Returns:
    export (numpy.ndarray): The class predictions for each graph in the dataset.
    """
    model.eval()

    class_list = []

    for data in tqdm(graph, desc="Eval"):
        out = model(data)[1]
        class_list.append(out.detach().cpu().numpy())

    export = np.asarray(class_list)
    return np.concatenate(export, axis=0)















