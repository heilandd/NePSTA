import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import squidpy as sq

def getImageAnnData(anndat):
    """
    Extracts the high-resolution image associated with the AnnData object.

    Parameters:
    anndat (AnnData): The AnnData object containing spatial transcriptomics data.

    Returns:
    img (numpy.ndarray): High-resolution image from the spatial dataset.
    """
    sample_name = list(anndat.uns["spatial"].keys())
    sample_name = sample_name[0]
    img = anndat.uns["spatial"][sample_name]["images"]["hires"]
    return img

def getScale(anndat):
    """
    Retrieves the scale factor for the high-resolution image associated with the AnnData object.

    Parameters:
    anndat (AnnData): The AnnData object containing spatial transcriptomics data.

    Returns:
    scale (float): Scale factor for the high-resolution tissue image.
    """
    sample_name = list(anndat.uns["spatial"].keys())
    sample_name = sample_name[0]
    scale = anndat.uns["spatial"][sample_name]["scalefactors"]["tissue_hires_scalef"]
    return scale

def getCounts(anndat):
    """
    Retrieves the gene expression count matrix from the AnnData object.

    Parameters:
    anndat (AnnData): The AnnData object containing spatial transcriptomics data.

    Returns:
    counts (numpy.ndarray or sparse matrix): Gene expression count matrix (cells x genes).
    """
    counts = anndat.X
    return counts

def getObs(anndat):
    """
    Retrieves the metadata associated with the cells (observations) in the AnnData object.

    Parameters:
    anndat (AnnData): The AnnData object containing spatial transcriptomics data.

    Returns:
    obs (pandas.DataFrame): Cell metadata including annotations, cell types, etc.
    """
    obs = anndat.obs
    return obs

def getcoords(anndat):
    """
    Retrieves the spatial coordinates of the cells in the AnnData object.

    Parameters:
    anndat (AnnData): The AnnData object containing spatial transcriptomics data.

    Returns:
    obsm (pandas.DataFrame or numpy.ndarray): Spatial coordinates of the cells.
    """
    obsm = anndat.obsm
    return obsm























