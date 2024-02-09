import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import squidpy as sq

def getImageAnnData(anndat):
  sample_name=list(anndat.uns["spatial"].keys())
  sample_name = sample_name[0]
  img = anndat.uns["spatial"][sample_name]["images"]["hires"]
  return(img)

def getScale(anndat):
  sample_name=list(anndat.uns["spatial"].keys())
  sample_name = sample_name[0]
  scale = anndat.uns["spatial"][sample_name]["scalefactors"]["tissue_hires_scalef"]
  return(scale)

def getCounts(anndat):
  counts = anndat.X
  return(counts)

def getObs(anndat):
  obs = anndat.obs
  return(obs)

def getcoords(anndat):
  obs = anndat.obsm
  return(obs)























