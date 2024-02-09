import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cell2location
from matplotlib import rcParams
import os

#rcParams['pdf.fonttype'] = 42 
#results_folder = r.results_folder
#vis_path=r.vis_path
#inf_aver_path=r.inf_aver_path
#adata_vis = r.annData

def runC2LfromAnndata(adata_vis,inf_aver_path,results_folder,N_cells_per_location=20,detection_alpha=20, max_epochs=1000, use_gpu=False):
  
  run_name = f'{results_folder}/cell2location_map'
  adata_file = f"{run_name}/sp.h5ad"
  
  
  if os.path.exists(adata_file):
    print("File exists!")
    adata_vis = sc.read_h5ad(adata_file)
  
  
  else:
    print("No Processed file found: Start Pipeline")
    #Prepare Data
    #adata_vis = sc.read_visium(vis_path)
    #adata_vis.obs['sample'] = list(adata_vis.uns['spatial'].keys())[0]
    
    adata_vis.var_names_make_unique()
    adata_vis.var['SYMBOL'] = adata_vis.var_names
    #adata_vis.var.set_index('gene_ids', drop=True, inplace=True)
    adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['SYMBOL']]
    adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
    if adata_vis.obsm['MT'].size!=0 :
      adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]
  
    import pandas as pd
    inf_aver = pd.read_csv(inf_aver_path, index_col=0)
    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
  
    # Find overlap of genes
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
  
    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis)
    mod = cell2location.models.Cell2location(adata_vis, cell_state_df=inf_aver,N_cells_per_location=N_cells_per_location,detection_alpha=detection_alpha)
    mod.train(max_epochs=max_epochs,batch_size=None,train_size=1,use_gpu=use_gpu)
    adata_vis = mod.export_posterior(adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': use_gpu})
  
    # Save model
    mod.save(f"{run_name}", overwrite=True)
  
    # Save anndata object with results
    adata_file = f"{run_name}/sp.h5ad"
    adata_vis.write(adata_file)
  
  cell_types = adata_vis.obsm["q05_cell_abundance_w_sf"]
  return(cell_types)

def runC2LfromVisium(results_folder,vis_path,inf_aver_path,N_cells_per_location=20,detection_alpha=20, max_epochs=1000, use_gpu=False):
  
  run_name = f'{results_folder}/cell2location_map'
  adata_file = f"{run_name}/sp.h5ad"
  
  
  if os.path.exists(adata_file):
    print("File exists!")
    adata_vis = sc.read_h5ad(adata_file)
  
  
  else:
    print("No Processed file found: Start Pipeline")
    #Prepare Data
    adata_vis = sc.read_visium(vis_path)
    adata_vis.obs['sample'] = list(adata_vis.uns['spatial'].keys())[0]
    adata_vis.var['SYMBOL'] = adata_vis.var_names
    adata_vis.var.set_index('gene_ids', drop=True, inplace=True)
    adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['SYMBOL']]
    adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
    adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]
  
    import pandas as pd
    inf_aver = pd.read_csv(inf_aver_path, index_col=0)
    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
  
    # Find overlap of genes
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
  
    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")
    mod = cell2location.models.Cell2location(adata_vis, cell_state_df=inf_aver,N_cells_per_location=N_cells_per_location,detection_alpha=detection_alpha)
    mod.train(max_epochs=max_epochs,batch_size=None,train_size=1,use_gpu=use_gpu)
    adata_vis = mod.export_posterior(adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': use_gpu})
  
    # Save model
    mod.save(f"{run_name}", overwrite=True)
  
    # Save anndata object with results
    adata_file = f"{run_name}/sp.h5ad"
    adata_vis.write(adata_file)
  
  cell_types = adata_vis.obsm["q05_cell_abundance_w_sf"]
  return(cell_types)
  
