import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cell2location
from matplotlib import rcParams
import os

def runC2LfromAnndata(adata_vis, inf_aver_path, results_folder, N_cells_per_location=20, detection_alpha=20, max_epochs=1000, use_gpu=False):
    """
    Runs the Cell2Location model on a given AnnData object.

    Parameters:
    adata_vis (AnnData): The AnnData object containing spatial transcriptomics data.
    inf_aver_path (str): Path to the CSV file containing inferred cell type average expression.
    results_folder (str): Directory to save the results.
    N_cells_per_location (int, optional): The expected number of cells per spatial location. Default is 20.
    detection_alpha (float, optional): Alpha parameter for detection model. Default is 20.
    max_epochs (int, optional): Maximum number of training epochs. Default is 1000.
    use_gpu (bool, optional): Whether to use GPU for training. Default is False.

    Returns:
    cell_types (DataFrame): Posterior distribution of cell types in each location (spatial coordinates) after the model has been trained.
    """
    
    run_name = f'{results_folder}/cell2location_map'
    adata_file = f"{run_name}/sp.h5ad"
    
    if os.path.exists(adata_file):
        print("File exists!")
        adata_vis = sc.read_h5ad(adata_file)
    
    else:
        print("No Processed file found: Start Pipeline")
        adata_vis.var_names_make_unique()
        adata_vis.var['SYMBOL'] = adata_vis.var_names
        adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['SYMBOL']]
        adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
        
        if adata_vis.obsm['MT'].size != 0:
            adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]
        
        import pandas as pd
        inf_aver = pd.read_csv(inf_aver_path, index_col=0)
        
        # Find shared genes and subset both AnnData and reference signatures
        intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
        
        # Subset the AnnData and reference signatures
        adata_vis = adata_vis[:, intersect].copy()
        inf_aver = inf_aver.loc[intersect, :].copy()
        
        # Prepare AnnData for Cell2Location model
        cell2location.models.Cell2location.setup_anndata(adata=adata_vis)
        mod = cell2location.models.Cell2location(adata_vis, cell_state_df=inf_aver, N_cells_per_location=N_cells_per_location, detection_alpha=detection_alpha)
        
        # Train the model
        mod.train(max_epochs=max_epochs, batch_size=None, train_size=1, use_gpu=use_gpu)
        
        # Export posterior
        adata_vis = mod.export_posterior(adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': use_gpu})
        
        # Save the model
        mod.save(f"{run_name}", overwrite=True)
        
        # Save the AnnData object with results
        adata_file = f"{run_name}/sp.h5ad"
        adata_vis.write(adata_file)
    
    # Extract cell types
    cell_types = adata_vis.obsm["q05_cell_abundance_w_sf"]
    return cell_types
