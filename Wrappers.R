#' @title  runCell2Location
#' @author Dieter Henrik Heiland
#' @description Run C2L Deconv
#' @inherit
#' @param object SPATA object
#' @return
#' @examples
#' @export
#'
runCell2Location <- function(object,
                             anndata=NULL,
                             inf_aver_path,
                             epochs=1000,
                             nr_cells=20,
                             vis_path=NULL,
                             temp_dir=NULL,
                             remove.temp=F,
                             verbose=T){


  message(paste0(Sys.time(), "### Create Temporal Folder"))

  if(is.null(temp_dir)){
    dir.create("Temp_folder")
    root_path <- paste0(getwd(), "/Temp_folder")
  }else{
    #setwd(temp_dir)
    #dir.create("Temp_folder")
    root_path <- temp_dir
  }

  if(file.exists(inf_aver_path)){
    message("inf_aver file path is correct")
  }



  if(is.null(vis_path)){
    confuns::give_feedback(msg = "Transform SPATA2Anndata", verbose = verbose)
    if(is.null(anndata)){
      annData <- asAnndata(object)
    }else{
      annData <- anndata
    }




    confuns::give_feedback(msg = "Import and check python environment", verbose = verbose)
    library(DeepSPATA)
    python_script_path <- system.file("python", "Cell2Location.py", package = "DeepSPATA")
    reticulate::source_python(python_script_path)

    cell_types_df <- runC2LfromAnndata(adata_vis=annData,
                                       inf_aver_path=inf_aver_path,
                                       N_cells_per_location=nr_cells,
                                       results_folder=temp_dir,
                                       max_epochs=epochs) %>% as.data.frame() %>% rownames_to_column("barcodes")
    names(cell_types_df) <- str_remove(names(cell_types_df), "q05cell_abundance_w_sf_")



  }else{

    results_folder = root_path
    vis_path_spata <- object@information$initiation$input$directory_10X

    if(dir.exists(vis_path_spata)){
      message("Visium File Path is correct")
      vis_path <- vis_path_spata
    }else{
      if(!is.null(vis_path)){
        if(dir.exists(vis_path)){
          message("Visium File Path is correct")
        }else{ stop("No correct visium path found")}
      }
    }


    message(paste0(Sys.time(), "  ### Run Deconvolution ###"))
    cell_types_df <-  runC2LfromVisium(results_folder,vis_path,inf_aver_path) %>% as.data.frame() %>% rownames_to_column("barcodes")
    names(cell_types_df) <- str_remove(names(cell_types_df), "q05cell_abundance_w_sf_")


  }

  object <- object %>% SPATA2::addFeatures(cell_types_df)

  return(object)

}



#' @title  asAnndata
#' @author Dieter Henrik Heiland
#' @description SPATA2obj into  Anndata
#' @inherit
#' @param object SPATA object
#' @return
#' @examples
#' @export
#'
asAnndata <- function(object){


  library(SPATA2)
  library(SingleCellExperiment)
  library(tidyverse)
  sc <- reticulate::import("scanpy")
  np <- reticulate::import("numpy")
  pd <- reticulate::import("pandas")


  sce <- SPATA2::asSingleCellExperiment(object)
  exprs <- assay(sce, "counts")
  col_data <- as.data.frame(colData(sce))
  row_data <- as.data.frame(rowData(sce))

  ## Create AnnData

  adata_seurat = sc$AnnData(X = t(exprs),
                            obs = col_data,
                            var = row_data)

  adata_seurat$obsm[["spatial"]] <- col_data[,c("x", "y")]
  adata_seurat$obsm[["pca"]] <- SPATA2::getPcaDf(object) %>% select(-barcodes, -sample) %>% as.matrix()
  adata_seurat$obsm[["umap"]] <- SPATA2::getUmapDf(object) %>% select(-barcodes, -sample) %>% as.matrix()


  return(adata_seurat)


}
