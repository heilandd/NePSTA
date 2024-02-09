fromAnnData <- function(anndat){
  
  
  
  img=getImageAnnData(anndat)
  img = EBImage::Image(img, colormode="Color") %>% EBImage::rotate(-90)
  
  scale=getScale(anndat)
  coords <- getcoords(anndat)$spatial %>% as.matrix()
  coords <- coords*scale
  coords <- coords %>% as.data.frame()
  names(coords) <- c("x", "y")
  
  # Create SPATA object
  
  
  coords_df <- 
    cbind(getObs(anndat), coords) %>% 
    rownames_to_column("barcodes") %>% 
    rename("row":=array_row) %>% 
    rename("col":=array_col) %>% 
    rename("tissue":=in_tissue) %>% 
    dplyr::select(barcodes,x,y,tissue,row,col )
  
  count_mtr <- anndat$X %>% as.matrix() %>% t() %>% Matrix::Matrix(sparse = T)
  
  obj <- SPATA2::initiateSpataObject_CountMtr(coords_df = coords_df, 
                                              count_mtr = count_mtr, 
                                              sample_name = "A", 
                                              image = img,
                                              image_class = "HistologyImage")
  return(obj)
  
}
toAnnData <- function(object){
  
  ## Transfer SPATA2 objects to anndata
  library(SingleCellExperiment)
  sc <- reticulate::import("scanpy")
  np <- reticulate::import("numpy")
  pd <- reticulate::import("pandas")
  bc <- getBarcodes(object)
  sce <- 
    Seurat::CreateSeuratObject(counts=SPATA2::getCountMatrix(object)) %>% 
    Seurat::as.SingleCellExperiment()
  exprs <- assay(sce, "counts")
  col_data <- as.data.frame(colData(sce))
  row_data <- as.data.frame(rowData(sce))
  
  
  ## Create AnnData
  
  adata = sc$AnnData(X = t(exprs),obs = col_data, var = row_data)
  adata$obsm[["spatial"]] <- getCoordsDf(object)[,c("x", "y")] %>% as.matrix()
  
  adata$obsm[["pca"]] <- 
    SPATA2::getPcaDf(object) %>% 
    filter(barcodes %in% bc) %>% 
    select(-barcodes, -sample) %>% 
    as.matrix()
  
  adata$obsm[["umap"]] <- 
    SPATA2::getUmapDf(object) %>% 
    filter(barcodes %in% bc) %>% 
    select(-barcodes, -sample) %>% 
    as.matrix()

  return(adata)
}


