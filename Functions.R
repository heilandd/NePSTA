###############################################################
###########   Graph Function
###############################################################

#' @title  CreateGraphSPATA
#' @author Dieter Henrik Heiland
#' @description Create a graph object from SPATA objects
#' @inherit
#' @param object SPATA object
#' @param genes genes to be included
#' @param add_node_feature features to be included
#' @return
#' @examples
#' @export
#'
CreateGraphSPATA <- function(object, genes, add_node_feature=NULL){

  require(SPATA2)
  require(igraph)

  obj <- object
  dist_matrix <- as.matrix(dist(getCoordsDf(obj)[,c("x", "y")]))
  dist_matrix %>% dim()
  test <- dist_matrix
  test[test==0] <- 1000
  threshold <- min(test)+1
  adj_matrix <- ifelse(dist_matrix <= threshold, 1, 0)
  adj_matrix %>% dim()
  rownames(adj_matrix) <- getCoordsDf(obj)$barcodes
  colnames(adj_matrix) <- getCoordsDf(obj)$barcodes
  net <- graph_from_adjacency_matrix(adj_matrix, mode="undirected")
  exp <- getExpressionMatrix(obj) %>% t()
  exp <- exp[names(V(net)), genes]


  pb <- progress::progress_bar$new(total = ncol(exp),
                                   format = "  Add Node Features [:bar] :percent eta: :eta",)

  for(i in 1:ncol(exp)){
    pb$tick()
    net <- igraph::set_vertex_attr(net, name= colnames(exp)[i], value = exp[,i]) }


  if(!is.null(add_node_feature)){
    print("empty")
  }



  return(net)

}




#' @title  CreateSUBGraphSPATA
#' @author Dieter Henrik Heiland
#' @description Create a graph object from SPATA objects
#' @inherit
#' @param object SPATA object
#' @param genes genes to be included
#' @param add_node_feature features to be included
#' @return
#' @examples
#' @export
#'
CreateSUBGraphSPATA <- function(net,
                                object,
                                names_ID,
                                Transcript_full=T,
                                CNV=T,
                                features=NULL,
                                random_n=NULL,
                                clinical_data,
                                hop=3){

  if(!is.null(random_n)){
    if(length(V(net))<random_n){random_n=NULL}}

  if(is.null(random_n)){
  message("Create Graphs from all spots")

    list_subgraph <-map(.x=1:length(V(net)), .f=function(i){

      ## Build subgraph
      central_node <- V(net)[i]

      subgraph_build <-
        ego(
          graph=net,
          order = hop-1,
          nodes = central_node,
          mode = c("all"),
          mindist = 0)
      subgraph_net <- induced_subgraph(net,unlist(subgraph_build))


      ## Add central node information
      central_node_index=rep(0,length(subgraph_net))
      central_node_index[names(V(subgraph_net))==names(central_node)]=1
      node_class <- central_node_index

      ##Add clinical information
      graph_information <- clinical_data #%>% filter(Samples_ID==names_ID) %>% as.data.frame()


      #select sub graph
      edges <- igraph::as_data_frame(subgraph_net, what = c("edges"))
      if(Transcript_full==T){

        order <- igraph::as_data_frame(subgraph_net, what = c("vertices")) %>% rownames()
        mat <- suppressMessages(SPATA2::getExpressionMatrix(object, verbose = F)) %>% t()
        Node_features_transcript <- mat[order, ]


      }else{
        Node_features_transcript <- igraph::as_data_frame(subgraph_net, what = c("vertices"))
      }

      if(CNV==T){
        order <- igraph::as_data_frame(subgraph_net, what = c("vertices")) %>% rownames()

        cnv_data <-
          SPATA2::getFeatureDf(object) %>%
          select(barcodes,paste0("Chr", 1:22)) %>%
          filter(barcodes %in% order) %>%
          as.data.frame()
        rownames(cnv_data) <- cnv_data$barcodes
        cnv_data$barcodes <- NULL
        Node_features_CNV <- cnv_data


      }else{
        Node_features_CNV=list()
      }

      if(!is.null(features)){
        order <- igraph::as_data_frame(subgraph_net, what = c("vertices")) %>% rownames()

        f_data <-
          suppressMessages(SPATA2::joinWithFeatures(object, features = features, verbose = F) ) %>%
          select(barcodes,{{features}}) %>%
          filter(barcodes %in% order) %>%
          as.data.frame()
        rownames(f_data) <- f_data$barcodes
        f_data$barcodes <- NULL

        for(i in features){
          if(!is.numeric(f_data[,i])){
            f_data[,paste0(i,"_num")] <- as.integer(as.factor(f_data[,i]))
          }
        }




        Node_features_add <- f_data


      }else{
        Node_features_add=list()
      }


      edges_features <- c()

      return(list(edges, Node_features_transcript, Node_features_CNV, Node_features_add, node_class, graph_information))


    }, .progress = T)
  }else{
    ## Create Random Graph
    message("Create Random Graphs")
    index <- sample(1:length(V(net)), random_n)
    list_subgraph <-map(.x=index, .f=function(i){

      ## Build subgraph
      central_node <- V(net)[i]

      subgraph_build <-
        ego(
          graph=net,
          order = hop-1,
          nodes = central_node,
          mode = c("all"),
          mindist = 0)
      subgraph_net <- induced_subgraph(net,unlist(subgraph_build))


      ## Add central node information
      central_node_index=rep(0,length(subgraph_net))
      central_node_index[names(V(subgraph_net))==names(central_node)]=1
      node_class <- central_node_index

      ##Add clinical information
      graph_information <- clinical_data #%>% filter(Samples_ID==names_ID) %>% as.data.frame()


      #select sub graph
      edges <- igraph::as_data_frame(subgraph_net, what = c("edges"))
      if(Transcript_full==T){

        order <- igraph::as_data_frame(subgraph_net, what = c("vertices")) %>% rownames()
        mat <- suppressMessages(SPATA2::getExpressionMatrix(object, verbose = F)) %>% t()
        Node_features_transcript <- mat[order, ]


      }else{
        Node_features_transcript <- igraph::as_data_frame(subgraph_net, what = c("vertices"))
      }

      if(CNV==T){
        order <- igraph::as_data_frame(subgraph_net, what = c("vertices")) %>% rownames()

        cnv_data <-
          SPATA2::getFeatureDf(object) %>%
          select(barcodes,paste0("Chr", 1:22)) %>%
          filter(barcodes %in% order) %>%
          as.data.frame()
        rownames(cnv_data) <- cnv_data$barcodes
        cnv_data$barcodes <- NULL
        Node_features_CNV <- cnv_data


      }else{
        Node_features_CNV=list()
      }

      if(!is.null(features)){
        order <- igraph::as_data_frame(subgraph_net, what = c("vertices")) %>% rownames()

        f_data <-
          SPATA2::joinWithFeatures(object, features = features, verbose = F) %>%
          select(barcodes,{{features}}) %>%
          filter(barcodes %in% order) %>%
          as.data.frame()
        rownames(f_data) <- f_data$barcodes
        f_data$barcodes <- NULL

        for(i in features){
          if(!is.numeric(f_data[,i])){
            f_data[,paste0(i,"_num")] <- as.integer(as.factor(f_data[,i]))
          }
        }




        Node_features_add <- f_data


      }else{
        Node_features_add=list()
      }


      edges_features <- c()






      return(list(edges, Node_features_transcript, Node_features_CNV, Node_features_add, node_class, graph_information))


    }, .progress = T)
  }

  return(list_subgraph)


}



#' @title  CreateFullSubgraph
#' @author Dieter Henrik Heiland
#' @description New Version of the CreateSUBGraphSPATA function which is faster
#' @inherit
#' @param object SPATA object
#' @param genes genes to be included
#' @param add_node_feature features to be included
#' @return
#' @examples
#' @export
#'
CreateFullSubgraph <- function(object,
                               celltypes,
                               features,
                               clinical_data,
                               hop=3){
  
  # Mask Genes --------------------------------------------------------------
  message("Mask Genes")
  object <- runGeneMaskObject(object, features)
  
  # Create Graph ------------------------------------------------------------
  library(SPATA2)
  library(igraph)
  dist_matrix <- as.matrix(dist(getCoordsDf(object)[,c("x", "y")]))
  
  test <- dist_matrix
  test[test==0] <- 1000
  threshold <- min(test)+1
  adj_matrix <- ifelse(dist_matrix <= threshold, 1, 0)
  
  rownames(adj_matrix) <- getCoordsDf(object)$barcodes
  colnames(adj_matrix) <- getCoordsDf(object)$barcodes
  net <- graph_from_adjacency_matrix(adj_matrix, mode="undirected")
  
  
  # Select Subgraphs --------------------------------------------------------
  
  message("Create Subgraphs")
  bcs <- SPATA2::getFeatureDf(object)$barcodes
  
  ## Alternative
  nx <- reticulate::import("networkx")
  G = nx$from_pandas_edgelist(igraph::as_data_frame(net, what="edges"), "from", "to")
  
  edges_subgraph <- map(1:length(bcs), .f=function(i){
    
    subnodes <- nx$single_source_shortest_path_length(G, bcs[i], cutoff=hop)
    
    nn=subnodes %>% as.data.frame() %>% t()
    colnames(nn)="NN"
    
    subgraph = G$subgraph(subnodes)
    subgraph <- map_dfr(subgraph$edges, ~ data.frame(from=.x[[1]], to=.x[[2]]))
    
    nodes <- names(subnodes)
    center_node <- ifelse(unlist(subnodes)==0, 1, 0)
    
    return(list(edges=subgraph, nodes=nodes, center_node=center_node, neighborhood=nn ))
    
  },.progress = T)
  
  
  # Import matrix -----------------------------------------------------------
  
  message("Add Subgraph gene expression")
  #Expression
  mat <- suppressMessages(SPATA2::getExpressionMatrix(object, verbose = F)) %>% t()
  mat <- mat[, features]
  exp <- map(edges_subgraph, ~ mat[.x$nodes,], .progress=T)
  
  fdata_names <- names(object@fdata[[1]])
  
  
  if(any(fdata_names==paste0("Chr", 1))){
    message("found CNV data")
    
    #CNV
    cnv_data <-
      SPATA2::getFeatureDf(object) %>%
      select(barcodes,paste0("Chr", 1:22))
    
    cnv <- map(edges_subgraph, .f=function(.x){
      cnv_1 <- cnv_data[cnv_data$barcodes %in%.x$nodes,] %>% select(-1) %>% as.matrix()
      rownames(cnv_1) <- .x$nodes
      return(cnv_1)
    } , .progress=T)
    
    
  }else{
    message("No CNV data")
    cnv <- map(1:length(edges_subgraph),~list())
  }
  
  if(any(fdata_names==celltypes[1])){
    message("found celltype annotation")
    
    #CNV
    ct_data <-
      SPATA2::getFeatureDf(object) %>%
      select(barcodes,all_of(celltypes))
    
    ct <- map(edges_subgraph, .f=function(.x){
      ct_1 <- ct_data[ct_data$barcodes %in%.x$nodes,] %>% select(-1) %>% as.matrix()
      rownames(ct_1) <- .x$nodes
      return(ct_1)
    } , .progress=T)
    
    
  }else{ct <- map(1:length(edges_subgraph),~list())}
  
  # Annotation --------------------------------------------------------------
  
  if(!any(getFeatureNames(object)=="Annotation")){
    object@fdata[[1]]$Annotation="Not_avaiable"
  }
  
  Annotation_mat <- joinWithFeatures(object, features = "Annotation") %>% select(Annotation) %>% as.matrix()
  rownames(Annotation_mat) <- bcs
  
  anno <- map(edges_subgraph, .f=function(.x){
    anno_1 <- data.frame(Annotation=Annotation_mat[.x$nodes, ])
    rownames(anno_1) <- .x$nodes
    return(anno_1)
  }, .progress=T)
  
  
  # Create / Merge List -----------------------------------------------------
  
  
  Subgraph <- map(1:length(edges_subgraph), .f=function(i){
    
    
    list(edges=edges_subgraph[[i]]$edges,
         expression = exp[[i]],
         CNV=cnv[[i]],
         Annotation= anno[[i]],
         CenterNode=edges_subgraph[[i]]$center_node,
         neighborhood=edges_subgraph[[i]]$neighborhood,
         clinical_data=clinical_data,
         celltype=ct[[i]])
    
    
  }, .progress=T)
  Subgraph <- map(Subgraph, .f=function(s){
    
    if(nrow(s[[1]])>1){
      masked_genes <- colSums(s[[2]])==0
    }else{
      masked_genes <- s[[2]]==0
    }
    
    masked_genes <- as.integer(masked_genes)
    s[["mask"]] <- masked_genes
    return(s)
    
  },.progress = T)
  
  
} 




#' @title  initiateSpataGraphObject
#' @author Dieter Henrik Heiland
#' @description Create a graph object from SPATA objects
#' @inherit
#' @param object SPATA object
#' @param genes genes to be included
#' @param add_node_feature features to be included
#' @return
#' @examples
#' @export
#'
initiateSpataGraphObject <- function(list_objects,
                                     add_node_feature=NULL,
                                     n_features=500,
                                     genes_select=NULL,
                                     random_n=NULL,
                                     hop=3,
                                     CNV=F,
                                     clinical_data){


  ## Aggregate Samples and get genes
  message(paste0(Sys.time(), " ### Aggregate SPATA Objects ### "))
  if(is.null(genes_select)){

    inter <- Reduce(intersect,  map(list_objects, ~SPATA2::getGenes(.x)))
    list_seurat <- map(list_objects, .f=function(.x){s <- Seurat::CreateSeuratObject(counts=SPATA2::getCountMatrix(.x))})
    s <- scCustomize::Merge_Seurat_List(list_seurat)
    s <- subset(s, features = inter)
    genes <- s %>% Seurat::FindVariableFeatures( nfeatures=n_features) %>% Seurat::VariableFeatures()

  }else{genes=genes_select}

  ## Whole Sample Graph
  message(paste0(Sys.time(), " ### Create Graph ### "))
  samples <- map(list_objects, ~SPATA2::getSampleName(.x)) %>% unlist
  WSG <- map(list_objects,~CreateGraphSPATA(.x, genes))
  names(WSG) <- samples
  Graph_object@WG <- WSG

  ## Create Subgraph
  message(paste0(Sys.time(), " ### Create Subgraphs ### "))
  list_sub=vector("list", length(Graph_object@WG))
  for(i in 1:length(Graph_object@WG)){

    list_subgraph <-   CreateSUBGraphSPATA(Graph_object@WG[[i]],
                                         object = list_objects[[i]],
                                         features=add_node_feature,
                                         clinical_data=clinical_data,
                                         names_ID = names(Graph_object@WG)[i],
                                         CNV=CNV,
                                         random_n=random_n)


    nodes <- map(list_subgraph, ~.x[[3]] %>% nrow()) %>% unlist()
    if(any(nodes==1)){list_subgraph <- list_subgraph[-c(which(nodes==1))]}

    list_sub[[i]] <- list_subgraph

  }
  names(list_sub) <- samples
  Graph_object@Subgraphs <- list_sub
  return(Graph_object)

}




#' @title  runGeneMaskObject
#' @author Dieter Henrik Heiland
#' @description Create a graph object from SPATA objects
#' @inherit
#' @param object SPATA object
#' @param genes genes to be included
#' @param add_node_feature features to be included
#' @return
#' @examples
#' @export
#'
runGeneMaskObject <- function(object, com_genes){

  mat <- getExpressionMatrix(object, mtr_name = "scaled")
  #dim(mat)

  ## Mask
  add_mask <- com_genes[!com_genes %in% intersect(rownames(mat), com_genes)]
  mask <- matrix(0, nrow=length(add_mask), ncol=ncol(mat))
  rownames(mask) <- add_mask;colnames(mask) <- colnames(mat)
  mat_mask <- rbind(mat, mask)
  dim(mat_mask)

  ## Remove genes that are not in com_genes
  mat_mask <- mat_mask[com_genes, ]
  dim(mat_mask)
  object <- addExpressionMatrix(object, expr_mtr = mat_mask,  mtr_name="scaled")

  return(object)
}
