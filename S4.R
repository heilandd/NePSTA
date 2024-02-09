
###############################################################
###########   S4 object
###############################################################

setClass("GraphObject",representation(WG="list",Subgraphs="list"))
Graph_object <-  new("GraphObject")

setMethod(f = "show",
          signature = "GraphObject",
          definition = function(object){
            print(paste0("The GraphObject contains: ", length(object@WG), " Spatial Graphs (Samples)"))

            for(i in 1:length(object@Subgraphs)){

              message(paste0("The sample (", i, ") contains: ", length(object@Subgraphs[[i]]), " Subgraphs"))

            }


          })
