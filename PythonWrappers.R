#' @title  InitializeGraphObjectR
#' @author Dieter Henrik Heiland
#' @description R Wrapper of the InitializeGraphObject python function
#' @inherit
#' @param input_list List of Subgraphs
#' @return
#' @examples
#' @export
#'
#'
InitializeGraphObjectR <- function(input_list) {

  library(reticulate)
  python_script_path <- system.file("python", "InitializeGraphObject.py", package = "DeepSPATA")
  reticulate::source_python(python_script_path)
  out <- InitializeGraphObject(input_list)
  return(out)
}
