#' The application server-side
#' 
#' @param input,output,session Internal parameters for {shiny}. 
#'     DO NOT REMOVE.
#' @import shiny
#' @noRd
#' 

#reticulate::source_python("classify_functions.py")

app_server <- function( input, output, session ) {
  # List the first level callModules here
   mod_classify_server("classify_ui_1")
}
