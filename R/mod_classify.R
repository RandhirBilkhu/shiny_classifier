#' classify UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd 
#'
#' @importFrom shiny NS tagList 
mod_classify_ui <- function(id){
  ns <- NS(id)
  tagList(
 
  )
}
    
#' classify Server Function
#'
#' @noRd 
mod_classify_server <- function(input, output, session){
  ns <- session$ns
 
}
    
## To be copied in the UI
# mod_classify_ui("classify_ui_1")
    
## To be copied in the server
# callModule(mod_classify_server, "classify_ui_1")
 
