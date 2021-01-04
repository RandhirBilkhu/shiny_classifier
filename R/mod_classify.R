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
    tableOutput(NS(id, "conf_matrix")),
    tableOutput(NS(id,  "fscore")),
    tableOutput(NS(id,  "precision")),
    tableOutput(NS(id,  "recall"))
    
  )
}
    
#' classify Server Function
#'
#' @noRd 
mod_classify_server <- function(id){
  #ns <- session$ns
   moduleServer(id, function(input, output, session){
     
     #data <- prep_data("bbc_data.csv")
     model <-log_reg(data[[1]] , data[[2]], t_size=0.33)
     
     category <- unique(unlist(data[[3]][3]) )
     conf_matrix <- as.data.frame(model[[1]] , row.names= category)
     colnames(conf_matrix) <- category
      
     output$conf_matrix <- renderTable({  conf_matrix} )
     output$fscore<- renderTable({ model[[2]]} )
     output$accuracy<- renderTable({ model[[3]]} )
     output$recall<- renderTable({ model[[4]]} )
   })
 }
    
## To be copied in the UI
# mod_classify_ui("classify_ui_1")
    
## To be copied in the server
# callModule(mod_classify_server, "classify_ui_1")
 
