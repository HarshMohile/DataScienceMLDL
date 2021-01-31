library(shiny)

ui <- fluidPage(
 titlePanel("Adipose Tissue"),
   sidebarLayout(
    sidebarPanel(
      numericInput("waist_circ", "Waist Circumference", 65)
    ),
    mainPanel(
      tableOutput("Pred_AT")
      
    )
  )
)

server <- function(input, output) {
  output$Pred_AT <- renderTable({
    
    wc_at <- read.csv("C:/Datasets_BA/360DigiTMG/DS_India/360DigiTMG DS India Module wise PPTs/Module 06 Simple Linear Regression/Data/wc-at.csv")
    
    reg_log <- lm(AT ~ Waist, data = wc_at)
    nw = data.frame(Waist = input$waist_circ)
    nw
    w = predict(reg_log, nw)
    w
  })
  
}

shinyApp(ui = ui, server = server)