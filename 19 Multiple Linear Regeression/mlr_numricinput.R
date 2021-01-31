library(shiny)
ui <- fluidPage(
  titlePanel("Prediction"),
  sidebarLayout(
    sidebarPanel(
      numericInput("num","Numeric input",1),
      numericInput("num1","Numeric input",1),
      numericInput("num2","Numeric input",1),
      numericInput("num3","Numeric input",1)
    ),
    mainPanel(
      tableOutput("distplot")
      
    )
  )
)
server <- function(input, output) {
  output$distplot <- renderTable({
    
    Cars <- read.csv("D:/Data Science/Data Science/RCodes/Multilinear Regression/Cars.csv")
    
    model.car <- lm(MPG~.,data=Cars)
    nw=data.frame(HP=input$num,VOL=input$num1,SP=input$num2,WT=input$num3)
    nw
    w=predict(model.car,nw)
    w
  })
  
}

shinyApp(ui = ui, server = server)
