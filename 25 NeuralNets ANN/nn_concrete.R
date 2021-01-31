library(shiny)
ui <- fluidPage(
  titlePanel("Prediction"),
  sidebarLayout(
    sidebarPanel(
      numericInput("num","Enter cement",141.36),
      numericInput("num1","Enter slag",42.25),
      numericInput("num2","Enter ash",124.3),
      numericInput("num3","Enter water",52),
      numericInput("num4","ENTER superplastic",10.5),
      numericInput("num5","Enter coarseagg",971.8),
      numericInput("num6","Enter fineagg",750),
      numericInput("num7","Enter age",35)
      
    ),
    mainPanel(
      tableOutput("distplot")
      
    )
  )
)
server <- function(input, output) {
  output$distplot <- renderTable({
    
    concrete1=read.csv("D:/Data Science/Data Science/Assignment/Neural Networks/concrete.csv")
    normalize<-function(x){
      return ( (x-min(x))/(max(x)-min(x)))
    }
    concrete1<-as.data.frame(lapply(concrete1,FUN=normalize))
    library(neuralnet)
    cmode=neuralnet(strength~cement+slag+ash+water+superplastic+coarseagg+fineagg+age,data=concrete1)
    nw=data.frame(cement =input$num,slag=input$num1,ash= input$num2,water=input$num3,
                  superplastic= input$num4,coarseagg=input$num5,fineagg=input$num6,age= input$num7 
    )
    nw
    conres=compute(cmode,nw)
    conres$net.result
    
    
  })
}

shinyApp(ui = ui, server = server)
