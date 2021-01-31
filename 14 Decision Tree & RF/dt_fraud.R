library(shiny)
ui <- fluidPage(
  titlePanel("Prediction"),
  sidebarLayout(
    sidebarPanel(
      textInput("num","Enter Undergrad","NO"),
      textInput("num1","Enter marital status","Single"),
      numericInput("num2","Enter city.population",50047),
      numericInput("num3","Enter work.experience",9),
      textInput("num4","ENTER urban","YES")
      
      
    ),
    mainPanel(
      tableOutput("distplot")
      
    )
  )
)
server <- function(input, output) {
  output$distplot <- renderTable({
    
    fc <- read.csv("D:/Data Science/Data Science/Assignment/Random Forests/Fraud_check.csv")
    fc$Taxable.Income=ifelse(fc$Taxable.Income<=30000,"risky","good")
    fc$Taxable.Income=as.factor(fc$Taxable.Income)
    library(C50)
    model=C5.0(fc[,-3],fc$Taxable.Income)
    
    nw=data.frame(Undergrad=input$num,Marital.Status= input$num1,City.Population= input$num2, 
                  Work.Experience= input$num3,Urban= input$num4)
    nw
    pre=predict(model,nw)
    pre
    
  })
}

shinyApp(ui = ui, server = server)
