library(shiny)
library(DMwR)
library(e1071)
ui <- fluidPage(
  titlePanel("Prediction"),
  sidebarLayout(
    sidebarPanel(
      numericInput("num","Enter age",35),
      textInput("num1","Enter workclass","Private"),
      textInput("num2","Enter education","HS-grad"),
      numericInput("num3","Enter education no",9),
      textInput("num4","ENTER marital status","Divorced"),
      textInput("num5","ENTER occupation","Handlers-cleaners"),
      textInput("num6","ENTER relationship","Husband"),
      textInput("num7","ENTER race","Black"),
      textInput("num8","ENTER sex","Male"),
      numericInput("num9","ENTER capitalgain",0),
      numericInput("num10","ENTER capitalloss",0),
      numericInput("num11","ENTER hoursperweek",15),
      textInput("num12","ENTER native","Cuba")
      
    ),
    mainPanel(
      tableOutput("distplot")
      
    )
  )
)
server <- function(input, output) {
  output$distplot <- renderTable({
    
    
    sal <- read.csv("D:/Data Science/Data Science/Assignment/Naive Bayes/SalaryData_Train.csv")
    bsal=SMOTE(Salary~.,data=sal)
    sal_specifier=naiveBayes(bsal[,-14],bsal$Salary)
    nw=data.frame(age=input$num,workclass= input$num1,education= input$num2,educationno= input$num3,maritalstatus= input$num4,
                  occupation=input$num5,relationship= input$num6,race= input$num7,sex= input$num8,
                  capitalgain= input$num9,capitalloss= input$num10,hoursperweek= input$num11,native= input$num12)
    
    nw
    sal_predict=predict(sal_specifier,nw)
    sal_predict
    
  })
  }

shinyApp(ui = ui, server = server)
