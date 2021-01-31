library(shiny)
ui <- fluidPage(
  titlePanel("Prediction"),
  sidebarLayout(
    sidebarPanel(
      numericInput("num","Enter CASENUUM",1),
      numericInput("num1","Enter CLMSEX",1),
      numericInput("num2","Enter CLMINSUR",1),
      numericInput("num3","Enter SEATBELT",1),
      numericInput("num4","ENTER CLMAGE",1),
      numericInput("num5","Enter LOSS",1)
    ),
    mainPanel(
      tableOutput("distplot")
      
    )
  )
)
server <- function(input, output) {
  output$distplot <- renderTable({
    
    claimants <- read.csv("D:/Data Science/Data Science/RCodes/Logistic Regression/claimants.csv")
    
    model <- glm(ATTORNEY~.,data=claimants,family = "binomial")
    nw=data.frame(CASENUM=input$num,CLMSEX=input$num1,CLMINSUR=input$num2,SEATBELT=input$num3,CLMAGE=input$num4,LOSS=input$num5)
    nw
    prob <- predict(model,nw,type="response")
    pred_values <- ifelse(prob>=0.5,1,0)
    pred_values
  })
  
}

shinyApp(ui = ui, server = server)
