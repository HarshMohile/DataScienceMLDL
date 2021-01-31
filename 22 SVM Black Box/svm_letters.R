library(shiny)
ui <- fluidPage(
  titlePanel("Prediction"),
  sidebarLayout(
    sidebarPanel(
      numericInput("num","Enter x.box",1),
      numericInput("num1","Enter y.box  ",1),
      numericInput("num2","Enterwidth  ",1),
      numericInput("num3","Enter high ",1),
      numericInput("num4","ENTER onpix ",1),
      numericInput("num5","ENTER x.bar ",1),
      numericInput("num6","ENTER y.bar ",1),
      numericInput("num7","ENTER x2bar ",1),
      numericInput("num8","ENTER y2bar",1),
      numericInput("num9","ENTER xybar",1),
      numericInput("num10","ENTER x2ybar ",1),
      numericInput("num11","ENTER xy2bar",1),
      numericInput("num12","ENTER x.ege ",1),
      numericInput("num13","ENTER x.egvy ",1),
      numericInput("num14","ENTER y.ege ",1),
      numericInput("num15","ENTER yegvx ",1)
      
    ),
    mainPanel(
      tableOutput("distplot")
      
    )
  )
)
server <- function(input, output) {
  output$distplot <- renderTable({
    
    claimants <- read.csv("D:/Data Science/Data Science/RCodes/Support Vector Machines/letters.csv")
    
    model1<-ksvm(lettr ~.,data = claimants,kernel = "vanilladot")
    nw=data.frame( x.box=input$num,y.box=input$num1,width=input$num2,high=input$num3,onpix= input$num4,x.bar= input$num5,
                   y.bar= input$num6,x2bar= input$num7,y2bar= input$num8,xybar= input$num9,x2ybar= input$num10,
                   xy2bar= input$num11,x.ege=input$num12,xegvy= input$num13,y.ege= input$num14,yegvx= input$num15)
    nw
    pred_vadot<-predict(model1,newdata=nw)
    pred_vadot
  })
  
}

shinyApp(ui = ui, server = server)
