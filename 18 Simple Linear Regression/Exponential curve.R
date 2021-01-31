ui <- fluidPage(sliderInput("exponent", 
                      label = "Choose an exponent",
                      min = 1,
                      max = 5,
                      value = 2
                      ),
          plotOutput("curve_plot")
)

server <- function(input, output, session){
     output$curve_plot <- renderPlot({
          curve(x^input$exponent, from = -5, to = 5)
     })
}

shinyApp(ui = ui, server = server)