#social media NA :
library("igraph")


fb <- read.csv(file.choose())
#colnames(fb) <- c("fb1","fb2","fb3","fb4","fb5","fb6","fb7","fb8","fb9")
ln <- read.csv(file.choose())
#colnames(ln) <- c("ln1","ln2","ln3","ln4","ln6","ln6","ln7","ln8","fb9","ln10","ln11","ln12","ln13")
insta <- read.csv(file.choose())
#colnames(insta) <- c("in1","in2","in3","in4","in6","in6","in7","in8")

class(fb)

fb[]
fb[1,]

#--------------------undirected------------------------------
g1 <- graph.adjacency(as.matrix(fb), mode="undirected")
plot(g1)
degree.cent <- centr_degree(g1, mode = "all")
degree.cent$res # 2

closeness.cent <- closeness(g1, mode="all")
closeness.cent

closeness_fb <- closeness(g1, mode = "in", normalized = TRUE)
max(closeness_fb)
index <- which(closeness_fb == max(closeness_fb))
closeness_fb[index]
plot(g1, layout=layout.sphere, main="sphere")
plot(g1, layout=layout.circle, main="circle")
plot(g1, layout=layout.random, main="random")


g2 <- graph.adjacency(as.matrix(ln), mode="directed")
plot(g2)
degree.cent <- centr_degree(g2, mode = "in")
degree.cent$res # 2 3 3 3 2 3 3 3 2 3 3 3 3

closeness.cent <- closeness(g2, mode="in")
closeness.cent
max(closeness.cent)



plot(g2, layout=layout.sphere, main="sphere")
plot(g2, layout=layout.circle, main="circle")
plot(g2, layout=layout.random, main="random")


g3 <- graph.adjacency(as.matrix(insta), mode="undirected")
plot(g3)
degree.cent <- centr_degree(g3, mode = "all")
degree.cent$res #  2 2 2 2 2 2 2 2 2
plot(g3, layout=layout.sphere, main="sphere")
plot(g3, layout=layout.circle, main="circle")
plot(g3, layout=layout.random, main="random")

#---------------------directed----------------------

g1 <- graph.adjacency(as.matrix(fb), mode="directed")
plot(g1)
plot(g1, layout=layout.sphere, main="sphere")
plot(g1, layout=layout.circle, main="circle")
plot(g1, layout=layout.random, main="random")


g2 <- graph.adjacency(as.matrix(ln), mode="directed")

plot(g2, layout=layout.sphere, main="sphere")
plot(g2, layout=layout.circle, main="circle")
plot(g2, layout=layout.random, main="random")


g3 <- graph.adjacency(as.matrix(insta), mode="directed")

plot(g3, layout=layout.sphere, main="sphere")
plot(g3, layout=layout.circle, main="circle")
plot(g3, layout=layout.random, main="random")


