library("igraph")

flight <- read.csv(file.choose(),header=FALSE)
head(flight)
#setting colnames
colnames(flight) <-c("ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time")
head(flight)


connecting_routes <- read.csv(file.choose(),header=FALSE)
head(flight)
#setting colnames
colnames(connecting_routes) <-c("flights", " ID", "main Airport","main Airport ID", "Destination ", "Destination ID", "haults", "machinary")
head(flight)

#from main airport to destination
conn_routesNW <- graph.edgelist(as.matrix(connecting_routes[, c(3, 5)]), directed = TRUE)
plot(conn_routesNW)
conn_routesNW <- graph.edgelist(as.matrix(connecting_routes[1:200, c(3, 5)]), directed = TRUE)
plot(conn_routesNW)
# a lot of indegree  coming towards on ARN and MNL NJC .If they are removed .It will break the network.

vcount(conn_routesNW) # 99  airports 
ecount(conn_routesNW)  # 200 connection since connecting_routes[1:200, c(3, 5)].200 is set as rows limit.

#INDEGREE
indegree <- degree(conn_routesNW, mode = "in")  # 13 max indegree of an airport
max(indegree)
index <- which(indegree == max(indegree))   #ARN has 13 indgeree which is max
indegree[index]
which(flight$IATA_FAA == "ARN")  # ARN is located at index 727 from fligth dataset.
ARNdata <- flight[727, ] # view its full detail in a dataframe
class(ARNdata)

#CLOSENESS
closeness_in <- closeness(conn_routesNW, mode = "in", normalized = TRUE)
max(closeness_in)
index <- which(closeness_in == max(closeness_in))
closeness_in[index]     # SVX  with closeness of 0.13
which(flight$IATA_FAA == "SVX")
flight[2896, ]
#Airport closer to other airports

#OUTDEGEREE
# outdegree of ariports ,of they are going to 
outdegree <- degree(conn_routesNW, mode = "out")
max(outdegree) # 13 outdegere
index <- which(outdegree == max(outdegree))
outdegree[index]          # ARN has highest outdegree and indegree
which(flight$IATA_FAA == "ARN")
flight[727, ]


#BETWEENESS
?betweenness
btwn <- betweenness(conn_routesNW, normalized = TRUE)
max(btwn)   #0.03040185
index <- which(btwn == max(btwn))
btwn[index]      # ARN
which(flight$IATA_FAA == "ARN")
flight[727,]


# Degree, closeness, and betweenness centralities together
centralities <- cbind(indegree, outdegree, closeness_in, btwn)
colnames(centralities) <- c("inDegree","outDegree","closenessIn","betweenness")
head(centralities)

cor(centralities)
plot(centralities[, "inDegree"], centralities[, "outDegree"]) # highest corr iwth indegree and outdegree

subset(centralities, (centralities[,"closenessIn"] > 0.050) & (centralities[,"betweenness"] > 0.35))
flight[which(flight$IATA_FAA == "ARN"), ]
flight[which(flight$IATA_FAA == "CDG"), ]
flight[which(flight$IATA_FAA == "ANC"), ]


# A high eigenvector score means
#that a node is connected to many nodes who themselves have high scores.
?eigen_centrality
eigenv <- eigen_centrality(conn_routesNW, directed = TRUE, scale = FALSE, weights = NULL)
eigenv$vector
max(eigenv$vector)
index <- which(eigenv$vector == max(eigenv$vector))
eigenv$vector[index]
which(flight$IATA_FAA == "ARN")
flight[727, ]

?page_rank
pg_rank <- page_rank(conn_routesNW, damping = 0.85) # do not put damping=1; the solution not necessarily converges; put a value close to 1.
pg_rank$vector
max(pg_rank$vector)
index <- which(pg_rank$vector == max(pg_rank$vector))
pg_rank$vector[index]
which(airports$IATA_FAA == "ARN")
airports[727, ]

