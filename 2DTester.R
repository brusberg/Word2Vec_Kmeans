data <- read.csv(file.choose(), header = FALSE)
# data$V1 <- as.numeric(as.character(data$V1))
data$V3 <- data$V3+1
plot(data$V1, data$V2, col=c("red", "blue" ,"green","orange", "yellow")[data$V3]);

# c("red","blue","green")[data$V3]
# plot(cluster::xclara$V1,cluster::xclara$V2)
# options(max.print=6000)
# 
# sink("comeone.txt")
# cluster::xclara
# sink()
# unlink("comeone.txt")