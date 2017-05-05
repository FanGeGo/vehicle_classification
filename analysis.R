require(gdata)
library(ggplot2)
library(RColorBrewer)
df = read.csv("./dataset/Video Data-03-08-2015/extended_dataset.csv", header = TRUE)
df = df[1:3000,]
boxplot(df$Class,col="red")

# Plot histogram for allthe whole dataset
barplot(prop.table(table(df$Class)), xlab = "Vehicle Class", ylab = "Percentage of frequency")
summary(df$Class)

# Plot a histogram for training and testing set
total_number = nrow(df)
training_set = df[1:floor(0.7*nrow(df)), ]
testing_set = df[(floor(0.7*nrow(df)) + 1):total_number, ]

barplot(prop.table(table(training_set$Class)), xlab = "Training set vehicle Class", ylab = "Percentage of frequency")
summary(training_set$Class)
table(training_set$Class)

barplot(prop.table(table(testing_set$Class)), xlab = "Testing set vehicle Class", ylab = "Percentage of frequency")
summary(testing_set$Class)
table(testing_set$Class)
