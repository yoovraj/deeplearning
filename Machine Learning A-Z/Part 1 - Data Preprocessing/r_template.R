dataset = read.csv('Data.csv')
dateset = dataset[,1:3]

print(dataset)

# split the dataset into training and test set

# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

training_Set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling
# training_Set[,2:3] = scale(training_Set[, 2:3])
# test_set[,2:3] = scale(test_set[,2:3])
