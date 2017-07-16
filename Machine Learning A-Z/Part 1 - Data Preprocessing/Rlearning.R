dataset = read.csv('Data.csv')
print(dataset)

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# encoding categorical data
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))

# split the dataset into training and test set

# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

training_Set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling
training_Set[,2:3] = scale(training_Set[, 2:3])
test_set[,2:3] = scale(test_set[,2:3])
