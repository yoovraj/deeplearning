dataset = read.csv('Salary_Data.csv')
dateset = dataset[,1:3]

print(dataset)

# split the dataset into training and test set

# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)

training_Set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling
# training_Set[,2:3] = scale(training_Set[, 2:3])
# test_set[,2:3] = scale(test_set[,2:3])

# Fitting Simple Linear Regression to this Training Set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_Set)
y_pred = predict(regressor, newdata = test_set)

# visualize the training set results
#install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_Set$YearsExperience, y = training_Set$Salary), 
             color = 'red') + 
  geom_line(aes(x = training_Set$YearsExperience, y = predict(regressor, newdata = training_Set)),
            color = 'blue') + 
  ggtitle('Salary vs Experience(Training set)') +
  xlab('Years of experience') +
  ylab('Salary')
