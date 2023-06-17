# source("lab5-classifiers1.R")


# make ABT (analytics base table) - mostly copied from my preprocessing script.
diabetes <- read.csv("diabetes.csv")
diabetes$Outcome <- as.factor(diabetes$Outcome)

zero_quartile_bin <- function(table) {
table1 <- table
FirstQuartile <- quantile(table[table != 0], 0.25)
SecondQuartile <- quantile(table[table != 0], 0.50)
ThirdQuartile <- quantile(table[table != 0], 0.75)

table1[table > ThirdQuartile] <- "FourthQuartile"
table1[table <= ThirdQuartile] <- "ThirdQuartile"
table1[table <= SecondQuartile] <- "SecondQuartile"
table1[table <= FirstQuartile] <- "FirstQuartile"
table1[table==0] <- "N/A"

table1 <- as.factor(table1)
return(table1)
}

diabetes$Glucose <- zero_quartile_bin(diabetes$Glucose)
diabetes$BloodPressure <- zero_quartile_bin(diabetes$BloodPressure)
diabetes$SkinThickness <- zero_quartile_bin(diabetes$SkinThickness)
diabetes$Insulin <- zero_quartile_bin(diabetes$Insulin)
diabetes$BMI <- zero_quartile_bin(diabetes$BMI)

print(str(diabetes))


#####################################################################################
#####################################################################################

library(OneR)
library(mlbench)
library(e1071)
library(caret)

library(car)
library(lattice)
library(Hmisc)
library(RWeka)
library(rpart)


print("-----------------------OneR-----------------------")
## OneR

# Apply supervised binning to the continuous data
diabetes.binned <- optbin(diabetes)

# Create separate training and test sets
# Use a 60:40 split of data for train:test
set.seed(0)
trainSet <- sample(seq_len(nrow(diabetes.binned)), nrow(diabetes.binned) * .6)
diabetes.binned.train <- diabetes.binned[trainSet,]
diabetes.binned.test <- diabetes.binned[-trainSet,]

# Create a 1R classification model
# OneR function assumes that the last feature is the target. If that is not the case
diabetes.oner.model <- OneR::OneR(diabetes.binned.train, verbose = TRUE)

# Look at the raw model (e.g. the tree's decisions)
print(diabetes.oner.model)

# Show the structure of the model
str(diabetes.oner.model)

# Show details regarding the model
summary(diabetes.oner.model)

# Create predictions based on the model
diabetes.oner.pred <- predict(diabetes.oner.model, diabetes.binned.test)

##### Evaluate the model #####
print("-----------------OneR Evaluation-----------------")
eval_model(diabetes.oner.pred, diabetes.binned.test)


####################################################################
print("-----------------------Naive Bayes-----------------------")

## Naive Bayes

# Create separate training and test sets
# Use a 60:40 split of data for train:test
set.seed(6)
trainSet <- sample(seq_len(nrow(diabetes)), nrow(diabetes) * .6)
diabetes.train <- diabetes[trainSet,]
diabetes.test <- diabetes[-trainSet,]

# Verify the class splits in the training set to explain the priors in the model
tbl.data <- table(diabetes$Outcome) / nrow(diabetes)
tbl.train <- table(diabetes.train$Outcome) / nrow(diabetes.train)
print(tbl.data)
print(tbl.train)

# Plot the original and training set class proportions
op <- par(mfrow=c(1,2))
bp <- barplot(tbl.data,
	ylim=c(0, .5),
	main = "Class Proportions in the\nDiabetes Data Set",
	xlab="Proportion",
	ylab="Class")
text(x = bp, y = tbl.data, label = round(tbl.data, digits=2),
	pos = 3, cex = 0.8, col = "red")

bp <- barplot(tbl.train,
	ylim=c(0, .5),
	main = "Class Proportions in the\nDiabetes Training Set",
	xlab="Proportion",
	ylab="Class")
text(x = bp, y = tbl.train, label = round(tbl.train, digits=2),
	pos = 3, cex = 0.8, col = "red")
par(op)

# Create a Naive Bayes classification model
diabetes.nb.model <- naiveBayes(Outcome ~ .,data = diabetes.train)

# Look at the raw model
print(diabetes.nb.model)

# Show the structure of the model
str(diabetes.nb.model)

# Show an overview of the model
summary(diabetes.nb.model)

# Create predictions based on the model
diabetes.nb.pred <- predict(diabetes.nb.model, diabetes.test)

##### Evaluate the model #####
print("-----------------Naive Bayes Evaluation-----------------")
eval_model(diabetes.nb.pred, diabetes.test)

####################################################################
print("-----------------Naive Bayes With Class Proportions Maintained-----------------")

## Maintains class proportions in the training data set

# Use the caret package's createDataPartition function to preserve proportions of classes in the training set
# Again, using a 60:40 split of data for train:test
set.seed(1)
trainSet <- createDataPartition(diabetes$Outcome, p=.6)[[1]]
diabetes.train <- diabetes[trainSet,]
diabetes.test <- diabetes[-trainSet,]

# Verify the class splits in the training set to explain the priors in the model
# Look at proportions instead of raw counts
tbl.data <- table(diabetes$Outcome) / nrow(diabetes)
tbl.train <- table(diabetes.train$Outcome) / nrow(diabetes.train)
print(tbl.data)
print(tbl.train)

# Plot the original and training set class proportions
op <- par(mfrow=c(1,2))
bp <- barplot(tbl.data,
	ylim=c(0, .5),
	main = "Class Proportions in the\nIris Data Set",
	xlab="Proportion",
	ylab="Class")
text(x = bp, y = tbl.data, label = round(tbl.data, digits=2),
	pos = 3, cex = 0.8, col = "red")

bp <- barplot(tbl.train,
	ylim=c(0, .5),
	main = "Class Proportions in the\nIris Training Set",
	xlab="Proportion",
	ylab="Class")
text(x = bp, y = tbl.train, label = round(tbl.train, digits=2),
	pos = 3, cex = 0.8, col = "red")
par(op)

# Create a Naive Bayes classification model
diabetes.nb.model <- naiveBayes(Outcome ~ .,data = diabetes.train)

# Look at the raw model
# Note the priors for the classes and the conditional
# probabilities for the DFs
print(diabetes.nb.model)

# Show the structure of the model
str(diabetes.nb.model)

# Show an overview of the model
summary(diabetes.nb.model)

# Create predictions based on the model
diabetes.nb.pred <- predict(diabetes.nb.model, diabetes.test)

##### Evaluate the model #####
print("-----------------Naive Bayes With Class Proportions Maintained Evaluation-----------------")
eval_model(diabetes.nb.pred, diabetes.test)

####################################################################
print("-----------------Decision Tree-----------------")

set.seed(1)
trainSet <- createDataPartition(diabetes$Outcome, p=.6)[[1]]
diabetes.train <- diabetes[trainSet,]
diabetes.test <- diabetes[-trainSet,]

# Build a decision tree for species using C4.5 (Weka's J48 implementation)
diabetes.model.nom <- J48(Outcome ~ ., data=diabetes.train)

# View details of the constructed tree
print(summary(diabetes.model.nom))

# Plot the decision tree
# Commented out because it was causing an error for unclear reasons ("Plotting Weka trees requires package 'partykit'.") Insofar as I can tell the same thing happens when running the example code.
# plot(diabetes.model.nom)


##### Evaluation #####
print("-----------------Decision Tree Evaluation -----------------")
# Create predictions from the decision tree model using the test set
diabetes.predict.nom <- predict(diabetes.model.nom, diabetes.test)

# Calculation of performance for nominal values uses a confusion matrix and related measures.
diabetes.eval.nom <- confusionMatrix(diabetes.predict.nom, diabetes.test$Outcome)

# Display the evaluation results for the decision tree
print(diabetes.eval.nom)

####################################################################
print("-----------------Rule Set-----------------")
# Build the rule set
diabetes.model.rules <- JRip(Outcome ~ ., data=diabetes.train)

# Display the rule set
print(diabetes.model.rules)

##### Evaluation #####
print("-----------------Rule Set Evaluation -----------------")
# Create predictions from the rule set using the test set
diabetes.predict.rules <- predict(diabetes.model.rules, diabetes.test)

# Calculation of performance for nominal values uses a confusion matrix and related measures.
diabetes.eval.rules <- confusionMatrix(diabetes.predict.rules, diabetes.test$Outcome)

# Display the evaluation results for the rule set
print(diabetes.eval.rules)
