# source("lab6-classifiers2.R")


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



# make version of ABT with factors as 0/1 numeric variables.
diabetes.numeric <- diabetes[, c(1, 7:9)]

#### Glucose
diabetes.numeric$GlucoseNA[diabetes$Glucose == "N/A"] <- 1
diabetes.numeric$GlucoseNA[diabetes$Glucose != "N/A"] <- 0

diabetes.numeric$GlucoseQ1[diabetes$Glucose == "FirstQuartile"] <- 1
diabetes.numeric$GlucoseQ1[diabetes$Glucose != "FirstQuartile"] <- 0

diabetes.numeric$GlucoseQ2[diabetes$Glucose == "SecondQuartile"] <- 1
diabetes.numeric$GlucoseQ2[diabetes$Glucose != "SecondQuartile"] <- 0

diabetes.numeric$GlucoseQ3[diabetes$Glucose == "ThirdQuartile"] <- 1
diabetes.numeric$GlucoseQ3[diabetes$Glucose != "ThirdQuartile"] <- 0

diabetes.numeric$GlucoseQ4[diabetes$Glucose == "FourthQuartile"] <- 1
diabetes.numeric$GlucoseQ4[diabetes$Glucose != "FourthQuartile"] <- 0


#### BloodPressure
diabetes.numeric$BloodPressureNA[diabetes$BloodPressure == "N/A"] <- 1
diabetes.numeric$BloodPressureNA[diabetes$BloodPressure != "N/A"] <- 0

diabetes.numeric$BloodPressureQ1[diabetes$BloodPressure == "FirstQuartile"] <- 1
diabetes.numeric$BloodPressureQ1[diabetes$BloodPressure != "FirstQuartile"] <- 0

diabetes.numeric$BloodPressureQ2[diabetes$BloodPressure == "SecondQuartile"] <- 1
diabetes.numeric$BloodPressureQ2[diabetes$BloodPressure != "SecondQuartile"] <- 0

diabetes.numeric$BloodPressureQ3[diabetes$BloodPressure == "ThirdQuartile"] <- 1
diabetes.numeric$BloodPressureQ3[diabetes$BloodPressure != "ThirdQuartile"] <- 0

diabetes.numeric$BloodPressureQ4[diabetes$BloodPressure == "FourthQuartile"] <- 1
diabetes.numeric$BloodPressureQ4[diabetes$BloodPressure != "FourthQuartile"] <- 0


#### SkinThickness
diabetes.numeric$SkinThicknessNA[diabetes$SkinThickness == "N/A"] <- 1
diabetes.numeric$SkinThicknessNA[diabetes$SkinThickness != "N/A"] <- 0

diabetes.numeric$SkinThicknessQ1[diabetes$SkinThickness == "FirstQuartile"] <- 1
diabetes.numeric$SkinThicknessQ1[diabetes$SkinThickness != "FirstQuartile"] <- 0

diabetes.numeric$SkinThicknessQ2[diabetes$SkinThickness == "SecondQuartile"] <- 1
diabetes.numeric$SkinThicknessQ2[diabetes$SkinThickness != "SecondQuartile"] <- 0

diabetes.numeric$SkinThicknessQ3[diabetes$SkinThickness == "ThirdQuartile"] <- 1
diabetes.numeric$SkinThicknessQ3[diabetes$SkinThickness != "ThirdQuartile"] <- 0

diabetes.numeric$SkinThicknessQ4[diabetes$SkinThickness == "FourthQuartile"] <- 1
diabetes.numeric$SkinThicknessQ4[diabetes$SkinThickness != "FourthQuartile"] <- 0


#### Insulin
diabetes.numeric$InsulinNA[diabetes$Insulin == "N/A"] <- 1
diabetes.numeric$InsulinNA[diabetes$Insulin != "N/A"] <- 0

diabetes.numeric$InsulinQ1[diabetes$Insulin == "FirstQuartile"] <- 1
diabetes.numeric$InsulinQ1[diabetes$Insulin != "FirstQuartile"] <- 0

diabetes.numeric$InsulinQ2[diabetes$Insulin == "SecondQuartile"] <- 1
diabetes.numeric$InsulinQ2[diabetes$Insulin != "SecondQuartile"] <- 0

diabetes.numeric$InsulinQ3[diabetes$Insulin == "ThirdQuartile"] <- 1
diabetes.numeric$InsulinQ3[diabetes$Insulin != "ThirdQuartile"] <- 0

diabetes.numeric$InsulinQ4[diabetes$Insulin == "FourthQuartile"] <- 1
diabetes.numeric$InsulinQ4[diabetes$Insulin != "FourthQuartile"] <- 0


#### BMI
diabetes.numeric$BMINA[diabetes$BMI == "N/A"] <- 1
diabetes.numeric$BMINA[diabetes$BMI != "N/A"] <- 0

diabetes.numeric$BMIQ1[diabetes$BMI == "FirstQuartile"] <- 1
diabetes.numeric$BMIQ1[diabetes$BMI != "FirstQuartile"] <- 0

diabetes.numeric$BMIQ2[diabetes$BMI == "SecondQuartile"] <- 1
diabetes.numeric$BMIQ2[diabetes$BMI != "SecondQuartile"] <- 0

diabetes.numeric$BMIQ3[diabetes$BMI == "ThirdQuartile"] <- 1
diabetes.numeric$BMIQ3[diabetes$BMI != "ThirdQuartile"] <- 0

diabetes.numeric$BMIQ4[diabetes$BMI == "FourthQuartile"] <- 1
diabetes.numeric$BMIQ4[diabetes$BMI != "FourthQuartile"] <- 0


# Create normalized features to be used for KNN
normFeature <- function(data) {
  (data - min(data)) / (max(data) - min(data))
}

diabetes.norm <- diabetes.numeric
diabetes.norm$Pregnancies <- normFeature(diabetes.norm$Pregnancies)
diabetes.norm$Age <- normFeature(diabetes.norm$Age)
diabetes.norm$DiabetesPedigreeFunction <- normFeature(diabetes.norm$DiabetesPedigreeFunction)


#####################################################################################
#####################################################################################

library(neuralnet)
library(e1071)
library(caret)

# for KNN
library(class)
library(dplyr)



# Split diabetes data into train and test sets (60% train)
set.seed(0)
trainSet <- createDataPartition(diabetes$Outcome, p=.6)[[1]]
diabetes.train <- diabetes[trainSet,]
diabetes.test <- diabetes[-trainSet,]

diabetes.numeric.train <- diabetes.numeric[trainSet,]
diabetes.numeric.test <- diabetes.numeric[-trainSet,]

diabetes.norm.train <- diabetes.norm[trainSet,]
diabetes.norm.test <- diabetes.norm[-trainSet,]


# Logistic regression model (two-class target)
print("------------------Logistic Regression------------------")

diabetes.model.logistic <- glm(Outcome ~ ., data=diabetes.train,
  family=binomial(link="logit"))

# Plot resulting model
png("LogisticRegression.png", width=900, height=500)
slope <- coef(diabetes.model.logistic)[2]/(-coef(diabetes.model.logistic)[3])
intercept <- coef(diabetes.model.logistic)[1]/(-coef(diabetes.model.logistic)[3])
plot(diabetes.train[,1:8], pch=as.numeric(diabetes.train$Outcome))
abline(intercept, slope)
dev.off()


# Evaluate logistic model
print("------------------Logistic regression evaluation------------------")

diabetes.pred.logistic <- predict(diabetes.model.logistic,
                                   diabetes.test, type="response")
diabetes.eval.logistic.conMat <-
                 confusionMatrix(as.factor(round(diabetes.pred.logistic + 1)),
                                 as.factor(as.numeric(diabetes.test$Outcome)))
print(diabetes.eval.logistic.conMat$table)



# SVM (discrete)
print("------------------SVM------------------")

diabetes.model.svm <- svm(Outcome~., data=diabetes.train)

# Plot the discrete model - commented out b/c data has >2 dimensions.
png("SVM.png", width=900, height=500)
plot(diabetes.model.svm, diabetes.train, Age ~ Pregnancies)
dev.off()


# Evaluate SVM model
print("------------------SVM evaluation------------------")

diabetes.eval.svm <- predict(diabetes.model.svm, diabetes.test)
diabetes.eval.svm.conMat <- confusionMatrix(
                     diabetes.eval.svm, diabetes.test$Outcome)
print(diabetes.eval.svm.conMat$table)



# Neural Network
print("------------------Neural Network------------------")

# Note: There is a bug in the neuralnet function that causes it to fail if the formula with form "target ~ ." is used
# See the workaround for creating an equivalent formula

# Workaround for "Outcome ~ ."
diabetes.nn.formula <- as.formula(paste("Outcome ~ ",
      paste(names(diabetes.numeric.train[!names(diabetes.numeric.train) %in% 'Outcome']),
            collapse = " + "), sep=""))


# Build NN model with default hidden layer (1 hidden layer with 1 node)
diabetes.model.nn1 <- neuralnet(diabetes.nn.formula, data=diabetes.numeric.train)

# Plot the network
plot(diabetes.model.nn1)


# neural network evaluation
print("------------------nn1 evaluation------------------")
diabetes.eval.nn1 <- neuralnet::compute(diabetes.model.nn1, diabetes.numeric.test)
predictions <- diabetes.eval.nn1$net.result
labels <- c("0", "1")
str(data.frame(max.col(predictions)))

diabetes.eval.nn1.prediction_labels <-
  data.frame(max.col(predictions)) %>%
  mutate(predictions=labels[max.col.predictions.]) %>%
  select(2) %>%
  unlist()

diabetes.eval.nn1.conMat <- confusionMatrix(as.factor(diabetes.eval.nn1.prediction_labels), diabetes.numeric.test$Outcome)
print(diabetes.eval.nn1.conMat$table)


# KNN
diabetes.knn.train <- diabetes.norm.train[,-4]
diabetes.knn.test <- diabetes.norm.test[,-4]
diabetes.knn.traincl <- diabetes.norm.train[,4]


# KNN evaluation
print("------------------KNN evaluation------------------")

# Use the knn() function (in the class library)
# Use KNN with 1, 3 and 10 neighbors
# Note there is no "model" other than the training instances
# In this case we simply need the predictions
diabetes.pred.knn.1 <- knn(diabetes.knn.train, diabetes.knn.test, diabetes.knn.traincl)
diabetes.pred.knn.3 <- knn(diabetes.knn.train, diabetes.knn.test, diabetes.knn.traincl, k=3)
diabetes.pred.knn.10 <- knn(diabetes.knn.train, diabetes.knn.test, diabetes.knn.traincl, k=10)

# Calculation of performance for KNN classifiers
diabetes.eval.knn.1 <- confusionMatrix(diabetes.pred.knn.1, diabetes.numeric.test$Outcome)
diabetes.eval.knn.3 <- confusionMatrix(diabetes.pred.knn.3, diabetes.numeric.test$Outcome)
diabetes.eval.knn.10 <- confusionMatrix(diabetes.pred.knn.10, diabetes.numeric.test$Outcome)

# Display the evaluation results for the KNN classifiers
print(diabetes.eval.knn.1$table)
print(diabetes.eval.knn.3$table)
print(diabetes.eval.knn.10$table)
