# source("Preprocessing.R")

# SETUP

library(FSelector)
library(rpart)

diabetes <- read.csv("diabetes.csv")


# FIXES

diabetes$Outcome <- as.factor(diabetes$Outcome)


# DERIVATIONS

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


quartile_bin <- function(table) {
table1 <- table
FirstQuartile <- quantile(table[table != 0], 0.25)
SecondQuartile <- quantile(table[table != 0], 0.50)
ThirdQuartile <- quantile(table[table != 0], 0.75)

table1[table > ThirdQuartile] <- "FourthQuartile"
table1[table <= ThirdQuartile] <- "ThirdQuartile"
table1[table <= SecondQuartile] <- "SecondQuartile"
table1[table <= FirstQuartile] <- "FirstQuartile"

table1 <- as.factor(table1)
return(table1)
}


diabetes$BinGlucose <- zero_quartile_bin(diabetes$Glucose)
diabetes$BinBloodPressure <- zero_quartile_bin(diabetes$BloodPressure)
diabetes$BinSkinThickness <- zero_quartile_bin(diabetes$SkinThickness)
diabetes$BinInsulin <- zero_quartile_bin(diabetes$Insulin)
diabetes$BinBMI <- zero_quartile_bin(diabetes$BMI)

diabetes$BinPregnancies <- quartile_bin(diabetes$Pregnancies)
diabetes$BinAge <- quartile_bin(diabetes$Age)
diabetes$BinDiabetesPedigreeFunction <- quartile_bin(diabetes$DiabetesPedigreeFunction)

#barplot(table(diabetes$BinBMI), main="Bins", xlab="Category", ylab="Frequency")
#print(sum(diabetes$BinBMI == "FirstQuartile"))
#print(sum(diabetes$BinBMI == "SecondQuartile"))
#print(sum(diabetes$BinBMI == "ThirdQuartile"))
#print(sum(diabetes$BinBMI == "FourthQuartile"))


# FEATURE SELECTION
diabetes.simple <- diabetes[, c(9:17)]
diabetes.original <- diabetes[, c(9:17)]
diabetes.fixed <- diabetes[, c(1, 7:14)]

# Random Forest
print("Random forest for modified set:")
diabetes.rf.scores <- random.forest.importance(Outcome ~ ., diabetes.fixed)
cutoff.k(diabetes.rf.scores, k = 4)
print(diabetes.rf.scores)
print(cutoff.k(diabetes.rf.scores, k = 4))

print("Random forest for simplified set:")
diabetes.rf.scores <- random.forest.importance(Outcome ~ ., diabetes.simple)
cutoff.k(diabetes.rf.scores, k = 4)
print(diabetes.rf.scores)
print(cutoff.k(diabetes.rf.scores, k = 4))


# Correlation & Entropy
print("Correlation & entropy for full, redundant set:")
result <- cfs(Outcome ~ ., diabetes)
ce <- as.simple.formula(result, "Outcome")
print(ce)

print("Correlation & entropy for modified set:")
result <- cfs(Outcome ~ ., diabetes.fixed)
ce <- as.simple.formula(result, "Outcome")
print(ce)

print("Correlation & entropy for simple set:")
result <- cfs(Outcome ~ ., diabetes.simple)
ce <- as.simple.formula(result, "Outcome")
print(ce)


# Outcome is fourth attribute in fixed
# Outcome is 1st attribute in simple

# I'm commenting out the part where the evaluators print things because exhaustive search generates enough lines to bury the rest of my output.


evaluator.diabetes.fixed <- function(subset) {
  # Use k-fold cross validation
  k <- 5
  splits <- runif(nrow(diabetes.fixed))
  results = sapply(1:k, function(i) {
#    print(paste("iteration", i))
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
#    print("test.idx created")
    train.idx <- !test.idx
#    print("train.idx created")
    test <- diabetes.fixed[test.idx, , drop=FALSE]
#    print("test created")
    train <- diabetes.fixed[train.idx, , drop=FALSE]
#    print(paste("train created. Build tree on ", subset, " as-formula", as.simple.formula(subset, "Outcome")))
    tree <- rpart(as.simple.formula(subset, "Outcome"), train)
#    print("tree created")
    error.rate = sum(test$Outcome != predict(tree, test, type="c")) / nrow(test)
#    print(paste("error rate:", error.rate))
    return(1 - error.rate)
  })
  ##print(subset)
  ##print(mean(results))
  return(mean(results))
}


writeLines("\nRun forward greedy search - diabetes.fixed\n")
subset <- forward.search(names(diabetes.fixed)[-4], evaluator.diabetes.fixed)
f <- as.simple.formula(subset, "Outcome")

writeLines("\nRun backward greedy search - diabetes.fixed\n")
subset <- backward.search(names(diabetes.fixed)[-4], evaluator.diabetes.fixed)
b <- as.simple.formula(subset, "Outcome")

writeLines("\nRun hill climb search - diabetes.fixed\n")
subset <- hill.climbing.search(names(diabetes.fixed)[-4], evaluator.diabetes.fixed)
h <- as.simple.formula(subset, "Outcome")

writeLines("\nRun exhaustive search - diabetes.fixed\n")
subset <- exhaustive.search(names(diabetes.fixed)[-4], evaluator.diabetes.fixed)
x <- as.simple.formula(subset, "Outcome")

print(f)
print(b)
print(h)
print(x)




evaluator.diabetes.simple <- function(subset) {
  # Use k-fold cross validation
  k <- 5
  splits <- runif(nrow(diabetes.simple))
  results = sapply(1:k, function(i) {
#    print(paste("iteration", i))
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
#    print("test.idx created")
    train.idx <- !test.idx
#    print("train.idx created")
    test <- diabetes.simple[test.idx, , drop=FALSE]
#    print("test created")
    train <- diabetes.simple[train.idx, , drop=FALSE]
#    print(paste("train created. Build tree on ", subset, " as-formula", as.simple.formula(subset, "Outcome")))
    tree <- rpart(as.simple.formula(subset, "Outcome"), train)
#    print("tree created")
    error.rate = sum(test$Outcome != predict(tree, test, type="c")) / nrow(test)
#    print(paste("error rate:", error.rate))
    return(1 - error.rate)
  })
  ##print(subset)
  ##print(mean(results))
  return(mean(results))
}


writeLines("\nRun forward greedy search - diabetes.simple\n")
subset <- forward.search(names(diabetes.simple)[-1], evaluator.diabetes.simple)
f1 <- as.simple.formula(subset, "Outcome")

writeLines("\nRun backward greedy search - diabetes.simple\n")
subset <- backward.search(names(diabetes.simple)[-1], evaluator.diabetes.simple)
b1 <- as.simple.formula(subset, "Outcome")

writeLines("\nRun hill climb search - diabetes.simple\n")
subset <- hill.climbing.search(names(diabetes.simple)[-1], evaluator.diabetes.simple)
h1 <- as.simple.formula(subset, "Outcome")

writeLines("\nRun exhaustive search - diabetes.simple\n")
subset <- exhaustive.search(names(diabetes.simple)[-1], evaluator.diabetes.simple)
x1 <- as.simple.formula(subset, "Outcome")

print(f1)
print(b1)
print(h1)
print(x1)
