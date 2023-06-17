# source("DQR.R")


library(corrplot)
library(aplpack)
library(googleVis)
library(ggplot2)
library(plyr)
library(reshape2)
library(waffle)



panel.cor <- function(x, y, digits = 2, cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # correlation coefficient
  r <- cor(x, y)
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste("r= ", txt, sep = "")
  text(0.5, 0.6, txt)

  # p-value calculation
  p <- cor.test(x, y)$p.value
  txt2 <- format(c(p, 0.123456789), digits = digits)[1]
  txt2 <- paste("p= ", txt2, sep = "")
  if(p<0.01) txt2 <- paste("p= ", "<0.01", sep = "")
  text(0.5, 0.4, txt2)
}



diabetes <- read.csv("diabetes.csv")

writeLines("\nstr() output for Diabetes data")
str(diabetes)

writeLines("\nData samples:")
print(head(diabetes))



png("Lab3.ScatterMatrix.png", width=900, height=500)
plot(diabetes, main="Scatter Plot Matrix for Diabetes Data")
dev.off()

png("Lab3.ScatterMatrixwithPearson.png", width=900, height=500)
pairs(diabetes,
  main="Scatter Plot Matrix with Pearson's r for Diabetes Data",
  upper.panel=panel.cor)
dev.off()

corr <- cor(diabetes)
png("Lab3.CorrelationPlot.png", width=900, height=500)
corrplot(corr, method="square")
dev.off()

writeLines("\nNumber of observations:")
print(nrow(diabetes))

#####################################################

print(summary(diabetes))

print(sum(diabetes$Outcome))
print(768 - sum(diabetes$Outcome))


png("AgeFrequencies.png", width=900, height=500)
hist(diabetes$Age, main="Age Frequencies", xlab="Age")
dev.off()

png("DPFFrequencies.png", width=900, height=500)
hist(diabetes$DiabetesPedigreeFunction, main="DPF Frequencies", xlab="Diabetes Pedigree Function")
dev.off()

png("BMIFrequencies.png", width=900, height=500)
hist(diabetes$BMI, main="BMI Frequencies", xlab="BMI")
dev.off()

png("InsulinFrequencies.png", width=900, height=500)
hist(diabetes$Insulin, main="Insulin Frequencies", xlab="Insulin")
dev.off()

png("SkinThicknessFrequencies.png", width=900, height=500)
hist(diabetes$SkinThickness, main="Skin Thickness Frequencies", xlab="Skin Thickness")
dev.off()

png("BloodPressureFrequencies.png", width=900, height=500)
hist(diabetes$BloodPressure, main="Blood Pressure Frequencies", xlab="Blood Pressure")
dev.off()

png("GlucoseFrequencies.png", width=900, height=500)
hist(diabetes$Glucose, main="Glucose Frequencies", xlab="Glucose")
dev.off()

png("PregnancyFrequencies.png", width=900, height=500)
hist(diabetes$Pregnancies, main="Pregnancy Frequencies", xlab="Pregnancies")
dev.off()


png("OutcomeFrequencies.png", width=900, height=500)
barplot(table(diabetes$Outcome), main="Outcome Frequencies", xlab="Category", ylab="Frequency")
dev.off()


png("AgeDotChart.png", width=900, height=500)
dotchart(diabetes$Age, main="Age Dot Chart")
dev.off()

png("BPStripChart.png", width=900, height=500)
stripchart(diabetes$BloodPressure, method="stack",
           xlab="Blood Pressure",
           main="Strip Chart with Y Limit 1-20 for Blood Pressure",
           pch=1, offset=1, cex=1, ylim=c(1,20))
dev.off()

png("AgeVSPregnanciesScatter.png", width=900, height=500)
plot(diabetes$Age, diabetes$Pregnancies, main="Scatter Plot of Age vs. Pregnancies", xlab="Age (Years)", ylab="Pregnancies")
dev.off()
