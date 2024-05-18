R
Copy code
# Load required libraries
library(readr)  # For reading data
library(dplyr)  # For data manipulation
library(caret)  # For data splitting
library(glmnet) # For logistic regression model
library(pROC)   # For ROC curve and AUC calculation

# Load the dataset
url <- "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
temp <- tempfile()
download.file(url, temp)
unzip(temp, exdir = ".")
bank_data <- read.csv("bank-full.csv", sep = ";", header = TRUE)
unlink(temp)

# Data preprocessing
bank_data$y <- ifelse(bank_data$y == "yes", 1, 0) # Convert response variable to binary
bank_data <- bank_data %>%
  select(-c(duration, contact, month, day, pdays, previous, poutcome)) # Remove unnecessary variables

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(bank_data$y, p = 0.7, list = FALSE)
train_data <- bank_data[train_index, ]
test_data <- bank_data[-train_index, ]

# Logistic regression model
model <- glm(y ~ ., data = train_data, family = "binomial")

# Model summary
summary(model)

# Predictions on test data
predictions <- predict(model, newdata = test_data, type = "response")

# Model evaluation
roc_curve <- roc(test_data$y, predictions)
auc <- auc(roc_curve)
print(paste("AUC:", auc))

# Confusion matrix
threshold <- 0.5
conf_matrix <- table(Actual = test_data$y, Predicted = ifelse(predictions > threshold, 1, 0))
print(conf_matrix)

# Accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

# Sensitivity and specificity
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))

# Plot ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue")
abline(a = 0, b = 1, lty = 2, col = "red")
