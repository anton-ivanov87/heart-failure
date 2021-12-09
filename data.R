#################################################
# Predicting mortality caused by Heart Failure
################################################

library(tidyverse)
library(caret)
library(psych)
library(reshape2)
library(gridExtra)

# Dataset available from:
# Heart failure clinical records. (2020). UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/machine-learning-databases/00519/

############################
# Preparation of the dataset
############################

data <- read.csv("heart_failure_clinical_records_dataset.csv")

# Time column will not be considered for the prediction since this information
# will not be available for real patients
data <- select(data, -time)
data_num <- data

# Binary variables are converted to factors, numeric variables are scaled
factors <- c(2, 4, 6, 10, 11, 12)
data[factors] <- lapply(data[factors], factor)
data_sc <- data
data_sc[-factors] <- sapply(data[-factors], scale)

# Output variable levels are converted to "No"/"Yes" as needed for some methods
levels(data_sc$DEATH_EVENT) = c("No","Yes")

# Train and test datasets are created
set.seed(2)
train_index <- createDataPartition(data$DEATH_EVENT, p = 0.8, list = FALSE)
train_not_sc <- train <- data[train_index,]
train <- data_sc[train_index,]
test <- data_sc[-train_index,]

# The proportion of negative and positive outocmes in the train and test sets
# is almost identical
prop.table(table(train$DEATH_EVENT))
prop.table(table(test$DEATH_EVENT))
prop.table(table(train$sex))

###################################
# Data exploration of the train set
###################################

str(train_not_sc)

# Distribution and correlations of all features
pairs.panels(train)

train_num <- data_num[train_index,]
cormat <- round(cor(train_num),2)
melted_cormat <- melt(cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  labs(x = NULL, y = NULL, fill = "Pearson's\nCorrelation") +
  scale_fill_gradient2(mid="#FBFEF9",low="#0C6291",high="#A63446", limits=c(-0.5,0.5)) +
  scale_x_discrete(guide = guide_axis(angle = 30))

# Age
train_not_sc %>%
  ggplot(aes(age, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_histogram(aes(y = ..density../3), bins = 20) +
  geom_density(alpha = 0.2)
  

train_not_sc %>%
  ggplot(aes(x = DEATH_EVENT, y = age)) +
  theme_gray() +
  geom_boxplot()

# Anaemia

train_not_sc %>%
  ggplot(aes(anaemia, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_bar(position = "fill")

# Creatinine phosphokinase

train_not_sc %>%
  ggplot(aes(creatinine_phosphokinase, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_histogram(aes(y = ..density../2)) +
  geom_density(alpha = 0.2) +
  scale_x_log10()

# Diabetes

train_not_sc %>%
  ggplot(aes(diabetes, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_bar(position = "fill")

# Ejection fraction

train_not_sc %>%
  ggplot(aes(ejection_fraction, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_histogram(aes(y = ..density../2), bins = 10) +
  geom_density(alpha = 0.2)

# High blood pressure

train_not_sc %>%
  ggplot(aes(high_blood_pressure, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_bar(position = "fill")

# Platelets

train_not_sc %>%
  ggplot(aes(platelets, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_histogram(aes(y = ..density../2), bins = 10) +
  geom_density(alpha = 0.2)

# Serum creatinine

train_not_sc %>%
  ggplot(aes(serum_creatinine, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_histogram(aes(y = ..density../2)) +
  geom_density(alpha = 0.2) +
  scale_x_log10()


# Serum sodium

train_not_sc %>%
  ggplot(aes(serum_sodium, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_histogram(aes(y = ..density../2)) +
  geom_density(alpha = 0.2)

# Sex

p1 <- train_not_sc %>%
  ggplot(aes(sex, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_bar(position = "fill")

p3 <- train_not_sc %>%
  ggplot(aes(sex, fill = smoking)) +
  theme_gray() +
  geom_bar(position = "fill")

# Smoking

p2 <- train_not_sc %>%
  ggplot(aes(smoking, fill = DEATH_EVENT)) +
  theme_gray() +
  geom_bar(position = "fill")

grid.arrange(p1, p2, nrow = 1)

#################
# Training models
#################

# The methods that will be used

methods <- c("nb", "glm", "knn", "svmLinear", "svmRadial", "svmPoly", "rpart", "treebag", "adaboost", "rf", "nnet")

# Accuracy by 'all negative' prediction

pred_neg <- c(1, rep(0, nrow(train) - 1))
pred_neg <- factor(pred_neg)
levels(pred_neg) = c("No","Yes")
cm_neg <- confusionMatrix(pred_neg, train$DEATH_EVENT, mode = "everything", positive = "Yes")
cm_neg$overall["Accuracy"]
cm_neg$byClass["F1"]

# Due to class imbalance F1-score will be used to measure model performance

f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred,
                                y_true = data$obs,
                                # level 2 ("Yes") will be used as positive level
                                positive = lev[2])
  c(F1 = f1_val)
}

# 5-fold cross-validation will be used to evaluate the models

fitControl <- trainControl(method = "cv",
                           number = 10,
                           p = 0.9,
                           classProbs = TRUE,
                           summaryFunction = f1)

# The function that will be used to run all the considered models

calc_model <- function(method, tuneGrid) {
  set.seed(3)
  fit <- train(DEATH_EVENT ~., # all features will be considered first
               data = train,
               method = method,
               metric = "F1",
               trControl = fitControl,
               tuneGrid = tuneGrid)
  # F1-score for the train set will be used to evaluate models prior to evaluation 
  # based on the test set
  F1_train <- max(fit$results$F1, na.rm = TRUE)
  # For the prediction of the test set metrics F1, Accuracy, Specificity and Kappa
  # will be shown and considered for evaluation
  pred <- predict(fit, train)
  cm <- confusionMatrix(pred, train$DEATH_EVENT, mode = "everything", positive = "Yes")
  accuracy <- cm$overall["Accuracy"]
  F1 <- cm$byClass["F1"]
  specificity <- cm$byClass["Specificity"]
  kappa <- cm$overall["Kappa"]
  results <- list()
  # The functions returns a list of trained models, confusion matrices and a dataframe with metrics
  results[[1]] <- fit
  results[[2]] <- cm
  results[[3]] <- data.frame(method = method, F1_train = F1_train, F1 = F1, accuracy = accuracy, specificity = specificity, kappa = kappa)
  return(results)
}

# Parameters for model tuning are provided as a list

tuneGrids <- list()

tuneGrids[[1]] <- expand.grid(fL = seq(0, 5, 1), 
                              usekernel = c(TRUE, FALSE), 
                              adjust = seq(0, 5, 1))

tuneGrids[2] <- list(NULL)

tuneGrids[[3]] <- data.frame(k = seq(1, 99, 2))

tuneGrids[[4]] <- data.frame(C = seq(0.5, 10, 0.5))

tuneGrids[5] <- list(NULL)

tuneGrids[6] <- list(NULL)

tuneGrids[[7]] <- data.frame(cp = seq(0, 0.05, len = 10))

tuneGrids[8] <- list(NULL)

tuneGrids[[9]] <- data.frame(nIter = seq(1, 19, 2),
                            method = "M1")

tuneGrids[[10]] <- data.frame(mtry = seq(1, 11, 1))

tuneGrids[[11]] <- expand.grid(size = seq(1, 10, 1),
               decay = seq(1, 10, 1))


# Models are trained and saved in a list

all_models <- mapply(calc_model, methods, tuneGrids)

# Dataframe containing all methods and metrics is created
df2 <- bind_rows(all_models[seq(3, 33, 3)], .id = "column_label")
rownames(df2) <- NULL
df2 <- df2[, -1]
df2

# All models are considered separately

# Naive Bayes

plot(all_models[[1]])

all_models[[2]]$table

#("nb", "glm", "knn", "svmLinear", "svmRadial", "svmPoly", "rpart", "treebag", "adaboost", "rf", "nnet")

# Logistic regression

all_models[[5]]$table

# K-nearest neighbors

plot(all_models[[7]])

all_models[[8]]$table

# SVM Linear

plot(all_models[[10]])

all_models[[11]]$table


# SVM Radial

plot(all_models[[13]])

all_models[[14]]$table

# SVM Polynomial

plot(all_models[[16]])

all_models[[17]]$table

# Decision tree

plot(all_models[[19]])

all_models[[20]]$table

# Bagged decision tree

all_models[[23]]$table

# Boosted decision tree

plot(all_models[[25]])

all_models[[26]]$table

# Random forest

plot(all_models[[28]])

all_models[[29]]$table

# ANN

plot(all_models[[31]])

all_models[[32]]$table

# Tuning parameters are redefined for SVM linear, Boosted decision tree and Random forest

tuneGrids[[4]] <- data.frame(C = seq(0.5, 100, 0.5))

tuneGrids[[9]] <- data.frame(nIter = seq(1, 19, 2),
                             method = "M1")

tuneGrids[[10]] <- data.frame(mtry = seq(1, 20, 1))

all_models <- mapply(calc_model, methods, tuneGrids)

# Tuning results are checked again
# SVM Linear

plot(all_models[[10]])

# Random forest

plot(all_models[[28]])

# Checking if reduction of features improves prediction

calc_model_red <- function(method, tuneGrid) {
  set.seed(3)
  fit <- train(DEATH_EVENT ~ age + anaemia + ejection_fraction + serum_creatinine +
                serum_sodium, # only features with correlation >= 0.05 are left
                data = train,
                method = method,
                metric = "F1",
                trControl = fitControl,
                tuneGrid = tuneGrid)
  # F1-score for the train set will be used to evaluate models prior to evaluation 
  # based on the test set
  F1_train <- max(fit$results$F1, na.rm = TRUE)
  # For the prediction of the test set metrics F1, Accuracy, Specificity and Kappa
  # will be shown and considered for evaluation
  pred <- predict(fit, train)
  cm <- confusionMatrix(pred, train$DEATH_EVENT, mode = "everything", positive = "Yes")
  accuracy <- cm$overall["Accuracy"]
  F1 <- cm$byClass["F1"]
  specificity <- cm$byClass["Specificity"]
  kappa <- cm$overall["Kappa"]
  results <- list()
  # The functions returns a list of trained models, confusion matrices and a dataframe with metrics
  results[[1]] <- fit
  results[[2]] <- cm
  results[[3]] <- data.frame(method = method, F1_train = F1_train, F1 = F1, accuracy = accuracy, specificity = specificity, kappa = kappa)
  return(results)
}

all_models_red <- mapply(calc_model_red, methods, tuneGrids)

df_red <- bind_rows(all_models_red[seq(3, 33, 3)], .id = "column_label")
rownames(df_red) <- NULL

# df_red <- df_red[, c(2, 3)]
df_red

# Performance of the best model
svmRadial_pred <- predict(all_models_red[[13]], test)
svmRadial_cm <- confusionMatrix(svmRadial_pred, test$DEATH_EVENT, mode = "everything", positive = "Yes")
svmRadial_cm

# 10-fold cross-validation will be used and comapred to the 5-fold c-v

fitControl <- trainControl(method = "cv",
                           number = 10,
                           p = 0.9,
                           classProbs = TRUE,
                           summaryFunction = f1)

all_models_red <- mapply(calc_model_red, methods, tuneGrids)
df_red <- bind_rows(all_models_red[seq(3, 33, 3)], .id = "column_label")
rownames(df_red) <- NULL

# df_red <- df_red[, c(2, 3)]
df_red

#############################################################################
# Ensemble learning: Ensemble of 11 models will be used to improve prediction
#############################################################################

# Predictions for all models are combined in a dataframe with pred_all function
pred_all <- function(method, tuneGrid) {
  set.seed(3)
  fit <- train(DEATH_EVENT ~ age + anaemia + ejection_fraction + serum_creatinine +
                 serum_sodium,
               data = train,
               method = method,
               metric = "F1",
               trControl = fitControl,
               tuneGrid = tuneGrid)
  pred <- predict(fit, train)
  df <- data.frame(method = pred)
  return(df)
}

all <- mapply(pred_all, methods, tuneGrids)
all <- as.data.frame(all)

# Output values are transformed to 1s and 0s
all <- ifelse(all == "Yes", 1, 0)

# Summary prediction is calculated
all_sums <- rowSums(all)
all_sums

# The function ensemble takes a vote threshold needed for a positive or negative vote
ensemble <- function(vote) {
  all_voted <- ifelse(all_sums > vote, 1, 0)
  all_voted <- as.factor(all_voted)
  levels(all_voted) = c("No", "Yes")
  cm_voted <- confusionMatrix(all_voted, train$DEATH_EVENT)
  return(cm_voted$byClass["F1"])
}

# Vote thresholds 1-11 are tried and corresponding F1-scores are calculated
vote <- seq(1, 11, 1)
F1_ens <- sapply(vote, ensemble)

# Relationship between F1-scores and vote thresholds
df_ens <- data.frame(vote = vote, F1_ens = F1_ens)
df_ens %>%
  ggplot(aes(vote, F1_ens)) +
  geom_line()

# Vote threshold 3 gives the best prediction, higher than that of the single models

all_voted <- ifelse(all_sums > 3, 1, 0)
all_voted <- as.factor(all_voted)
levels(all_voted) = c("No", "Yes")
cm_voted <- confusionMatrix(all_voted, train$DEATH_EVENT)
cm_voted$byClass["F1"]

# Evaluation of the ensemble with the test set

pred_all_test <- function(method, tuneGrid) {
  set.seed(3)
  fit <- train(DEATH_EVENT ~ age + anaemia + ejection_fraction + serum_creatinine +
                 serum_sodium,
               data = train,
               method = method,
               metric = "F1",
               trControl = fitControl,
               tuneGrid = tuneGrid)
  pred <- predict(fit, test)
  df <- data.frame(method = pred)
  return(df)
}

all_test <- mapply(pred_all_test, methods, tuneGrids)
all_test <- as.data.frame(all_test)
all_test <- ifelse(all_test == "Yes", 1, 0)
all_sums_test <- rowSums(all_test)
all_sums_test

all_voted_test <- ifelse(all_sums_test > 3, 1, 0)
all_voted_test <- as.factor(all_voted_test)
levels(all_voted_test) = c("No", "Yes")
cm_voted_test <- confusionMatrix(all_voted_test, test$DEATH_EVENT)
cm_voted_test$byClass["F1"]
cm_voted_test

