if (!require("dplyr")) install.packages("dplyr")
library("dplyr")
if (!require(ggplot2)) install.packages('ggplot2')
library(ggplot2)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(corrplot)) install.packages('corrplot')
library(corrplot)
if (!require(gridExtra)) install.packages('gridExtra')
library(gridExtra)
if (!require(knitr)) install.packages('knitr')
library(knitr)
if (!require(magrittr)) install.packages('magrittr')
library(magrittr)

################################################################################
#  Happiness prediction                                                        #
#  --------------------------------------------------------------------------  #
#  Mgr. Lucie Schaynová, Ph.D., 2021, schaynova.lucie@seznam.cz                #
################################################################################
#  --------------------------------------------------------------------------  #
#  data source: https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021?select=world-happiness-report-2021.csv  #
################################################################################

# Load data
data <- read.csv("C:\\personal\\Data Science\\EdX certificate\\second final project\\final_project_new\\world-happiness-report-2021.csv")
#data <- read.csv(url("https://github.com/lsustkova/Happiness-project/blob/main/world-happiness-report-2021.csv"))
# Exclude columns which are not described in documentation
data <- data %>% select(-Standard.error.of.ladder.score, -upperwhisker, -lowerwhisker, -Dystopia...residual)
# Exclude columns which recalculate data from other columns
data <- data %>% select(-Explained.by..Log.GDP.per.capita, -Explained.by..Social.support, -Explained.by..Healthy.life.expectancy, -Explained.by..Freedom.to.make.life.choices, -Explained.by..Generosity, -Explained.by..Perceptions.of.corruption, -Ladder.score.in.Dystopia)
# Rename the column ï..Country.name to Country.name
names(data)[1] <- "Country.name" 

########################### 1) Introduction

str(data)

########################### 2) Data analysis

# Replace all NA
data[is.na(data)] <- 0

# Divide to train and test data set
set.seed(3)
indexes <- createDataPartition(data$Ladder.score, times = 1, 0.2, list = FALSE)
train_data <- data %>% slice(-indexes)
test_data <- data %>% slice(indexes)

train_data %>%  summary()

# average `Ladder.score` per `Regional.indicator`
countries <- train_data %>% select(Regional.indicator,Ladder.score) %>% group_by(Regional.indicator) %>% summarize(Average.score = mean(Ladder.score)) %>% arrange(is.na(Average.score), Average.score)
countries

# the highest happiness and the lowest happiness
low_states <- train_data %>% select(Country.name,Ladder.score) 
tail(low_states,3)

# plot of correlations
correlation_matrix <- train_data %>% 
  select(-Country.name, -Regional.indicator) %>%cor()
corrplot(correlation_matrix, type = "upper", tl.col = "black", method = "circle", cl.ratio = 0.3, tl.cex = 0.6, tl.srt = 70)

# plots of all features
GDP_plot <- train_data %>% ggplot(aes(Logged.GDP.per.capita,Ladder.score))+
  geom_point(color = "purple1") + 
  geom_smooth(color = "purple3") +
  xlab("Logged GDP per capita") + 
  ylab("Ladder score") + 
  ggtitle("Ladder score by GDP")

Social_plot <- train_data %>% ggplot(aes(Social.support,Ladder.score))+
  geom_point(color = "red1") + 
  geom_smooth(color = "red3") +
  xlab("Social support") + 
  ylab("Ladder score") + 
  ggtitle("Ladder score by Social support")

Healthy_plot <- train_data %>% ggplot(aes(Healthy.life.expectancy,Ladder.score))+
  geom_point(color = "orchid1") + 
  geom_smooth(color = "orchid3") +
  xlab("Healthy life expectancy") + 
  ylab("Ladder score") + 
  ggtitle("Ladder score by Life expectancy")

Freedom_plot <- train_data %>% ggplot(aes(Freedom.to.make.life.choices,Ladder.score))+
  geom_point(color = "green1") + 
  geom_smooth(color = "green3") +
  xlab("Freedom to make life choices") + 
  ylab("Ladder score") + 
  ggtitle("Ladder score by Freedom")

Generosity_plot <- train_data %>% ggplot(aes(Generosity,Ladder.score))+
  geom_point(color = "blue1") + 
  geom_smooth(color = "blue3") +
  xlab("Generosity") + 
  ylab("Ladder score") + 
  ggtitle("Ladder score by Generosity")

Corruption_plot <- train_data %>% ggplot(aes(Perceptions.of.corruption,Ladder.score))+
  geom_point(color = "orange1") + 
  geom_smooth(color = "orange3") +
  xlab("Perceptions of corruption") + 
  ylab("Ladder score") + 
  ggtitle("Ladder score by Corruption")

grid.arrange(GDP_plot, Social_plot, Healthy_plot, Freedom_plot, Generosity_plot, Corruption_plot)

########################### 3) Machine learning methods

############## 3.1.) Multivariate Regression (MVR)

# we do not need countries columns for training
train_data <- train_data %>% select(-Country.name, -Regional.indicator)

# cut number of rows into 5 intervals
folds <- cut(seq(1, nrow(train_data)), breaks = 5, labels = FALSE)

# RMSE in each iteration 
RMSEs_11 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_11 <- train_data[-indexes, ]
  new_test_data_11 <- train_data[indexes, ]
  linear_model_11 <- lm(Ladder.score ~ Logged.GDP.per.capita, data = new_train_data_11)
  prediction_11 <- predict(linear_model_11, new_test_data_11)
  current_RMSE <- sqrt(mean((new_test_data_11$Ladder.score - prediction_11)^2) )
  # we need to identify and show the best model from the 5 runs
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_11 <- linear_model_11
  }
  RMSEs_11 <- c(RMSEs_11, current_RMSE)
}
message('average RMSE:')
mean(RMSEs_11)
message("Coefficients of the best model:")
summary(best_model_11)$coefficients
message("Multiple R-squared:") 
summary(best_model_11)$r.squared

# add `Social.support` feature:
RMSEs_12 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_12 <- train_data[-indexes, ]
  new_test_data_12 <- train_data[indexes, ]
  linear_model_12 <- lm(Ladder.score ~ Logged.GDP.per.capita + Social.support, data = new_train_data_12)
  prediction_12 <- predict(linear_model_12, new_test_data_12)
  current_RMSE <- sqrt(mean((new_test_data_12$Ladder.score - prediction_12)^2) )
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_12 <- linear_model_12
  }
  RMSEs_12 <- c(RMSEs_12, current_RMSE)
}
message('average RMSE:')
mean(RMSEs_12)
message("Coefficients of the best model:")
summary(best_model_12)$coefficients
message("Multiple R-squared:")
summary(best_model_12)$r.squared

# add `Healthy.life.expectancy` feature:
RMSEs_13 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_13 <- train_data[-indexes, ]
  new_test_data_13 <- train_data[indexes, ]
  linear_model_13 <- lm(Ladder.score ~ Logged.GDP.per.capita + Social.support +  Healthy.life.expectancy, data = new_train_data_13)
  prediction_13 <- predict(linear_model_13, new_test_data_13)
  current_RMSE <- sqrt(mean((new_test_data_13$Ladder.score - prediction_13)^2) )
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_13 <- linear_model_13
  }
  RMSEs_13 <- c(RMSEs_13, current_RMSE)
}
message("average RMSE:")
mean(RMSEs_13)
message("Coefficients of the best model:")
summary(best_model_13)$coefficients
message("Multiple R-squared:")
summary(best_model_13)$r.squared

# add `Freedom.to.make.life.choices` feature:
RMSEs_14 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_14 <- train_data[-indexes, ]
  new_test_data_14 <- train_data[indexes, ]
  linear_model_14 <- lm(Ladder.score ~ Logged.GDP.per.capita + Social.support + Healthy.life.expectancy + Freedom.to.make.life.choices, data = new_train_data_14)
  prediction_14 <- predict(linear_model_14, new_test_data_14)
  current_RMSE <- sqrt(mean((new_test_data_14$Ladder.score - prediction_14)^2) )
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_14 <- linear_model_14
  }
  RMSEs_14 <- c(RMSEs_14, current_RMSE)
}
message("average RMSE:")
mean(RMSEs_14)
message("Coefficients of the best model:")
summary(best_model_14)$coefficients
message("Multiple R-squared:")
summary(best_model_14)$r.squared

# add `Generosity` feature:
RMSEs_15 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_15 <- train_data[-indexes, ]
  new_test_data_15 <- train_data[indexes, ]
  linear_model_15 <- lm(Ladder.score ~ Logged.GDP.per.capita + Social.support + Healthy.life.expectancy + Freedom.to.make.life.choices + Generosity, data = new_train_data_15)
  prediction_15 <- predict(linear_model_15, new_test_data_15)
  current_RMSE <- sqrt(mean((new_test_data_15$Ladder.score - prediction_15)^2) )
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_15 <- linear_model_15
  }
  RMSEs_15 <- c(RMSEs_15, current_RMSE)
}
message("average RMSE:")
mean(RMSEs_15)
message("Coefficients of the best model:")
summary(best_model_15)$coefficients
message("Multiple R-squared:")
summary(best_model_15)$r.squared

# add `Perceptions.of.corruption` feature:
RMSEs_16 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_16 <- train_data[-indexes, ]
  new_test_data_16 <- train_data[indexes, ]
  linear_model_16 <- lm(Ladder.score ~ Logged.GDP.per.capita + Social.support + Healthy.life.expectancy + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption, data = new_train_data_16)
  prediction_16 <- predict(linear_model_16, new_test_data_16)
  current_RMSE <- sqrt(mean((new_test_data_16$Ladder.score - prediction_16)^2) )
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_16 <- linear_model_16
  }
  RMSEs_16 <- c(RMSEs_16, current_RMSE)
}
message('average RMSE:')
mean(RMSEs_16)
message("Coefficients of the best model:")
summary(best_model_16)$coefficients
message("Multiple R-squared:")
summary(best_model_16)$r.squared

# final features
RMSEs_17 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_17 <- train_data[-indexes, ]
  new_test_data_17 <- train_data[indexes, ]
  linear_model_17 <- lm(Ladder.score ~ Logged.GDP.per.capita + Social.support  + Freedom.to.make.life.choices, data = new_train_data_17)
  prediction_17 <- predict(linear_model_17, new_test_data_17)
  current_RMSE <- sqrt(mean((new_test_data_17$Ladder.score - prediction_17)^2) )
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_17 <- linear_model_17
  }
  RMSEs_17 <- c(RMSEs_17, current_RMSE)
}
message('average RMSE:')
mean(RMSEs_17)
message("Coefficients of the best model:")
summary(best_model_17)$coefficients
message("Multiple R-squared:")
summary(best_model_17)$r.squared

# use original testing data to test our best linear model
new_column_1 <- predict(linear_model_17, test_data) 
results <- data.frame(new_column_1)
message("RMSE:")
sqrt(mean((test_data$Ladder.score - new_column_1)^2))

plot(test_data$Ladder.score, type="l")
lines(results$new_column_1, col="red")

############## 3.2.) k-Nearest Neighbors (KNN)
RMSEs_21 <- c()
minimal_RMSE <- 1000000
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_21 <- train_data[-indexes, ]
  new_test_data_21 <- train_data[indexes, ]
  knn <- train(Ladder.score ~ Logged.GDP.per.capita + Social.support + 
                 Freedom.to.make.life.choices ,
               data = new_train_data_21, method = "knn")
  prediction_knn <- predict(knn, new_test_data_21) 
  current_RMSE <- sqrt(mean((new_test_data_21$Ladder.score - prediction_knn)^2)) 
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_21 <- knn
  }
  RMSEs_21 <- c(RMSEs_21, current_RMSE)
}
message("average RMSE:")
mean(RMSEs_21)

# use original testing data
new_column_3 <- predict(best_model_21, test_data)
results <- data.frame(new_column_3)
message("RMSE:")
sqrt(mean((test_data$Ladder.score - new_column_3)^2)) 

#plot(knn)
plot(test_data$Ladder.score, type="l") 
lines(results$new_column_3, col="red")

############## 3.3.) Neural Networks (NN)
RMSEs_31 <- c()
minimal_RMSE <- 1000000
set.seed(123)
for (i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_31 <- train_data[-indexes, ]
  new_test_data_31 <- train_data[indexes, ]
  neural_networks <- train(Ladder.score ~ Logged.GDP.per.capita + Social.support + 
                             Freedom.to.make.life.choices ,
                           data = new_train_data_31, method = "nnet", maxit = 1000, 
                           trace = FALSE, linout = 1)
  prediction_31 <- predict(neural_networks, new_test_data_31)
  current_RMSE <- sqrt(mean((new_test_data_31$Ladder.score - prediction_31)^2))
  if (current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_31 <- neural_networks
  }
  RMSEs_31 <- c(RMSEs_31, current_RMSE)
}
message("average RMSE:")
mean(RMSEs_31)
summary(best_model_31)

# use testing data to test our best linear model
new_column_2 <- predict(best_model_31, test_data)
results <- data.frame(new_column_2)
message("RMSE:")
sqrt(mean((test_data$Ladder.score - new_column_2)^2))

#plot(neural_networks)
plot(test_data$Ladder.score, type = "l")
lines(results$new_column_2, col = "red")

############## 3.4.) Generalized Linear Model (GLM)
RMSEs_41 <- c()
minimal_RMSE <- 1000000
for(i in 1:5){
  indexes <- which(folds == i, arr.ind = TRUE)
  new_train_data_41 <- train_data[indexes, ]
  new_test_data_41 <- train_data[-indexes, ]
  glm <- train(Ladder.score ~ Logged.GDP.per.capita + Social.support +                
                 Freedom.to.make.life.choices, data = new_train_data_41, method = "glm") 
  prediction_41 <- predict(glm, new_test_data_41)
  current_RMSE <- sqrt(mean((new_test_data_41$Ladder.score - prediction_41)^2))
  if(current_RMSE < minimal_RMSE){
    minimal_RMSE <- current_RMSE
    best_model_41 <- glm
  }
  RMSEs_41 <- c(RMSEs_41, current_RMSE)
}
message("average RMSE:")
mean(RMSEs_41)
#summary(best_model_41)

# use testing data to test our best linear model
new_column_4 <- predict(best_model_41, test_data)
results <- data.frame(new_column_4)
message("RMSE:")
sqrt(mean((test_data$Ladder.score - new_column_4)^2))

plot(test_data$Ladder.score, type = "l")
lines(results$new_column_4, col = "red")

