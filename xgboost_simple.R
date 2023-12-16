library(xgboost)

data <- read.csv("filepath//xgboost_data.csv")

set.seed(123)
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

dtrain <- xgb.DMatrix(data = as.matrix(train_data[, -1]), label = train_data$target_variable)
dtest <- xgb.DMatrix(data = as.matrix(test_data[, -1]), label = test_data$target_variable)

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  verbose_eval = FALSE
)

importance_scores <- xgb.importance(model = model)
print(importance_scores)

xgb.plot.importance(importance_matrix = importance_scores)

write.csv(importance_scores, file = "filepath//xgboost_features_extraced.csv")
