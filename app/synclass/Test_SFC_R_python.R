library(xgboost)
library(gplots)

mydata = read.table("/home/rios/projects/thrombosis/data/thrombosis_non_thrombosis_v4.csv", sep="\t", header = T)

run_XGBoost <- function(trainSet, testSet, positive, negative){
  
  training <- trainSet[,1:(ncol(trainSet)-1)]
  
  myClass <- as.vector(trainSet$type)
  
  myClass[which(myClass == positive)] <- 0
  myClass[which(myClass == negative)] <- 1

  dTrain <- xgb.DMatrix(as.matrix(training), label=myClass)
  
  testing <- as.data.frame(testSet[, 1:(ncol(testSet)-1)])
  
  dTest <- xgb.DMatrix(as.matrix(testing), label=rep(1, nrow(testing)))
  
  params <- list(booster = "gbtree",
                 objective = "binary:logistic",
                 nthread=4)
  
  myFit <- xgb.train (params = params,
                      data = dTrain,
                      nrounds = 100,
                      maximize = T,
                      eval_metric = "auc")
  
  myPred <- predict(myFit, dTest)
  
  return(myPred)
}

get_statistical_weights = function(inputSet){
  
  myWeights = vector(length = ncol(inputSet)-1)
  
  for(i in 1:(ncol(inputSet)-1)){
    myWeights[i] = wilcox.test(inputSet[,i]~inputSet[,ncol(inputSet)])$statistic
  }
  
  return(1/myWeights)
}

mydata = mydata[,2:ncol(mydata)]

test_idx = c(2,3,416,417)
testSet = mydata[test_idx, ]
trainSet = mydata[-test_idx, ]

result = matrix(nrow=nrow(testSet), ncol = (ncol(trainSet)-1))

for(i in 1:(ncol(trainSet)-1)){
  
  trainSet_att = trainSet[,c(i, ncol(trainSet))]
  testSet_att = testSet[,c(i, ncol(testSet))]
  
  result[, i] = run_XGBoost(trainSet = trainSet_att, testSet = testSet_att,
                            positive = "Thrombosis",
                            negative = "Non_thrombosis")
}

run_XGBoost(trainSet = trainSet[,c(2, ncol(trainSet))], 
            testSet = testSet[,c(2, ncol(testSet))],
            positive = "Thrombosis",
            negative = "Non_thrombosis")

# saving to compare with python
write.csv(result, file="/tmp/output_xgb.csv")


testSet$pred = rowMeans(result)

statistical_weights = get_statistical_weights(trainSet)

testSet$pred = apply(result, 1, function(x) weighted.mean(x, statistical_weights))

