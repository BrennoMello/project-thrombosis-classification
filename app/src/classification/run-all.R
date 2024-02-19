#######################################################
# Main Code implemented to run all experiments using
# classification
#######################################################
# This code is part of the Hema-Class framework
# Date: December, 2020
#
# Developers: Tiago Lopes, 
#		Ricardo Rios, 
#		Tatiane Nogueira, 
#		Rodrigo Mello
#
# GNU General Public License v3.0
# Permissions of this strong copyleft license are 
#	conditioned on making available complete 
#	source code of licensed works and 
#	modifications, which include larger works 
#	using a licensed work, under the same license. 
#	Copyright and license notices must be 
#	preserved. Contributors provide an express 
#	grant of patent rights.
#######################################################

###
# chose which dataset you'd like to analyze
###
rm(list=ls())
### default dataset
source("src/preprocessing/load-data-classif.R")

###
# loading machine learning methods
###
source("src/classification/dt.R")
source("src/classification/knn.R")
source("src/classification/dwnn.R")
source("src/classification/rf.R")
source("src/classification/svm.R")
source("src/classification/naive.R")
source("src/classification/xgboost.R")
library(UBL)
#run.methods<-c("dt", "knn", "dwnn", "rf", "svrpol", "svrrad", "naive", "xgboost")
#run.methods<-c("svrrad", "naive", "xgboost") # preparing ensemble
#run.methods<-c("svrrad", "naive") # preparing ensemble
#run.methods<-c("xgboost") # preparing ensemble
run.methods<-c("svrrad", "svrpol", "naive", "rf") # preparing ensemble
#run.methods<-c("dt", "naive", "rf")
#run.methods<-c("dt", "naive")

###
# create matrices of results
###
result.mcc<-matrix(nrow=length(cv.10), ncol=length(run.methods))
result.acc<-matrix(nrow=length(cv.10), ncol=length(run.methods))
result.kappa<-matrix(nrow=length(cv.10), ncol=length(run.methods))
result.auc<-matrix(nrow=length(cv.10), ncol=length(run.methods))
result.F1<-matrix(nrow=length(cv.10), ncol=length(run.methods))

pos.name = "Thrombosis"
neg.name = "Non_thrombosis"
target.name = "type"
data.aug = TRUE
threshold.interval = seq(0.45, 0.7, 0.05)
measure.train = list(tnr)

for(threshold.class in threshold.interval){
  cat("*****\n*****Threshold ", threshold.class, "*****\n")

  for(i in 1:length(cv.10)){
    cat("*****\n*****Iteration ", i, "*****\n")
    final.prob<-c()  
    
    if (data.aug){
      #####with data augmentation#####
      aug.data = GaussNoiseClassif(type ~ ., dat=hemo.data[-unlist(cv.10[[i]]),], C.perc="extreme")  
      train.tsk <- mlr::makeClassifTask(data = rbind(aug.data, hemo.data[-unlist(cv.10[[i]]),]), target = target.name)  
    }else{
      ######without data augmentation#####
      train.tsk <- mlr::makeClassifTask(data = hemo.data[-unlist(cv.10[[i]]),], target = target.name, positive = pos.name)
    }
    #####
    test.tsk <- mlr::makeClassifTask(data = hemo.data[unlist(cv.10[[i]]),], target = target.name, positive = pos.name)

    final.prob<-cbind(final.prob, unlist(cv.10[[i]]))
    
    for(j in 1:length(run.methods)){
      cat("*****Running ", run.methods[j], "\n")
      switch (run.methods[j],
              dt = {
                ###
                # running dt
                ###
                eval<-dt.classif(train.task = train.tsk, test.task = test.tsk, 
                                measure = list(acc), 
                                save.model=paste(sep="", "results/models/model-", run.methods[j], "-", i, ".mod"), 
                                threshold = 0.5)
                
                if(((eval$data$response %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name)
                }  
                final.prob<-cbind(final.prob, eval$data$prob.Thrombosis)              
              },
              knn = {
                eval<-knn.classif(train.task = mlr::makeClassifTask(data = hemo.data, target = "Activity"), 
                                  test.task = unlist(cv.10[[i]]))
                
                if(((eval %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval, positive = pos.name)
                }
              },
              dwnn = {
                eval<-dwnn.classif(train.task = mlr::makeClassifTask(data = hemo.data, target = "Activity"), 
                                  test.task = unlist(cv.10[[i]]))
                
                if(((eval %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval, positive = pos.name)
                }
              },
              rf = {
                eval<-randomForest.classif(train.task = train.tsk, test.task = test.tsk, 
                                          measure = measure.train, 
                                          save.model=paste(sep="", "results/models/model-", run.methods[j], "-", i, ".mod"), 
                                          threshold = threshold.class)
                
                if(((eval$data$response %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name)
                }
                final.prob<-cbind(final.prob, eval$data$prob.Thrombosis)
              },
              svrpol = {
                eval<-svm.classif(train.task = train.tsk, test.task = test.tsk, 
                                  measure = measure.train, 
                                  save.model=paste(sep="", "results/models/model-", run.methods[j], "-", i, ".mod"), 
                                  pol=TRUE, threshold = threshold.class)
                
                if(((eval$data$response %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name)
                }
                final.prob<-cbind(final.prob, eval$data$prob.Thrombosis)
              },
              svrrad = {
                eval<-svm.classif(train.task = train.tsk, test.task = test.tsk, 
                                  measure = measure.train, 
                                  save.model=paste(sep="", "results/models/model-", run.methods[j], "-", i, ".mod"), 
                                  pol=FALSE, threshold = threshold.class)
                
                if(((eval$data$response %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name)
                }
                final.prob<-cbind(final.prob, eval$data$prob.Thrombosis)
              },
              naive = {
                eval<-naiveBayes.classif(train.task = train.tsk, test.task = test.tsk, 
                                        measure = measure.train, 
                                        save.model=paste(sep="", "results/models/model-", run.methods[j], "-", i, ".mod"), 
                                        threshold = threshold.class)
                
                if(((eval$data$response %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name)
                }
                final.prob<-cbind(final.prob, eval$data$prob.Thrombosis)
              },
              xgboost = {
                eval<-xgboost.classif(train.task = train.tsk, test.task = test.tsk, 
                                      measure = measure.train, 
                                      save.model=paste(sep="", "results/models/model-", run.methods[j], "-", i, ".mod"), 
                                      threshold = threshold.class)
                
                if(((eval$data$response %>% table()) > 0) %>% all()){
                  result.acc[i,j] <- measureACC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # mean((train[unlist(cv.10[[i]]),ncol(train)] - eval$data$response)^2)
                  result.kappa[i,j] <- measureKAPPA(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response) # sqrt(mse[i])
                  result.mcc[i,j] <- measureMCC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name) 
                  result.auc[i,j] <- measureAUC(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name, negative = neg.name)
                  result.F1[i,j] <- measureF1(hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)], eval$data$response, positive = pos.name)
                }
                final.prob<-cbind(final.prob, eval$data$prob.Thrombosis)
              }
      )
      
    }    
    final.prob<-cbind(final.prob, 
                      hemo.data[unlist(cv.10[[i]]),ncol(hemo.data)]%>% as.character())
    colnames(final.prob)<-c("id", paste(sep="", "high.prob.", run.methods), "truth")
    write.csv(final.prob, 
              file=paste(sep="", "results/models/predictions-", i, ".csv"),
              row.names = F)
    
  }

  #Saving mean results
  mean.results = rbind(
    apply(result.acc %>% na.omit(), 2, mean) %>% round(digits = 2),
    apply(result.kappa %>% na.omit(), 2, mean) %>% round(digits = 2),
    apply(result.mcc %>% na.omit(), 2, mean) %>% round(digits = 2),
    apply(result.auc %>% na.omit(), 2, mean) %>% round(digits = 2),
    apply(result.F1 %>% na.omit(), 2, mean) %>% round(digits = 2)
  ) %>% t()
  rownames(mean.results) = run.methods
  colnames(mean.results) = c('ACC', 'KAPPA', 'MCC', 'AUC', 'F1')
  mean.results %>% write.csv(file=paste(sep="", "results/models/test-", threshold.class, ".csv"))

  #Saving individual testes
  colnames(result.acc) = run.methods
  result.acc %>% write.csv(file=paste(sep="", "results/models/test-acc-", threshold.class, ".csv"))
  colnames(result.kappa) = run.methods
  result.kappa %>% write.csv(file=paste(sep="", "results/models/test-kappa-", threshold.class, ".csv"))
  colnames(result.mcc) = run.methods
  result.mcc %>% write.csv(file=paste(sep="", "results/models/test-mcc-", threshold.class, ".csv"))
  colnames(result.auc) = run.methods
  result.auc %>% write.csv(file=paste(sep="", "results/models/test-auc-", threshold.class, ".csv"))
  colnames(result.F1) = run.methods
  result.F1 %>% write.csv(file=paste(sep="", "results/models/test-F1-", threshold.class, ".csv"))

  threshold.class = threshold.class+0.05
}
