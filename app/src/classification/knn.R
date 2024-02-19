#######################################################
# Code implemented to model the dataset using KNN 
#
#######################################################
# Date: January, 2021
#
# Developers: Tiago Lopes, 
#               Ricardo Rios, 
#               Tatiane Nogueira, 
#               Rodrigo Mello
#
#
# GNU General Public License v3.0
#
# Permissions of this strong copyleft license are 
#       conditioned on making available complete 
#       source code of licensed works and 
#       modifications, which include larger works 
#       using a licensed work, under the same license. 
#       Copyright and license notices must be 
#       preserved. Contributors provide an express 
#       grant of patent rights.
#######################################################

#' This method runs a grid search to look for the best
#' KNN models
#' 
#' @param train.task - train dataset
#' @param test.task  - test dataset
#' @param measure    - list of measures used to seek the best parametrization
#' @param save.model - file name to save all random forest model and configuration
#' @param max.k  - highest k-nearest neighbors used by the search methods
#' @return predicted values

knn.classif<-function(train.task, test.task, measure = list(acc), save.model=NULL, max.k=60){
  distances <- dist(train.task$env$data[, -ncol(train.task$env$data)], method = "euclidean") %>% as.matrix()
  
  best.acc = 1000
  best.k = 2
  best.out = c()
  label=train.task$env$data[, ncol(train.task$env$data)] %>% as.numeric()
  for(i in 2:max.k){
    out <- distMat.KernelKnn(distances, TEST_indices = test.task, y = label, k = i, 
                             regression = F, weights_function = NULL, threads = detectCores(),
                             Levels = unique(label)) %>% apply(., 1, which.max)
            

    current.acc<-measureACC(train.task$env$data[test.task, ncol(train.task$env$data)] %>% as.numeric(), out)
    if(best.acc > current.acc){
      best.k = i
      best.acc = current.acc
      best.out<-out
    }
  }
  cat("Best k: ", best.k, "\n")
  best.out<-as.character(best.out)
  best.out[which(best.out == "1")]<-levels(train.task$env$data[, ncol(train.task$env$data)])[1]
  best.out[which(best.out == "2")]<-levels(train.task$env$data[, ncol(train.task$env$data)])[2]
  invisible(best.out)
}  

