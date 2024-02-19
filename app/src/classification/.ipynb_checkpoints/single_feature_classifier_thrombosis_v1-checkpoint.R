# Tiago Lopes
# 25/Feb/2022

start_time <- Sys.time()

options(warn=-1, scipen=10000000)

library(xgboost)
library(gplots)


# This is the input file, with all instances, features and class labels
#mydata = read.table("data/thrombosis_non_thrombosis_v2.csv", sep="\t", header = T)
#mydata = read.table("data/thrombosis_non_thrombosis_v3.csv", sep="\t", header = T)
mydata = read.table("data/hemophilia-A-FVIII.csv", sep="\t", header = T)

mydata = mydata[order(mydata$type),]

# These are the class labels, derived directly from the input file
class1 = unique(mydata$type)[1]
class2 = unique(mydata$type)[2]

testSize_smaller_class = 10
classThreshold = 0.5
num_repetitions = 50

# This function is usally the last to be called.
# It plots the presence / abscence matrix, 
# containing the instances used in the test sets over all repetitions

plot_presence_matrix = function(){
    my_palette <- colorRampPalette(c("black", "yellow"))(n = 2)
    
    heatmap.2(test_presence_matrix, trace="none", Rowv = NA, Colv = NA, col=my_palette, sepwidth=c(0.01,0.01),
              sepcolor="white",
              colsep = 1:ncol(test_presence_matrix),
              rowsep = 1:nrow(test_presence_matrix))
    
    #plot(rowSums(test_presence_matrix), las=1, ylab="Number of appearances in the test set", xlab="Instance #")
    #abline(v=22.5)
    #grid()
    
}

# This function runs the XGBoost classifier
# It does not use a validation set to optimize the parameters.

run_XGBoost <- function(trainSet, testSet){
    
    #print("Running the XGBoost classifier")
    
    ##################################################
    # Create the xgbMatrix for the training set
    ##################################################
    training <- trainSet[,1:(ncol(trainSet)-1)]
    
    myClass <- as.vector(trainSet$type)
    
    myClass[which(myClass == class1)] <- 0
    myClass[which(myClass == class2)] <- 1
    
    #training$class <- NULL
    
    dTrain <- xgb.DMatrix(as.matrix(training), label=myClass)
    
    ##################################################
    # Create the XGMatrix for the test instance
    ##################################################
    
    testing <- as.data.frame(testSet[, 1:(ncol(testSet)-1)])
    
    dTest <- xgb.DMatrix(as.matrix(testing), label=rep(1, nrow(testing)))
    
    
    #####################################################################################
    
    
    # These parameters can be optimized, using a grid-search
    params <- list(booster = "gbtree", 
                   objective = "binary:logistic", 
                   #eta=0.01, 
                   #gamma=0, 
                   #max_depth=10, 
                   #min_child_weight=1, 
                   #subsample=1, 
                   #colsample_bytree=1, 
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



# Time to assemble the train and test set

mydata$node = NULL

pos1 = which(mydata$type == class1)
pos2 = which(mydata$type == class2)

smallerClass_dataset = vector()
largerClass_dataset = vector()

if(length(pos1) > length(pos2)){
    smallerClass_dataset = mydata[pos2,]
    largerClass_dataset = mydata[pos1,]
}else{
    smallerClass_dataset = mydata[pos1,]
    largerClass_dataset = mydata[pos2,]
}


accuracy_vec = matrix(nrow=num_repetitions, ncol = 2)
colnames(accuracy_vec) = c(class1, class2)

test_presence_matrix_smallerClass = matrix(0, nrow = nrow(smallerClass_dataset), ncol = num_repetitions)
test_presence_matrix_largerClass = matrix(0, nrow = nrow(largerClass_dataset), ncol = num_repetitions)

set.seed(100)

for(reps in 1:num_repetitions){
    
    print(paste("** Performing classification procedure. Repetion", reps, "of", num_repetitions))
    
    # First, assemble the test set
    # Step 1: select instances from the smaller and from the larger classes
    testInstances_smaller_class = sample(1:nrow(smallerClass_dataset), testSize_smaller_class)
    testInstances_larger_class = sample(1:nrow(largerClass_dataset), testSize_smaller_class)
    
    # Indicate in the presence matrix the instances that have been selected.
    #test_presence_matrix_smallerClass[testInstances_smaller_class, reps] = 1
    #test_presence_matrix_largerClass[testInstances_larger_class, reps] = 1
    
    # Step 2: Combine both sets and shuffle
    testSet = rbind(smallerClass_dataset[testInstances_smaller_class,], largerClass_dataset[testInstances_larger_class,])
    testSet = testSet[sample(1:nrow(testSet)),]
    
    # Second, assemble the training set
    # Step 1: select instances from the smaller and from the larger classes
    trainSet_smaller_class = smallerClass_dataset[-testInstances_smaller_class,]
    
    # Step 2: Reduce the size of the larger class, to have the same number of instances as the smaller class.
    trainSet_larger_class = largerClass_dataset[-testInstances_larger_class,]
    trainSet_larger_class = largerClass_dataset[sample(1:nrow(trainSet_larger_class), nrow(trainSet_smaller_class)),]
    
    # Step 3: Combine both sets and shuffle
    trainSet = rbind(trainSet_smaller_class, trainSet_larger_class)
    trainSet = trainSet[sample(1:nrow(trainSet)),]
    
    
    ################################################################################
    ##
    ## Now create train and test sets using only one attribute at a time.
    ## This is the single feature classification.
    ##
    ################################################################################
    
    result = matrix(nrow=nrow(testSet), ncol = (ncol(trainSet)-1))
    
    for(i in 1:(ncol(trainSet)-1)){
        #print(paste("Testing feature", colnames(trainSet)[i]))
        
        trainSet_att = trainSet[,c(i, ncol(trainSet))]
        testSet_att = testSet[,c(i, ncol(testSet))]
        
        result[, i] = run_XGBoost(trainSet = trainSet_att, testSet = testSet_att)
    }
    
    testSet$pred = rowMeans(result)
    
    statistical_weights = get_statistical_weights(trainSet)
    
    testSet$pred = apply(result, 1, function(x) weighted.mean(x, statistical_weights))
    
    # Check accuracy
    acc_class1 = length(which(testSet$type == class1 & testSet$pred < (1 - classThreshold))) / length(which(testSet$type == class1 & (testSet$pred > classThreshold | testSet$pred < (1 - classThreshold)))) * 100
    acc_class2 = length(which(testSet$type == class2 & testSet$pred > classThreshold)) / length(which(testSet$type == class2 & (testSet$pred > classThreshold | testSet$pred < (1 - classThreshold)))) * 100
    
    accuracy_vec[reps, 1] = acc_class1
    accuracy_vec[reps, 2] = acc_class2
}

test_presence_matrix = rbind(test_presence_matrix_smallerClass, test_presence_matrix_largerClass)

if(any(is.na(accuracy_vec))){
    accuracy_vec[(is.na(accuracy_vec))]=0
}

finalAcc = colMeans(accuracy_vec)

print("Done!")
print("")
print(paste("Final accuracy:", class1, ":", finalAcc[1], class2, ":", finalAcc[2]))
print("")

end_time <- Sys.time()
print(end_time - start_time)



