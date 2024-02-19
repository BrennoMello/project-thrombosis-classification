binarize.features.by.class <- function(data, ignore.class.pos=1){
    for (i in (1:ncol(data))[-ignore.class.pos]){
        cutoff = boxplot(data[,i], plot=F)$stats[3,1]
        temp = rep("low", nrow(data))
        temp[which(data[,i] > cutoff)] = "high"
        data[,i] = temp
    }
    data
}