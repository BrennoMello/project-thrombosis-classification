#######################################################
# Code implemented to load the analyzed dataset 
#	as well as the required libraries
#
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

# loading main packages
library(dplyr)
library(caTools)
library(reshape)
library(ggplot2)
library(mlr)
library(parallelMap)
library(parallel)
library(KernelKnn)
library(BBmisc)

source("src/preprocessing/cvClass.R")
source("src/preprocessing/binarize.R")

plotVariablesByClass<-function(data){
    by.class = names(table(hemo.data$type))

    for (i in by.class){
        ### box plot attributes
        melt_df <- melt(subset(
            hemo.data %>% filter(type == i), 
            select = -c(type)))

        p<-ggplot(melt_df, aes(x=variable, y=value)) + 
        geom_boxplot(fill="slateblue", alpha=0.2) + 
            theme(text = element_text(size=20),
            axis.text.x = element_text(angle=90, hjust=1)) +
            xlab("Variable") + ylab("Value")

        png(file=paste(sep="", "results/models/boxplot-", i, ".png"))
        print(p)
        dev.off()
    }

}


#hemo.data<-read.table(file="dataset/RIN - Supplementary Table XXX - 2R7E structure residue network.csv", sep=";", header = T)
hemo.data<-read.table(file="data/thrombosis_non_thrombosis_v2.csv", sep="\t", header = T)
hemo.data<-subset(hemo.data, select = -c(node))
hemo.data$type = as.factor(hemo.data$type)

#normalizing all atts
for (i in 1:(ncol(hemo.data)-1)){
    hemo.data[, i] = (hemo.data[, i]-min(hemo.data[, i]))/(max(hemo.data[, i])-min(hemo.data[, i]))
}

#write.table(train, file = "/home/rios/programming/python/hemo-2r7e.csv", row.names = F, sep=",")
plotVariablesByClass(hemo.data)

#hemo.data = binarize.features.by.class(hemo.data, ncol(hemo.data))

hemo.data = hemo.data %>% subset(select = c(areaSAS, areaSES, relSESA, type))

cv.10<-cv.bin.strat.class(dataset=hemo.data, label.index=length(names(hemo.data)), seed=123456, cv=10)

