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


#hemo.data<-read.table(file="dataset/RIN - Supplementary Table XXX - 2R7E structure residue network.csv", sep=";", header = T)
hemo.data<-read.table(file="datasets/RIN-2R7E.csv", sep=";", header = T)
hemo.data<-subset(hemo.data, select = -c(res, node))

activity<-hemo.data$Activity
### should we use normalization?
#hemo.data<-normalize(hemo.data, method = "range", range = c(0, 1))

valid.data<-which(!is.na(hemo.data$Activity))
train<-hemo.data[valid.data,]
test<-hemo.data[-valid.data,]

train$Activity<-activity[valid.data] %>% 
  cut(breaks=c(-Inf, 50, Inf), labels=c("low","high")) %>% droplevels()

rm(activity, valid.data)

### box plot attributes
temp_df <- subset(train, select = -c(Activity))
temp_df <- train
melt_df <- melt(temp_df)

p<-ggplot(melt_df, aes(x=variable, y=value)) + 
  geom_boxplot(fill="slateblue", alpha=0.2) + 
  theme(text = element_text(size=20),
        axis.text.x = element_text(angle=90, hjust=1)) +
  xlab("Variable") + ylab("Value")

#pdf(file="results/boxplot.pdf")
#print(p)
#dev.off()
###

#write.table(train, file = "/home/rios/programming/python/hemo-2r7e.csv", row.names = F, sep=",")

cv.10<-cv.bin.strat.class(dataset=train, seed=123456, cv=10)
