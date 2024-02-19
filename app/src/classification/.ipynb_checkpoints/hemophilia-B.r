#hemo.data<-read.table(file="HemB_Dataset_SENAI_cleaned_v3-label.csv", sep=",", header = T)
hemo.data<-read.table(file="HemB_Dataset_SENAI_v5a.csv", sep="\t", header = T)
hemo.data<-subset(hemo.data, select = -c(cDNA, AA_HGVS, AA_Legacy, Domain, Protein_Change, aa1, aa2))
#head(hemo.data)
#hemo.data<-subset(hemo.data, select = -c(degree, kcore))
#head(hemo.data)

### should we use normalization?
hemo.data<-normalize(hemo.data, method = "range", range = c(0, 1))

### remove NA
cat('Data length [L C]:', dim(hemo.data), "\n")
na.values<-apply(hemo.data, 1, function (x) any(is.na(x)))
cat ('Na ratio: ', sum(na.values)/nrow(hemo.data))
hemo.data<-hemo.data[-c(which(na.values)), ]
cat('Data length [L C]:', dim(hemo.data), "\n")