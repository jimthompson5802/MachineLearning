Practical Machine Learning - Project
========================================================

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(plyr)
```

Data Preparation
----------------


```r
raw.data <- read.csv("./data/pml-training.csv",
                     stringsAsFactors=FALSE)
raw.data <- subset(raw.data,new_window == "no")

## eliminate columns that are constant or near zero variance
nz.idx <- nearZeroVar(raw.data)
raw.data <- raw.data[,-nz.idx]

## eliminate identifier attributes that are not generalizable
raw.data <- subset(raw.data,select=-c(X,user_name,cvtd_timestamp,
                                num_window,
                                raw_timestamp_part_1,
                                raw_timestamp_part_2))

## convert classe into factor variable for training and prediciton
raw.data <- mutate(raw.data,
                   classe=factor(classe))

## extract subset of the training data for model building and validation
set.seed(123)
raw.idx <- createDataPartition(raw.data$classe,p=0.25,list=FALSE)
raw.data <- raw.data[raw.idx,]


## split model building subset into training and test sets
set.seed(456)
train.idx <- createDataPartition(raw.data$classe,p=0.6,list=FALSE)

train <- raw.data[train.idx,]
test <- raw.data[-train.idx,]
```


Model Training
----------

```r
# library(e1071)
## set up for parallel processing
library(doSNOW)
hosts <- c(rep("localhost",2))
cl <- makeSOCKcluster(hosts)
registerDoSNOW(cl)

trCtrl <- trainControl(method="repeatedcv", number=10, repeats=5)

set.seed(789)
system.time(gbm.mdl1 <- train(classe~.,train,method="gbm",verbose=FALSE,
                  metric="Accuracy",
                  trControl=trCtrl))
```

```
##    user  system elapsed 
##    6.43    0.09  347.21
```

```r
## stop cluster
stopCluster(cl)

print(gbm.mdl1)
```

```
## Stochastic Gradient Boosting 
## 
## 2885 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## 
## Summary of sample sizes: 2597, 2596, 2597, 2598, 2596, 2597, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
##   1                  50       0.7       0.7    0.02         0.03    
##   1                  100      0.8       0.8    0.02         0.03    
##   1                  200      0.8       0.8    0.02         0.03    
##   2                  50       0.8       0.8    0.02         0.03    
##   2                  100      0.9       0.9    0.02         0.02    
##   2                  200      0.9       0.9    0.02         0.02    
##   3                  50       0.9       0.8    0.02         0.02    
##   3                  100      0.9       0.9    0.01         0.02    
##   3                  200      0.9       0.9    0.01         0.02    
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```


```r
pred.classe <- predict(gbm.mdl1,test)


confusionMatrix(pred.classe,test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 537  14   1   1   4
##          B   4 346  19   2  11
##          C   3   9 313  20  11
##          D   1   1   2 286   7
##          E   2   2   0   5 319
## 
## Overall Statistics
##                                         
##                Accuracy : 0.938         
##                  95% CI : (0.926, 0.948)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.922         
##  Mcnemar's Test P-Value : 4.19e-06      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.982    0.930    0.934    0.911    0.906
## Specificity             0.985    0.977    0.973    0.993    0.994
## Pos Pred Value          0.964    0.906    0.879    0.963    0.973
## Neg Pred Value          0.993    0.983    0.986    0.983    0.979
## Prevalence              0.285    0.194    0.174    0.164    0.183
## Detection Rate          0.280    0.180    0.163    0.149    0.166
## Detection Prevalence    0.290    0.199    0.185    0.155    0.171
## Balanced Accuracy       0.984    0.953    0.954    0.952    0.950
```


```r
vi <- data.frame(varImp(gbm.mdl1)[1])
sorted.vi.idx <- rev(order(vi))
sorted.vi.idx <- sorted.vi.idx[1:20]


par(mar=c(5,12,4,2)) # increase y-axis margin.

barplot(vi[rev(sorted.vi.idx),1],
        names.arg=rownames(vi)[rev(sorted.vi.idx)],
        horiz=TRUE,
        las=2,
        xlab="Importance",
        main="Top 20 Important Attributes"
        )
```

![plot of chunk ModelAnalysis](figure/ModelAnalysis.png) 


Submission
----------

```r
##
# Instructor provided function to generate submission data for grading
##
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


sub.data <- read.csv("./data/pml-testing.csv",
                     stringsAsFactors=FALSE)

sub.classe <- predict(gbm.mdl1,sub.data)

pml_write_files(sub.classe)
```


