Predicting Quality of Weight Lifting Exercise Through Motion Data
====================================
Author: Jim Thompson

Summary
-------
This project develops a predictive model for assessing a participant's ability to 
correctly perform the Unilateral Dumbbell Biceps Curl^1.  Participant's performance
is assessed into one of five classes.  One class represents performing the exercise
correctly, the remaining four classes represent common mistakes.  Data used for prediction are motion data captured by sensors attached to various parts of the participant's body and dumbbell.  

The predictive model developed in this work is based on a boosted tree machine 
learning algorithm.  The specific implementation used is the **gbm** package from R.
The model correctly predicts 93.8% for the out-of-sample test cases.

The remainder of this paper describes data preparation for modeling, model training and
assessing the model's performance.


Data Preparation
--------------
Specify required R packages.

```r
library(caret)
library(plyr)
```
Data for model [training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and [test submission](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) were obtained through Coursera.

After downloading the data, we begin by reading the training data.

```r
raw.data <- read.csv("./data/pml-training.csv",
                     stringsAsFactors=FALSE)


## save names of attributes
raw.names <- names(raw.data)
raw.rows <- nrow(raw.data)
```
The data contains 19,622 observations each with 160 attributes.
Exercise performance is categorized into one of 5 classes, caputured in the attribute "classe".  Here is the percent distribution values in the classe attribute:

```r
dist.raw <- round(100*table(raw.data$classe)/raw.rows,2)
names(dist.raw) <- c("A","B","C","D","E")

## Percent distribution of performance classes
dist.raw
```

```
##     A     B     C     D     E 
## 28.44 19.35 17.44 16.39 18.38
```
Class "A" represents performing the exercise correctly.  The other classes indicate
various types of mistakes.


After a brief review of the training and test data for submission, the following 
steps were taken to prepare data for model building and performance assessment.
* Keep only observations that are generated during movement.  This makes the training
data similar to the data used for submission.
* Eliminate attributes that are near or at zero variance.  Low or no variance attributes
do not provide any value in the modeling processing.
* Eliminate attributes that are not generalizable, such as time stamps and user
identifiers.  These attributes are unique to the training data and can lead data leakage, where the
model will overfit the training data and not predict well on other data sets.


```r
## keep only raw motion data
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

## data set for training
train <- raw.data[train.idx,]

## data set to assess out-of-sample model performance
test <- raw.data[-train.idx,]
```

From the original 19,622 observations, we extract a 25% 
(4,805) random sample 
for model building.  The reason for selecting this subset is to speed up training and
validation work.  This extract is then split into a 60% (2,885) for training 
and 40% (1,920) for assessing model performance.
Out of the original 160 attributes, we end up using 53 
for training and validation.  See Appdendix for details on attributes used or eliminated.

Here is the percent distribution of the classes in the training and test data

```r
dist.train <- round(100*table(train$classe)/nrow(train),2)
names(dist.train) <- c("A","B","C","D","E")

dist.test <- round(100*table(test$classe)/nrow(test),2)
names(dist.test) <- c("A","B","C","D","E")

## combine the distribution of classe values for the original, training and test data sets
df <- data.frame(classe=LETTERS[1:5],Original=as.vector(dist.raw),
           Training=as.vector(dist.train),
            Testing=as.vector(dist.test),
            stringsAsFactors=FALSE)
```

```
## Percent Distrbutions of classe values
## Original/Training/Test Data Sets
```

```
##   classe Original Training Testing
## 1      A    28.44    28.46   28.49
## 2      B    19.35    19.34   19.38
## 3      C    17.44    17.44   17.45
## 4      D    16.39    16.40   16.35
## 5      E    18.38    18.37   18.33
```
From this we see the training and testing data have same distribution of classe values as the original
raw data.

Model Training
----------
Gradient Boosting algorithm (**gbm** R package) is used for the machine learning algorithm. 
Repeated cross-validation is used to determine values for the algorithm's 
hyper-parameters.  For purposes of assessing model performance, we use the accuracy, i.e.,
the percent of test case that are assigned to the correct classe value.

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
##    6.33    0.17  399.62
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
##   2                  50       0.8       0.8    0.02         0.02    
##   2                  100      0.9       0.9    0.02         0.02    
##   2                  200      0.9       0.9    0.02         0.02    
##   3                  50       0.9       0.8    0.02         0.02    
##   3                  100      0.9       0.9    0.02         0.02    
##   3                  200      0.9       0.9    0.02         0.02    
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```

Model Performance
-----------------
Using the **gbm** model with the optimal hyper-parameters, determined above, we assess
model performance.  First we present the confusion matrix.  Next we see that the
model is 93.8% accurate on the out-of-sample data.  The 95% confidence interval for accuracy
is from 92.6% to 94.8%.


```r
## make prediction of classe value on the out-of-sample test data
pred.classe <- predict(gbm.mdl1,test)

## report accuracy of predictions on out-of-sample data
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

One feature of R's **gbm** package is the ability to identify attributes that are 
important in predicting the classe attribute.  Below shows the top 20 explanatory
variables in descending order.

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
The following code generates submission files for the project.

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

##
# read in test set for submission
##
sub.data <- read.csv("./data/pml-testing.csv",
                     stringsAsFactors=FALSE)

## predict classe for the submission test set
sub.classe <- predict(gbm.mdl1,sub.data)

## create the submission files
pml_write_files(sub.classe)
```
Following are the predicted classe value for the test cases.

```r
df <- data.frame(Test.Case=1:length(sub.classe),Predicted.classe=sub.classe,
           row.names=NULL,stringsAsFactors=FALSE)
```

```
## Predicted 'classe' Values for Project Submission
```

```
##    Test.Case Predicted.classe
## 1          1                B
## 2          2                A
## 3          3                B
## 4          4                A
## 5          5                A
## 6          6                E
## 7          7                D
## 8          8                B
## 9          9                A
## 10        10                A
## 11        11                B
## 12        12                C
## 13        13                B
## 14        14                A
## 15        15                E
## 16        16                E
## 17        17                A
## 18        18                B
## 19        19                B
## 20        20                B
```


Reference
---------
1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **Qualitative Activity Recognition of Weight Lifting Exercises**. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.  [See Documentation](http://groupware.les.inf.puc-rio.br/har)

Appendix
--------
These are the attributes selected for building the **gbm** model.

```r
names(train)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```


Following is the list of attributes removed from analysis because of near zero
variance or attributes, such as time stamps or identifiers, that will not
generalize.

```r
setdiff(raw.names,names(train))
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "kurtosis_roll_belt"      
##   [9] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [11] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [13] "skewness_yaw_belt"        "max_roll_belt"           
##  [15] "max_picth_belt"           "max_yaw_belt"            
##  [17] "min_roll_belt"            "min_pitch_belt"          
##  [19] "min_yaw_belt"             "amplitude_roll_belt"     
##  [21] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [23] "var_total_accel_belt"     "avg_roll_belt"           
##  [25] "stddev_roll_belt"         "var_roll_belt"           
##  [27] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [29] "var_pitch_belt"           "avg_yaw_belt"            
##  [31] "stddev_yaw_belt"          "var_yaw_belt"            
##  [33] "var_accel_arm"            "avg_roll_arm"            
##  [35] "stddev_roll_arm"          "var_roll_arm"            
##  [37] "avg_pitch_arm"            "stddev_pitch_arm"        
##  [39] "var_pitch_arm"            "avg_yaw_arm"             
##  [41] "stddev_yaw_arm"           "var_yaw_arm"             
##  [43] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [45] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [47] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [49] "max_roll_arm"             "max_picth_arm"           
##  [51] "max_yaw_arm"              "min_roll_arm"            
##  [53] "min_pitch_arm"            "min_yaw_arm"             
##  [55] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [57] "amplitude_yaw_arm"        "kurtosis_roll_dumbbell"  
##  [59] "kurtosis_picth_dumbbell"  "kurtosis_yaw_dumbbell"   
##  [61] "skewness_roll_dumbbell"   "skewness_pitch_dumbbell" 
##  [63] "skewness_yaw_dumbbell"    "max_roll_dumbbell"       
##  [65] "max_picth_dumbbell"       "max_yaw_dumbbell"        
##  [67] "min_roll_dumbbell"        "min_pitch_dumbbell"      
##  [69] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
##  [71] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"  
##  [73] "var_accel_dumbbell"       "avg_roll_dumbbell"       
##  [75] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
##  [77] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
##  [79] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
##  [81] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
##  [83] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
##  [85] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
##  [87] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
##  [89] "max_roll_forearm"         "max_picth_forearm"       
##  [91] "max_yaw_forearm"          "min_roll_forearm"        
##  [93] "min_pitch_forearm"        "min_yaw_forearm"         
##  [95] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
##  [97] "amplitude_yaw_forearm"    "var_accel_forearm"       
##  [99] "avg_roll_forearm"         "stddev_roll_forearm"     
## [101] "var_roll_forearm"         "avg_pitch_forearm"       
## [103] "stddev_pitch_forearm"     "var_pitch_forearm"       
## [105] "avg_yaw_forearm"          "stddev_yaw_forearm"      
## [107] "var_yaw_forearm"
```



