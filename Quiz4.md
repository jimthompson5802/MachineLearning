Machine Learning - Quiz 4
========================================================

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.0.3
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.0.3
```

```r
library(plyr)
```

```
## Warning: package 'plyr' was built under R version 3.0.3
```



Question 1
----------

```r
library(ElemStatLearn)
library(plyr)
data(vowel.train)
data(vowel.test) 

vowel.train <- mutate(vowel.train,y=factor(y))
vowel.test <- mutate(vowel.test,y=factor(y))

library(caret)
set.seed(33833)
rf.mdl <- train(y~.,data=vowel.train,method="rf",verbose=FALSE,metric="Accuracy")

gbm.mdl <- train(y~.,data=vowel.train,method="gbm",verbose=FALSE,metric="Accuracy")
```

```r
print(rf.mdl)
```

```
## Random Forest 
## 
## 528 samples
##  10 predictors
##  11 classes: '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 528, 528, 528, 528, 528, 528, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.9       0.9    0.02         0.02    
##   6     0.9       0.9    0.02         0.02    
##   10    0.9       0.9    0.03         0.03    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
print(gbm.mdl)
```

```
## Stochastic Gradient Boosting 
## 
## 528 samples
##  10 predictors
##  11 classes: '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 528, 528, 528, 528, 528, 528, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
##   1                  50       0.7       0.7    0.04         0.04    
##   1                  100      0.8       0.7    0.04         0.04    
##   1                  200      0.8       0.8    0.03         0.04    
##   2                  50       0.8       0.8    0.03         0.03    
##   2                  100      0.8       0.8    0.03         0.04    
##   2                  200      0.9       0.8    0.03         0.04    
##   3                  50       0.8       0.8    0.03         0.04    
##   3                  100      0.9       0.8    0.03         0.03    
##   3                  200      0.9       0.9    0.02         0.03    
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```

```r
pred.rf <- predict(rf.mdl,vowel.test)
pred.gbm <- predict(gbm.mdl,vowel.test)

confusionMatrix(pred.rf,vowel.test$y)$overall["Accuracy"]
```

```
## Accuracy 
##   0.6061
```

```r
confusionMatrix(pred.gbm,vowel.test$y)$overall["Accuracy"]
```

```
## Accuracy 
##   0.5303
```

```r
idx <- pred.rf == pred.gbm

pred.comb <- pred.rf[idx]
test.comb <- vowel.test[idx,"y"]

confusionMatrix(pred.comb,test.comb)$overall["Accuracy"]
```

```
## Accuracy 
##   0.6623
```

Question 2
----------

```
## Warning: package 'AppliedPredictiveModeling' was built under R version
## 3.0.3
```

```
## Loading required package: MASS
```

```
## Warning: package 'MASS' was built under R version 3.0.3
## Warning: variables are collinear
## Warning: variables are collinear
## Warning: variables are collinear
```

```
## note: only 2 unique complexity parameters in default grid. Truncating the grid to 2 .
```

```
## Error: no applicable method for 'predict' applied to an object of class
## "NULL"
```

```r
test.pred.rf <- predict(rf.mdl,testing)
test.pred.gbm <- predict(gbm.mdl,testing)
test.pred.lda <- predict(lda.mdl,testing)

test.stack.df <- data.frame(pred.rf=test.pred.rf,pred.gbm=test.pred.gbm,
                            pred.lda=test.pred.lda)
test.pred.stack <- predict(stack.rf.mdl,test.stack.df)

confusionMatrix(test.pred.rf,testing$diagnosis)$overall["Accuracy"]
```

```
## Accuracy 
##   0.7683
```

```r
confusionMatrix(test.pred.gbm,testing$diagnosis)$overall["Accuracy"]
```

```
## Accuracy 
##   0.7927
```

```r
confusionMatrix(test.pred.lda,testing$diagnosis)$overall["Accuracy"]
```

```
## Accuracy 
##   0.7683
```

```r
confusionMatrix(test.pred.stack,testing$diagnosis)$overall["Accuracy"]
```

```
## Accuracy 
##   0.7927
```

Question 3
----------

```r
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)
lasso.mdl <- train(CompressiveStrength~.,data=training,method="lasso",
                metric="RMSE")
```

```
## Loading required package: elasticnet
## Loading required package: lars
## Loaded lars 1.2
```

```r
plot(lasso.mdl$finalModel,xvar="penalty",use.color=TRUE)
```

![plot of chunk Question3](figure/Question3.png) 

Question 4
----------

```r
library(lubridate)
```

```
## Warning: package 'lubridate' was built under R version 3.0.3
```

```
## 
## Attaching package: 'lubridate'
## 
## The following object is masked from 'package:plyr':
## 
##     here
```

```r
library(forecast)
```

```
## Warning: package 'forecast' was built under R version 3.0.3
```

```
## Loading required package: zoo
```

```
## Warning: package 'zoo' was built under R version 3.0.3
```

```
## 
## Attaching package: 'zoo'
## 
## The following objects are masked from 'package:base':
## 
##     as.Date, as.Date.numeric
## 
## Loading required package: timeDate
## 
## Attaching package: 'timeDate'
## 
## The following objects are masked from 'package:e1071':
## 
##     kurtosis, skewness
## 
## This is forecast 5.4
```

```r
dat = read.csv("./data/gaData.csv")
training = dat[year(dat$date) == 2011,]
tstrain = ts(training$visitsTumblr)

bat.mdl <- bats(tstrain)

fcast <- forecast(bat.mdl,235)

test <- dat[366:600,"visitsTumblr"]

cat("test set inside the 95% interval")
```

```
## test set inside the 95% interval
```

```r
100 * sum(fcast$lower[,2] <= test & test <= fcast$upper[,2])/length(test)
```

```
## [1] 96.17
```



Question 5
----------

```r
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

library(e1071)
set.seed(325)
svm.mdl <- svm(CompressiveStrength~.,data=training)

pred <- predict(svm.mdl,testing)

mse <- sum((pred-testing$CompressiveStrength)^2)/length(pred);mse
```

```
## [1] 45.09
```

```r
sqrt(mse)
```

```
## [1] 6.715
```

