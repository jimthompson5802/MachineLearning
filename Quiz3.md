Machine Learning - Quiz 3
========================================================

Question 1
----------

```r
library(AppliedPredictiveModeling)
```

```
## Warning: package 'AppliedPredictiveModeling' was built under R version
## 3.0.3
```

```r
data(segmentationOriginal)
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
train <- segmentationOriginal[segmentationOriginal$Case == "Train",]
test <- segmentationOriginal[segmentationOriginal$Case == "Test",]

set.seed(125)
fit <- train(Class~.,method="rpart",data=train)
```

```
## Loading required package: rpart
```

```
## Warning: package 'rpart' was built under R version 3.0.3
## Warning: package 'e1071' was built under R version 3.0.3
```

```r
print(fit)
```

```
## CART 
## 
## 1009 samples
##  118 predictors
##    2 classes: 'PS', 'WS' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 1009, 1009, 1009, 1009, 1009, 1009, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.03  0.8       0.5    0.02         0.04    
##   0.2   0.7       0.5    0.02         0.04    
##   0.3   0.7       0.4    0.06         0.2     
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03.
```

```r
plot(fit$finalModel,uniform=TRUE)
text(fit$finalModel, use.n=TRUE, all=TRUE, cex=.8)
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1.png) 

```r
data.a <- data.frame(TotalIntench2 = 23,000, FiberWidthCh1 = 10, PerimStatusCh1=20)
data.b <- data.frame( TotalIntench2 = 50,000, FiberWidthCh1 = 10,VarIntenCh4 = 100)
data.c <- data.frame(TotalIntench2 = 57,000, FiberWidthCh1 = 8,VarIntenCh4 = 100)
data.d <- data.frame(FiberWidthCh1 = 8,VarIntenCh4 = 100, PerimStatusCh1=2) 

predict(fit,data.a)
```

```
## Error: object 'Cell' not found
```

Question 3
----------

```r
library(pgmm)
data(olive)
olive = olive[,-1]

library(tree)
```

```
## Warning: package 'tree' was built under R version 3.0.3
```

```r
fit.q3 <- tree(Area~.,olive)
print(fit.q3)
```

```
## node), split, n, deviance, yval
##       * denotes terminal node
## 
## 1) root 572 3000 5  
##   2) Eicosenoic < 6.5 249  500 7  
##     4) Linoleic < 1053.5 151  100 8  
##       8) Oleic < 7895 95   20 8 *
##       9) Oleic > 7895 56   20 9 *
##     5) Linoleic > 1053.5 98   20 5 *
##   3) Eicosenoic > 6.5 323  200 3  
##     6) Oleic < 7770.5 304  100 3 *
##     7) Oleic > 7770.5 19   20 1 *
```

```r
newdata = as.data.frame(t(colMeans(olive)))

predict(fit.q3,newdata)
```

```
##     1 
## 2.875
```


Question 4
----------

```r
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

fit.q4 <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl,
                data=trainSA,
                method="glm",
                family="binomial")

print(fit.q4)
```

```
## Generalized Linear Model 
## 
## 231 samples
##   9 predictors
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 231, 231, 231, 231, 231, 231, ... 
## 
## Resampling results
## 
##   RMSE  Rsquared  RMSE SD  Rsquared SD
##   0.4   0.2       0.02     0.06       
## 
## 
```

```r
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

missClass(testSA$chd,predict(fit.q4,testSA))
```

```
## [1] 0.3117
```

```r
missClass(trainSA$chd,predict(fit.q4,trainSA))
```

```
## [1] 0.2727
```


Question 5
----------

```r
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 
library(plyr)
```

```
## Warning: package 'plyr' was built under R version 3.0.3
```

```
## 
## Attaching package: 'plyr'
## 
## The following object is masked from 'package:ElemStatLearn':
## 
##     ozone
```

```r
vowel.train <- mutate(vowel.train,y=factor(y))
vowel.test <- mutate(vowel.test,y=factor(y))

library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(33833)
# fit.q5 <- train(y~.,method="rf",data=vowel.train)
fit.q5 <- randomForest(y~.,data=vowel.train)

print(fit.q5)
```

```
## 
## Call:
##  randomForest(formula = y ~ ., data = vowel.train) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 3
## 
##         OOB estimate of  error rate: 3.22%
## Confusion matrix:
##     1  2  3  4  5  6  7  8  9 10 11 class.error
## 1  48  0  0  0  0  0  0  0  0  0  0     0.00000
## 2   1 47  0  0  0  0  0  0  0  0  0     0.02083
## 3   0  0 48  0  0  0  0  0  0  0  0     0.00000
## 4   0  0  0 47  0  1  0  0  0  0  0     0.02083
## 5   0  0  0  0 45  2  0  0  0  0  1     0.06250
## 6   0  0  0  1  0 42  0  0  0  0  5     0.12500
## 7   0  0  0  0  2  0 44  2  0  0  0     0.08333
## 8   0  0  0  0  0  0  0 48  0  0  0     0.00000
## 9   0  0  0  0  0  0  1  0 47  0  0     0.02083
## 10  0  0  0  0  0  0  1  0  0 47  0     0.02083
## 11  0  0  0  0  0  0  0  0  0  0 48     0.00000
```

```r
rownames(varImp(fit.q5))[rev(order(varImp(fit.q5)$Overall))]
```

```
##  [1] "x.2"  "x.1"  "x.5"  "x.6"  "x.8"  "x.4"  "x.9"  "x.3"  "x.7"  "x.10"
```

