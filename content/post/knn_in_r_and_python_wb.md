---
title: "KNN in R and Python"
author: "Xinyu Zhang"
date: "April 13, 2019"
output:
  html_document:
    keep_md: true
---



In this post, I will demonstrate KNN in both R and python.

## Data set and task

The data set I will use is the OJ data set from ISLR(Introduction to Statistical Learning) library in R. You can also download it {{% staticref "files/OJ.CSV" "newtab" %}}here{{% /staticref %}}.

The task is to predict if a customer will purchase CH(Citrus Hill) or MM(Minute Maid) orange juice using the predictors in the data set. For demonstration purpose, I will only use the following response and predictors:

 - *Purchase*: A factor with levels CH and MM indicating whether the customer purchased Citrus Hill or Minute Maid Orange Juice
 
 - *StoreID*: Store ID
 
 - *SalePriceMM*: Sale price for MM

 - *SalePriceCH*: Sale price for CH

Let's take a look at the data set first.

#### R


```r
library(ISLR)
library(caret)
library(crossval)
library(pROC)
library(tidyverse)
library(reticulate)
library(xtable)
library(ggplot2)
library(metR)
library(gridExtra)

data=OJ %>% select(Purchase,StoreID,SalePriceMM,SalePriceCH) %>%
  mutate(Purchase=as.factor(Purchase),
         StoreID=as.factor(StoreID),
         SalePriceMM=as.numeric(SalePriceMM),
         SalePriceCH=as.numeric(SalePriceCH))

summary(data)
```

```
##  Purchase StoreID  SalePriceMM     SalePriceCH   
##  CH:653   1:157   Min.   :1.190   Min.   :1.390  
##  MM:417   2:222   1st Qu.:1.690   1st Qu.:1.750  
##           3:196   Median :2.090   Median :1.860  
##           4:139   Mean   :1.962   Mean   :1.816  
##           7:356   3rd Qu.:2.130   3rd Qu.:1.890  
##                   Max.   :2.290   Max.   :2.090
```

#### Python


```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score, make_scorer, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from itertools import product

OJ = pd.read_csv('E:/Self study/Post/KNN/OJ.csv')
data=OJ.loc[:,['Purchase','StoreID','SalePriceMM','SalePriceCH']]

pd.crosstab(index=data["Purchase"],columns="count")
```

```
## col_0     count
## Purchase       
## CH          653
## MM          417
```

```python
pd.crosstab(index=data["StoreID"],columns="count")
```

```
## col_0    count
## StoreID       
## 1          157
## 2          222
## 3          196
## 4          139
## 7          356
```

```python
data.describe()
```

```
##            StoreID  SalePriceMM  SalePriceCH
## count  1070.000000  1070.000000  1070.000000
## mean      3.959813     1.962047     1.815561
## std       2.308984     0.252697     0.143384
## min       1.000000     1.190000     1.390000
## 25%       2.000000     1.690000     1.750000
## 50%       3.000000     2.090000     1.860000
## 75%       7.000000     2.130000     1.890000
## max       7.000000     2.290000     2.090000
```

## Data preprocessing

KNN is designed for continuous predictors. To handle categorical predictors, we need to define a distance function $D\_{ij}$ for categorical variable $X\_{cat}$.

$\begin{aligned}
D\_{ij}=\begin{cases}1 & \text{if }x\_{cat}\text{ is different for observation }i\text{ and }j \\\\\\ 0 &\text{otherwise}\end{cases}\\\\\\
\end{aligned}$

This is equivalent to using $\sqrt{0.5}$ instead of 1 in one hot encoding on the categorical variable and keeping all levels instead of removing one of them. For example, a categorical variable with 3 levels

<!-- html table generated in R 3.4.4 by xtable 1.8-3 package -->
<!-- Tue Apr 16 05:11:19 2019 -->
<table border=1>
<tr> <th>  </th> <th> Cat </th>  </tr>
  <tr> <td> 1 </td> <td align="right"> A </td> </tr>
  <tr> <td> 2 </td> <td align="right"> B </td> </tr>
  <tr> <td> 3 </td> <td align="right"> B </td> </tr>
  <tr> <td> 4 </td> <td align="right"> B </td> </tr>
  <tr> <td> 5 </td> <td align="right"> C </td> </tr>
  <tr> <td> 6 </td> <td align="right"> C </td> </tr>
   </table>

becomes:

<!-- html table generated in R 3.4.4 by xtable 1.8-3 package -->
<!-- Tue Apr 16 05:11:20 2019 -->
<table border=1>
<tr> <th>  </th> <th> Cat_A </th> <th> Cat_B </th> <th> Cat_C </th>  </tr>
  <tr> <td> 1 </td> <td align="right"> 0.7071 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.0000 </td> </tr>
  <tr> <td> 2 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.7071 </td> <td align="right"> 0.0000 </td> </tr>
  <tr> <td> 3 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.7071 </td> <td align="right"> 0.0000 </td> </tr>
  <tr> <td> 4 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.7071 </td> <td align="right"> 0.0000 </td> </tr>
  <tr> <td> 5 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.7071 </td> </tr>
  <tr> <td> 6 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.0000 </td> <td align="right"> 0.7071 </td> </tr>
   </table>

One can definetly use a different distance function for categorical predictors. The scaling value can be picked using cross validation instead of $\sqrt{0.5}$.

Same goes for the continuous predictors. Instead of the Euclidean distance which I will be using here, one can consider using Manhattan or other distance functions.

Since I will use the Euclidean distance, continuous predictors should be standardized before putting into the KNN model.

#### R


```r
data_scaled=data %>% mutate(SalePriceMM=scale(SalePriceMM),
                            SalePriceCH=scale(SalePriceCH))

X=model.matrix(Purchase~-1+StoreID+SalePriceMM+SalePriceCH,data_scaled)
factor=diag(c(sqrt(0.5),sqrt(0.5),sqrt(0.5),sqrt(0.5),sqrt(0.5),1,1))
X_reconst=X%*%factor
train=data.frame(response=data_scaled$Purchase,X_reconst)
```

#### Python

The StandardScaler API in sklearn package divides the centered column by its population standard error instead of the sample standard error. For consistency, we will manually standardize our data using the sample standard error


```python
data_scaled=data.copy()
data_scaled[['SalePriceMM']]=(data_scaled['SalePriceMM']-np.average(data_scaled['SalePriceMM']))/np.std(data_scaled['SalePriceMM'], ddof=1)
data_scaled[['SalePriceCH']]=(data_scaled['SalePriceCH']-np.average(data_scaled['SalePriceCH']))/np.std(data_scaled['SalePriceCH'], ddof=1)

X_reconst=data_scaled.copy()
X_reconst[['StoreID_1','StoreID_2','StoreID_3','StoreID_4','StoreID_7']]=pd.DataFrame([[np.nan,np.nan,np.nan,np.nan,np.nan]],index=X_reconst.index)
X_reconst.loc[X_reconst['StoreID']==1,['StoreID_1','StoreID_2','StoreID_3','StoreID_4','StoreID_7']]=0.5**0.5,0,0,0,0
X_reconst.loc[X_reconst['StoreID']==2,['StoreID_1','StoreID_2','StoreID_3','StoreID_4','StoreID_7']]=0,0.5**0.5,0,0,0
X_reconst.loc[X_reconst['StoreID']==3,['StoreID_1','StoreID_2','StoreID_3','StoreID_4','StoreID_7']]=0,0,0.5**0.5,0,0
X_reconst.loc[X_reconst['StoreID']==4,['StoreID_1','StoreID_2','StoreID_3','StoreID_4','StoreID_7']]=0,0,0,0.5**0.5,0
X_reconst.loc[X_reconst['StoreID']==7,['StoreID_1','StoreID_2','StoreID_3','StoreID_4','StoreID_7']]=0,0,0,0,0.5**0.5

train=pd.DataFrame({'response':X_reconst['Purchase'],'X1':X_reconst['StoreID_1'],'X2':X_reconst['StoreID_2'],'X3':X_reconst['StoreID_3'],'X4':X_reconst['StoreID_4'],'X5':X_reconst['StoreID_7'],'X6':X_reconst['SalePriceMM'],'X7':X_reconst['SalePriceCH']})
train_y=train['response'].astype('object').values
train_X=train.drop(columns=['response'])
```

## KNN

#### R

I will use caret library to perform a 10-fold cross validation to choose the best K which gives the highest accuracy. Caret uses kknn library to build the KNN model.


```r
set.seed(1)
trControl=trainControl(method="cv",number=10)
knn_fit=train(response~.,method="knn",tuneGrid=expand.grid(k = 1:30),trControl  = trControl,metric="Accuracy",data=train)
results=knn_fit$results
best_index=c(which(results$Accuracy==max(results$Accuracy)),
             which(results$Kappa==max(results$Kappa)))
cat("The best K is",knn_fit$bestTune$k,"based on 10-fold CV.")
```

```
## The best K is 17 based on 10-fold CV.
```

The CV evaluation score table is below. Evaluation metrics are [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification) and [kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa#Calculation).


```r
knn_fit$results
```

```
##     k  Accuracy     Kappa AccuracySD    KappaSD
## 1   1 0.6896819 0.3321063 0.04254971 0.09263948
## 2   2 0.6990280 0.3538687 0.05042353 0.10916349
## 3   3 0.6971848 0.3487796 0.04771307 0.10376887
## 4   4 0.6972111 0.3483215 0.05214732 0.10806761
## 5   5 0.6944162 0.3434870 0.04234239 0.08312180
## 6   6 0.7028628 0.3612155 0.04500631 0.09085001
## 7   7 0.6954122 0.3458162 0.04268293 0.08405527
## 8   8 0.7009592 0.3591907 0.03651074 0.06655244
## 9   9 0.7028718 0.3637702 0.04131470 0.07956975
## 10 10 0.7075099 0.3714192 0.04162599 0.08380993
## 11 11 0.7103225 0.3778771 0.03770324 0.07344567
## 12 12 0.7094139 0.3774971 0.04149781 0.08513499
## 13 13 0.7149869 0.3883344 0.03519244 0.07140077
## 14 14 0.7150218 0.3875047 0.03676830 0.07345591
## 15 15 0.7113097 0.3793108 0.04247936 0.08903697
## 16 16 0.7084796 0.3731874 0.04560078 0.09423627
## 17 17 0.7178346 0.3913480 0.04111672 0.08718350
## 18 18 0.7131703 0.3799901 0.03662609 0.07670596
## 19 19 0.7066456 0.3629696 0.03782947 0.08194424
## 20 20 0.7084798 0.3669370 0.03190863 0.06975017
## 21 21 0.7122268 0.3749979 0.03415286 0.07566456
## 22 22 0.7094056 0.3678481 0.02932583 0.06609833
## 23 23 0.7065842 0.3612636 0.02842313 0.06559356
## 24 24 0.7094054 0.3685464 0.03122861 0.06904943
## 25 25 0.7112398 0.3746705 0.03891331 0.08496743
## 26 26 0.7140522 0.3797552 0.03803511 0.08506928
## 27 27 0.7140522 0.3803761 0.03463166 0.07776308
## 28 28 0.7159300 0.3846383 0.03395814 0.07434971
## 29 29 0.7159039 0.3844158 0.03446271 0.07655765
## 30 30 0.7130828 0.3783018 0.03754836 0.08681035
```

Using the best K = 17. We can make predictions on the data set and calculate the confusion matrix.


```r
pred_prob=predict(knn_fit,train,type = "prob")$CH
pred_class=ifelse(pred_prob >= 0.5, "CH", "MM")
table(pred_class,train[, "response"])
```

```
##           
## pred_class  CH  MM
##         CH 537 172
##         MM 116 245
```

Below are the other evaluation metrics if we assume CH to be the positive (or 1) case and MM to be the negative (or 0) case.


```r
cm = confusionMatrix(train[, "response"], pred_class, negative="MM")
diagnosticErrors(cm)
```

```
##       acc      sens      spec       ppv       npv       lor 
## 0.7308411 0.8223583 0.5875300 0.7574048 0.6786704 1.8861716 
## attr(,"negative")
## [1] "MM"
```

## Python

I use the neighbors API from sklearn package to perform the same 10-fold CV in order to find the best K.


```python
knn_model = neighbors.KNeighborsClassifier(algorithm="brute")
param_grid = {'n_neighbors': np.arange(1, 31)}
scoring = {"Accuracy": make_scorer(accuracy_score), "Kappa": make_scorer(cohen_kappa_score)}
kappa_scorer = make_scorer(cohen_kappa_score)

knn_fit = GridSearchCV(knn_model, param_grid, scoring=scoring, cv=10, refit='Accuracy', return_train_score=True)
random.seed(1)
knn_fit.fit(train_X, train_y)
```

```
## GridSearchCV(cv=10, error_score='raise-deprecating',
##        estimator=KNeighborsClassifier(algorithm='brute', leaf_size=30, metric='minkowski',
##            metric_params=None, n_jobs=None, n_neighbors=5, p=2,
##            weights='uniform'),
##        fit_params=None, iid='warn', n_jobs=None,
##        param_grid={'n_neighbors': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
##        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])},
##        pre_dispatch='2*n_jobs', refit='Accuracy', return_train_score=True,
##        scoring={'Accuracy': make_scorer(accuracy_score), 'Kappa': make_scorer(cohen_kappa_score)},
##        verbose=0)
## 
## C:\PROGRA~3\ANACON~1\lib\site-packages\sklearn\model_selection\_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
##   DeprecationWarning)
```

```python
results=knn_fit.cv_results_
print("The best K is " + str(knn_fit.best_params_['n_neighbors']) + " based on 10-fold CV.")
```

```
## The best K is 30 based on 10-fold CV.
```

Note that 30 is returned as the best K while it is 17 from R. This can be caused by different CV partitions when using different packages. R caret doesn't return the average and the standard error of the training set for CV procedure.

The CV evaluation score table is below.


```python
pd.DataFrame({'K':results['param_n_neighbors'].data,
              'Train Accuracy Average':results['mean_train_Accuracy'],
              'Train Accuracy Standard Error':results['std_train_Accuracy'],
              'Test Accuracy Average':results['mean_test_Accuracy'],
              'Test Accuracy Standard Error':results['std_test_Accuracy'],
              'Train Kappa Average':results['mean_train_Kappa'],
              'Train Kappa Standard Error':results['std_train_Kappa'],
              'Test Kappa Average':results['mean_test_Kappa'],
              'Test Kappa Standard Error':results['std_test_Kappa']})
```

```
##      K  Train Accuracy Average  ...  Test Kappa Average  Test Kappa Standard Error
## 0    1                0.654315  ...            0.144728                   0.137301
## 1    2                0.650784  ...            0.074590                   0.101229
## 2    3                0.666465  ...            0.151834                   0.073513
## 3    4                0.708932  ...            0.247408                   0.090333
## 4    5                0.718380  ...            0.288623                   0.090168
## 5    6                0.716509  ...            0.262517                   0.116245
## 6    7                0.720555  ...            0.291231                   0.102555
## 7    8                0.720766  ...            0.230787                   0.113243
## 8    9                0.717232  ...            0.270816                   0.107517
## 9   10                0.718170  ...            0.254045                   0.127403
## 10  11                0.722119  ...            0.292632                   0.101089
## 11  12                0.721080  ...            0.299281                   0.101347
## 12  13                0.723676  ...            0.301652                   0.106219
## 13  14                0.723989  ...            0.312110                   0.112813
## 14  15                0.724612  ...            0.284426                   0.122221
## 15  16                0.724923  ...            0.299349                   0.120607
## 16  17                0.725754  ...            0.312776                   0.119652
## 17  18                0.726686  ...            0.308376                   0.107192
## 18  19                0.723879  ...            0.325104                   0.100905
## 19  20                0.723881  ...            0.309256                   0.098674
## 20  21                0.724298  ...            0.313155                   0.084381
## 21  22                0.721495  ...            0.328060                   0.084876
## 22  23                0.723782  ...            0.324447                   0.089069
## 23  24                0.719937  ...            0.312622                   0.080163
## 24  25                0.720560  ...            0.323535                   0.090037
## 25  26                0.719627  ...            0.331236                   0.056184
## 26  27                0.721600  ...            0.335115                   0.069910
## 27  28                0.719524  ...            0.331585                   0.050655
## 28  29                0.722015  ...            0.334613                   0.070841
## 29  30                0.718897  ...            0.339948                   0.066611
## 
## [30 rows x 9 columns]
```

We use the best K = 30 on the data set to make predictions. The confusion matrix is below.


```python
pred_prob=knn_fit.predict_proba(train_X)[:,0]
pred_class=knn_fit.predict(train_X)

cm=pd.DataFrame(confusion_matrix(train_y, pred_class).T)
cm.rename(columns={0:'Actual CM',1:'Actual MM'},index={0:'Predicted CM',1:'Predicted MM'},inplace=True)
cm
```

```
##               Actual CM  Actual MM
## Predicted CM        546        192
## Predicted MM        107        225
```

Other evaluation metrics:


```python
print(classification_report(train_y, pred_class, digits=3))
```

```
##               precision    recall  f1-score   support
## 
##           CH      0.740     0.836     0.785       653
##           MM      0.678     0.540     0.601       417
## 
##    micro avg      0.721     0.721     0.721      1070
##    macro avg      0.709     0.688     0.693      1070
## weighted avg      0.716     0.721     0.713      1070
```

Note that in binary classification, recall of the positive class is also known as sensitivity. Recall of the negative class is specificity.

## KNN plots

I will make three plots here.

The first one is the CV accuracy and kappa plot.

#### R

Ggplot2 library is used.


```r
ggplot(results,aes(x=k)) +
  geom_ribbon(aes(ymin = Accuracy - AccuracySD, ymax = Accuracy + AccuracySD), fill = "#E69F00") +
  geom_ribbon(aes(ymin = Kappa - KappaSD, ymax = Kappa + KappaSD), fill = "#66FFCC") +
  geom_line(aes(y = Accuracy, colour = "Accuracy")) +
  geom_line(aes(y = Kappa, colour = "Kappa")) +
  geom_point(aes(y = Accuracy, colour = "Accuracy"),size=2) +
  geom_point(aes(y = Kappa, colour = "Kappa"),size=2) +
  geom_segment(aes(x=best_index[1],y=0,xend=best_index[1],yend=results[best_index[1],"Accuracy"],colour="Accuracy")) +
  geom_segment(aes(x=best_index[2],y=0,xend=best_index[2],yend=results[best_index[2],"Kappa"],colour="Kappa")) +
  annotate("text",x=best_index[1],y=results[best_index[1],"Accuracy"]+0.02,label=round(results[best_index[1],"Accuracy"],4)) +
  annotate("text",x=best_index[2],y=results[best_index[2],"Kappa"]+0.02,label=round(results[best_index[2],"Kappa"],4)) +
  labs(title="KNN CV Test Accuracy and Kappa", x="Number of nearest neighbors", y="CV evaluation score") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_colour_discrete(name = "Evaluation metrics")
```

{{< figure library="1" src="knn r 1.png" title="KNN CV Test Accuracy and Kappa" >}}

#### Python

Matplotlib package is used. The code is sourced from sklearn [demonstration](https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#plotting-the-result).


```python
plt.figure(figsize=(20,10))
plt.title("KNN CV Train and Test Accuracy and Kappa",
          fontsize=16)

plt.xlabel("K")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0, 30)
ax.set_ylim(0, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_n_neighbors'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
  for sample, style in (('train', '--'), ('test', '-')):
    sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
    sample_score_std = results['std_%s_%s' % (sample, scorer)]
    ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
    ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

  best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
  best_score = results['mean_test_%s' % scorer][best_index]

  # Plot a dotted vertical line at the best score for that scorer marked by x
  ax.plot([X_axis[best_index], ] * 2, [0, best_score],
          linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

  # Annotate the best score for that scorer
  ax.annotate("%0.2f" % best_score,
            (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()
```

{{< figure library="1" src="knn python 1.png" title="KNN CV Train and Test Accuracy and Kappa" >}}

The second plot is the ROC curve using the predictions made by the best K.

#### R

The basic plot command is used.


```r
roc = roc(train[, "response"], pred_prob)

plot(roc,legacy.axes = T)
mtext(paste("Area under the curve:", round(auc(roc),4)),side=3,line=2)
title(main="KNN ROC Curve",line=3)
```

{{< figure library="1" src="knn r 2.png" title="KNN ROC Curve" >}}

#### Python

Matplotlib package is used.


```python
train_y_bin=(np.array(train_y) == 'CH').astype(int)
fpr, tpr, th = roc_curve(train_y_bin, pred_prob)
roc_auc = auc(fpr,tpr)

plt.figure(figsize=(10,10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

{{< figure library="1" src="knn python 2.png" title="KNN ROC Curve" >}}

The last plot is the decision boundary plot for different stores (1, 2, 3, 4 and 7).

#### R

Contour plot from ggplot2 library is used.


```r
n_grid = 100
X6_range=seq(min(train[,"X6"]), max(train[,"X6"]), length.out=n_grid)
X7_range=seq(min(train[,"X7"]), max(train[,"X7"]), length.out=n_grid)
grid_X6_X7=expand.grid(X6=X6_range,X7=X7_range)
grid_1=data.frame(X1=sqrt(0.5),X2=0,X3=0,X4=0,X5=0,grid_X6_X7)
grid_2=data.frame(X1=0,X2=sqrt(0.5),X3=0,X4=0,X5=0,grid_X6_X7)
grid_3=data.frame(X1=0,X2=0,X3=sqrt(0.5),X4=0,X5=0,grid_X6_X7)
grid_4=data.frame(X1=0,X2=0,X3=0,X4=sqrt(0.5),X5=0,grid_X6_X7)
grid_5=data.frame(X1=0,X2=0,X3=0,X4=0,X5=sqrt(0.5),grid_X6_X7)

for (i in 1:5) {
  plot_name=paste("plot_",i,sep="")
  grid_name=paste("grid_",i,sep="")
  col_name=paste("X",i,sep="")
  tt=paste("KNN Decision Boundary for Store ",i,sep="")
  assign(plot_name,
         ggplot(data.frame(grid_X6_X7,prob=predict(knn_fit,get(grid_name),type = "prob")$CH), aes(X6, X7)) +
         stat_contour(aes(z=prob), breaks=c(0.5), size=0.7) +
         geom_point(aes(X6, X7, colour=response),data=train[train[,col_name]!=0,]) +
         labs(title=tt, x="SalePriceMM", y="SalePriceCH") +
         theme(plot.title = element_text(hjust = 0.5)) +
         scale_colour_discrete(name = "Purchase", labels=c("CH", "MM")) +
         geom_text_contour(aes(z = prob), breaks=c(0.5), stroke = 0.2))
}

grid.arrange(plot_1,plot_2,plot_3,plot_4,plot_5,ncol=2)
```

{{< figure library="1" src="knn r 3.png" title="KNN Decision Boundaries" >}}

#### Python

The contour plot in matplotlib package does not interpolate Z values as good as ggplot2 in R, so I choose the color plot (pcolormesh) in matplotlib to draw the decision regions.


```python
delta = 0.02
X6_grid, X7_grid = np.meshgrid(np.arange(min(train['X6'])-0.2,max(train['X6'])+0.2, delta),
                               np.arange(min(train['X7'])-0.2,max(train['X7'])+0.2, delta))
grid={}
grid['grid_1']=pd.DataFrame({'X1':0.5**0.5,'X2':0,'X3':0,'X4':0,'X5':0,'X6':X6_grid.ravel(),'X7':X7_grid.ravel()})
grid['grid_2']=pd.DataFrame({'X1':0,'X2':0.5**0.5,'X3':0,'X4':0,'X5':0,'X6':X6_grid.ravel(),'X7':X7_grid.ravel()})
grid['grid_3']=pd.DataFrame({'X1':0,'X2':0,'X3':0.5**0.5,'X4':0,'X5':0,'X6':X6_grid.ravel(),'X7':X7_grid.ravel()})
grid['grid_4']=pd.DataFrame({'X1':0,'X2':0,'X3':0,'X4':0.5**0.5,'X5':0,'X6':X6_grid.ravel(),'X7':X7_grid.ravel()})
grid['grid_5']=pd.DataFrame({'X1':0,'X2':0,'X3':0,'X4':0,'X5':0.5**0.5,'X6':X6_grid.ravel(),'X7':X7_grid.ravel()})

pred_prob_grid={}
pred_class_grid={}
prob_threshold=0.5
for i in grid:
    pred_prob_grid[i]=knn_fit.predict_proba(grid[i])[:,0]
    pred_class_grid[i]=np.array(["CH" if j>=prob_threshold else "MM" for j in pred_prob_grid[i]])
    
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

f, axarr = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(20, 20))
f.delaxes(axarr[2][1])

for idx,grid_ind,tt,tb_X in zip(product([0,1,2],[0, 1]),
                                grid,
                                ["KNN Decision Boundary for Store 1","KNN Decision Boundary for Store 2","KNN Decision Boundary for Store 3","KNN Decision Boundary for Store 4","KNN Decision Boundary for Store 7"],
                                ["X1","X2","X3","X4","X5"]):
    Z = pred_class_grid[grid_ind].reshape(X6_grid.shape)

    axarr[idx[0], idx[1]].pcolormesh(X6_grid, X7_grid, Z=='CH' ,cmap=cmap_light)
    axarr[idx[0], idx[1]].scatter(train.loc[train[tb_X]!=0,"X6"],
                                  train.loc[train[tb_X]!=0,"X7"],
                                  c=(train_y[train[tb_X]!=0]=='CH').astype(int),
                                  cmap=cmap_bold,
                                  edgecolor='k',
                                  s=30)
    axarr[idx[0], idx[1]].set_title(tt, fontsize=20)
    axarr[idx[0], idx[1]].set_xlim(X6_grid.min(), X6_grid.max())
    axarr[idx[0], idx[1]].set_ylim(X7_grid.min(), X7_grid.max())
    axarr[idx[0], idx[1]].set_xlabel("SalePriceMM", fontsize=15)
    axarr[idx[0], idx[1]].set_ylabel("SalePriceCH", fontsize=15)
    
plt.show()
```

{{< figure library="1" src="knn python 3.png" title="KNN Decision Boundaries" >}}
