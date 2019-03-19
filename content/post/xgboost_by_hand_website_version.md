---
title: "Explain Xgboost Algorithm by Manually Calculating a Binary Classification Problem"
author: "Xinyu Zhang"
date: "March 18, 2019"
output:
  html_document:
    keep_md: true
    toc: true
    number_sections: true
    toc_float: true
---



In this post, I will explain xgoost algorithm and manually solve a simple binary classification problem using the algorithm.

# Xgboost Algorithm for binary classification

In generalized linear regression (GLM), we have $g(E[Y])=X\beta$ where the right hand side is the linear combination of predictors. In xgboost, it puts predictors into multiple trees (rounds) to come up with leaf score (weight) $w\_{ki}$ for each tree $k$ and observation $i$. $w\_{ki}$ is summed over all trees so that $w\_{\cdot{i}}$ is the final leaf score for observation $i$. The vector $W$ formed by these $W\_i=w\_{\cdot{i}}$ is what appears on the right hand side of $g(E[Y])=X\beta$ instead of $X\beta$. In GLM, the common link function for a response variable following a bernoulli distribution is the logit canonical link. Xgboost also uses the logit link when specifying the "binary:logistic" objective. In GLM, we maximize the log likelihood of the estimator $\widehat{\beta}$ to find the desired $\widehat{\beta}$. In xgboost, the log likelihood of $\widehat{W}$ is maximized which is equivalent to minimizing the loss function $LOSS(\widehat{W})=-l(\widehat{W})$.

$$
\small
\begin{aligned}
\underset{\widehat{W}}{\operatorname{argmax}}\;l(\widehat{W})
&=\ln(\prod\_i(\frac{1}{1+e^{-\widehat{W\_i}}})^{y\_i}(1-\frac{1}{1+e^{-\widehat{W\_i}}})^{1-y\_i})\\\\\\
&=\ln(\prod\_i(\frac{1}{1+e^{-\widehat{W\_i}}})^{y\_i}(\frac{1}{1+e^{\widehat{W\_i}}})^{1-y\_i})\\\\\\
&=-\sum\_iy\_i\ln(1+e^{-\widehat{W\_i}})-\sum\_i(1-y\_i)\ln(1+e^{\widehat{W\_i}})\\\\\\
\underset{\widehat{W}}{\operatorname{argmin}}\;LOSS(\widehat{W})
&=-l(\widehat{W})
=\sum\_iy\_i\ln(1+e^{-\widehat{W\_i}})+(1-y\_i)\ln(1+e^{\widehat{W\_i}})
\end{aligned}
$$

Adding the regularization term $\sum\_k\pi(\widehat{f\_k})$, we have the objective function $obj(\widehat{W})=LOSS(\widehat{W})+\sum\_k\pi(\widehat{f\_k})$. $f\_k$ is the function maps observations $i$ to $w\_{ki}$ using tree $k$. $\pi(f\_k)$ measures the complexity of tree $k$.

In order to add tree $\widehat{f\_{(t)}}$ to the existing tree collection ${\widehat{f\_{(1)}},\widehat{f\_{(2)}},...,\widehat{f\_{(t-1)}}}$ at time $(t)$, we need to minimize the objective function $obj(\widehat{W^{(t)}})=LOSS(\widehat{W^{(t-1)}}+\widehat{f\_{(t)}(X)})+\pi(\widehat{f\_{(t)}})+constant$. $\widehat{W^{(t)}}$ is the leaf score mapped from our data $X$ using the newly added tree $f\_{(t)}$. Using Taylor expansin of the loss function up to the second order we have:

$$
\small
\begin{aligned}
obj(\widehat{W^{(t)}})
&=LOSS(\widehat{W^{(t-1)}})+\sum\_i\frac{\partial LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W\_i^{(t-1)}}}\widehat{f\_{(t)}(x\_i)}\\\\\\
&\quad+\frac{1}{2}\sum\_i\frac{\partial^2 LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W\_i^{(t-1)}}^2}{\widehat{f\_{(t)}(x\_i)}}^2+\pi(\widehat{f\_{(t)}})+constant\\\\\\
&=\sum\_i\frac{\partial LOSS(\widehat{W^{(t-1)}})}{\partial \widehat{W\_i^{(t-1)}}}\widehat{f\_{(t)}(x\_i)}+\frac{1}{2}\sum\_i\frac{\partial^2LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W\_i^{(t-1)}}^2}{\widehat{f\_{(t)}(x\_i)}}^2\\\\\\
&\quad+\pi(\widehat{f\_{(t)}})+constant
\end{aligned}
$$

Let's derive the first (gradient) and second (hessian) order derivative of the loss function for the binary classification problem.

$$
\small
\begin{aligned}
\frac{\partial LOSS(\widehat{W^{(t)}})}{\partial \widehat{W\_i^{(t)}}}
&=\frac{\partial y\_i\ln(1+e^{-\widehat{W\_i^{(t)}}})+(1-y\_i)\ln(1+e^{\widehat{W\_i^{(t)}}})}{\partial \widehat{W\_i^{(t)}}}\\\\\\
&=\frac{-y\_i+(1-y\_i)e^{\widehat{W\_i^{(t)}}}}{1+e^{\widehat{W\_i^{(t)}}}}\\\\\\
&=\frac{-(y\_ie^{-\widehat{W\_i^{(t)}}}+y\_i)+1}{e^{-\widehat{W\_i^{(t)}}}+1}\\\\\\
&=\frac{1}{1+e^{-\widehat{W\_i^{(t)}}}}-y\_i\\\\\\
&=\widehat{p\_i^{(t)}}-y\_i\\\\\\
\\\\\\
\frac{\partial^2LOSS(\widehat{W^{(t)}})}{\partial\widehat{W\_i^{(t)}}^2}
&=-(1+e^{-\widehat{W\_i^{(t)}}})^{-2}(-e^{-\widehat{W\_i^{(t)}}})\\\\\\
&=\frac{1}{1+e^{-\widehat{W\_i^{(t)}}}}\cdot\frac{e^{-\widehat{W\_i^{(t)}}}}{1+e^{-\widehat{W\_i^{(t)}}}}\\\\\\
&=\frac{1}{1+e^{-\widehat{W\_i^{(t)}}}}\cdot(1-\frac{1}{1+e^{-\widehat{W\_i^{(t)}}}})\\\\\\
&=\widehat{p\_i^{(t)}}\cdot(1-\widehat{p\_i^{(t)}})
\end{aligned}
$$

$\widehat{p\_i^{(t)}}$ is the predicted success probability after adding tree $f\_{(t)}$. Plug the gradient and the hessian into the objective function we get:

$$
\small
\begin{aligned}
obj(\widehat{W^{(t)}})
&=\sum\_i\frac{\partial LOSS(\widehat{W^{(t-1)}})}{\partial \widehat{W\_i^{(t-1)}}}\widehat{f\_{(t)}(x\_i)}+\frac{1}{2}\sum\_i\frac{\partial^2LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W\_i^{(t-1)}}^2}{\widehat{f\_{(t)}(x\_i)}}^2\\\\\\
&\quad+\pi(\widehat{f\_{(t)}})+constant\\\\\\
&=\sum\_i(\widehat{p\_i^{(t-1)}}-y\_i)\cdot \widehat{f\_{(t)}(x\_i)}+\frac{1}{2}\sum\_i\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})\cdot {\widehat{f\_{(t)}(x\_i)}}^2\\\\\\
&\quad+\pi(\widehat{f\_{(t)}})+constant
\end{aligned}
$$

Before we define the tree complexity measure. Let me introduce another representation of $W\_i^{(t)}$, the leaf score of observation $i$ in tree $f\_{(t)}$. Assume tree $f\_{k}$ has $L$ leaves and $V^{k}$ is a vector of length $L$ representing the leaf score of the tree. Each leaf node $l$ has leaf score $V\_l^{k}$. Let's also assume function $q(i)$ maps the observation $i$ to leaf $l\;\epsilon\;\{1,2,...,L\}$ of the tree. Then we can write $W\_i^{(t)}$ as $V\_{q(i)}^{(t)}$.

The complexity measure of a tree $f\_k$ is defined as $\pi(f\_k)=\gamma L+\frac{1}{2}\lambda\sum\_lV\_l^2$. $\gamma$ (gamma) and $\lambda$ (lambda) are tuning parameters. It takes both the number of leaves and the leaf score in to account. Notice the sum of squares in the second part is not the L2 norm of $W$, but the L2 norm of $V$ instead.

Finally, we put it all together:

$$
\small
\begin{aligned}
obj(\widehat{V^{(t)}})
&=\sum\_i(\widehat{p\_i^{(t-1)}}-y\_i)\cdot \widehat{f\_{(t)}(x\_i)}+\frac{1}{2}\sum\_i\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})\cdot {\widehat{f\_{(t)}(x\_i)}}^2\\\\\\
&\quad+\gamma L\_{(t)}+\frac{1}{2}\lambda\sum\_l{\widehat{V\_l^{(t)}}}^2+constant\\\\\\
&=\sum\_l[(\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}-y\_i)\cdot \widehat{V\_l^{(t)}}+\frac{1}{2}(\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda)\cdot{\widehat{V\_l^{(t)}}}^2]\\\\\\
&\quad+\gamma L\_{(t)}+constant
\end{aligned}
$$

$\widehat{V\_l^{(t)}}$ is the leaf score vector of the estimated tree $\widehat{f\_{(t)}}$. $L\_{(t)}$ is the number of leaves in the estimated tree. $I\_l$ is the collection of index of observations who are mapped to leaf $l$ of the estimated tree.

Since $\widehat{V\_l^{(t)}}$ are independent to each other for different value of $l$ and the objective function is a convex quadratic function of $\widehat{V\_l^{(t)}}$, the maximum $-\frac{1}{2}\sum\_l\frac{(\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}-y\_i)^2}{\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}+\gamma L\_{(t)}$ is attained at $\widehat{V\_l^{(t)}}=-\frac{\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}-y\_i}{\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}$.

At this point, we know how to calculate the best leaf score if we know the mapping from observations to leaves in each estimated tree. But how do we construct each tree? Imaging if we split a leaf $l$ into left leaf $l\_L$ and right leaf $l\_R$. Then the reduce (Gain) in the objective function is:

$$
\small
\begin{aligned}
Gain&=-\frac{1}{2}\frac{\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}-y\_i}{\sum\_{i\in I\_l}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}+\gamma L\_{(t)}\\\\\\
&\quad-(-\frac{1}{2}\frac{\sum\_{i\in{I\_{l\_L}}}\widehat{p\_i^{(t-1)}}-y\_i}{\sum\_{i\in {I\_{l\_L}}}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}-\frac{1}{2}\frac{\sum\_{i\in{I\_{l\_R}}}\widehat{p\_i^{(t-1)}}-y\_i}{\sum\_{i\in {I\_{l\_R}}}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}+\gamma (L\_{(t)}+1))\\\\\\
&=\frac{1}{2}[\frac{\sum\_{i\in{I\_{l\_L}}}\widehat{p\_i^{(t-1)}}-y\_i}{\sum\_{i\in {I\_{l\_L}}}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}+\frac{\sum\_{i\in{I\_{l\_R}}}\widehat{p\_i^{(t-1)}}-y\_i}{\sum\_{i\in {I\_{l\_R}}}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}\\\\\\
&\quad-\frac{\sum\_{i\in {I\_{l\_L}}}\widehat{p\_i^{(t-1)}}-y\_i+\sum\_{i\in {I\_{l\_R}}}\widehat{p\_i^{(t-1)}}-y\_i}{\sum\_{i\in I\_{l\_L}}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\sum\_{i\in I\_{l\_R}}\widehat{p\_i^{(t-1)}}\cdot(1-\widehat{p\_i^{(t-1)}})+\lambda}]-\gamma
\end{aligned}
$$

Split is allowed only when we have a positive gain.

# Match xgboost result with manual calculation

The data set I used is the agaricus data set in xgboost library. To make everything simpler, I only used the first 2000 observations with the response variable = 1 and the first 3000 observations with the response variable = 0. Also, only the first three predictors (cap-shape=bell, cap-shape=conical and cap-shape=convex) are used in the model.


```r
library(xgboost)
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')

train_ind=c(which(agaricus.train$label=="1")[1:2000],which(agaricus.train$label=="0")[1:3000])
train_X = agaricus.train$data[train_ind,1:3]
train_y = agaricus.train$label[train_ind]
train_data = xgb.DMatrix(train_X, label = train_y, missing = NA)
train_X@Dimnames[[2]]
```

```
## [1] "cap-shape=bell"    "cap-shape=conical" "cap-shape=convex"
```

```r
test_X = agaricus.test$data[1:10,1:3]
test_y = agaricus.test$label[1:10]
xgb.test.data = xgb.DMatrix(test_X, label = test_y, missing = NA)
```

The following parameters are used:

eta = 1. It is the shrinkage parameter multiplied to the leaf score after finishing each tree.

gamma = 0. It is the $\gamma$ in our formula.

max.depth = 3. It is the maxinum depth of each three beside the root node.

min_child_weight = 1. It is the minimum sum of the hessian required in a possible child node for a certrain node. Sum of hessian is a measure of the number of observations. It is exactly the number of observations when the loss function is the residual sum of squares and no regularization term. The Cover in the tree plot below is same as sum of hessian.

max_delta_step = 0. The leaf score is capped by max_delta_step before getting multiplied by eta. It is most effective in situations like extreme imbalanced logistic regression. In which case, the hessian is almost 0 and the leaf nodes are nearly infinite, so eta is not enough. 0 means no capping.
l
ambda = 1. It is the $\lambda$ in our formula.

nround = 2. It is the number of trees to grow or number of iterations.

objective = "binary:logistic". It is to specify the objective function is what we used in our formula.

base_score = 0.5. It is the initial success probability. Xgboost will drive it to a more reasonable leaf score.

eval_metric = "error". Misclassification rate is used for evaluation.

verbose = 0. It tells xgboost not to print "error" after finishing each tree.


```r
bst = xgboost(data = train_data,
              eta = 1,
              gamma = 0,
              max.depth = 3,
              min_child_weight = 1,
              max_delta_step = 0,
              lambda = 1,
              nround = 2,
              objective = "binary:logistic",
              base_score = 0.5,
              eval_metric = "error",
              verbose = 0)

#xgb.model.dt.tree(train_X@Dimnames[[2]], model = bst)
xgb.plot.tree(model=bst)
```

<!--html_preserve--><div id="htmlwidget-8d09a06382d67c5b3542" style="width:672px;height:480px;" class="grViz html-widget"></div>
<script type="application/json" data-for="htmlwidget-8d09a06382d67c5b3542">{"x":{"diagram":"digraph {\n\ngraph [layout = \"dot\",\n       rankdir = \"LR\"]\n\nnode [color = \"DimGray\",\n      style = \"filled\",\n      fontname = \"Helvetica\"]\n\nedge [color = \"DimGray\",\n     arrowsize = \"1.5\",\n     arrowhead = \"vee\",\n     fontname = \"Helvetica\"]\n\n  \"1\" [label = \"Tree 1\ncap-shape=bell\nCover: 1189.18127\nGain: 4.80300188\", shape = \"rectangle\", fontcolor = \"black\", fillcolor = \"Beige\"] \n  \"2\" [label = \"cap-shape=convex\nCover: 1147.43542\nGain: 0.0110983625\", shape = \"rectangle\", fontcolor = \"black\", fillcolor = \"Beige\"] \n  \"3\" [label = \"Leaf\nCover: 41.745903\nValue: -0.346059561\", shape = \"oval\", fontcolor = \"black\", fillcolor = \"Khaki\"] \n  \"4\" [label = \"Leaf\nCover: 524.148438\nValue: -0.00805134326\", shape = \"oval\", fontcolor = \"black\", fillcolor = \"Khaki\"] \n  \"5\" [label = \"Leaf\nCover: 623.286987\nValue: -0.00180733623\", shape = \"oval\", fontcolor = \"black\", fillcolor = \"Khaki\"] \n  \"6\" [label = \"Tree 0\ncap-shape=bell\nCover: 1250\nGain: 75.0913696\", shape = \"rectangle\", fontcolor = \"black\", fillcolor = \"Beige\"] \n  \"7\" [label = \"cap-shape=convex\nCover: 1183.25\nGain: 9.77643013\", shape = \"rectangle\", fontcolor = \"black\", fillcolor = \"Beige\"] \n  \"8\" [label = \"Leaf\nCover: 66.75\nValue: -1.4243542\", shape = \"oval\", fontcolor = \"black\", fillcolor = \"Khaki\"] \n  \"9\" [label = \"Leaf\nCover: 549.75\nValue: -0.438492954\", shape = \"oval\", fontcolor = \"black\", fillcolor = \"Khaki\"] \n  \"10\" [label = \"Leaf\nCover: 633.5\nValue: -0.255319148\", shape = \"oval\", fontcolor = \"black\", fillcolor = \"Khaki\"] \n\"1\"->\"2\" [label = \"< -9.53674316e-07\", style = \"bold\"] \n\"2\"->\"4\" [label = \"< -9.53674316e-07\", style = \"bold\"] \n\"6\"->\"7\" [label = \"< -9.53674316e-07\", style = \"bold\"] \n\"7\"->\"9\" [label = \"< -9.53674316e-07\", style = \"bold\"] \n\"1\"->\"3\" [style = \"bold\", style = \"solid\"] \n\"2\"->\"5\" [style = \"solid\", style = \"solid\"] \n\"6\"->\"8\" [style = \"solid\", style = \"solid\"] \n\"7\"->\"10\" [style = \"solid\", style = \"solid\"] \n}","config":{"engine":"dot","options":null}},"evals":[],"jsHooks":[]}</script><!--/html_preserve-->

The Gain is same as our Gain formula but without $\frac{1}{2}$.

The Cover is the sum of hesssian in that node as mentioned earlier.

The Value is the leaf score in that leaf.

Now let's manually perform the xgboost. We first initiate the estimated success probability as 0.5. It will only be updated when a tree is finished not at each node in the tree. Then we calculate and report the Gain, node Cover, child node Cover, leaf score and leaf success probability if a split is allowed. Since all of our predictors are binary, I will assign the predictor = 0 as the left child node and the predictor = 1 as the right child node.


```r
p = rep(0.5,nrow(train_X))

L = which(train_X[,'cap-shape=bell']==0)
R = which(train_X[,'cap-shape=bell']==1)

pL = p[L]
pR = p[R]

yL = train_y[L]
yR = train_y[R]

GL = sum(pL-yL)
GR = sum(pR-yR)
G = GL+GR

HL = sum(pL*(1-pL))
HR = sum(pR*(1-pR))
H = HL+HR

gain = (GL^2/(HL+1)+GR^2/(HR+1)-G^2/(H+1))

WL = -GL/(HL+1)
WR = -GR/(HR+1)
pL_new = 1/(1+exp(-WL))
pR_new = 1/(1+exp(-WR))

cat('"cap-shape=bell" is used to make the split. The gain is ',
    gain,
    ".\nThe node cover is ",
    H,
    ".\nThe left child node cover is ",
    HL,
    ".\nThe right child node cover is ",
    HR,
    ".\nThe left leaf score is ",
    WL,
    ".\nThe right leaf score is ",
    WR,
    ".\nThe left leaf success probability is ",
    pL_new,
    ".\nThe right leaf success probability is ",
    pR_new,
    ".",
    sep = "")
```

```
## "cap-shape=bell" is used to make the split. The gain is 75.09137.
## The node cover is 1250.
## The left child node cover is 1183.25.
## The right child node cover is 66.75.
## The left leaf score is -0.340722.
## The right leaf score is -1.424354.
## The left leaf success probability is 0.4156341.
## The right leaf success probability is 0.1939799.
```

Let's check out the report of other possible splits.


```
## "cap-shape=conical" is used to make the split. The gain is 1.547051.
## The node cover is 1250.
## The left child node cover is 1249.5.
## The right child node cover is 0.5.
## The left leaf score is -0.4006397.
## The right leaf score is 0.6666667.
## The left leaf success probability is 0.4011586.
## The right leaf success probability is 0.6607564.
```

Since the right child node cover is smaller than the min_child_weight = 1, the split is not allowed.


```
## "cap-shape=convex" is used to make the split. The gain is 26.5321.
## The node cover is 1250.
## The left child node cover is 616.5.
## The right child node cover is 633.5.
## The left leaf score is -0.5473684.
## The right leaf score is -0.2553191.
## The left leaf success probability is 0.3664752.
## The right leaf success probability is 0.4365147.
```

Since "cap-shape=bell" gives the largest gain among all the allowed splits. We will use it to make the initial split of our tree.

Let's see the report for all possible splits for "cap-shape=bell" = 0.


```
## "cap-shape=conical" is used to make the split. The gain is 1.407313.
## The node cover is 1183.25.
## The left child node cover is 1182.75.
## The right child node cover is 0.5.
## The left leaf score is -0.3417107.
## The right leaf score is 0.6666667.
## The left leaf success probability is 0.415394.
## The right leaf success probability is 0.6607564.
```

Since the right child node cover is smaller than 1, the split is not allowed.


```
## "cap-shape=convex" is used to make the split. The gain is 9.776436.
## The node cover is 1183.25.
## The left child node cover is 549.75.
## The right child node cover is 633.5.
## The left leaf score is -0.438493.
## The right leaf score is -0.2553191.
## The left leaf success probability is 0.3921001.
## The right leaf success probability is 0.4365147.
```

We will use this only allowed split.

Let's see the report for all possible split for "cap-shape=bell" = 0 and "cap-shape=convex" = 0.


```
## "cap-shape=conical" is used to make the split. The gain is 1.642492.
## The node cover is 549.75.
## The left child node cover is 549.25.
## The right child node cover is 0.5.
## The left leaf score is -0.4407088.
## The right leaf score is 0.6666667.
## The left leaf success probability is 0.3915721.
## The right leaf success probability is 0.6607564.
```

Since the right child node cover is smaller than 1, the split is not allowed.

There are no observations with "cap-shape=conical" = 1 when "cap-shape=bell" = 1, so the child node cover of the split using "cap-shape=conical" is smaller than 1 and the split is not allowed. Same goes for spitting "cap-shape=bell" = 1 with "cap-shape=convex".

Therefore, for the first tree we will use the right child node of the initial split with "cap-shape=bell" as a leaf. Its leaf score is -1.424354 and its leaf success probability is 0.1939799. We will also use the two child nodes of splitting "cap-shape=bell" = 0 as leaves. The left leaf score is -0.438493. The right leaf score is -0.2553191. The left leaf success probability is 0.3921001. The right leaf success probability is 0.4365147.

Before we construct the second tree. We will update the estimated success probability using the first tree.


```r
p[which(train_X[,'cap-shape=bell']==1)] = 0.1939799
p[which(train_X[,'cap-shape=bell']==0 & train_X[,'cap-shape=convex']==0)] = 0.3921001
p[which(train_X[,'cap-shape=bell']==0 & train_X[,'cap-shape=convex']==1)] = 0.4365147
```

Let's see the report of all the possible splits of the second tree.


```
## "cap-shape=bell" is used to make the split. The gain is 4.803007.
## The node cover is 1189.181.
## The left child node cover is 1147.435.
## The right child node cover is 41.7459.
## The left leaf score is -0.004664058.
## The right leaf score is -0.3460597.
## The left leaf success probability is 0.498834.
## The right leaf success probability is 0.4143383.
```


```
## "cap-shape=conical" is used to make the split. The gain is 1.043546.
## The node cover is 1189.181.
## The left child node cover is 1188.705.
## The right child node cover is 0.4767152.
## The left leaf score is -0.01795807.
## The right leaf score is 0.8233136.
## The left leaf success probability is 0.4955106.
## The right leaf success probability is 0.6949393.
```

Since the right child node cover is smaller than 1, the split is not allowed.


```
## "cap-shape=convex" is used to make the split. The gain is 0.2991243.
## The node cover is 1189.181.
## The left child node cover is 565.8943.
## The right child node cover is 623.287.
## The left leaf score is -0.03355256.
## The right leaf score is -0.001807261.
## The left leaf success probability is 0.4916126.
## The right leaf success probability is 0.4995482.
```

Since "cap-shape=bell" gives the largest gain among all the allowed splits. We will use it to make the initial split of our second tree.

Let's see the report for all possible splits for "cap-shape=bell" = 0.


```
## "cap-shape=conical" is used to make the split. The gain is 1.013628.
## The node cover is 1147.435.
## The left child node cover is 1146.959.
## The right child node cover is 0.4767152.
## The left leaf score is -0.005725092.
## The right leaf score is 0.8233136.
## The left leaf success probability is 0.4985687.
## The right leaf success probability is 0.6949393.
```

Since the right child node cover is smaller than 1, the split is not allowed.


```
## "cap-shape=convex" is used to make the split. The gain is 0.01109842.
## The node cover is 1147.435.
## The left child node cover is 524.1484.
## The right child node cover is 623.287.
## The left leaf score is -0.008051286.
## The right leaf score is -0.001807261.
## The left leaf success probability is 0.4979872.
## The right leaf success probability is 0.4995482.
```

We will use this only allowed split.

Let's see the report for all possible split for "cap-shape=bell" = 0 and "cap-shape=convex" = 0.


```
## "cap-shape=conical" is used to make the split. The gain is 1.023428.
## The node cover is 524.1484.
## The left child node cover is 523.6717.
## The right child node cover is 0.4767152.
## The left leaf score is -0.01037586.
## The right leaf score is 0.8233136.
## The left leaf success probability is 0.4974061.
## The right leaf success probability is 0.6949393.
```

Since the right child node cover is smaller than 1, the split is not allowed.

There are no observations with "cap-shape=conical" = 1 when "cap-shape=bell" = 1, so the child node cover of the split using "cap-shape=conical" is smaller than 1 and the split is not allowed. Same goes for spitting "cap-shape=bell" = 1 with "cap-shape=convex".

Therefore, for the first tree we will use the right child node of the initial split with "cap-shape=bell" as a leaf. Its leaf score is 41.7459 and its leaf success probability is 0.4143383. We will also use the two child nodes of splitting "cap-shape=bell" = 0 as leaves. The left leaf score is -0.008051286. The right leaf score is -0.001807261. The left leaf success probability is 0.4979872. The right leaf success probability is 0.4995482.

All gain, cover, leaf score and leaf success probability match the tree plot using the xgboost library.
