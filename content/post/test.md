---
title: "test"
author: "Xinyu Zhang"
date: "March 18, 2019"
output:
  html_document:
    keep_md: true
    toc: true
    number_sections: true
    toc_float: true
---



This is inline: $\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon$

# Xgboost Algorithm for binary classification

In generalized linear regression (GLM), we have $g(E[Y])=X\beta$ where the right hand side is the linear combination of predictors. In xgboost, it puts predictors into multiple trees (rounds) to come up with leaf score (weight) $w\_{ki}$ for each tree $k$ and observation $i$. $w\_{ki}$ is summed over all trees so that $w\_{\cdot{i}}$ is the final leaf score for observation $i$. The vector $W$ formed by these $W\_i=w\_{\cdot{i}}$ is what appears on the right hand side of $g(E[Y])=X\beta$ instead of $X\beta$. In GLM, the common link function for a response variable following a bernoulli distribution is the logit canonical link. Xgboost also uses the logit link when specifying the "binary:logistic" objective. In GLM, we maximize the log likelihood of the estimator $\widehat{\beta}$ to find the desired $\widehat{\beta}$. In xgboost, the log likelihood of $\widehat{W}$ is maximized which is equivalent to minimizing the loss function $LOSS(\widehat{W})=-l(\widehat{W})$.

$$\left [ - \frac{\hbar^2}{2 m} \frac{\partial^2}{\partial x^2} + V \right ] \Psi = i \hbar \frac{\partial}{\partial t} \Psi$$
