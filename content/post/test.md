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

$$
\begin{aligned}
\underset{\widehat{W}}{\operatorname{argmax}}\;l(\widehat{W})
&=\ln(\prod_i(\frac{1}{1+e^{-\widehat{W_i}}})^{y_i}(1-\frac{1}{1+e^{-\widehat{W_i}}})^{1-y_i})\\\\\\
&=\ln(\prod_i(\frac{1}{1+e^{-\widehat{W_i}}})^{y_i}(\frac{1}{1+e^{\widehat{W_i}}})^{1-y_i})\\\\\\
&=-\sum_iy_i\ln(1+e^{-\widehat{W_i}})-\sum_i(1-y_i)\ln(1+e^{\widehat{W_i}})\\\\\\
\underset{\widehat{W}}{\operatorname{argmin}}\;LOSS(\widehat{W})
&=-l(\widehat{W})
=\sum_iy_i\ln(1+e^{-\widehat{W_i}})+(1-y_i)\ln(1+e^{\widehat{W_i}})
\end{aligned}
$$

Adding the regularization term $\sum\_k\pi(\widehat{f\_k})$, we have the objective function $obj(\widehat{W})=LOSS(\widehat{W})+\sum\_k\pi(\widehat{f\_k})$. $f\_k$ is the function maps observations $i$ to $w\_{ki}$ using tree $k$. $\pi(f\_k)$ measures the complexity of tree $k$.

In order to add tree $\widehat{f\_{(t)}}$ to the existing tree collection ${\widehat{f\_{(1)}},\widehat{f\_{(2)}},...,\widehat{f\_{(t-1)}}}$ at time $(t)$, we need to minimize the objective function $obj(\widehat{W^{(t)}})=LOSS(\widehat{W^{(t-1)}}+\widehat{f\_{(t)}(X)})+\pi(\widehat{f\_{(t)}})+constant$. $\widehat{W^{(t)}}$ is the leaf score mapped from our data $X$ using the newly added tree $f\_{(t)}$. Using Taylor expansin of the loss function up to the second order we have:

$$
\begin{aligned}
obj(\widehat{W^{(t)}})
&=LOSS(\widehat{W^{(t-1)}})+\sum_i\frac{\partial LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W_i^{(t-1)}}}\widehat{f_{(t)}(x_i)}+\frac{1}{2}\sum_i\frac{\partial^2 LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W_i^{(t-1)}}^2}{\widehat{f_{(t)}(x_i)}}^2+\pi(\widehat{f_{(t)}})+constant\\\\\\
&=\sum_i\frac{\partial LOSS(\widehat{W^{(t-1)}})}{\partial \widehat{W_i^{(t-1)}}}\widehat{f_{(t)}(x_i)}+\frac{1}{2}\sum_i\frac{\partial^2LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W_i^{(t-1)}}^2}{\widehat{f_{(t)}(x_i)}}^2+\pi(\widehat{f_{(t)}})+constant
\end{aligned}
$$


asd
