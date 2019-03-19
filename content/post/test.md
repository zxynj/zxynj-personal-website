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
&=\ln(\prod_i(\frac{1}{1+e^{-\widehat{W\_i}}})^{y\_i}(1-\frac{1}{1+e^{-\widehat{W\_i}}})^{1-y\_i})
\end{aligned}
$$

$$
\begin{aligned}
\underset{\widehat{W}}{\operatorname{argmax}}\;l(\widehat{W})
&=\ln(\prod_i(\frac{1}{1+e^{-\widehat{W_i}}})^{y_i}(1-\frac{1}{1+e^{-\widehat{W_i}}})^{1-y_i})\\\\\\
&=\ln(\prod_i(\frac{1}{1+e^{-\widehat{W_i}}})^{y_i}(\frac{1}{1+e^{\widehat{W_i}}})^{1-y_i})\\\\\\
&=-\sum_iy_i\ln(1+e^{-\widehat{W_i}})-\sum_i(1-y_i)\ln(1+e^{\widehat{W_i}})\\\\\\
\underset{\widehat{W}}{\operatorname{argmin}}\;LOSS(\widehat{W})&=-l(\widehat{W})=\sum_iy_i\ln(1+e^{-\widehat{W_i}})+(1-y_i)\ln(1+e^{\widehat{W_i}})
\end{aligned}
$$



