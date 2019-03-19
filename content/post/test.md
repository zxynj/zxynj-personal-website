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
&=\ln(\prod\_i(\frac{1}{1+e^{-\widehat{W\_i}}})^{y\_i}(1-\frac{1}{1+e^{-\widehat{W\_i}}})^{1-y\_i})
=\ln(\prod\_i(\frac{1}{1+e^{-\widehat{W\_i}}})^{y\_i}(\frac{1}{1+e^{\widehat{W\_i}}})^{1-y\_i})\\\\\\
&=-\sum\_iy\_i\ln(1+e^{-\widehat{W\_i}})-\sum\_i(1-y\_i)\ln(1+e^{\widehat{W\_i}})\\\\\\
\underset{\widehat{W}}{\operatorname{argmin}}\;LOSS(\widehat{W})&=-l(\widehat{W})=\sum\_iy\_i\ln(1+e^{-\widehat{W\_i}})+(1-y\_i)\ln(1+e^{\widehat{W\_i}})
\end{aligned}
$$



