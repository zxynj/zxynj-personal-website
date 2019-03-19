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



In this post, I will explain xgoost algorithm and manually solve a simple binary classification problem using the algorithm.

# Xgboost Algorithm for binary classification

In generalized linear regression (GLM), we have $g(E[Y])=X\beta$ where the right hand side is the linear combination of predictors. In xgboost, it puts predictors into multiple trees (rounds) to come up with leaf score (weight) $w\\_{ki}$ for each tree $k$ and observation $i$. $w\\_{ki}$ is summed over all trees so that
