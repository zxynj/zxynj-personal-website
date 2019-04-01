---
title: "Two Sample Bootstrap Test in R"
author: "Xinyu Zhang"
date: "March 30, 2019"
output:
  html_document:
    keep_md: true
---



In this post, I will provide a bootstrap algorithm to test the difference between two population means and implement it in R.

## A little discussion before the code

We all know how to do two sample test when the two populations follow normal distribution or when the sample size is large enough for the central limit theorem to kick in. What should we do if the two populations follow different distributions and the sample size is not large enough? How about using a nonparametric test such as Wilcoxon rank-sum test (Mann-Whitney U test)? Wilcoxon rank-sum test has the assumption that the distribution of the two population needs to be the same under the null hypothesis. See the [Wikipedia link](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Assumptions_and_formal_statement_of_hypotheses). So, this is a no go. What about permutation test? Permutation test requires the label of the two groups to be exchangeable under the null hypothesis. See the [Wikipedia link](https://en.wikipedia.org/wiki/Resampling_(statistics)#Permutation_tests). We can't use this either. Finally, let's turn to bootstrap hypothesis test which does not make assumptions about the form of the distribution of the two populations.

First we need a test statistic for the following hypothesis test:

$$
\begin{aligned}
H\_0
&:\mu\_1-\mu\_2=\mu\\\\\\
H\_1
&:\mu\_0-\mu\_1>\mu\\\\\\
\end{aligned}
$$

A reasonable test statistic is a standardized sample mean difference. We will use the same [test statistics](https://en.wikipedia.org/wiki/Welch%27s_t-test#Calculations) in two sample t test with unequal variance since the two populations can have different variances. Say we have two samples: $x\_{11},x\_{12},...,x\_{1m}$ and $x\_{21},x\_{22},...,x\_{2n}$. $\overline{X\_1}$ and $\overline{X\_2}$ are the sample means. $s\_1^2$ and $s\_2^2$ are the sample variances. $m$ and $n$ are the sample sizes. The test statistics is:

$$
test\;statistic=\frac{\overline{X\_1}-\overline{X\_2}-\mu}{\sqrt{\frac{s\_1^2}{m}+\frac{s\_2^2}{n}}}
$$

Then we will bootstrap $s$ times to simulate the distribution of our test statistics under the null hypothesis. Each time we will generate a pair of sample $X\_1$ and $X\_2$ under the null hypothesis. Then we calculate the test statistics from the two samples until we have a nice distribution of the test statistics. To generate pairs of sample $X\_1$ and $X\_2$ under the null hypothesis, we need to find a way to shift their sample mean difference to $\mu$ without changing their distribution (or their sample variances at least). One way to do this is first we center both original samples. Then we shift the first sample by $\mu$ before our resampling. The modified original samples are $x\_{1i}-\overline{X\_1}+\mu$ and $x\_{2j}-\overline{X\_2}$ for $i$ in $\{1,2,...,m\}$ and $j$ in $\{1,2,...,n\}$

Finally, we can use the simulated test statistics to come up with a critical region so that we can make decision for our test.

## R function and example


```r
bs.two.sample.test = function(x, y,
                              alternative = c("two.sided", "less", "greater"),
                              mu = 0, nrep = 1000, parallel = FALSE) {
  ts.original = (mean(x) - mean(y) - mu) /
    (sd(x)/sqrt(length(x)) + sd(y)/sqrt(length(y)))
  x.sample = matrix(sample(scale(x, scale = FALSE) + mu, length(x) * nrep, replace=TRUE), ncol = nrep)
  y.sample = matrix(sample(scale(y, scale = FALSE), length(y) * nrep, replace=TRUE), ncol = nrep)
  if (parallel == FALSE) {
    ts.dist.sim = sapply(1:nrep, function(t) (mean(x.sample[,t], na.rm = TRUE) - mean(y.sample[,t], na.rm = TRUE) - mu) /
                           (sd(x.sample[,t], na.rm = TRUE) / sqrt(length(x)) + sd(y.sample[,t], na.rm = TRUE) / sqrt(length(y))))
  }
  else {
    library(parallel)
    cl = makeCluster(detectCores()/2)
    clusterExport(cl = cl, varlist=c("x.sample", "y.sample", "mu", "nrep", "x", "y"), envir=environment())
    ts.dist.sim = parSapply(cl = cl, 1:nrep, function(t) (mean(x.sample[,t], na.rm = TRUE) - mean(y.sample[,t], na.rm = TRUE) - mu) /
                              (sd(x.sample[,t], na.rm = TRUE) / sqrt(length(x)) + sd(y.sample[,t], na.rm = TRUE) / sqrt(length(y))))
    stopCluster(cl)
  }
  if (alternative == "two.sided") {
    p.value = 2 * min(mean(ts.dist.sim >= ts.original), mean(ts.dist.sim <= ts.original))
    cat("\n", "   Two Sample Bootstrap Test\n","\n",
        "data: ", deparse(substitute(x)), " and ", deparse(substitute(y)), "\n",
        nrep, " replications\n",
        "test statistic of the original sample: ", ts.original, ", p-value: ", p.value, "\n",
        "alternative hypothesis: true difference in means is not equal to ", mu, "\n",
        sep = "")
  }
  else if (alternative == "less") {
    p.value = mean(ts.dist.sim <= ts.original)
    cat("\n", "   Two Sample Bootstrap Test\n","\n",
        "data: ", deparse(substitute(x)), " and ", deparse(substitute(y)), "\n",
        nrep, " replications\n",
        "test statistic of the original sample: ", ts.original, ", p-value: ", p.value, "\n",
        "alternative hypothesis: true difference in means is less than ", mu, "\n",
        sep = "")
  }
  else if (alternative == "greater") {
    p.value = mean(ts.dist.sim >= ts.original)
    cat("\n", "   Two Sample Bootstrap Test\n","\n",
        "data: ", deparse(substitute(x)), " and ", deparse(substitute(y)), "\n",
        nrep, " replications\n",
        "test statistic of the original sample: ", ts.original, ", p-value: ", p.value, "\n",
        "alternative hypothesis: true difference in means is greater than ", mu, "\n",
        sep = "")
  }
  else {
    stop("Please use the supported value for alternative.\n")
  }
  return(invisible(list(ts.original = ts.original, p.value = p.value, ts.dist.sim = ts.dist.sim)))
  }
```

**Arguments**

 - *x*: a numeric vector of the first sample.

 - *y*: numeric vector of the second sample.

 - *alternative*: character string specifying the alternative hypothesis, must be one of "two.sided" (default), "greater" or "less".

 - *mu*: a number indicating the difference in means under the null hypothesis. Default is 0.

 - *nrep*: an integer indicating the number of replications used in bootstrap resampling. When nrep is larger than 5000, it is suggested to set *parallel* = TRUE. Default is FALSE.

 - *parallel*: a logic indicating if paralle processing need to be used. If TRUE (default) then parallel library is used along with half of the total cores.

**Value**

A list containing the following components:

 - *ts.original*: the value of the test statistic of the original sample.

 - *p.value*: the p-value for the test.

 - *ts.dist.sim*: the simulated test statistics under the null hypothesis.

**Examples**

```r
set.seed(123)
x = runif(10, 1, 3) #mean = 2, var = 1/3
y = rexp(20, 1) #mean = 1, var = 1

bs.two.sample.test(x, y, alternative = "two.sided", mu = 0.5, nrep = 1000)
```

```
## 
##    Two Sample Bootstrap Test
## 
## data: x and y
## 1000 replications
## test statistic of the original sample: 1.570439, p-value: 0.066
## alternative hypothesis: true difference in means is not equal to 0.5
```

Lastly, let's do a null hypothesis test to check the type I error. I first generate 10000 pairs of sample X and Y from unif(1,3) and 1 + exp(1) respectively. Each X has a size of 10 numbers. Each Y has a size of 20 numbers. Then I use the function we defined to perform two sample boostrap test on each pair of samples and record their p-values. Finally, I count the number of p-values smaller than 0.05 to see if their proportion is close to 0.05.


```r
set.seed(123)
nrep.p.value = 50000

x2 = matrix(runif(10*nrep.p.value, 1, 3), ncol = nrep.p.value)
y2 = matrix(1 + rexp(20*nrep.p.value, 1), ncol = nrep.p.value)

library(parallel)
cl2 = makeCluster(detectCores()/2)
clusterExport(cl = cl2, varlist=c("x", "y","bs.two.sample.test"), envir=environment())

p.value.vec = parSapply(cl = cl2, 1:nrep.p.value, function(s) bs.two.sample.test(x[,s], y[,s], mu=0)$p.value)
mean(p.value.vec < 0.05)

stopCluster(cl2)
```


```
## [1] 0.0416
```


The proportion is 0.0416 not very close to 0.05, but at least it's conservative.



