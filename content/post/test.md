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



$$
\begin{aligned}
obj(\widehat{W^{(t)}})
&=LOSS(\widehat{W^{(t-1)}})+\sum_i\frac{\partial LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W_i^{(t-1)}}}\widehat{f_{(t)}(x_i)}+\frac{1}{2}\sum_i\frac{\partial^2 LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W_i^{(t-1)}}^2}{\widehat{f_{(t)}(x_i)}}^2+\pi(\widehat{f_{(t)}})+constant
\end{aligned}
$$


$$
\begin{aligned}
\sum_i\frac{\partial LOSS(\widehat{W^{(t-1)}})}{\partial \widehat{W_i^{(t-1)}}}\widehat{f_{(t)}(x_i)}+\frac{1}{2}\sum_i\frac{\partial^2LOSS(\widehat{W^{(t-1)}})}{\partial\widehat{W_i^{(t-1)}}^2}{\widehat{f_{(t)}(x_i)}}^2+\pi(\widehat{f_{(t)}})+constant
\end{aligned}
$$
