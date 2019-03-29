---
title: "Explain Lightgbm Algorithm from Source Code and API Parameters"
author: "Xinyu Zhang"
date: "March 24, 2019"
output:
  html_document:
    keep_md: true
---




In this post, I will use try to explain Lightgbm's key algorithm from its source code and API parameters.

## Objective function: adding L1 regularization

Remember in [Xgboost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) the objective function is: $obj^{(t)}=\sum\_{j=1}^T [2\cdot G\_jw\_j+(H\_j+\lambda)w\_j^2]+2\gamma T$. Its maximum $-\sum\_{j=1}^T \frac{G\_j^2}{H\_j+\lambda}+2\gamma T$ is attained at $w\_j^*=-\frac{G\_j}{H\_j+\lambda}$. $obj^{(t)}$ represents the objective function in the $(t)$th iteration. $T$ is the number of leaves of the estimated tree in the $(t)$th iteration. $G\_j$ and $H\_j$ are the sum of gradients and hessians of all observations ended up in $j$th leaf. $w\_j$ is the leaf score of $j$th leaf. $\lambda$ is the weight of the L2 regularization. The L2 regularization is applied to the leaf scores of a tree. $\gamma$ is the weight of the number of leaves regularization.

The resulting split gain is: $Gain=[\frac{G\_L^2}{H\_L^2+\lambda}+\frac{G\_R^2}{H\_R^2+\lambda}-\frac{(G\_L+G\_R)^2}{H\_L+H\_R+\lambda}]-\gamma$. $G\_L$ and $G\_R$ are the sum of gradients in left child leaf and right child leaf. $H\_L$ and $H\_R$ are the sum of hessians in left child leaf and right child leaf. $\lambda$ and $\gamma$ are the same as before.

It is reasonable to also add a L1 regularization of the leaf score into the objective function. If we use $L\_1$ as the weight of the L1 regularization and $L\_2$ as the weight of the L2 regularization instead of $\lambda$, then we have:

$$
\begin{aligned}
obj^{(t)}
&=\sum\_{j=1}^T [2\cdot (G\_jw\_j+L\_1|w\_j|)+(H\_j+L\_2)w\_j^2]+2\gamma T\\\\\\
&=\sum\_{j=1}^T [2w\_j\cdot (G\_j+sgn(w\_j)L\_1)+(H\_j+L\_2)w\_j^2]+2\gamma T
\end{aligned}
$$
Its maximum $-\sum\_{j=1}^T \frac{(G\_j+sgn(w\_j)L\_1)^2}{H\_j+L\_2}+2\gamma T$ is attained at $w\_j^*=-\frac{G\_j+sgn(w\_j)L\_1}{H\_j+L\_2}$. The leaf objective function is $-\frac{(G\_j+sgn(w\_j)L\_1)^2}{H\_L^2+L\_2}$.

How does lightgbm implement it? In the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 438-450, we see:


```rcpp
  static double ThresholdL1(double s, double l1) {
    const double reg_s = std::max(0.0, std::fabs(s) - l1);
    return Common::Sign(s) * reg_s;
  }

  static double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step) {
    double ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
    if (max_delta_step <= 0.0f || std::fabs(ret) <= max_delta_step) {
      return ret;
    } else {
      return Common::Sign(ret) * max_delta_step;
    }
  }
```

The leaf score is defined as $w\_j^*=-\frac{sgn(G\_j) \cdot max(0,|G\_j|-L\_1)}{H\_j+\lambda}$.

From the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 495-498, we see:


```rcpp
  static double GetLeafSplitGainGivenOutput(double sum_gradients, double sum_hessians, double l1, double l2, double output) {
    const double sg_l1 = ThresholdL1(sum_gradients, l1);
    return -(2.0 * sg_l1 * output + (sum_hessians + l2) * output * output);
  }
```

The output here is the leaf score. The leaf objective function is defined as $-\frac{(sgn(G\_j) \cdot max(0,|G\_j|-L\_1))^2}{H\_L^2+L\_2}$.

Lightgbm uses a different numerator from ours, but are they really that different? Let's have a look at the square of the two numerators of the leaf objective function.

$$
[sgn(G\_j) \cdot max(0,|G\_j|-L\_1)]^2=G\_j^2+L\_1^2\pm2G\_jL\_1=[G\_j+sgn(w\_j)L\_1]^2
$$

Therefore in some cases these two definitions work the same.

The corresponding API parameters are *lambda_l1* and *lambda_l2*.

* *lambda_l1*, default = 0.0, type = double, aliases: reg_alpha, constraints: lambda_l1 >= 0.
    + L1 regularization
* *lambda_l2*, default = 0.0, type = double, aliases: reg_lambda, lambda, constraints: lambda_l2 >= 0.0
    + L2 regularization

## Tree structure: leaf-wise tree growth

Lightgbm uses leaf-wise tree growth which means for every split it checks split gain not only for the nodes on the same level (of the same tree depth) but for all existing nodes. Therefore, it is very easy to grow a deep tree. The API parameter *max_depth* sets a limit for the tree depth to prevent overfitting.

* *max_depth*, default = -1, type = int.
    + limit the max depth for tree model. This is used to deal with over-fitting when #data is small. Tree still grows leaf-wise.

We can see the implementation in the following [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/serial_tree_learner.cpp) line 59-84.


```rcpp
    if (tree->leaf_depth(left_leaf) >= config_->max_depth) {
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
```

The API parameter *num_leaves* is used to set a limit for the number of leaves to prevent overfitting as well.

* *num_leaves*, default = 31, type = int, aliases: num_leaf, max_leaves, max_leaf, constraints: num_leaves > 1.
    + max number of leaves in one tree.

We can see the implementation in the following [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/serial_tree_learner.cpp) line 185.


```rcpp
  for (int split = init_splits; split < config_->num_leaves - 1; ++split) {
```

## Dealing with categorical variables.

When dealing with categorical variables we commonly use one vs many (one hot encoding) or many vs many. Even though many vs many can become complicated very easily when the number of levels in a categorical variable is large, but it tend to build a less deeper tree when comparing to one vs many method since it can have a much larger gain at the split.

Lightgbm uses one vs many when the number of bins are smaller than or equal to the API parameter *max_cat_to_onehot*.

* *max_cat_to_onehot*, default = 4, type = int, constraints: max_cat_to_onehot > 0.
    + when number of categories of one feature smaller than or equal to max_cat_to_onehot, one-vs-other split algorithm will be used.

It goes over every level (bin) and use it as one of the child node and the rest levels as the other.


```rcpp
    if (use_onehot) {
      for (int t = 0; t < used_bin; ++t) {
        // if data not enough, or sum hessian too small
        if (data_[t].cnt < meta_->config->min_data_in_leaf
            || data_[t].sum_hessians < meta_->config->min_sum_hessian_in_leaf) continue;
        data_size_t other_count = num_data - data_[t].cnt;
        // if data not enough
        if (other_count < meta_->config->min_data_in_leaf) continue;

        double sum_other_hessian = sum_hessian - data_[t].sum_hessians - kEpsilon;
        // if sum hessian too small
        if (sum_other_hessian < meta_->config->min_sum_hessian_in_leaf) continue;

        double sum_other_gradient = sum_gradient - data_[t].sum_gradients;
```

Above is the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 130-143. We see that it says the current level is not splittable when the number of data in the level is smaller than the API parameter *min_data_in_leaf* or the sum of hessians in the level is smaller than the API parameter *min_sum_hessian_in_leaf*.

* *min_data_in_leaf*, default = 20, type = int, aliases: min_data_per_leaf, min_data, min_child_samples, constraints: min_data_in_leaf >= 0.
    + minimal number of data in one leaf. Can be used to deal with over-fitting.
* *min_sum_hessian_in_leaf*, default = 1e-3, type = double, aliases: min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight, constraints: min_sum_hessian_in_leaf >= 0.0.
    + minimal sum hessian in one leaf. Like min_data_in_leaf, it can be used to deal with over-fitting.

We should also notice that it calculates the other child node's data count and sum hessians using the parent node and the current level. This is the historgram subtraction speedup mentioned in the lightgbm [features](https://lightgbm.readthedocs.io/en/latest/Features.html) page.

If the current level is splittable and the objective function is larger than the sum of the parent node's objective function and the API paramter *min_gain_to_split*, the split gain and the split is recorded. They are updated as the model goes over all levels are finds one that provides the largest split gain.

* *min_gain_to_split*,  default = 0.0, type = double, aliases: min_split_gain, constraints: min_gain_to_split >= 0.0.
    + the minimal gain to perform split.

See the code below from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 154-160.


```rcpp
        if (current_gain > best_gain) {
          best_threshold = t;
          best_sum_left_gradient = data_[t].sum_gradients;
          best_sum_left_hessian = data_[t].sum_hessians + kEpsilon;
          best_left_count = data_[t].cnt;
          best_gain = current_gain;
        }
```

When many vs many is used, lightgbm first selects levels (bins) with data count larger than the API parameter *cat_smooth*.

* *cat_smooth*, default = 10.0, type = double, constraints: cat_smooth >= 0.0.
    + used for the categorical features.
    + this can reduce the effect of noises in categorical features, especially for categories with few data.

See the code below from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 163-167.


```rcpp
      for (int i = 0; i < used_bin; ++i) {
        if (data_[i].cnt >= meta_->config->cat_smooth) {
          sorted_idx.push_back(i);
        }
      }
```

Then it sorts all levels based on their $\frac{G}{H+cat_smooth}$. See the code below from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 172-178.


```rcpp
      auto ctr_fun = [this](double sum_grad, double sum_hess) {
        return (sum_grad) / (sum_hess + meta_->config->cat_smooth);
      };
      std::sort(sorted_idx.begin(), sorted_idx.end(),
                [this, &ctr_fun](int i, int j) {
        return ctr_fun(data_[i].sum_gradients, data_[i].sum_hessians) < ctr_fun(data_[j].sum_gradients, data_[j].sum_hessians);
      });
```

First it iterates over the sorted levels from left (small) to right (large). It groups up the current level and all levels to the left of it and put them in the left child node. The rest are put in the right child node. It also checks to see if the number of data in the left level group is larger or equal to the API paramter *min_data_per_group*. Then it iterates over the sorted levels from right to left using a similar method. The best split is found in this process.

* *min_data_per_group*, default = 100, type = int, constraints: min_data_per_group > 0.
    + minimal number of data per categorical group.

The following code is the from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 180-197. It initiates the iteration direction 1 (small to large) and -1 (large to small), then starts the two iterations.


```rcpp
      std::vector<int> find_direction(1, 1);
      std::vector<int> start_position(1, 0);
      find_direction.push_back(-1);
      start_position.push_back(used_bin - 1);
      const int max_num_cat = std::min(meta_->config->max_cat_threshold, (used_bin + 1) / 2);

      is_splittable_ = false;
      for (size_t out_i = 0; out_i < find_direction.size(); ++out_i) {
        auto dir = find_direction[out_i];
        auto start_pos = start_position[out_i];
        data_size_t min_data_per_group = meta_->config->min_data_per_group;
        data_size_t cnt_cur_group = 0;
        double sum_left_gradient = 0.0f;
        double sum_left_hessian = kEpsilon;
        data_size_t left_count = 0;
        for (int i = 0; i < used_bin && i < max_num_cat; ++i) {
          auto t = sorted_idx[start_pos];
          start_pos += dir;
```

From the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) line 212 we see the use of *min_data_per_group*.


```rcpp
          if (cnt_cur_group < min_data_per_group) continue;
```

Note:  
1. This is not an exhaustive best group split search. It essentially groups up the levels with similar $\frac{G}{H+cat_smooth}$ and see which group split is the best.  
2. The reason for the model to iterate over both directions is to deal with the missing values. When it iterates from left to right the missing values are always put in the right child node. Similarly they are put in the left child node when the model iterates from right to left. This way, both situations of the final location of the missing value are evaluated. This is called sparsity aware split-finding which is also used in [xgboost](https://arxiv.org/pdf/1603.02754.pdf).  
3. When the model iterates from left to right, the left level group which is lower bounded by *min_data_per_group* is put in the left child node which is lower bounded by *min_data_in_leaf*. So in this case it is actually lower bounded by the larger of the two parameters.

## Dealing with continuous variables

In the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/io/dataset.cpp) line 789-800, we see:


```rcpp
    if (!is_constant_hessian) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_gradients[i] = gradients[data_indices[i]];
        ordered_hessians[i] = hessians[data_indices[i]];
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_gradients[i] = gradients[data_indices[i]];
      }
    }
```

Lightgbm first sorts the data by their gradients, as well as hessian if they are not a constant. From these sorted data the model constructs their histogram under the constrains of API parameters: *max_bin*, *min_data_in_bin* and *bin_construct_sample_cnt*. Notice here lightgbm doesn't use all data. Instead, it samples a portion of data to speed up the histogram construction process.

* *max_bin*, default = 255, type = int, constraints: max_bin > 1.
    + max number of bins that feature values will be bucketed in.
    + small number of bins may reduce training accuracy but may increase general power (deal with over-fitting).
    + LightGBM will auto compress memory according to max_bin. For example, LightGBM will use uint8_t for feature value if max_bin=255.
* *min_data_in_bin*, , default = 3, type = int, constraints: min_data_in_bin > 0.
    + minimal number of data inside one bin.
    + use this to avoid one-data-one-bin (potential over-fitting).
* *bin_construct_sample_cnt*, default = 200000, type = int, aliases: subsample_for_bin, constraints: bin_construct_sample_cnt > 0.
    + number of data that sampled to construct histogram bins.
    + setting this to larger value will give better training result, but will increase data loading time
    + set this to larger value if data is very sparse

The finding best split process is very much similar to the process for categorical variables. When the data is very sparse, it is reasonable to treat 0 as missing data to speed up the split process. This feature is controlled by the API parameter *zero_as_missing*.

* *use_missing*, default = true, type = bool.
    + set this to false to disable the special handle of missing value.
* *zero_as_missing*, , default = false, type = bool.
    + set this to true to treat all zero as missing values (including the unshown values in libsvm/sparse matrices).
    + set this to false to use na for representing missing values.

## Exclusive Feature Bundling (EFB)

Common dimention reduction methods such as principle component analysis or projection pursuit assume features to contain redundent information. This might not be true in practice. High dimensional data are usually very sparse, lightgbm use EFB not to filter out the redundent information but to combine some not so mutually exclusive sparse feature in order to acchieve dimension reduction.

We say two features are mutually exclusive when either one takes a nonzero value A, the other is never A. Therefore, two features are very likely to be mutually exclusive when they rarely take nonzero values simultaneously. In another word, if both two features only have a small number of nonzero values (low conflict counts), then they should be combined together. EFB does not simply combine these two features by summing up their values. It shifts the value of the second feature by the range of the first feature before making the summation. For example, say our first feature has a range of [0,10] and its first observation is 5. Our second feature has a range of [0,20] and its first observation is 15. Then the first observation of the combined feature is 5 + (10 + 15) = 30. This way the distribution of the individual feature pertains to the combined feature to a certain degree.

The following code is line 49-60 in the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/io/dataset.cpp). We see it counts the number of 0 in each feature.


```rcpp
int GetConfilctCount(const std::vector<bool>& mark, const int* indices, int num_indices, int max_cnt) {
  int ret = 0;
  for (int i = 0; i < num_indices; ++i) {
    if (mark[indices[i]]) {
      ++ret;
      if (ret > max_cnt) {
        return -1;
      }
    }
  }
  return ret;
}
```

The following code is line 106-136 in the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/io/dataset.cpp).


```rcpp
    for (auto gid : search_groups) {
      const int rest_max_cnt = max_error_cnt - group_conflict_cnt[gid];
      int cnt = GetConfilctCount(conflict_marks[gid], sample_indices[fidx], num_per_col[fidx], rest_max_cnt);
      if (cnt >= 0 && cnt <= rest_max_cnt) {
        data_size_t rest_non_zero_data = static_cast<data_size_t>(
          static_cast<double>(cur_non_zero_cnt - cnt) * num_data / total_sample_cnt);
        if (rest_non_zero_data < filter_cnt) { continue; }
        need_new_group = false;
        features_in_group[gid].push_back(fidx);
        group_conflict_cnt[gid] += cnt;
        group_non_zero_cnt[gid] += cur_non_zero_cnt - cnt;
        MarkUsed(conflict_marks[gid], sample_indices[fidx], num_per_col[fidx]);
        if (is_use_gpu) {
          group_num_bin[gid] += bin_mappers[fidx]->num_bin() + (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0);
        }
        break;
      }
    }
    if (need_new_group) {
      features_in_group.emplace_back();
      features_in_group.back().push_back(fidx);
      group_conflict_cnt.push_back(0);
      conflict_marks.emplace_back(total_sample_cnt, false);
      MarkUsed(conflict_marks.back(), sample_indices[fidx], num_per_col[fidx]);
      group_non_zero_cnt.emplace_back(cur_non_zero_cnt);
      if (is_use_gpu) {
        group_num_bin.push_back(1 + bin_mappers[fidx]->num_bin() + (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0));
      }
    }
  }
  return features_in_group;
```

It combines a feature into the current feature bundle when the their conflict count is lower or equal to max_error_cnt which is determined by the API parameter *max_conflict_rate*. Otherwise, a new bundle (group) is created.

* *max_conflict_rate*, default = 0.0, type = double, constraints: 0.0 <= max_conflict_rate < 1.0.
    + max conflict rate for bundles in EFB.
    + set this to 0.0 to disallow the conflict and provide more accurate results.
    + set this to a larger value to achieve faster speed.

The following code is line 152 in the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/io/dataset.cpp). We see it calcualte max_error_cnt using the API parameter *max_conflict_rate*.


```rcpp
  const data_size_t max_error_cnt = static_cast<data_size_t>(total_sample_cnt * max_conflict_rate);
```

We are not finished yet. After the bundling, EFB further splits some bundles with features less than 5 if their zero count is too small (their sparse_rate is less than the API parameter *sparse_threshold*).

* *max_conflict_rate*, default = 0.8, type = double, constraints: 0.0 < sparse_threshold <= 1.0.
    + the threshold of zero elements percentage for treating a feature as a sparse one.

See the following code from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/io/dataset.cpp) line 182-202.


```rcpp
  for (size_t i = 0; i < features_in_group.size(); ++i) {
    if (features_in_group[i].size() <= 1 || features_in_group[i].size() >= 5) {
      ret.push_back(features_in_group[i]);
    } else {
      int cnt_non_zero = 0;
      for (size_t j = 0; j < features_in_group[i].size(); ++j) {
        const int fidx = features_in_group[i][j];
        cnt_non_zero += static_cast<int>(num_data * (1.0f - bin_mappers[fidx]->sparse_rate()));
      }
      double sparse_rate = 1.0f - static_cast<double>(cnt_non_zero) / (num_data);
      // take apart small sparse group, due it will not gain on speed
      if (sparse_rate >= sparse_threshold && is_enable_sparse) {
        for (size_t j = 0; j < features_in_group[i].size(); ++j) {
          const int fidx = features_in_group[i][j];
          ret.emplace_back();
          ret.back().push_back(fidx);
        }
      } else {
        ret.push_back(features_in_group[i]);
      }
    }
```

* *is_enable_sparse*,  default = true, type = bool, aliases: is_sparse, enable_sparse, sparse.
    + used to enable/disable sparse optimization.

## Gradient-based One-Side Sampling (GOSS)

In gradient boost tree, after finishing one tree we use the entire data set again as the training set for the next tree. We weight every observation the same. Lightgbm assigns larger weights to observations with a larger gradient. Then it samples data into the training set based on these weights so that observations with less optimized leaf score will have a higher chance to get into the training set for the next tree construction. Specifically, GOSS first sorts the data according to the absolute value of their gradient. Then a top proportion is selected into the training set. Among the rest, another proportion is randomly sampled into the training set.

The following code is from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/boosting/goss.hpp) line 91-102.


```rcpp
    std::vector<score_t> tmp_gradients(cnt, 0.0f);
    for (data_size_t i = 0; i < cnt; ++i) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
        tmp_gradients[i] += std::fabs(gradients_[idx] * hessians_[idx]);
      }
    }
    data_size_t top_k = static_cast<data_size_t>(cnt * config_->top_rate);
    data_size_t other_k = static_cast<data_size_t>(cnt * config_->other_rate);
    ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, static_cast<int>(tmp_gradients.size()), top_k - 1);
    score_t threshold = tmp_gradients[top_k - 1];
```

First, we see that lightgbm sorts the data based on the absolute value of the product of their gradients and hessians (not just the absolute value of their gradients as in the [paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)). Then it determines the number of the top proportion that will be sampled (#data $\times$ *top_rate*) and the number of the rest data will be randomly sampled (#data $\times$ *other_rate*).*Top_rate* and *other_rate* are two API parameters. A threshold is also defined which is the cutoff point of the absolute value of the product of gradient and hessian so that the top proportion of the data can be selected.

* *Top_rate*, default = 0.2, type = double, constraints: 0.0 <= top_rate <= 1.0.
    + used only in goss.
    + the retain ratio of large gradient data.
* *other_rate*, default = 0.1, type = double, constraints: 0.0 <= other_rate <= 1.0.
    + used only in goss.
    + the retain ratio of small gradient data.


```rcpp
      if (grad >= threshold) {
        buffer[cur_left_cnt++] = start + i;
        ++big_weight_cnt;
      } else {
        data_size_t sampled = cur_left_cnt - big_weight_cnt;
        data_size_t rest_need = other_k - sampled;
        data_size_t rest_all = (cnt - i) - (top_k - big_weight_cnt);
        double prob = (rest_need) / static_cast<double>(rest_all);
        if (cur_rand.NextFloat() < prob) {
          buffer[cur_left_cnt++] = start + i;
          for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
            size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
            gradients_[idx] *= multiply;
            hessians_[idx] *= multiply;
          }
        } else {
          buffer_right[cur_right_cnt++] = start + i;
        }
      }
```

The code above is from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/boosting/goss.hpp) line 114-132. We see that data with the absolute value of the product of gradient and hessian larger than the threshold is added to the buffer (training set). For the rest data, #data $\times$ other_rate number of data are randomly sampled. Notice, before adding these into the buffer multiply is timed to their gradients and hessians. The multiply is defined in the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/boosting/goss.hpp) line 104 as below.


```rcpp
    score_t multiply = static_cast<score_t>(cnt - top_k) / other_k;
```
 
It is essentially $\frac{1-top\_rate}{other\_rate}$. This multiplier guarantees that when sampling from the rest data, the sum of gradients of these undersampled data match the sum of gradients of the rest data. Same goes for hessian.

There are also some requirements to use GOSS. See the code from the [source code](https://github.com/Microsoft/LightGBM/blob/master/src/boosting/goss.hpp) line 59-84.


```rcpp
    CHECK(config_->top_rate + config_->other_rate <= 1.0f);
    CHECK(config_->top_rate > 0.0f && config_->other_rate > 0.0f);
    if (config_->bagging_freq > 0 && config_->bagging_fraction != 1.0f) {
      Log::Fatal("Cannot use bagging in GOSS");
    }
    Log::Info("Using GOSS");

    bag_data_indices_.resize(num_data_);
    tmp_indices_.resize(num_data_);
    tmp_indice_right_.resize(num_data_);
    offsets_buf_.resize(num_threads_);
    left_cnts_buf_.resize(num_threads_);
    right_cnts_buf_.resize(num_threads_);
    left_write_pos_buf_.resize(num_threads_);
    right_write_pos_buf_.resize(num_threads_);

    is_use_subset_ = false;
    if (config_->top_rate + config_->other_rate <= 0.5) {
      auto bag_data_cnt = static_cast<data_size_t>((config_->top_rate + config_->other_rate) * num_data_);
      bag_data_cnt = std::max(1, bag_data_cnt);
      tmp_subset_.reset(new Dataset(bag_data_cnt));
      tmp_subset_->CopyFeatureMapperFrom(train_data_);
      is_use_subset_ = true;
    }
    // flag to not bagging first
    bag_data_cnt_ = num_data_;
```

We see that *top_rate* + *other_rate* has to be less or equal to 0.5 to use GOSS. Bagging and GOSS cannot use together.

* *bagging_fraction*, default = 1.0, type = double, aliases: sub_row, subsample, bagging, constraints: 0.0 < bagging_fraction <= 1.
    + like feature_fraction, but this will randomly select part of data without resampling.
    + can be used to speed up training.
    + can be used to deal with over-fitting.
    + Note: to enable bagging, bagging_freq should be set to a non zero value as well.
* *bagging_freq*, default = 0, type = int, aliases: subsample_fr.
    + frequency for bagging.
    + 0 means disable bagging; k means perform bagging at every k iteration.
    + Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as well.
