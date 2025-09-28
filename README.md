
<!-- README.md is generated from README.Rmd. Please edit that file -->

# AgreementPhi

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![CRAN
status](https://www.r-pkg.org/badges/version/AgreementPhi)](https://CRAN.R-project.org/package=AgreementPhi)

<!-- badges: end -->

The `AgreementPhi` package allows to accurately estimate the general
agreement among raters across a collection of items. It provides a
general tool to deal with percentage and ordinal data.

## Installation

You can install `AgreementPhi` using the `devtools` package:

``` r
devtools::install_github("giuseppealfonzetti/AgreementPhi")
```

## Example

Generate a synthetic dataset with continuous ratings in (0,1)

``` r
library(AgreementPhi)
set.seed(123)

# setting dimension
items <- 100 
budget_per_item <- 10
n_obs <- items * budget_per_item

# item-specific intercepts to generate the data
alphas <- runif(items)

# true agreement (between 0 and 1)
agr <- .7

# generate continuous rating in (0,1)
dt <- sim_data(
  J = items,
  B = budget_per_item,
  AGREEMENT = agr,
  ALPHA = alphas,
  DATA_TYPE = "continuous",
  SEED = 123
)
```

We fit the model on the continuous ratings data and compare the
agreement estimated between diffferent methods

``` r
# fit via profile likelihood
fit_c_p <- agreement(
  RATINGS = dt$rating,
  ITEM_INDS = dt$id_item,
  METHOD = "profile")
fit_c_p$pl_agreement
#> [1] 0.7261697

# fit via modified profile likelihood
fit_c_mp <- agreement(
  RATINGS = dt$rating,
  ITEM_INDS = dt$id_item,
  METHOD = "modified")
fit_c_mp$mpl_agreement
#> [1] 0.6872219
```

We can plot the relative loglikelihood profiles

``` r
rll <- get_rll(fit_c_mp, PLOT = TRUE)
```

<img src="man/figures/README-unnamed-chunk-2-1.png" width="100%" />

and also construct confidence intervals for the estimated agreement

``` r
get_ci(fit_c_mp)$agreement_ci
#> [1] 0.6558677 0.7185760
```

Consider now ratings collected on Likert-type rating. To allow for a
direct comparison with the previous dataset, we directly discretise the
continuous ratings generated previously

``` r
rating_k3 <- cont2ord(dt$rating, K=3)
rating_k5 <- cont2ord(dt$rating, K=5)
rating_k7 <- cont2ord(dt$rating, K=7)
```

Now, we evaluate the agreement on the discretised data. The fitting
function automatically detects the ordinal ratings scale.

``` r
fit_k3_p <- agreement(
  RATINGS = rating_k3,
  ITEM_INDS = dt$id_item,
  METHOD = "profile")
fit_k3_p$pl_agreement
#> [1] 0.7599315

fit_k3_mp <- agreement(
  RATINGS = rating_k3,
  ITEM_INDS = dt$id_item,
  METHOD = "modified")
fit_k3_mp$mpl_agreement
#> [1] 0.7167846

fit_k5_p <- agreement(
  RATINGS = rating_k5,
  ITEM_INDS = dt$id_item,
  METHOD = "profile")
fit_k5_p$pl_agreement
#> [1] 0.7333303

fit_k5_mp <- agreement(
  RATINGS = rating_k5,
  ITEM_INDS = dt$id_item,
  METHOD = "modified")
fit_k5_mp$mpl_agreement
#> [1] 0.6940051

fit_k7_p <- agreement(
  RATINGS = rating_k7,
  ITEM_INDS = dt$id_item,
  METHOD = "profile")
fit_k7_p$pl_agreement
#> [1] 0.7446112

fit_k7_mp <- agreement(
  RATINGS = rating_k7,
  ITEM_INDS = dt$id_item,
  METHOD = "modified")
fit_k7_mp$mpl_agreement
#> [1] 0.7067203
```

Also in case of ordinal data we can plot the profiles of the relative
log-likelihoods

``` r
rll <- get_rll(fit_k3_mp, PLOT = TRUE)
```

<img src="man/figures/README-unnamed-chunk-4-1.png" width="100%" />

and construct confidence intervals

``` r
get_ci(fit_k3_mp)$agreement_ci
#> [1] 0.6637606 0.7698086
get_ci(fit_k7_mp)$agreement_ci
#> [1] 0.6716127 0.7418278
```
