
<!-- README.md is generated from README.Rmd. Please edit that file -->

# AgreementPhi

<!-- badges: start -->

[![R-CMD-check](https://github.com/giuseppealfonzetti/AgreementPhi/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/giuseppealfonzetti/AgreementPhi/actions/workflows/R-CMD-check.yaml)
[![Codecov test
coverage](https://codecov.io/gh/giuseppealfonzetti/AgreementPhi/graph/badge.svg)](https://app.codecov.io/gh/giuseppealfonzetti/AgreementPhi)
[![Docs](https://img.shields.io/badge/docs-homepage-blue.svg)](https://giuseppealfonzetti.github.io/AgreementPhi/)
<!-- [![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental) -->
<!-- badges: end -->

The `AgreementPhi` package is the companion of “Alfonzetti G., Bellio
R., Vidoni P. *Accurate agreement estimation in crowdsourced relevance
assessments*”. It allows the accurate estimation of the general $\Phi$
agreement measure among multiple crowd-workers assessing a given
collection of items (Checco et al. 2017).

## Installation

You can install `AgreementPhi` using the `devtools` package:

``` r
devtools::install_github("giuseppealfonzetti/AgreementPhi")
```

## Example

Generate a synthetic dataset with continuous ratings in (0,1)

``` r
library(AgreementPhi)
set.seed(321)
items <- 200
budget_per_item <- 8
alphas <- runif(items, -2, 2)
agr <- .8

dt <- sim_data(
  J = items,
  B = budget_per_item,
  AGREEMENT = agr,
  DATA_TYPE = "continuous",
  ALPHA = alphas
)
```

Prepare the data with `rating_data()`, which validates the input and
reports diagnostics

``` r
rd <- rating_data(
  RATINGS = dt$rating, 
  ITEM_INDS = dt$id_item, 
  WORKER_INDS = dt$id_worker
  )
```

Fit the model using `agreement()`

``` r
fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"), VERBOSE = TRUE)
#> 
#> MODEL PARAMETERS
#>  - Constant effects: workers
#>  - Nuisance effects: items
#> Non-adjusted agreement: 0.835331
#> Adjusted agreement: 0.79202
#> Done!
```

Extract estimates

``` r
coef(fit)[1:10]
#>         phi     alpha_1     alpha_2     alpha_3     alpha_4     alpha_5 
#>  6.53680906  1.84746562  1.68557164 -1.08680905 -1.36272134 -0.73792367 
#>     alpha_6     alpha_7     alpha_8     alpha_9 
#> -0.49287874 -0.37423723 -0.54713560  0.01762487
length(coef(fit))
#> [1] 201
```

Construct confidence intervals

``` r
confint(fit)
#> $parameters
#>     Estimate Std. Error    2.5 %   97.5 %
#> phi 6.536809  0.2408701 6.064712 7.008906
#> 
#> $agreement
#>            Estimate Std. Error     2.5 %    97.5 %
#> agreement 0.7920203 0.01203441 0.7684333 0.8156073
```

# References

- Checco A., Roitero K., Maddalena E., Mizzaro S., Demartini G., (2017).
  “Let’s Agree to Disagree: Fixing Agreement Measures for
  Crowdsourcing.” *Proceedings of the AAAI Conference on Human
  Computation and Crowdsourcing* **5**: 11–20.
  [doi](https://doi.org/10.1609/hcomp.v5i1.13306)
