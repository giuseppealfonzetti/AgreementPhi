
<!-- README.md is generated from README.Rmd. Please edit that file -->

# AgreementPhi

<!-- badges: start -->

[![R-CMD-check](https://github.com/giuseppealfonzetti/AgreementPhi/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/giuseppealfonzetti/AgreementPhi/actions/workflows/R-CMD-check.yaml)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![CRAN
status](https://www.r-pkg.org/badges/version/AgreementPhi)](https://CRAN.R-project.org/package=AgreementPhi)
[![Codecov test
coverage](https://codecov.io/gh/giuseppealfonzetti/AgreementPhi/graph/badge.svg)](https://app.codecov.io/gh/giuseppealfonzetti/AgreementPhi)
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
# setting dimension
items <- 200
budget_per_item <- 8
n_obs <- items * budget_per_item

# item-specific intercepts to generate the data
alphas <- runif(items, -2, 2)

# true agreement (between 0 and 1)
agr <- .8

# generate continuous rating in (0,1)
dt <- sim_data(
  J = items,
  B = budget_per_item,
  AGREEMENT = agr,
  DATA_TYPE = "continuous",
  ALPHA = alphas
)
```

Fit the model using the `agreement()` function

``` r
# estimation via modified profile likelihood
fit <- agreement(
  RATINGS = dt$rating,
  ITEM_INDS = dt$id_item,
  WORKER_INDS = dt$id_worker,
  METHOD = "modified",
  NUISANCE = c("items"),
  VERBOSE = TRUE)
#> 
#> DATA
#>  - Detected 200 items and 200 workers.
#>  - Detected continuous data on the (0,1) range.
#>  - Average number of observed ratings per item is 8.
#>  - Average number of observed ratings per worker is 8.
#> 
#> MODEL PARAMETERS
#>  - Constant effects: workers
#>  - Nuisance effects: items
#> Non-adjusted agreement: 0.835331
#> Adjusted agreement: 0.792021
#> Done!
```

Construct confidence intervals

``` r
# get standard error and confidence interval
ci <- get_ci(fit)
ci 
#> $agreement_est
#> [1] 0.7920213
#> 
#> $agreement_se
#> [1] 0.01203436
#> 
#> $agreement_ci
#> [1] 0.7684344 0.8156082
```

# References

- Checco A., Roitero K., Maddalena E., Mizzaro S., Demartini G., (2017).
  “Let’s Agree to Disagree: Fixing Agreement Measures for
  Crowdsourcing.” *Proceedings of the AAAI Conference on Human
  Computation and Crowdsourcing* **5**: 11–20.
  [doi](https://doi.org/10.1609/hcomp.v5i1.13306)
