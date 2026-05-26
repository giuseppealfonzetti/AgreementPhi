# AgreementPhi

The `AgreementPhi` package is the companion of “Alfonzetti G., Bellio
R., Vidoni P. *Accurate agreement estimation in crowdsourced relevance
assessments*”. It allows the accurate estimation of the general $`\Phi`$
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

Prepare the data with
[`rating_data()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/rating_data.md),
which validates the input and reports diagnostics

``` r

rd <- rating_data(dt$rating, dt$id_item, dt$id_worker)
#>  - Detected 200 items and 200 workers.
#>  - Detected continuous data on the (0,1) range.
#>  - Average number of observed ratings per item is 8.
#>  - Average number of observed ratings per worker is 8.
```

Fit the model using
[`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md)

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

Construct confidence intervals

``` r

ci <- confint(fit)
ci
#> $agreement_est
#> [1] 0.7920203
#> 
#> $agreement_se
#> [1] 0.01203441
#> 
#> $agreement_ci
#> [1] 0.7684333 0.8156073
```

# References

- Checco A., Roitero K., Maddalena E., Mizzaro S., Demartini G., (2017).
  “Let’s Agree to Disagree: Fixing Agreement Measures for
  Crowdsourcing.” *Proceedings of the AAAI Conference on Human
  Computation and Crowdsourcing* **5**: 11–20.
  [doi](https://doi.org/10.1609/hcomp.v5i1.13306)
