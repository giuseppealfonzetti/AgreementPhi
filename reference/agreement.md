# Compute Agreement

Compute the \\\Phi\\ agreement proposed in Checco et al. (2017) via
profile likelihood methods. Three data types are supported, detected
automatically from the supplied `rating_data` object:

- **Ordinal**: integer-valued in {1, 2, ..., K}.

- **Continuous**: real-valued in the open interval `(0, 1)`.

- **Inflated interval**: real-valued in `[0, 1]` with point masses at 0
  and/or 1. Fitted via the ordered beta mixture model. One-way only (no
  workers in `DATA`).

## Usage

``` r
agreement(
  DATA,
  METHOD = c("modified", "profile"),
  ALPHA_START = NULL,
  BETA_START = NULL,
  TAU = NULL,
  PHI_START = NULL,
  NUISANCE = c("items", "workers"),
  CONTROL = list(),
  VERBOSE = FALSE
)
```

## Arguments

- DATA:

  A `rating_data` object created by
  [`rating_data()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/rating_data.md).

- METHOD:

  Choose between `"modified"` or `"profile"`. Default is `"modified"`.

  - `"modified"`: Uses modified profile likelihood with
    Barndorff-Nielsen correction.

  - `"profile"`: Uses standard profile likelihood.

- ALPHA_START:

  Starting values for item-specific intercepts. Vector of length J.
  Default is `init_alpha()`. Ignored for the inflated interval model.

- BETA_START:

  Starting values for worker-specific intercepts. Vector of length W-1.
  Default is `rep(0, W-1)`. Ignored for the inflated interval model.

- TAU:

  Thresholds for discretisation of the underlying beta distribution.
  Ignored for the inflated interval model.

- PHI_START:

  Starting value for the beta precision parameter. Must be positive.
  Default is `agr2prec(0.5)`. Ignored for the inflated interval model.

- NUISANCE:

  Vector containing either `"items"`, `"workers"`, or both. Defines
  which fixed effects to profile out during estimation. Ignored for the
  inflated interval model.

- CONTROL:

  Control options for the optimization.

  `SEARCH_RANGE`

  :   Search range for precision parameter optimization. The algorithm
      searches in \[1e-8, PHI_START + SEARCH_RANGE\]. Must be positive.
      Default: `8`.

  `MAX_ITER`

  :   Maximum number of iterations for precision parameter optimization.
      Must be a positive integer. Default: `100`.

  `PROF_SEARCH_RANGE`

  :   Search range for profiling out item intercepts (alpha). The
      algorithm searches in \[alpha_j - PROF_SEARCH_RANGE, alpha_j +
      PROF_SEARCH_RANGE\] for each item j. Applies to both
      continuous/ordinal and inflated interval data. Must be positive.
      Default: `10`.

  `PROF_MAX_ITER`

  :   Maximum number of iterations for profiling optimization. Must be a
      positive integer. Default: `500`.

  `ALT_MAX_ITER`

  :   Maximum iterations for alternating profiling. Non-inflated only.
      Must be a positive integer. Default: `50`.

  `ALT_TOL`

  :   Relative convergence tolerance for alternating profiling.
      Non-inflated only. Must be positive. Default: `1e-3`.

  `BOUNDARY`

  :   Boundary value for cutpoints when one boundary is absent. Inflated
      interval only. Must be positive. Default: `100`.

- VERBOSE:

  Print optimization progress. Default `FALSE`.

## Value

An S3 object of class `agreement_fit` with the following components:

- `data_type`:

  Detected data type: `"ordinal"`, `"continuous"`, or `"inflated"`.

- `method`:

  Estimation method used: `"profile"` or `"modified"`.

- `alpha`:

  Estimated item-specific intercepts (vector of length J).

- `beta`:

  Estimated worker-specific intercepts. `NULL` for one-way models.

- `k0`:

  Estimated lower cutpoint on the logit scale. Inflated interval model
  only.

- `k1`:

  Estimated upper cutpoint on the logit scale. Inflated interval model
  only.

- `profile`:

  List with `$precision` (profile MLE of \\\phi\\) and `$agreement`
  (corresponding \\\Phi\\).

- `modified`:

  List with `$precision` (MPL estimate of \\\phi\\) and `$agreement`
  (corresponding \\\Phi\\). `NA` when `METHOD = "profile"`.

- `loglik`:

  Profile log-likelihood at the MLE.

- `se`:

  Named vector of standard errors. For inflated interval data: `phi`,
  `k0`, `k1`.

- `vcov`:

  Variance-covariance matrix of `(phi, k0, k1)`. Inflated interval model
  only.

- `convergence`:

  Optimizer convergence code. Inflated interval model only.

## References

- Checco A., Roitero K., Maddalena E., Mizzaro S., Demartini G., (2017).
  "Let's Agree to Disagree: Fixing Agreement Measures for
  Crowdsourcing." *Proceedings of the AAAI Conference on Human
  Computation and Crowdsourcing* **5**: 11–20.
  [doi](https://doi.org/10.1609/hcomp.v5i1.13306)

## Examples

``` r
# \donttest{
set.seed(321)

items <- 50
budget_per_item <- 5
alphas <- runif(items, -2, 2)
agr <- .6

dt_oneway <- sim_data(
  J = items, B = budget_per_item, AGREEMENT = agr,
  ALPHA = alphas, DATA_TYPE = "continuous", SEED = 123
)
rd <- rating_data(dt_oneway$rating, dt_oneway$id_item, dt_oneway$id_worker)
#>  - Detected 50 items and 49 workers.
#>  - Detected continuous data on the (0,1) range.
#>  - Average number of observed ratings per item is 5.
#>  - Average number of observed ratings per worker is 5.1.
fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"))
confint(fit)
#> $parameters
#>     Estimate Std. Error    2.5 %   97.5 %
#> phi 4.462539  0.4359353 3.608121 5.316956
#> 
#> $agreement
#>            Estimate Std. Error    2.5 %    97.5 %
#> agreement 0.6576837 0.03584846 0.587422 0.7279454
#> 
plot(fit)


dt_inflated <- sim_data(
  J = items, B = budget_per_item, AGREEMENT = agr,
  ALPHA = alphas, DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 123
)
rd_inf <- rating_data(dt_inflated$rating, dt_inflated$id_item)
#>  - Detected 49 non-degenerate items.
#>  - Detected inflated interval data on the [0,1] range.
#>  - Average number of observed ratings per item is 5.04.
fit_inf <- agreement(rd_inf, METHOD = "modified")
confint(fit_inf)
#> $parameters
#>      Estimate Std. Error     2.5 %    97.5 %
#> phi  5.168256  0.6335260  3.926568  6.409945
#> k0  -1.891239  0.2105952 -2.303998 -1.478480
#> k1   2.172110  0.2019442  1.776307  2.567914
#> 
#> $agreement
#>            Estimate Std. Error     2.5 %    97.5 %
#> agreement 0.3489285 0.03463438 0.2810463 0.4168106
#> 
# }
```
