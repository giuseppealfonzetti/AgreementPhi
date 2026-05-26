# Plot an agreement_fit object

Plots the relative log-likelihood curve(s) for a fitted agreement model.
Calls
[`get_range_ll()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/get_range_ll.md)
to evaluate the likelihood over a grid. A precomputed grid can be
supplied via `RANGE_LL` to avoid recomputation.

## Usage

``` r
# S3 method for class 'agreement_fit'
plot(x, RANGE_LL = NULL, RANGE = 0.2, GRID_LENGTH = 15, CONFIDENCE = 0.95, ...)
```

## Arguments

- x:

  An `agreement_fit` object from
  [`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md).

- RANGE_LL:

  Optional. A data frame returned by
  [`get_range_ll()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/get_range_ll.md).
  If `NULL` (default), the grid is computed internally using `RANGE` and
  `GRID_LENGTH`.

- RANGE:

  Range of agreement values around the MLE to evaluate. Default `0.2`.

- GRID_LENGTH:

  Number of grid points. Default `15`.

- CONFIDENCE:

  Confidence level for the shaded interval. Default `0.95`.

- ...:

  Ignored (required for S3 consistency).

## Value

Invisibly returns `x`.

## Examples

``` r
set.seed(1)
dt <- sim_data(J = 30, B = 5, AGREEMENT = 0.6,
               ALPHA = rep(0, 30), DATA_TYPE = "continuous", SEED = 1)
rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
plot(fit)

```
