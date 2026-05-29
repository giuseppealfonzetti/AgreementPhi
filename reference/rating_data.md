# Prepare rating data for analysis

Validates and preprocesses a raw ratings dataset. Returns a
`rating_data` S3 object that can be passed to
[`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md),
[`plot()`](https://rdrr.io/r/graphics/plot.default.html), and
[`print()`](https://rdrr.io/r/base/print.html). Degenerate items (all
ratings identical) are detected and their recoded indices stored in
`$degen_ids`; no observations are removed here. The decision to drop
degenerate items before fitting is delegated to
[`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md).

## Usage

``` r
rating_data(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  ITEM_LABELS = NULL,
  WORKER_LABELS = NULL,
  K = NULL,
  VERBOSE = FALSE
)
```

## Arguments

- RATINGS:

  Ratings vector. Ordinal: integers in {1,...,K}. Continuous: reals in
  `(0,1)`. Inflated interval: reals in `[0,1]` with exact 0s or 1s.

- ITEM_INDS:

  Integer index vector of item allocations (same length as `RATINGS`).

- WORKER_INDS:

  Integer index vector of worker allocations.

- ITEM_LABELS:

  Optional character vector of item labels (same length as `RATINGS`).
  Each unique item index must map to exactly one label. When provided,
  label names are used for `alpha` coefficients in
  [`coef()`](https://rdrr.io/r/stats/coef.html).

- WORKER_LABELS:

  Optional character vector of worker labels (same length as `RATINGS`).
  Requires `WORKER_INDS`. When provided, label names are used for `beta`
  coefficients in [`coef()`](https://rdrr.io/r/stats/coef.html).

- K:

  Number of ordinal categories. If `NULL`, inferred as `max(RATINGS)`.
  Provide explicitly when boundary categories may be absent from the
  data.

- VERBOSE:

  Print data diagnostics on construction. Default `TRUE`.

## Value

An S3 object of class `rating_data`.

## Examples

``` r
dt <- sim_data(J = 20, B = 5, AGREEMENT = 0.6,
               ALPHA = rep(0, 20), DATA_TYPE = "continuous", SEED = 1)
rd <- rating_data(dt$rating, dt$id_item, dt$id_worker)
print(rd)
#> - Data type: continuous 
#> - Items: 20 
#> - Workers: 20 
#> - Average budget per item: 5 
#> - Average load per worker: 5 
#> - n: 100 
```
