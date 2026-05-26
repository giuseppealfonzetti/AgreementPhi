# Plot a rating_data object

Plot a rating_data object

## Usage

``` r
# S3 method for class 'rating_data'
plot(x, ...)
```

## Arguments

- x:

  A `rating_data` object from
  [`rating_data()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/rating_data.md).

- ...:

  Ignored.

## Value

Invisibly returns `x`. Called for its side effect (a rating matrix
plot).

## Examples

``` r
dt <- sim_data(J = 20, B = 5, AGREEMENT = 0.6,
               ALPHA = rep(0, 20), DATA_TYPE = "continuous", SEED = 1)
rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
plot(rd)

```
