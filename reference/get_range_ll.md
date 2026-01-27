# Get log-likelihood range

Get log-likelihood range

## Usage

``` r
get_range_ll(X, RANGE = 0.2, GRID_LENGTH = 15)
```

## Arguments

- X:

  Object fitted with
  [`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md)
  function.

- RANGE:

  Range around agreement mle.

- GRID_LENGTH:

  Number of points to be evaluated within RANGE.

## Value

Return a data.frame with GRID_LENGTH rows and columns `precision`,
`agreement`, `profile` and `modified`.
