# From model parameters to agreement

From model parameters to agreement

## Usage

``` r
par2agr(
  PHI,
  ALPHA = NULL,
  BETA = NULL,
  K0 = NULL,
  K1 = NULL,
  ADJUST = FALSE,
  N_DEGENERATE = 0L
)
```

## Arguments

- PHI:

  dispersion parameter

- ALPHA:

  item-specific intercepts

- BETA:

  worker-specific intercepts

- K0:

  zero-inflation threshold

- K1:

  one-inflation threshold

- ADJUST:

  logical; if `TRUE`, degenerate items (dropped from estimation, i.e.
  not in `ALPHA`) are included in the overall mean with a unit
  contribution. Requires `ALPHA`.

- N_DEGENERATE:

  number of degenerate items dropped before estimation. Used only when
  `ADJUST = TRUE`.

## Value

return agreement measure according to the estimated parameters
