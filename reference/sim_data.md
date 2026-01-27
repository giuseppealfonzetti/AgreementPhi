# Simulate ordinal or continuous (0,1) ratings

Simulate ordinal or continuous (0,1) ratings

## Usage

``` r
sim_data(
  J,
  B,
  W = J,
  AGREEMENT,
  ALPHA,
  BETA = NULL,
  DATA_TYPE = c("ordinal", "continuous"),
  K = 6,
  SEED = 123
)
```

## Arguments

- J:

  Number of items.

- B:

  Budget per item (i.e. number of workers assigned to each item).

- W:

  Maximum number of workers.

- AGREEMENT:

  General agreement.

- ALPHA:

  Item-specific intercepts.

- BETA:

  Worker-specific intercepts.

- DATA_TYPE:

  Choose between `ordinal` or `continuous`.

- K:

  Number of categories in case of ordinal data.

- SEED:

  RNG seed.

## Value

Returns a dataframe with columns id_items, id_worker and rating

## Examples

``` r
set.seed(123)

# generate from one-way model
# (varying item effects, worker effects fixed to zero)
dt1way <- sim_data(
 J = 50,
 B = 5,
 AGREEMENT = .8,
 ALPHA = runif(50, 0, 1),
 DATA_TYPE = "continuous",
 SEED = 123
)
# generate from two-way model
# (varying item effects, varying worker effects)
dt2way <- sim_data(
 J = 50,
 W = 40,
 B = 5,
 AGREEMENT = .8,
 ALPHA = runif(50, 0, 1),
 BETA = runif(40, 0, 1),
 DATA_TYPE = "continuous",
 SEED = 123
)
```
