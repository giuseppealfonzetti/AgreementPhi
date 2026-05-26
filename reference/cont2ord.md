# Discretise continuous data

Discretise continuous data

## Usage

``` r
cont2ord(X, K, TRESHOLDS = NULL)
```

## Arguments

- X:

  Vector of continuous data in (0,1).

- K:

  Number of ordinal categories.

- TRESHOLDS:

  Threshold vector of length K-1. If null, thresholds are assumed to be
  equispaced.

## Value

Discretised vector

## Examples

``` r
x <- c(0,runif(5,0,1),1)
x
#> [1] 0.0000000 0.8004792 0.1600811 0.4991498 0.1388658 0.6885982 1.0000000
cont2ord(x, K=3)
#> [1] 1 3 1 2 1 3 3

```
