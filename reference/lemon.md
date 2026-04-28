# Squeeze \[0,1\] data

Squeeze \[0,1\] data

## Usage

``` r
lemon(X, U = NULL)
```

## Arguments

- X:

  Vector of continuous data in \[0,1\].

- U:

  Squeezing parameter. If NULL, default chosen as per Smithson et al.
  (2006).

## Value

Squeezed vector

## References

- Smithson, Michael, and Jay Verkuilen. 2006. “A Better Lemon Squeezer?
  Maximum-Likelihood Regression with Beta-Distributed Dependent
  Variables.” *Psychological Methods* **11(1)**: 54–71.
  [doi](https://psycnet.apa.org/doi/10.1037/1082-989X.11.1.54)

## Examples

``` r
x <- c(0,runif(5,0,1),1)
x
#> [1] 0.0000000 0.5028886 0.7134387 0.7176965 0.8379953 0.4422286 1.0000000
lemon(x)
#> [1] 0.07142857 0.50247595 0.68294748 0.68659700 0.78971028 0.45048167 0.92857143
```
