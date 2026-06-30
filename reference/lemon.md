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

- Smithson, Michael, and Jay Verkuilen. 2006. "A Better Lemon Squeezer?
  Maximum-Likelihood Regression with Beta-Distributed Dependent
  Variables." *Psychological Methods* **11(1)**: 54-71.
  [doi](https://psycnet.apa.org/doi/10.1037/1082-989X.11.1.54)

## Examples

``` r
x <- c(0,runif(5,0,1),1)
x
#> [1] 0.0000000 0.4806419 0.2078129 0.7177213 0.3334211 0.5294274 1.0000000
lemon(x)
#> [1] 0.07142857 0.48340737 0.24955389 0.68661829 0.35721806 0.52522347 0.92857143
```
