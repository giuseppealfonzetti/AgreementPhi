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
#> [1] 0.00000000 0.44222862 0.44805073 0.37934982 0.03246628 0.47011544 1.00000000
lemon(x)
#> [1] 0.07142857 0.45048167 0.45547205 0.39658556 0.09925681 0.47438467 0.92857143
```
