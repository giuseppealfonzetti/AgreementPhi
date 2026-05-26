# Extract coefficients from an agreement fit

Extract coefficients from an agreement fit

## Usage

``` r
# S3 method for class 'agreement_fit'
coef(object, ...)
```

## Arguments

- object:

  An `agreement_fit` object from
  [`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md).

- ...:

  Ignored.

## Value

A named numeric vector. Always contains `phi` and `alpha_1...alpha_J`.
For inflated data: also `k0` and `k1`. For two-way models (workers
profiled as nuisance): also `beta_1...beta_W`.

## Examples

``` r
set.seed(1)
dt <- sim_data(J = 20, B = 5, AGREEMENT = 0.6,
               ALPHA = rep(0, 20), DATA_TYPE = "continuous", SEED = 1)
rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
coef(fit)
#>          phi      alpha_1      alpha_2      alpha_3      alpha_4      alpha_5 
#>  3.393244096 -0.555118810 -0.136443219 -0.007767511 -0.688755083 -0.608161674 
#>      alpha_6      alpha_7      alpha_8      alpha_9     alpha_10     alpha_11 
#> -0.631546909  1.119820121 -0.111824190 -0.095840207 -0.404821959 -0.227166568 
#>     alpha_12     alpha_13     alpha_14     alpha_15     alpha_16     alpha_17 
#>  0.056172009 -0.189093898  0.039604126 -0.312190505 -0.273567686 -0.299999841 
#>     alpha_18     alpha_19     alpha_20 
#>  0.072199155  0.683324524  1.019688226 
```
