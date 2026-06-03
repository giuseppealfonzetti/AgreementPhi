# Confidence intervals for an agreement fit

Confidence intervals for an agreement fit

## Usage

``` r
# S3 method for class 'agreement_fit'
confint(object, parm = NULL, level = 0.95, ...)
```

## Arguments

- object:

  Object fitted with
  [`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md).

- parm:

  Ignored (included for S3 compatibility).

- level:

  Confidence level. Default `0.95`.

- ...:

  Ignored.

## Value

A named list with two elements, each a numeric matrix with columns
`Estimate`, `Std. Error`, and the lower/upper confidence bounds:

- `parameters`:

  Parameter-scale estimates. One row (`phi`) for non-inflated data;
  three rows (`phi`, `k0`, `k1`) for inflated data.

- `agreement`:

  Agreement-scale estimate. Always one row (`agreement`).
