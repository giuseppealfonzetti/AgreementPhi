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

For non-inflated data: a list with `agreement_est`, `agreement_se`, and
`agreement_ci`. For inflated data: a list with `phi_est`, `phi_se`,
`phi_ci`, `k0_est`, `k0_se`, `k0_ci`, `k1_est`, `k1_se`, `k1_ci`.
