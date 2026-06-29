# Confidence intervals for model-based probability of item degeneracy

Applies the delta method to
[`prob_degenerate()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/prob_degenerate.md),
propagating parameter uncertainty to per-item probabilities. For the
inflated (ordered beta) model, uncertainty in the item intercepts is
propagated via the partitioned information matrix (full delta method
over the item intercept and the two cutpoints); for the ordinal model
each item intercept is treated as fixed at its MLE (plug-in). Not
defined for two-way models.

## Usage

``` r
confint_prob_degenerate(object, level = 0.95)
```

## Arguments

- object:

  An `agreement_fit` object from
  [`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md).

- level:

  Confidence level. Default `0.95`.

## Value

A named matrix with one row per item and columns `Estimate`,
`Std. Error`, and the two percentile bounds.
