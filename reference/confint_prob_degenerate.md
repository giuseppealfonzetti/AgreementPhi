# Confidence intervals for model-based probability of item degeneracy

Applies the delta method to
[`prob_degenerate()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/prob_degenerate.md),
propagating parameter uncertainty to per-item probabilities. Item
intercepts α_j are treated as fixed at their MLE (plug-in). Not defined
for two-way models.

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
