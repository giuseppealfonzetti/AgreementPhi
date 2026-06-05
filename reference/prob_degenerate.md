# Model-based probability of item degeneracy

For each item, computes the probability that all raters give the same
rating according to the fitted model. Returns 0 for continuous fits
(exact ties have zero probability under a continuous distribution). Not
defined for two-way models (raises an error).

## Usage

``` r
prob_degenerate(object)
```

## Arguments

- object:

  An `agreement_fit` object from
  [`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md).

## Value

A named numeric vector of length J (all items, including any degenerate
items detected before fitting). Degenerate items always get
probability 1. Names are `item_1, ..., item_J` or `item_<label>` when
item labels are available.
