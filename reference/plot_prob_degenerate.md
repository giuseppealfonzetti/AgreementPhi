# Forest plot of model-based probability of item degeneracy

Plots per-item P(degenerate) estimates and their confidence intervals
from
[`confint_prob_degenerate()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/confint_prob_degenerate.md).
Items are colour-coded: observed degenerate items (P = 1, CI collapsed)
in orange; non-degenerate items in blue.

## Usage

``` r
plot_prob_degenerate(x, LEVEL = 0.95, SORT = TRUE, ...)
```

## Arguments

- x:

  An `agreement_fit` object from
  [`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md).

- LEVEL:

  Confidence level passed to
  [`confint_prob_degenerate()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/confint_prob_degenerate.md).
  Default `0.95`.

- SORT:

  Logical; sort items by estimate before plotting. Default `TRUE`.

- ...:

  Ignored (required for S3 consistency).

## Value

Invisibly returns the matrix from
[`confint_prob_degenerate()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/confint_prob_degenerate.md),
in the order plotted.
