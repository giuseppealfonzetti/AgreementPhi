# Intoduction

## Basic usage

The `AgreementPhi` package exports a utility function to simulate data
by providing the true agreement and item effects. Consider for example
to simulate continuous ratings for 200 items, collecting 8 relevance
assessments per item, for a total of 1600 responses. We set the true
agreement at $`\Phi=0.4`$

``` r

library(AgreementPhi)
set.seed(321)
items <- 200
budget_per_item <- 8
alphas <- runif(items, -1, 1)
agr <- .4

dt <- sim_data(
  J = items,
  B = budget_per_item,
  AGREEMENT = agr,
  DATA_TYPE = "continuous",
  ALPHA = alphas
)
```

The simulated 1600 ratings are stored in `dt$ratings`, while
`dt$id_item` and `dt$id_worker` store the item and worker indices
related to each rating

``` r

names(dt)
#> [1] "id_item"   "id_worker" "rating"
head(dt$id_item)
#> [1] 20 44 45 51 83 92
head(dt$id_worker)
#> [1] 1 1 1 1 1 1
head(dt$rating)
#> [1] 0.6375834 0.7420828 0.9866412 0.4995354 0.2111970 0.0271688
length(dt$rating)
#> [1] 1600
```

Use
[`rating_data()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/rating_data.md)
to validate the input and construct a `rating_data` object. The function
reports diagnostics and, for two-way data, supports
[`plot()`](https://rdrr.io/r/graphics/plot.default.html)

``` r

rd <- rating_data(dt$rating, dt$id_item, dt$id_worker)
#>  - Detected 200 items and 200 workers.
#>  - Detected continuous data on the (0,1) range.
#>  - Average number of observed ratings per item is 8.
#>  - Average number of observed ratings per worker is 8.
```

``` r

plot(rd)
```

![](intro_files/figure-html/plot_data-1.png)

The core function of the `AgreementPhi` package is
[`agreement()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/agreement.md),
which implements the numerical algorithms to estimate the $`\Phi`$
agreement via profile and modified profile likelihood methods. It takes
a `rating_data` object as its first argument. For the estimation via
profile likelihood, you can specify `METHOD="profile"`

``` r

fit_profile <- agreement(rd, NUISANCE = c("items"), METHOD = "profile", VERBOSE = TRUE)
#> 
#> MODEL PARAMETERS
#>  - Constant effects: workers
#>  - Nuisance effects: items
#> Done!
```

When the `VERBOSE` option is chosen, the function prints on screen some
useful information about data dimensions and sparsity. In addition, it
also provides an overview of how items and worker effects are treated.
In this case, for example, worker effects are considered as constant
(set at zero by default), while items effects are profiled out as
nuisance parameters. The agreement and precision estimates are available
at

``` r

fit_profile$profile
#> $precision
#> [1] 2.446563
#> 
#> $agreement
#> [1] 0.4444125
```

while the maximum likelihood estimates of the nuisance parameters are
available at

``` r

head(fit_profile$alpha)
#> [1]  1.0698711  1.4333241 -0.2898244 -0.9135686  0.3879414  0.1008767
```

To use the modified likelihood approach, it is enough to change the
`METHOD` argument to `modified`.

``` r

fit_modified <- agreement(rd, NUISANCE = c("items"), METHOD = "modified", VERBOSE = TRUE)
#> 
#> MODEL PARAMETERS
#>  - Constant effects: workers
#>  - Nuisance effects: items
#> Non-adjusted agreement: 0.444413
#> Adjusted agreement: 0.400505
#> Done!
```

As it can be read from the verbose output, when `METHOD = "modified"`,
the proposed algorithm first optimises the profile likelihood to
evaluate the maximum likelihood estimators needed to construct the
modified profile likelihood. Thus, both estimates can be retrieved from
the fitted object

``` r

fit_modified$profile
#> $precision
#> [1] 2.446563
#> 
#> $agreement
#> [1] 0.4444125
fit_modified$modified
#> $precision
#> [1] 2.129941
#> 
#> $agreement
#> [1] 0.4005054
```

Once the point estimates are computed, we can draw inference on
agreement by using the
[`confint()`](https://rdrr.io/r/stats/confint.html) S3 method to
construct confidence intervals. The function will automatically
recognise if the estimates are related to the profile or modified
likelihood approach by looking at the fitted object

``` r

ci_profile <- confint(fit_profile)
ci_profile
#> $agreement_est
#> [1] 0.4444125
#> 
#> $agreement_se
#> [1] 0.0102727
#> 
#> $agreement_ci
#> [1] 0.4242784 0.4645466

ci_modified <- confint(fit_modified)
ci_modified
#> $agreement_est
#> [1] 0.4005054
#> 
#> $agreement_se
#> [1] 0.0103424
#> 
#> $agreement_ci
#> [1] 0.3802347 0.4207762
```

For convenience,
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) visualises the
relative log-likelihood profile and confidence interval in one step

``` r

plot(fit_modified)
```

![](intro_files/figure-html/plot_fit-1.png)

If you need the grid data for further analysis, compute it with
[`get_range_ll()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/get_range_ll.md)
and pass it back to
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) to avoid
recomputation

``` r

range_ll <- get_range_ll(fit_modified)
plot(fit_modified, RANGE_LL = range_ll)
```

![](intro_files/figure-html/plot_fit_precomputed-1.png)
