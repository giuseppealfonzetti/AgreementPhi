# Analysing crowdsourced relevance assessments (Roitero et al. 2021)

## The data

The `roitero2021` dataset is a subset of the crowdsourcing experiment in
Roitero et al. (2021), restricted to assessments collected on the
100-point continuous (S100) relevance scale. Each row is one worker’s
`relevance_score` for one `document_id` within a `topic_id`. The data
are fetched on demand from the [upstream
repository](https://github.com/KevinRoitero/CrowdsourcingRelevanceScales).

``` r

roitero2021 <- download_roitero2021()
str(roitero2021)
#> 'data.frame':    56472 obs. of  11 variables:
#>  $ topic_id       : int  445 445 445 445 445 445 445 445 445 445 ...
#>  $ unit_id        : int  0 0 0 0 0 0 0 0 1 1 ...
#>  $ document_id    : chr  "FT941-7858" "FT941-5312" "FT943-9241" "FT924-8156" ...
#>  $ gold           : chr  "null" "null" "null" "H" ...
#>  $ doc_pos        : int  0 1 2 3 4 5 6 7 0 1 ...
#>  $ worker_id      : chr  "gAAAAABgpOXVqNEjGuhKGPD07r9ZE4aIETRpKrbp78KF-qmEVxiruzngZVFS7Y8DuOGnqYngOpkQ-CtyeHxF6fZ_kiP-RCn9Gw==" "gAAAAABgpOXVqNEjGuhKGPD07r9ZE4aIETRpKrbp78KF-qmEVxiruzngZVFS7Y8DuOGnqYngOpkQ-CtyeHxF6fZ_kiP-RCn9Gw==" "gAAAAABgpOXVqNEjGuhKGPD07r9ZE4aIETRpKrbp78KF-qmEVxiruzngZVFS7Y8DuOGnqYngOpkQ-CtyeHxF6fZ_kiP-RCn9Gw==" "gAAAAABgpOXVqNEjGuhKGPD07r9ZE4aIETRpKrbp78KF-qmEVxiruzngZVFS7Y8DuOGnqYngOpkQ-CtyeHxF6fZ_kiP-RCn9Gw==" ...
#>  $ relevance_score: num  100 60 0 100 75 0 0 0 98 0 ...
#>  $ cumulative_time: num  49.1 72 33.9 26.9 55.2 ...
#>  $ comment        : chr  "It's about women as priests in England, spot on." "A small part of the text is about it \"Women priests Saturday sees the first ordination in Britain of women pri"| __truncated__ "Nothing at all to be found." "Once again about female priests in England, completely relevant." ...
#>  $ trec           : int  1 0 1 1 1 0 0 0 1 0 ...
#>  $ sormunen       : int  2 NA 0 3 1 0 NA NA 1 0 ...
```

For simplicity, here we focus on a single topic and look at how the
related items were assessed.

``` r

topic <- subset(roitero2021, topic_id == 418)
nrow(topic)
#> [1] 3216
```

We retain only workers who “correctly” responded to the gold items
(high-relevance `"H"` above threshold and non-relevant `"N"` below
threshold) and then keep only non-gold items.

``` r


select_workers <- intersect(
  unique(topic[topic$relevance_score > 50 & topic$gold == "H", ]$unit_id),
  unique(topic[topic$relevance_score < 50 & topic$gold == "N", ]$unit_id)
)

topic <- subset(topic, unit_id %in% select_workers & gold == "null")
nrow(topic)
#> [1] 2088
```

As scores were collected on a 0–100 scale, we map them onto `[0,1]` by
simply dividing by 100. To build a `rating_data` object we use the
[`rating_data()`](https://giuseppealfonzetti.github.io/AgreementPhi/reference/rating_data.md)
function. Since exact 0s and 100s are present in the data, the function
automatically detects the inflated beta ratings.

``` r

ratings <- topic$relevance_score / 100
items <- as.integer(factor(topic$document_id))
rd <- rating_data(ratings, items)
rd
#> - Data type: inflated 
#> - Inflation: zeros = 46.1% / ones = 2.4% 
#> - Items: 241 ( 1 degenerate )
#> - Average budget per item: 8.66 
#> - n: 2088
```

As printed, the checks detected two degenerate items. These correspond
to documents where the same value is given by all raters

``` r

degen_docs <- levels(factor(topic$document_id))[rd$degen_ids]
d <- subset(topic, document_id %in% degen_docs, select = c(document_id, relevance_score))
d[order(d$document_id), ]
#>        document_id relevance_score
#> 141537 FBIS3-26358               0
#> 141872 FBIS3-26358               0
#> 142406 FBIS3-26358               0
#> 143087 FBIS3-26358               0
#> 143612 FBIS3-26358               0
#> 143936 FBIS3-26358               0
```

## Fitting the agreement model

The inflated interval model is a one-way model: the item effects are
profiled out as nuisance parameters. The argument `METHOD = "modified"`
denotes that the modified profile likelihood is used for estimation

``` r

fit <- agreement(rd, NUISANCE = "items", METHOD = "modified")
```

We can extract estimated coefficients with the familiar
[`coef()`](https://rdrr.io/r/stats/coef.html) method

``` r

coef(fit)[1:10]
#>       phi        k0        k1   alpha_1   alpha_2   alpha_3   alpha_4   alpha_5 
#>  2.841631 -1.328231  2.974278 -1.597678 -1.382612 -1.729650 -1.646931 -1.248158 
#>   alpha_6   alpha_7 
#> -1.330361 -1.540069
```

where alphas for degenrate items are reported with infinite values

``` r

coef(fit)[paste0("alpha_", rd$degen_ids)]
#> alpha_12 
#>     -Inf
```

The [`confint()`](https://rdrr.io/r/stats/confint.html) method returns
the confidence intervals for parameter estimates (`phi`, `k0`, `k1`) as
well as for agreement. Note that, for the latter, intervals are
constructed via delta method

``` r

confint(fit)
#> $parameters
#>      Estimate Std. Error     2.5 %    97.5 %
#> phi  2.841631 0.12506931  2.596500  3.086763
#> k0  -1.328231 0.06043951 -1.446690 -1.209772
#> k1   2.974278 0.15372395  2.672985  3.275572
#> 
#> $agreement
#>            Estimate Std. Error     2.5 %    97.5 %
#> agreement 0.2828532 0.01201713 0.2593001 0.3064064
```

## Item effects

The original data also contain TREC labels, which correspond to
evaluations given by experts on a binary scale (relevant = 1 /
non-relevant = 0). We can plot the estimated $`\alpha`$ vector to
visualise how the model captured the different relevance levels across
items

``` r

doc_levels <- levels(factor(topic$document_id))
alphas <- coef(fit)[grep("^alpha", names(coef(fit)))]
trec <- tapply(topic$trec, topic$document_id, function(x) x[!is.na(x)][1])
boxplot(
  alphas ~ factor(trec[doc_levels]),
  xlab = "TREC", 
  ylab = "Item effect",
  col = c("#F28E2B", "#4E79A7"))
#> Warning in bplt(at[i], wid = width[i], stats = z$stats[, i], out =
#> z$out[z$group == : Outlier (-Inf) in boxplot 1 is not drawn
```

![](roitero2021_files/figure-html/item-effects-1.png)
