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

topic <- subset(roitero2021, topic_id == 403)
nrow(topic)
#> [1] 1456
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
#> [1] 432
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
#> - Inflation: zeros = 45.8% / ones = 4.6% 
#> - Items: 108 ( 16 degenerate )
#> - Average budget per item: 4 
#> - n: 432
```

As printed, the checks detected two degenerate items. These correspond
to documents where the same value is given by all raters

``` r

degen_docs <- levels(factor(topic$document_id))[rd$degen_ids]
d <- subset(topic, document_id %in% degen_docs, select = c(document_id, relevance_score))
d[order(d$document_id), ]
#>             document_id relevance_score
#> 157105      FBIS4-20504               0
#> 157767      FBIS4-20504               0
#> 158383      FBIS4-20504               0
#> 157161      FBIS4-44913               0
#> 157331      FBIS4-44913               0
#> 157468      FBIS4-44913               0
#> 158122      FBIS4-44913               0
#> 157747      FBIS4-46649               0
#> 157972      FBIS4-46649               0
#> 157133 FR940525-1-00086               0
#> 157335 FR940525-1-00086               0
#> 158314 FR940525-1-00086               0
#> 157823 FR940603-0-00134               0
#> 158271 FR940603-0-00134               0
#> 157455 FR940603-0-00153               0
#> 157442 FR940725-2-00141               8
#> 157251 FR940902-1-00048               0
#> 157368 FR940902-1-00048               0
#> 157806 FR940902-1-00048               0
#> 157988 FR940902-1-00048               0
#> 158406 FR940902-1-00048               0
#> 157206 FR940913-2-00004               0
#> 158281 FR940913-2-00004               0
#> 157822 FR941130-0-00122               0
#> 157255        FT911-805               0
#> 157390        FT911-805               0
#> 157970        FT911-805               0
#> 158184        FT911-805               0
#> 158237        FT911-805               0
#> 158388        FT911-805               0
#> 157163       FT944-2249               0
#> 158261       FT944-2249               0
#> 158385       FT944-2249               0
#> 157123       FT944-2266               0
#> 157638       FT944-2266               0
#> 157967       FT944-2266               0
#> 158269       FT944-2266               0
#> 157152       FT944-2690               0
#> 157433       FT944-2690               0
#> 157966       FT944-2690               0
#> 158318       FT944-2690               0
#> 158127    LA091489-0031               0
#> 157729    LA120689-0083               3
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
#>        phi         k0         k1    alpha_1    alpha_2    alpha_3    alpha_4 
#>  2.7809964 -1.2809263  2.5849451 -1.7610438 -0.8840457 -0.8055866 -1.4486444 
#>    alpha_5    alpha_6    alpha_7 
#> -1.2831001 -2.1263205 -0.5893108
```

where alphas for degenrate items are reported with infinite values

``` r

coef(fit)[paste0("alpha_", rd$degen_ids)]
#>   alpha_8  alpha_11  alpha_13  alpha_29  alpha_32  alpha_34  alpha_37  alpha_40 
#>      -Inf      -Inf      -Inf      -Inf      -Inf      -Inf -1.064533      -Inf 
#>  alpha_44  alpha_48  alpha_53  alpha_62  alpha_64  alpha_65 alpha_100 alpha_108 
#>      -Inf      -Inf      -Inf      -Inf      -Inf      -Inf      -Inf -1.405803
```

The [`confint()`](https://rdrr.io/r/stats/confint.html) method returns
the confidence intervals for parameter estimates (`phi`, `k0`, `k1`) as
well as for agreement. Note that, for the latter, intervals are
constructed via delta method

``` r

confint(fit)
#> $parameters
#>      Estimate Std. Error     2.5 %     97.5 %
#> phi  2.780996  0.2943545  2.204072  3.3579206
#> k0  -1.280926  0.1440354 -1.563230 -0.9986222
#> k1   2.584945  0.2596668  2.076008  3.0938826
#> 
#> $agreement
#>            Estimate Std. Error     2.5 %    97.5 %
#> agreement 0.3562843 0.02045248 0.3161982 0.3963704
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
