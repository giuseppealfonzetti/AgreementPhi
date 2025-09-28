items <- 100
budget_per_item <- 10
n_obs <- items * budget_per_item
k <- 5
alphas <- rnorm(items)
agr <- runif(1)

dt <- sim_data(
  J = items,
  B = budget_per_item,
  AGREEMENT = .1,
  ALPHA = alphas,
  DATA_TYPE = "continuous",
  K = k,
  SEED = 123
)
