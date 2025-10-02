#### one-way ####
items <- 100
budget_per_item <- 10
k <- 10
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
head(dt)
length(unique(dt$id_worker))

#### two-way ####
items <- 100
budget_per_item <- 10
workers <- 300
k <- 10
alphas <- rnorm(items)
betas <- rnorm(workers)
agr <- runif(1)
dt2 <- sim_data(
  J = items,
  B = budget_per_item,
  W = workers,
  AGREEMENT = .1,
  ALPHA = alphas,
  # BETA = betas,
  DATA_TYPE = "continuous",
  K = k,
  SEED = 123
)

head(dt2)
length(unique(dt2$id_worker))
as.vector(replicate(items, sample(1:workers, budget_per_item, replace = FALSE)))
max(table(dt2$id_worker, dt2$id_item))
