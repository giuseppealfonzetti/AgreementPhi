test_that("oneway continuous works", {
  set.seed(321)
  # setting dimension
  items <- 200
  budget_per_item <- 5
  n_obs <- items * budget_per_item
  workers <- 200
  # item-specific intercepts to generate the data
  alphas <- runif(items, -2, 2)
  betas <- rep(0, workers)

  agr <- .7
  dt_oneway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )

  fit <- agreement(
    RATINGS = dt_oneway$rating,
    ITEM_INDS = dt_oneway$id_item,
    WORKER_INDS = dt_oneway$id_worker,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )

  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)

  agr <- .3
  dt_oneway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )

  fit <- agreement(
    RATINGS = dt_oneway$rating,
    ITEM_INDS = dt_oneway$id_item,
    WORKER_INDS = dt_oneway$id_worker,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )

  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)

  fitp <- agreement(
    RATINGS = dt_oneway$rating,
    ITEM_INDS = dt_oneway$id_item,
    WORKER_INDS = dt_oneway$id_worker,
    METHOD = "profile",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )

  expect_equal(fitp$profile$agreement, fit$profile$agreement)
})

test_that("oneway ordinal works", {
  set.seed(321)
  # setting dimension
  items <- 50
  budget_per_item <- 5
  n_obs <- items * budget_per_item
  workers <- 50
  # item-specific intercepts to generate the data
  alphas <- runif(items, -2, 2)
  betas <- rep(0, workers)

  agr <- .7
  dt_oneway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "ordinal",
    K = 6,
    SEED = 123
  )

  fit <- agreement(
    RATINGS = dt_oneway$rating,
    ITEM_INDS = dt_oneway$id_item,
    WORKER_INDS = dt_oneway$id_worker,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )

  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )

  fitp <- agreement(
    RATINGS = dt_oneway$rating,
    ITEM_INDS = dt_oneway$id_item,
    WORKER_INDS = dt_oneway$id_worker,
    METHOD = "profile",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )

  expect_equal(fitp$profile$agreement, fit$profile$agreement)
})

test_that("twoway continuous works", {
  set.seed(321)
  # setting dimension
  items <- 100
  budget_per_item <- 10
  n_obs <- items * budget_per_item
  workers <- 100
  alphas <- runif(items, -2, 2)
  betas <- c(0, runif(workers - 1, -2, 2))

  agr <- .7
  dt_twoway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )

  fit <- agreement(
    RATINGS = dt_twoway$rating,
    ITEM_INDS = dt_twoway$id_item,
    WORKER_INDS = dt_twoway$id_worker,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = TRUE
  )

  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)

  agr <- .3
  dt_twoway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )

  fit <- agreement(
    RATINGS = dt_twoway$rating,
    ITEM_INDS = dt_twoway$id_item,
    WORKER_INDS = dt_twoway$id_worker,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = TRUE
  )

  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)
})

test_that("twoway ordinal works", {
  set.seed(321)
  # setting dimension
  items <- 10
  budget_per_item <- 5
  n_obs <- items * budget_per_item
  workers <- 20
  alphas <- runif(items, -1, 1)
  betas <- c(0, runif(workers - 1, -1, 1))

  agr <- .4
  dt_twoway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "ordinal",
    K = 6,
    SEED = 123
  )

  fit <- agreement(
    RATINGS = dt_twoway$rating,
    ITEM_INDS = dt_twoway$id_item,
    WORKER_INDS = dt_twoway$id_worker,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = TRUE
  )

  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
})

test_that("agreement handles ordinal data with missing boundary categories", {
  dt <- sim_data(
    J = 30, B = 5, AGREEMENT = 0.6,
    ALPHA = runif(30, -2, 2),
    DATA_TYPE = "ordinal", K = 10, SEED = 42
  )
  dt_sub <- dt[dt$rating >= 2 & dt$rating <= 9, ]

  expect_no_error(
    fit <- agreement(
      RATINGS = dt_sub$rating,
      ITEM_INDS = dt_sub$id_item,
      WORKER_INDS = dt_sub$id_worker,
      K = 10,
      NUISANCE = "items"
    )
  )
  expect_true(fit$profile$agreement >= 0 && fit$profile$agreement <= 1)
  expect_equal(fit$cpp_args$K, 10L)
})
