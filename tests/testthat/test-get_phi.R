test_that("cpp_get_phi recovers agreement for continuous data", {
  set.seed(123)
  items <- 15
  workers <- 20
  budget <- 8
  agr <- 0.7
  dt <- sim_data(
    J = items,
    B = budget,
    W = workers,
    AGREEMENT = agr,
    ALPHA = rnorm(items),
    BETA = rnorm(workers),
    DATA_TYPE = "continuous"
  )

  ctrl <- list(
    SEARCH_RANGE = 5,
    MAX_ITER = 50,
    PROF_SEARCH_RANGE = 5,
    PROF_MAX_ITER = 50,
    ALT_MAX_ITER = 10,
    ALT_TOL = 1e-3
  )

  args <- c(
    list(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      ALPHA_START = rep(0, items),
      BETA_START = rep(0, workers),
      TAU_START = c(0, 1),
      PHI_START = agr2prec(0.5),
      J = items,
      W = workers,
      K = 1,
      METHOD = "profile",
      DATA_TYPE = "continuous",
      WORKER_NUISANCE = TRUE,
      THRESHOLDS_NUISANCE = FALSE,
      VERBOSE = FALSE
    ),
    ctrl
  )

  res <- do.call(cpp_get_phi, args)
  expect_equal(length(res), 2L)
  expect_equal(prec2agr(res[1]), agr, tolerance = 0.1)
})

test_that("cpp_get_phi returns modified/profile estimates for ordinal data", {
  set.seed(321)
  items <- 20
  workers <- 20
  budget <- 6
  agr <- 0.65
  K <- 6
  dt <- sim_data(
    J = items,
    B = budget,
    W = workers,
    AGREEMENT = agr,
    ALPHA = rnorm(items),
    BETA = rnorm(workers),
    K = K,
    DATA_TYPE = "ordinal"
  )
  counts <- tabulate(dt$rating, nbins = K)
  tau_start <- c(0, cumsum(counts / sum(counts))[-K], 1)

  ctrl <- list(
    SEARCH_RANGE = 5,
    MAX_ITER = 50,
    PROF_SEARCH_RANGE = 5,
    PROF_MAX_ITER = 50,
    ALT_MAX_ITER = 15,
    ALT_TOL = 1e-3
  )

  args <- c(
    list(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      ALPHA_START = rep(0, items),
      BETA_START = rep(0, workers),
      TAU_START = tau_start,
      PHI_START = agr2prec(0.5),
      J = items,
      W = workers,
      K = K,
      METHOD = "modified",
      DATA_TYPE = "ordinal",
      WORKER_NUISANCE = TRUE,
      THRESHOLDS_NUISANCE = TRUE,
      VERBOSE = FALSE
    ),
    ctrl
  )

  res <- do.call(cpp_get_phi, args)
  expect_equal(length(res), 3L)
  expect_lt(abs(prec2agr(res[1]) - agr), .2)
  expect_lt(abs(prec2agr(res[3]) - agr), .2)
})
