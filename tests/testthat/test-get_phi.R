#### Continuous ####
test_that("get_phi twoway recovers true agreement | continuous", {
  set.seed(123)
  items <- 20
  budget <- 20
  workers <- 50
  agr <- .7
  alphas <- rnorm(items)
  betas <- rnorm(workers)
  K <- 10
  dt <- sim_data(
    J = items,
    B = budget,
    W = workers,
    K = K,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "continuous"
  )

  p_mle <- get_phi_profile_twoway(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    LAMBDA_START = rep(0, items + workers - 1),
    PHI_START = agr2prec(0.5),
    K = 1,
    J = items,
    W = workers,
    DATA_TYPE = "continuous",
    SEARCH_RANGE = 10,
    MAX_ITER = 100,
    PROF_MAX_ITER = 100,
    VERBOSE = FALSE
  )

  mp_mle <- get_phi_modified_profile_twoway(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    LAMBDA_START = rep(0, items + workers - 1),
    PHI_START = agr2prec(0.5),
    K = K,
    J = items,
    W = workers,
    DATA_TYPE = "continuous",
    SEARCH_RANGE = 10,
    MAX_ITER = 100,
    PROF_MAX_ITER = 100,
    VERBOSE = FALSE
  )

  prec2agr(mp_mle$pl_precision)
  prec2agr(mp_mle$mpl_precision)
  expect_lt(abs(prec2agr(mp_mle$mpl_precision) - agr), 0.1)
})
#### ordinal ####
if (0) {
  test_that("get_phi twoway recovers true agreement | continuous", {
    set.seed(123)
    items <- 20
    budget <- 20
    workers <- 50
    agr <- .7
    alphas <- rnorm(items)
    betas <- rnorm(workers)
    K <- 10
    dt <- sim_data(
      J = items,
      B = budget,
      W = workers,
      K = K,
      AGREEMENT = agr,
      ALPHA = alphas,
      BETA = betas,
      DATA_TYPE = "ordinal"
    )

    p_mle <- get_phi_profile_twoway(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      LAMBDA_START = rep(0, items + workers - 1),
      PHI_START = agr2prec(0.5),
      K = K,
      J = items,
      W = workers,
      DATA_TYPE = "ordinal",
      SEARCH_RANGE = 10,
      MAX_ITER = 100,
      PROF_MAX_ITER = 100,
      VERBOSE = FALSE
    )

    mp_mle <- get_phi_modified_profile_twoway(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      LAMBDA_START = rep(0, items + workers - 1),
      PHI_START = agr2prec(0.5),
      K = K,
      J = items,
      W = workers,
      DATA_TYPE = "ordinal",
      SEARCH_RANGE = 10,
      MAX_ITER = 100,
      PROF_MAX_ITER = 100,
      VERBOSE = FALSE
    )

    prec2agr(mp_mle$pl_precision)
    prec2agr(mp_mle$mpl_precision)
    expect_lt(abs(prec2agr(mp_mle$mpl_precision) - agr), 0.1)
  })
}
