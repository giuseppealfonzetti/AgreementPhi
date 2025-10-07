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

#### alternating minimization #####
# test_that("get_phi matches cpp profile via alterating maximization | continuous", {
#   set.seed(123)
#   items <- 100
#   budget <- 10
#   n_obs <- items * budget
#   workers <- 50
#   agr <- .7
#   alphas <- rnorm(items)
#   betas <- rnorm(workers)
#   K <- 10
#   dt <- sim_data(
#     J = items,
#     B = budget,
#     W = workers,
#     K = K,
#     AGREEMENT = agr,
#     ALPHA = alphas,
#     BETA = betas,
#     DATA_TYPE = "continuous"
#   )

#   # tictoc::tic()
#   p_mle <- get_phi_profile_twoway(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     LAMBDA_START = rep(0, items + workers - 1),
#     PHI_START = agr2prec(0.5),
#     K = 1,
#     J = items,
#     W = workers,
#     DATA_TYPE = "continuous",
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_MAX_ITER = 100,
#     VERBOSE = FALSE
#   )
#   # tictoc::toc()

#   # tictoc::tic()
#   cpp_res <- cpp_twoway_get_phi_profile(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     ALPHA_START = rep(0, items),
#     BETA_START = rep(0, workers),
#     PHI_START = agr2prec(0.5),
#     K = 1,
#     J = items,
#     W = workers,
#     CONTINUOUS = TRUE,
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_SEARCH_RANGE = 10,
#     PROF_MAX_ITER = 100,
#     ALT_MAX_ITER = 10,
#     ALT_TOL = 1e-5
#   )
#   # tictoc::toc()

#   expect_equal(
#     p_mle$precision,
#     cpp_res[1],
#     tolerance = 1e-2
#   )

#   expect_equal(
#     p_mle$loglik,
#     -cpp_res[2],
#     tolerance = 1e-2
#   )

#   # tictoc::tic()
#   mp_mle <- get_phi_modified_profile_twoway(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     LAMBDA_START = rep(0, items + workers - 1),
#     PHI_START = agr2prec(0.5),
#     K = K,
#     J = items,
#     W = workers,
#     DATA_TYPE = "continuous",
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_MAX_ITER = 100,
#     VERBOSE = FALSE
#   )
#   # tictoc::toc()

#   # tictoc::tic()
#   cpp_res_mp <- cpp_twoway_get_phi_modified_profile(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     ALPHA_START = rep(0, items),
#     BETA_START = rep(0, workers),
#     PHI_START = agr2prec(0.5),
#     K = 1,
#     J = items,
#     W = workers,
#     CONTINUOUS = TRUE,
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_SEARCH_RANGE = 10,
#     PROF_MAX_ITER = 100,
#     ALT_MAX_ITER = 10,
#     ALT_TOL = 1e-5,
#     VERBOSE = TRUE
#   )
#   # tictoc::toc()

#   expect_equal(
#     mp_mle$mpl_precision,
#     cpp_res_mp[1],
#     tolerance = 1e-2
#   )

#   expect_equal(
#     mp_mle$loglik / n_obs,
#     cpp_res_mp[2] / n_obs,
#     tolerance = 1e-2
#   )
# })
# test_that("get_phi matches cpp profile via alterating maximization | ordinal", {
#   set.seed(123)
#   items <- 30
#   budget <- 10
#   n_obs <- items * budget
#   workers <- 20
#   agr <- .7
#   alphas <- rnorm(items)
#   betas <- rnorm(workers)
#   K <- 10
#   dt <- sim_data(
#     J = items,
#     B = budget,
#     W = workers,
#     K = K,
#     AGREEMENT = agr,
#     ALPHA = alphas,
#     BETA = betas,
#     DATA_TYPE = "ordinal"
#   )

#   # tictoc::tic()
#   p_mle <- get_phi_profile_twoway(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     LAMBDA_START = rep(0, items + workers - 1),
#     PHI_START = agr2prec(0.5),
#     K = K,
#     J = items,
#     W = workers,
#     DATA_TYPE = "ordinal",
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_MAX_ITER = 10,
#     VERBOSE = FALSE
#   )
#   # tictoc::toc()

#   # tictoc::tic()
#   cpp_res <- cpp_twoway_get_phi_profile(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     ALPHA_START = rep(0, items),
#     BETA_START = rep(0, workers),
#     PHI_START = agr2prec(0.5),
#     K = K,
#     J = items,
#     W = workers,
#     CONTINUOUS = FALSE,
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_SEARCH_RANGE = 10,
#     PROF_MAX_ITER = 10,
#     ALT_MAX_ITER = 10,
#     ALT_TOL = 1e-5
#   )
#   # tictoc::toc()

#   expect_equal(
#     p_mle$precision,
#     cpp_res[1],
#     tolerance = 1e-2
#   )

#   expect_equal(
#     p_mle$loglik,
#     -cpp_res[2],
#     tolerance = 1e-2
#   )

#   # tictoc::tic()
#   mp_mle <- get_phi_modified_profile_twoway(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     LAMBDA_START = rep(0, items + workers - 1),
#     PHI_START = agr2prec(0.5),
#     K = K,
#     J = items,
#     W = workers,
#     DATA_TYPE = "ordinal",
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_MAX_ITER = 10,
#     VERBOSE = FALSE
#   )
#   # tictoc::toc()

#   # tictoc::tic()
#   cpp_res_mp <- cpp_twoway_get_phi_modified_profile(
#     Y = dt$rating,
#     ITEM_INDS = as.integer(dt$id_item),
#     WORKER_INDS = as.integer(dt$id_worker),
#     ALPHA_START = rep(0, items),
#     BETA_START = rep(0, workers),
#     PHI_START = agr2prec(0.5),
#     K = K,
#     J = items,
#     W = workers,
#     CONTINUOUS = FALSE,
#     SEARCH_RANGE = 10,
#     MAX_ITER = 100,
#     PROF_SEARCH_RANGE = 10,
#     PROF_MAX_ITER = 10,
#     ALT_MAX_ITER = 10,
#     ALT_TOL = 1e-5,
#     VERBOSE = FALSE
#   )
#   # tictoc::toc()

#   expect_equal(
#     mp_mle$mpl_precision,
#     cpp_res_mp[1],
#     tolerance = 1e-2
#   )

#   expect_equal(
#     mp_mle$loglik / n_obs,
#     cpp_res_mp[2] / n_obs,
#     tolerance = 1e-2
#   )
# })
