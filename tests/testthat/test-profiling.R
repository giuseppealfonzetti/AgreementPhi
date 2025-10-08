#### Profiling two way continuous ####
test_that("profiled lambda maximizes likelihood for fixed phi", {
  set.seed(123)
  dt <- sim_data(
    J = 5,
    B = 3,
    W = 8,
    AGREEMENT = 0.7,
    ALPHA = rnorm(5),
    BETA = rnorm(8),
    DATA_TYPE = "continuous"
  )

  phi <- agr2prec(0.7)
  lambda_start <- rep(0, 5 + 8 - 1)

  lambda_hat <- twoway_profiling_bfgs(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_start,
    phi,
    J = 5,
    W = 8,
    DATA_TYPE = "continuous"
  )

  ll_profiled <- cpp_continuous_twoway_joint_loglik(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_hat,
    phi,
    J = 5,
    W = 8,
    GRADFLAG = 0
  )$ll

  for (i in 1:10) {
    set.seed(i)
    lambda_perturb <- lambda_hat + rnorm(length(lambda_hat), 0, 0.1)
    ll_perturb <- cpp_continuous_twoway_joint_loglik(
      dt$rating,
      as.integer(dt$id_item),
      as.integer(dt$id_worker),
      lambda_perturb,
      phi,
      J = 5,
      W = 8,
      GRADFLAG = 0
    )$ll

    expect_gte(ll_profiled, ll_perturb)
  }
})

test_that("gradient near zero at profiled values", {
  set.seed(456)
  dt <- sim_data(
    J = 4,
    B = 4,
    W = 6,
    AGREEMENT = 0.6,
    ALPHA = rnorm(4),
    BETA = rnorm(6),
    DATA_TYPE = "continuous",
    K = 5
  )

  phi <- agr2prec(0.6)
  lambda_start <- rep(0, 4 + 6 - 1)

  lambda_hat <- twoway_profiling_bfgs(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_start,
    phi,
    J = 4,
    W = 6,
    DATA_TYPE = "continuous"
  )

  result <- cpp_continuous_twoway_joint_loglik(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_hat,
    phi,
    J = 4,
    W = 6,
    GRADFLAG = 1
  )

  grad_norm <- sqrt(sum(result$dlambda^2))
  expect_lt(grad_norm, 1e-3)
})

#### Profiling two way ordinal ####
test_that("profiled lambda maximizes likelihood for fixed phi", {
  set.seed(123)
  items <- 50
  budget <- 10
  workers <- 20
  k <- 10
  dt <- sim_data(
    J = items,
    B = budget,
    W = workers,
    K = k,
    AGREEMENT = 0.7,
    ALPHA = rnorm(items),
    BETA = rnorm(workers),
    DATA_TYPE = "ordinal"
  )

  phi <- agr2prec(0.7)
  lambda_start <- rep(0, items + workers - 1)

  lambda_hat <- twoway_profiling_bfgs(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_start,
    phi,
    J = items,
    W = workers,
    K = k,
    DATA_TYPE = "ordinal"
  )

  ll_profiled <- cpp_ordinal_twoway_joint_loglik(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_hat,
    phi,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 0
  )$ll

  for (i in 1:10) {
    set.seed(i)
    lambda_perturb <- lambda_hat + rnorm(length(lambda_hat), 0, 0.2)
    ll_perturb <- cpp_ordinal_twoway_joint_loglik(
      dt$rating,
      as.integer(dt$id_item),
      as.integer(dt$id_worker),
      lambda_perturb,
      phi,
      J = items,
      W = workers,
      K = k,
      GRADFLAG = 0
    )$ll

    expect_gte(ll_profiled, ll_perturb)
  }
})

# test_that("gradient near zero at profiled values", {
#   items <- 100
#   workers <- 100
#   b <- 10
#   n_obs <- items * b
#   alphas <- rnorm(items)
#   betas <- c(0, rnorm(workers - 1))
#   agr <- 0.3
#   k <- 6
#   dt <- sim_data(
#     J = items,
#     B = b,
#     W = workers,
#     AGREEMENT = agr,
#     ALPHA = alphas,
#     BETA = betas,
#     K = k,
#     DATA_TYPE = "ordinal"
#   )

#   phi <- agr2prec(0.6)
#   lambda_start <- rep(0, items + workers - 1)

#   lambda_hat <- twoway_profiling_bfgs(
#     dt$rating,
#     as.integer(dt$id_item),
#     as.integer(dt$id_worker),
#     lambda_start,
#     phi,
#     K = k,
#     J = items,
#     W = workers,
#     DATA_TYPE = "ordinal"
#   )

#   result <- cpp_ordinal_twoway_joint_loglik(
#     dt$rating,
#     as.integer(dt$id_item),
#     as.integer(dt$id_worker),
#     lambda_hat,
#     phi,
#     K = k,
#     J = items,
#     W = workers,
#     GRADFLAG = 1
#   )

#   grad_norm <- sqrt(sum(result$dlambda^2))
#   expect_lt(grad_norm, 1e-3)
# })

#### Profilinf two way continuous

test_that("Alteranting Maximization matches bfgs | continuous", {
  set.seed(123)
  items <- 200
  workers <- 500
  b <- 10
  n_obs <- items * b
  alphas <- rnorm(items)
  betas <- c(0, rnorm(workers - 1))
  agr <- 0.3
  dt <- sim_data(
    J = items,
    B = b,
    W = workers,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "continuous"
  )

  phi <- agr2prec(agr)
  lambda_start <- rep(0, items + workers - 1)

  # tictoc::tic()
  lambda_hat <- twoway_profiling_bfgs(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_start,
    phi,
    J = items,
    W = workers,
    DATA_TYPE = "continuous"
  )
  # tictoc::toc()

  # tictoc::tic()
  lambda_alt <- cpp_continuous_profiling(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, items),
    BETA = rep(0, workers),
    PHI = phi,
    J = items,
    W = workers,
    PROF_UNI_RANGE = 5L,
    PROF_UNI_MAX_ITER = 100L,
    PROF_MAX_ITER = 10L,
    TOL = 1e-5
  )
  # tictoc::toc()

  bfgs_ll <- cpp_continuous_twoway_joint_loglik(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_hat,
    phi,
    J = items,
    W = workers,
    GRADFLAG = 0
  )$ll /
    n_obs
  alt_ll <- cpp_continuous_twoway_joint_loglik(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    c(lambda_alt[[1]], lambda_alt[[2]][-1]),
    phi,
    J = items,
    W = workers,
    GRADFLAG = 0
  )$ll /
    n_obs

  expect_equal(bfgs_ll, alt_ll, tolerance = 1e-2)
})
#### Profilinf two way ordinal

test_that("Alteranting Maximization matches bfgs | ordinal", {
  set.seed(123)
  items <- 30
  workers <- 10
  b <- 10
  n_obs <- items * b
  alphas <- rnorm(items)
  betas <- c(0, rnorm(workers - 1))
  agr <- 0.3
  k <- 6
  dt <- sim_data(
    J = items,
    B = b,
    W = workers,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    K = k,
    DATA_TYPE = "ordinal"
  )

  phi <- agr2prec(agr)
  lambda_start <- rep(0, items + workers - 1)

  # tictoc::tic()
  lambda_hat <- twoway_profiling_bfgs(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_start,
    phi,
    J = items,
    W = workers,
    K = k,
    DATA_TYPE = "ordinal"
  )
  # tictoc::toc()

  # tictoc::tic()
  lambda_alt <- cpp_ordinal_profiling(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, items),
    BETA = rep(0, workers),
    PHI = phi,
    J = items,
    W = workers,
    K = k,
    PROF_UNI_RANGE = 5L,
    PROF_UNI_MAX_ITER = 100L,
    PROF_MAX_ITER = 10L,
    TOL = 1e-5
  )
  # tictoc::toc()

  bfgs_ll <- cpp_ordinal_twoway_joint_loglik(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    lambda_hat,
    phi,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 0
  )$ll /
    n_obs
  alt_ll <- cpp_ordinal_twoway_joint_loglik(
    dt$rating,
    as.integer(dt$id_item),
    as.integer(dt$id_worker),
    c(lambda_alt[[1]], lambda_alt[[2]][-1]),
    phi,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 0
  )$ll /
    n_obs

  expect_equal(bfgs_ll, alt_ll, tolerance = 1e-2)
})
