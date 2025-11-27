##### Threshold profiling ####

test_that("cpp_ordinal_get_lambda2 profiles thresholds", {
  set.seed(42)
  J <- 100
  B <- 10
  W <- 20
  K <- 6

  alphas <- rnorm(J, 0, 0.5)
  betas <- c(0, rnorm(W - 1, 0, 0.3))
  thr <- sort(runif(K - 1, 0.1, 0.9))
  tau_true <- c(0, thr, 1)

  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = 0.5,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "continuous",
    SEED = 99
  )

  rating_k <- cont2ord(dt$rating, K = K, TRESHOLDS = thr)
  rating_k[rating_k < 1] <- 1
  rating_k[rating_k > K] <- K

  tau_start <- seq(0, 1, length.out = K + 1)

  profiled <- cpp_ordinal_get_lambda2(
    Y = rating_k,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, J),
    BETA = rep(0, W),
    TAU = tau_start,
    PHI = agr2prec(0.5),
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = FALSE,
    WORKER_NUISANCE = FALSE,
    THRESHOLDS_NUISANCE = TRUE,
    PROF_UNI_RANGE = 2,
    PROF_UNI_MAX_ITER = 50L,
    PROF_MAX_ITER = 20L,
    TOL = 1e-4
  )
  profiled
  alphas
  betas
  thr

  tau_hat <- profiled$tau

  expect_equal(length(tau_hat), K + 1)
  expect_equal(tau_hat[1], 0)
  expect_equal(tau_hat[K + 1], 1)

  expect_equal(tau_hat[2:K], tau_true[2:K], tolerance = 0.1)
})
