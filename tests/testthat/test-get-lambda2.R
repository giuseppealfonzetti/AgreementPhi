test_that("cpp_ordinal_get_lambda2 returns coherent estimates", {
  set.seed(1)
  J <- 200
  W <- 200
  B <- 4
  K <- 5
  agr <- 0.65

  alphas <- rnorm(J, 0, 0.4)
  betas <- c(0, rnorm(W - 1, 0, 0.25))
  thr <- sort(runif(K - 1, 0.05, 0.95))
  tau_start <- seq(0, 1, length.out = K + 1)

  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "continuous",
    SEED = 99
  )

  rating_k <- cont2ord(dt$rating, K = K, TRESHOLDS = thr)
  table(rating_k)

  fixed_tau <- cpp_ordinal_get_lambda2(
    Y = rating_k,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = alphas,
    BETA = betas,
    TAU = tau_start,
    PHI = agr2prec(agr),
    J = J,
    W = W,
    K = K,
    WORKER_NUISANCE = FALSE,
    THRESHOLDS_NUISANCE = FALSE,
    PROF_UNI_RANGE = 5,
    PROF_UNI_MAX_ITER = 20L,
    PROF_MAX_ITER = 15L,
    TOL = 1e-4
  )

  expect_equal(length(fixed_tau$alpha), J)
  expect_equal(length(fixed_tau$beta), W)
  expect_equal(fixed_tau$beta[1], 0)
  expect_equal(length(fixed_tau$tau), K + 1)
  expect_equal(fixed_tau$tau, tau_start)

  init_tau <- function(y, K) {
    counts <- tabulate(factor(y, levels = seq_len(K)), nbins = K)
    cum_p <- cumsum(counts) / sum(counts)
    c(0, cum_p[-K], 1)
  }
  tau_start <- init_tau(y = rating_k, K = K)

  profiled_tau <- cpp_ordinal_get_lambda2(
    Y = rating_k,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = alphas,
    BETA = betas,
    TAU = tau_start,
    PHI = agr2prec(agr),
    J = J,
    W = W,
    K = K,
    WORKER_NUISANCE = TRUE,
    THRESHOLDS_NUISANCE = TRUE,
    PROF_UNI_RANGE = 5,
    PROF_UNI_MAX_ITER = 60L,
    PROF_MAX_ITER = 100,
    TOL = 1e-4
  )

  profiled_tau$tau
  c(0, thr, 1)

  expect_equal(length(profiled_tau$tau), K + 1)
  expect_equal(profiled_tau$tau[1], 0)
  expect_equal(profiled_tau$tau[K + 1], 1)
  expect_true(all(diff(profiled_tau$tau) > 0))
  expect_true(any(abs(profiled_tau$tau - tau_start) > 1e-6))
})
