test_that("continuous joint loglik gradients match numerical derivatives", {
  skip_if_not_installed("numDeriv")
  set.seed(111)
  J <- 6
  W <- 5
  B <- 4
  agr <- 0.6
  alphas <- rnorm(J)
  betas <- c(0, rnorm(W - 1))
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "continuous"
  )
  lambda <- c(alphas, betas[-1])
  phi <- agr2prec(agr)

  loglik <- function(par) {
    cpp_continuous_twoway_joint_loglik(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      LAMBDA = par,
      PHI = phi,
      J = J,
      W = W,
      WORKER_NUISANCE = TRUE,
      GRADFLAG = 0L
    )$ll
  }

  analytic <- cpp_continuous_twoway_joint_loglik(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    LAMBDA = lambda,
    PHI = phi,
    J = J,
    W = W,
    WORKER_NUISANCE = TRUE,
    GRADFLAG = 2L
  )

  num_grad <- numDeriv::grad(loglik, lambda)
  num_hess <- -diag(numDeriv::hessian(loglik, lambda))

  expect_equal(as.vector(analytic$dlambda), num_grad, tolerance = 1e-4)
  expect_equal(
    c(analytic$jalphaalpha, analytic$jbetabeta),
    num_hess,
    tolerance = 1e-2
  )
})

test_that("ordinal joint loglik gradients match numerical derivatives", {
  skip_if_not_installed("numDeriv")
  set.seed(222)
  J <- 10
  W <- 10
  B <- 5
  K <- 6
  agr <- 0.55
  alphas <- rnorm(J)
  betas <- c(0, rnorm(W - 1))
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    K = K,
    DATA_TYPE = "ordinal"
  )
  tau <- seq(0, 1, length.out = K + 1)
  lambda <- c(alphas, betas[-1])
  phi <- agr2prec(agr)

  loglik <- function(par) {
    cpp_ordinal_twoway_joint_loglik(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      LAMBDA = par,
      TAU = tau,
      PHI = phi,
      J = J,
      W = W,
      K = K,
      WORKER_NUISANCE = TRUE,
      GRADFLAG = 0L
    )$ll
  }

  analytic <- cpp_ordinal_twoway_joint_loglik(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    LAMBDA = lambda,
    TAU = tau,
    PHI = phi,
    J = J,
    W = W,
    K = K,
    WORKER_NUISANCE = TRUE,
    GRADFLAG = 2L
  )

  num_grad <- numDeriv::grad(loglik, lambda)
  num_hess <- -diag(numDeriv::hessian(loglik, lambda))

  expect_equal(as.vector(analytic$dlambda), num_grad, tolerance = 1e-3)
  expect_equal(
    c(analytic$jalphaalpha, analytic$jbetabeta),
    num_hess,
    tolerance = 1e-1
  )
})

test_that("continuous log_det_obs_info matches numerical Hessian", {
  skip_if_not_installed("numDeriv")
  set.seed(333)
  J <- 4
  W <- 5
  B <- 4
  agr <- 0.7
  alphas <- rnorm(J)
  betas <- c(0, rnorm(W - 1))
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "continuous"
  )
  lambda <- c(alphas, betas[-1])
  phi <- agr2prec(agr)

  obs_logdet <- cpp_continuous_twoway_log_det_obs_info(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    LAMBDA = lambda,
    PHI = phi,
    J = J,
    W = W,
    WORKER_NUISANCE = TRUE
  )

  loglik <- function(par) {
    cpp_continuous_twoway_joint_loglik(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      LAMBDA = par,
      PHI = phi,
      J = J,
      W = W,
      WORKER_NUISANCE = TRUE,
      GRADFLAG = 0L
    )$ll
  }

  num_logdet <- log(det(-numDeriv::hessian(loglik, lambda)))
  expect_equal(obs_logdet, num_logdet, tolerance = 1e-4)
})

test_that("ordinal log_det_obs_info matches numerical Hessian", {
  skip_if_not_installed("numDeriv")
  set.seed(444)
  J <- 4
  W <- 10
  B <- 5
  K <- 5
  agr <- 0.65
  alphas <- rnorm(J)
  betas <- c(0, rnorm(W - 1))
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    K = K,
    DATA_TYPE = "ordinal"
  )
  tau <- seq(0, 1, length.out = K + 1)
  lambda <- c(alphas, betas[-1])
  phi <- agr2prec(agr)

  obs_logdet <- cpp_ordinal_twoway_log_det_obs_info(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    LAMBDA = lambda,
    TAU = tau,
    PHI = phi,
    K = K,
    J = J,
    W = W,
    WORKER_NUISANCE = TRUE
  )

  loglik <- function(par) {
    cpp_ordinal_twoway_joint_loglik(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      LAMBDA = par,
      TAU = tau,
      PHI = phi,
      J = J,
      W = W,
      K = K,
      WORKER_NUISANCE = TRUE,
      GRADFLAG = 0L
    )$ll
  }

  num_logdet <- log(det(-numDeriv::hessian(loglik, lambda)))
  expect_equal(obs_logdet, num_logdet, tolerance = 1e-3)
})
