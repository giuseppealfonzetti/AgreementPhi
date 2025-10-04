#### Derivatives continuous ####
set.seed(123)
items <- 3
workers <- 5
budget <- 4
agr <- 0.7
alphas <- rnorm(J, 0, 0.5)
betas <- c(0, rnorm(W - 1, 0, 0.3))

dt2 <- sim_data(
  J = items,
  B = budget,
  W = workers,
  AGREEMENT = agr,
  ALPHA = alphas,
  BETA = betas,
  DATA_TYPE = "continuous",
  SEED = 123
)

lambda_test <- c(alphas, betas[-1])
phi_test <- agr2prec(agr)

# Function to compute log-likelihood
Rf <- function(LAMBDA) {
  cpp_continuous_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = LAMBDA,
    PHI = phi_test,
    J = items,
    W = workers,
    GRADFLAG = 0
  )$ll
}

test_that("Gradient matches numerical derivative", {
  cpp_res <- cpp_continuous_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    GRADFLAG = 1
  )

  numgrad <- numDeriv::grad(Rf, lambda_test)

  expect_equal(
    as.vector(cpp_res$dlambda),
    numgrad,
    tolerance = 1e-5,
    label = "Analytical gradient",
    expected.label = "Numerical gradient"
  )
})

test_that("J_alphaalpha diagonal matches numerical Hessian", {
  cpp_res <- cpp_continuous_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    GRADFLAG = 2
  )

  numhess <- numDeriv::hessian(Rf, lambda_test)

  # Check diagonal of α block
  expect_equal(
    as.vector(cpp_res$jalphaalpha),
    -diag(numhess[1:items, 1:items]),
    tolerance = 1e-4,
    label = "Analytical J_alphaalpha diagonal",
    expected.label = "Numerical Hessian diagonal (alpha block)"
  )
})

test_that("J_betabeta diagonal matches numerical Hessian", {
  cpp_res <- cpp_continuous_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    GRADFLAG = 2
  )

  numhess <- numDeriv::hessian(Rf, lambda_test)

  beta_idx <- (items + 1):(items + workers - 1)
  expect_equal(
    as.vector(cpp_res$jbetabeta),
    -diag(numhess[beta_idx, beta_idx]),
    tolerance = 1e-4,
    label = "Analytical J_betabeta diagonal",
    expected.label = "Numerical Hessian diagonal (beta block)"
  )
})

test_that("J_alphabeta matches numerical Hessian", {
  cpp_res <- cpp_continuous_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    GRADFLAG = 2
  )

  hess_numerical <- numDeriv::hessian(Rf, lambda_test)

  # Check cross-derivative block
  beta_idx <- (items + 1):(items + workers - 1)
  expect_equal(
    cpp_res$jalphabeta,
    -hess_numerical[1:items, beta_idx],
    tolerance = 1e-4,
    label = "Analytical J_alphabeta",
    expected.label = "Numerical Hessian (cross block)"
  )
})

#### logdet obs info continuous ####
test_that("log_det_obs_info matches numerical Hessian for continuous data", {
  set.seed(123)
  J <- 3
  W <- 5
  B <- 4
  AGREEMENT <- 0.7
  ALPHA <- rnorm(J, 0, 0.5)
  BETA <- c(0, rnorm(W - 1, 0, 0.3))

  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = AGREEMENT,
    ALPHA = ALPHA,
    BETA = BETA,
    DATA_TYPE = "continuous",
    SEED = 123
  )

  y <- dt$rating
  item_inds <- as.integer(dt$id_item)
  worker_inds <- as.integer(dt$id_worker)
  lambda <- c(ALPHA, BETA[-1])
  phi <- agr2prec(AGREEMENT)

  # Analytic log determinant
  log_det_obs <- cpp_continuous_twoway_log_det_obs_info(
    Y = y,
    ITEM_INDS = item_inds,
    WORKER_INDS = worker_inds,
    LAMBDA = lambda,
    PHI = phi,
    J = J,
    W = W
  )

  # Numerical Hessian
  loglik <- function(lam) {
    cpp_continuous_twoway_joint_loglik(
      Y = y,
      ITEM_INDS = item_inds,
      WORKER_INDS = worker_inds,
      LAMBDA = lam,
      PHI = phi,
      J = J,
      W = W,
      GRADFLAG = 0L
    )$ll
  }

  hess <- numDeriv::hessian(loglik, lambda)
  log_det_numeric <- log(det(-hess))

  expect_equal(log_det_obs, log_det_numeric, tolerance = 1e-4)
})


#### log det E0d0d1 ####
test_that("log_det_obs_info matches numerical Hessian for continuous data", {
  set.seed(123)
  items <- 5
  workers <- 6
  budget <- 3
  agr <- 0.7
  alphas <- rnorm(items, 0, 0.5)
  betas <- c(0, rnorm(workers - 1, 0, 0.3))

  dt <- sim_data(
    J = items,
    B = budget,
    W = workers,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "continuous",
    SEED = 123
  )

  lambda_test <- c(alphas, betas[-1])
  phi_test <- agr2prec(agr)

  item_inds <- as.integer(dt$id_item)
  worker_inds <- as.integer(dt$id_worker)

  # Analytic log determinant
  log_det_obs_J <- cpp_continuous_twoway_log_det_obs_info(
    Y = dt$rating,
    ITEM_INDS = item_inds,
    WORKER_INDS = worker_inds,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers
  )
  log_det_I <- cpp_continuous_twoway_log_det_E0d0d1(
    ITEM_INDS = item_inds,
    WORKER_INDS = worker_inds,
    LAMBDA0 = lambda_test,
    PHI0 = phi_test,
    LAMBDA1 = lambda_test,
    PHI1 = phi_test,
    J = items,
    W = workers
  )

  expect_equal(log_det_obs_J, log_det_I, tolerance = 1e-1)
})

#### Derivatives ordinal ####
items <- 20
budget_per_item <- 5
workers <- 10
k <- 10
alphas <- rnorm(items)
betas <- c(0, rnorm(workers - 1))
agr <- runif(1)
dt2 <- sim_data(
  J = items,
  B = budget_per_item,
  W = workers,
  AGREEMENT = .1,
  ALPHA = alphas,
  BETA = betas,
  DATA_TYPE = "ordinal",
  K = k,
  SEED = 123
)

lambda_test <- c(alphas, betas[-1])
phi_test <- agr2prec(agr)

# Function to compute log-likelihood
Rf <- function(LAMBDA) {
  cpp_ordinal_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = LAMBDA,
    PHI = phi_test,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 0
  )$ll
}

lambda_test <- c(alphas, betas[-1])
phi_test <- agr2prec(agr)

test_that("Gradient matches numerical derivative", {
  cpp_res <- cpp_ordinal_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 1
  )

  numgrad <- numDeriv::grad(Rf, lambda_test)

  expect_equal(
    as.vector(cpp_res$dlambda),
    numgrad,
    tolerance = 1e-1,
    label = "Analytical gradient",
    expected.label = "Numerical gradient"
  )
})

test_that("J_alphaalpha diagonal matches numerical Hessian", {
  cpp_res <- cpp_ordinal_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 2
  )

  numhess <- numDeriv::hessian(Rf, lambda_test)

  # Check diagonal of α block
  expect_equal(
    as.vector(cpp_res$jalphaalpha),
    -diag(numhess[1:items, 1:items]),
    tolerance = 1e-1,
    label = "Analytical J_alphaalpha diagonal",
    expected.label = "Numerical Hessian diagonal (alpha block)"
  )
})

test_that("J_betabeta diagonal matches numerical Hessian", {
  cpp_res <- cpp_ordinal_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 2
  )

  numhess <- numDeriv::hessian(Rf, lambda_test)

  beta_idx <- (items + 1):(items + workers - 1)
  expect_equal(
    as.vector(cpp_res$jbetabeta),
    -diag(numhess[beta_idx, beta_idx]),
    tolerance = 1e-1,
    label = "Analytical J_betabeta diagonal",
    expected.label = "Numerical Hessian diagonal (beta block)"
  )
})

test_that("J_alphabeta matches numerical Hessian", {
  cpp_res <- cpp_ordinal_twoway_joint_loglik(
    Y = dt2$rating,
    ITEM_INDS = dt2$id_item,
    WORKER_INDS = dt2$id_worker,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    K = k,
    GRADFLAG = 2
  )

  hess_numerical <- numDeriv::hessian(Rf, lambda_test)

  # Check cross-derivative block
  beta_idx <- (items + 1):(items + workers - 1)
  expect_equal(
    cpp_res$jalphabeta,
    -hess_numerical[1:items, beta_idx],
    tolerance = 1e-1,
    label = "Analytical J_alphabeta",
    expected.label = "Numerical Hessian (cross block)"
  )
})

#### logdet obs info continuous ####
test_that("log_det_obs_info matches numerical Hessian for ordinal data", {
  set.seed(123)
  J <- 3
  W <- 5
  B <- 4
  k <- 10
  AGREEMENT <- 0.7
  ALPHA <- rnorm(J, 0, 0.5)
  BETA <- c(0, rnorm(W - 1, 0, 0.3))

  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = AGREEMENT,
    ALPHA = ALPHA,
    BETA = BETA,
    K = k,
    DATA_TYPE = "ordinal",
    SEED = 123
  )

  y <- dt$rating
  item_inds <- as.integer(dt$id_item)
  worker_inds <- as.integer(dt$id_worker)
  lambda <- c(ALPHA, BETA[-1])
  phi <- agr2prec(AGREEMENT)

  # Analytic log determinant
  log_det_obs <- cpp_ordinal_twoway_log_det_obs_info(
    Y = y,
    ITEM_INDS = item_inds,
    WORKER_INDS = worker_inds,
    LAMBDA = lambda,
    PHI = phi,
    J = J,
    W = W,
    K = k
  )

  # Numerical Hessian
  loglik <- function(lam) {
    cpp_ordinal_twoway_joint_loglik(
      Y = y,
      ITEM_INDS = item_inds,
      WORKER_INDS = worker_inds,
      LAMBDA = lam,
      PHI = phi,
      J = J,
      W = W,
      K = k,
      GRADFLAG = 0L
    )$ll
  }

  hess <- numDeriv::hessian(loglik, lambda)
  log_det_numeric <- log(det(-hess))

  expect_equal(log_det_obs, log_det_numeric, tolerance = 1e-4)
})

#### log det E0d0d1 ####
test_that("log_det_obs_info matches numerical Hessian for continuous data", {
  set.seed(123)
  items <- 5
  workers <- 6
  budget <- 3
  agr <- 0.7
  alphas <- rnorm(items, 0, 0.5)
  betas <- c(0, rnorm(workers - 1, 0, 0.3))

  k <- 10
  dt <- sim_data(
    J = items,
    B = budget,
    W = workers,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    K = k,
    DATA_TYPE = "ordinal",
    SEED = 123
  )

  lambda_test <- c(alphas, betas[-1])
  phi_test <- agr2prec(agr)

  item_inds <- as.integer(dt$id_item)
  worker_inds <- as.integer(dt$id_worker)

  # Analytic log determinant
  log_det_obs_J <- cpp_ordinal_twoway_log_det_obs_info(
    Y = dt$rating,
    ITEM_INDS = item_inds,
    WORKER_INDS = worker_inds,
    LAMBDA = lambda_test,
    PHI = phi_test,
    J = items,
    W = workers,
    K = k
  )
  log_det_I <- cpp_ordinal_twoway_log_det_E0d0d1(
    ITEM_INDS = item_inds,
    WORKER_INDS = worker_inds,
    LAMBDA0 = lambda_test,
    PHI0 = phi_test,
    LAMBDA1 = lambda_test,
    PHI1 = phi_test,
    J = items,
    W = workers,
    K = k
  )

  expect_equal(log_det_obs_J, log_det_I, tolerance = 1e-1)
})
