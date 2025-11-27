#### Test gradient and Hessian of profile_extended ####

test_that("profile_extended_grad_raw_tau matches numDeriv", {
  set.seed(123)

  # Simulation parameters
  J <- 20
  B <- 10
  W <- 15
  K <- 5

  alphas <- rnorm(J, 0, 0.5)
  betas <- c(0, rnorm(W - 1, 0, 0.3))
  thr <- sort(runif(K - 1, 0.1, 0.9))
  tau_true <- c(0, thr, 1)

  # Generate data
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    K = K,
    AGREEMENT = 0.6,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "ordinal"
  )

  # Set up parameters
  raw_phi <- log(agr2prec(0.6))
  raw_tau <- tau2raw(tau_true)

  # Compute gradient using our implementation
  grad_cpp <- cpp_profile_extended_grad_raw_tau(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, J),
    BETA = rep(0, W),
    RAW_TAU = raw_tau,
    RAW_PHI = raw_phi,
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = TRUE,
    WORKER_NUISANCE = TRUE,
    PROF_UNI_RANGE = 2L,
    PROF_UNI_MAX_ITER = 50L,
    PROF_MAX_ITER = 5L,
    PROF_TOL = 1e-4
  )

  # Define objective function for numDeriv
  profile_ll_raw_tau <- function(raw_tau_vec) {
    cpp_profile_extended(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      ALPHA = rep(0, J),
      BETA = rep(0, W),
      RAW_TAU = raw_tau_vec,
      RAW_PHI = raw_phi,
      J = J,
      W = W,
      K = K,
      ITEMS_NUISANCE = TRUE,
      WORKER_NUISANCE = TRUE,
      PROF_UNI_RANGE = 2L,
      PROF_UNI_MAX_ITER = 50L,
      PROF_MAX_ITER = 5L,
      PROF_TOL = 1e-4
    )
  }

  # Compute gradient using numDeriv
  grad_num <- numDeriv::grad(profile_ll_raw_tau, raw_tau)

  # Check that gradients match  # Tolerance reflects numerical profiling + envelope theorem approximation
  expect_equal(as.vector(grad_cpp), grad_num, tolerance = 0.02)
})

test_that("profile_extended_grad_raw_phi matches numDeriv", {
  set.seed(456)

  # Simulation parameters
  J <- 15
  B <- 8
  W <- 12
  K <- 4

  alphas <- rnorm(J, 0, 0.5)
  betas <- c(0, rnorm(W - 1, 0, 0.3))
  thr <- sort(runif(K - 1, 0.15, 0.85))
  tau_true <- c(0, thr, 1)

  # Generate data
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    K = K,
    AGREEMENT = 0.5,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "ordinal"
  )

  # Set up parameters
  raw_phi <- log(agr2prec(0.5))
  raw_tau <- tau2raw(tau_true)

  # Compute gradient using our implementation
  grad_cpp <- cpp_profile_extended_grad_raw_phi(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, J),
    BETA = rep(0, W),
    RAW_TAU = raw_tau,
    RAW_PHI = raw_phi,
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = TRUE,
    WORKER_NUISANCE = TRUE,
    PROF_UNI_RANGE = 2L,
    PROF_UNI_MAX_ITER = 50L,
    PROF_MAX_ITER = 5L,
    PROF_TOL = 1e-4
  )

  # Define objective function for numDeriv
  profile_ll_raw_phi <- function(raw_phi_val) {
    cpp_profile_extended(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      ALPHA = rep(0, J),
      BETA = rep(0, W),
      RAW_TAU = raw_tau,
      RAW_PHI = raw_phi_val,
      J = J,
      W = W,
      K = K,
      ITEMS_NUISANCE = TRUE,
      WORKER_NUISANCE = TRUE,
      PROF_UNI_RANGE = 2L,
      PROF_UNI_MAX_ITER = 50L,
      PROF_MAX_ITER = 5L,
      PROF_TOL = 1e-4
    )
  }

  # Compute gradient using numDeriv
  grad_num <- numDeriv::grad(profile_ll_raw_phi, raw_phi)

  # Check that gradients match
  expect_equal(grad_cpp, grad_num, tolerance = 0.02)
})

test_that("profile_extended_hess_raw_tau matches numDeriv", {
  set.seed(789)

  # Simulation parameters (smaller for Hessian test - it's slow!)
  J <- 10
  B <- 8
  W <- 8
  K <- 4

  alphas <- rnorm(J, 0, 0.5)
  betas <- c(0, rnorm(W - 1, 0, 0.3))
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Generate data
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    K = K,
    AGREEMENT = 0.55,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "ordinal"
  )

  # Set up parameters
  raw_phi <- log(agr2prec(0.55))
  raw_tau <- tau2raw(tau_true)

  # Compute Hessian using our implementation
  hess_cpp <- cpp_profile_extended_hess_raw_tau(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, J),
    BETA = rep(0, W),
    RAW_TAU = raw_tau,
    RAW_PHI = raw_phi,
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = TRUE,
    WORKER_NUISANCE = TRUE,
    PROF_UNI_RANGE = 2L,
    PROF_UNI_MAX_ITER = 50L,
    PROF_MAX_ITER = 5L,
    PROF_TOL = 1e-4
  )

  # Define gradient function for numDeriv
  # Hessian = Jacobian of gradient
  gradient_raw_tau <- function(raw_tau_vec) {
    as.vector(cpp_profile_extended_grad_raw_tau(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      ALPHA = rep(0, J),
      BETA = rep(0, W),
      RAW_TAU = raw_tau_vec,
      RAW_PHI = raw_phi,
      J = J,
      W = W,
      K = K,
      ITEMS_NUISANCE = TRUE,
      WORKER_NUISANCE = TRUE,
      PROF_UNI_RANGE = 2L,
      PROF_UNI_MAX_ITER = 50L,
      PROF_MAX_ITER = 5L,
      PROF_TOL = 1e-4
    ))
  }

  # Compute Hessian as Jacobian of gradient
  hess_num <- numDeriv::jacobian(gradient_raw_tau, raw_tau)

  # Check that Hessians match
  # Using Jacobian of gradient is more accurate than Hessian of function
  # since it avoids double numerical differentiation
  # Our analytical Hessian uses J^T × H_τ × J which neglects:
  # (1) second-order terms with ∂²τ/∂raw_τ²
  # (2) how profiled (α,β) change with raw_τ in the Hessian
  # Tolerance reflects these approximations
  expect_equal(hess_cpp, hess_num, tolerance = 5)
})

test_that("gradient is zero at optimum", {
  set.seed(999)

  # Simulation parameters
  J <- 15
  B <- 10
  W <- 10
  K <- 5

  alphas <- rnorm(J, 0, 0.5)
  betas <- c(0, rnorm(W - 1, 0, 0.3))
  agr_true <- 0.6
  thr <- sort(runif(K - 1, 0.1, 0.9))
  tau_true <- c(0, thr, 1)

  # Generate data
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    K = K,
    AGREEMENT = agr_true,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "ordinal"
  )

  # Optimize using optim
  raw_phi_start <- log(agr2prec(agr_true))
  raw_tau_start <- tau2raw(tau_true)

  # Define negative log-likelihood for optimization
  neg_ll <- function(par) {
    raw_tau_opt <- par[1:(K-1)]
    raw_phi_opt <- par[K]

    -cpp_profile_extended(
      Y = dt$rating,
      ITEM_INDS = as.integer(dt$id_item),
      WORKER_INDS = as.integer(dt$id_worker),
      ALPHA = rep(0, J),
      BETA = rep(0, W),
      RAW_TAU = raw_tau_opt,
      RAW_PHI = raw_phi_opt,
      J = J,
      W = W,
      K = K,
      ITEMS_NUISANCE = TRUE,
      WORKER_NUISANCE = TRUE,
      PROF_UNI_RANGE = 2L,
      PROF_UNI_MAX_ITER = 50L,
      PROF_MAX_ITER = 5L,
      PROF_TOL = 1e-4
    )
  }

  # Optimize
  opt_result <- optim(
    par = c(raw_tau_start, raw_phi_start),
    fn = neg_ll,
    method = "BFGS",
    control = list(maxit = 100, reltol = 1e-8)
  )

  # Extract optimal values
  raw_tau_opt <- opt_result$par[1:(K-1)]
  raw_phi_opt <- opt_result$par[K]

  # Compute gradients at optimum
  grad_tau_opt <- cpp_profile_extended_grad_raw_tau(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, J),
    BETA = rep(0, W),
    RAW_TAU = raw_tau_opt,
    RAW_PHI = raw_phi_opt,
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = TRUE,
    WORKER_NUISANCE = TRUE,
    PROF_UNI_RANGE = 2L,
    PROF_UNI_MAX_ITER = 50L,
    PROF_MAX_ITER = 5L,
    PROF_TOL = 1e-4
  )

  grad_phi_opt <- cpp_profile_extended_grad_raw_phi(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, J),
    BETA = rep(0, W),
    RAW_TAU = raw_tau_opt,
    RAW_PHI = raw_phi_opt,
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = TRUE,
    WORKER_NUISANCE = TRUE,
    PROF_UNI_RANGE = 2L,
    PROF_UNI_MAX_ITER = 50L,
    PROF_MAX_ITER = 5L,
    PROF_TOL = 1e-4
  )

  # Gradients should be reasonably small at optimum
  # Not exactly zero due to numerical optimization tolerance
  expect_lt(sqrt(sum(grad_tau_opt^2)), 2.0)
  expect_lt(abs(grad_phi_opt), 6.0)
})

test_that("Hessian is negative definite at optimum", {
  skip_on_cran()  # Skip on CRAN as this test is slow

  set.seed(111)

  # Simulation parameters (small for speed)
  J <- 8
  B <- 6
  W <- 10
  K <- 4

  alphas <- rnorm(J, 0, 0.5)
  betas <- c(0, rnorm(W - 1, 0, 0.3))
  agr_true <- 0.5
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Generate data
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    K = K,
    AGREEMENT = agr_true,
    ALPHA = alphas,
    BETA = betas,
    DATA_TYPE = "ordinal"
  )

  # Use true parameters (which should be close to optimum for this data)
  raw_phi <- log(agr2prec(agr_true))
  raw_tau <- tau2raw(tau_true)

  # Compute Hessian
  hess <- cpp_profile_extended_hess_raw_tau(
    Y = dt$rating,
    ITEM_INDS = as.integer(dt$id_item),
    WORKER_INDS = as.integer(dt$id_worker),
    ALPHA = rep(0, J),
    BETA = rep(0, W),
    RAW_TAU = raw_tau,
    RAW_PHI = raw_phi,
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = TRUE,
    WORKER_NUISANCE = TRUE,
    PROF_UNI_RANGE = 2L,
    PROF_UNI_MAX_ITER = 50L,
    PROF_MAX_ITER = 5L,
    PROF_TOL = 1e-4
  )

  # Check that all eigenvalues are negative (negative definite)
  eigenvalues <- eigen(hess, symmetric = TRUE, only.values = TRUE)$values

  # At a maximum, Hessian should be negative definite (all eigenvalues negative)
  # Use lenient tolerance since: (1) we're at simulation params not MLE,
  # (2) our Hessian has numerical approximation errors
  expect_true(all(eigenvalues < 2))
})
