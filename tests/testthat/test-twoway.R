#### Derivatives continuous ####
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
  DATA_TYPE = "continuous",
  K = k,
  SEED = 123
)

lambda_test <- c(alphas, betas[-1])
phi_test <- agr2prec(agr)

# Function to compute log-likelihood
Rf <- function(LAMBDA) {
  cpp_twoway_joint_loglik(
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
  cpp_res <- cpp_twoway_joint_loglik(
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
  cpp_res <- cpp_twoway_joint_loglik(
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
  cpp_res <- cpp_twoway_joint_loglik(
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
  cpp_res <- cpp_twoway_joint_loglik(
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
