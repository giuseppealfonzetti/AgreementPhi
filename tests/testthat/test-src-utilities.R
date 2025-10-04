#### log Beta(a,b) ####
RlogB <- function(PARS) {
  log(beta(PARS[1], PARS[2]))
}

set.seed(123)
test_that("log Beta(a,b) | gradient", {
  skip_if_not_installed("numDeriv")

  x <- runif(2)
  numgrad <- numDeriv::grad(RlogB, x)
  cpp_res <- cpp_beta_funs(A = x[1], B = x[2])

  expect_equal(numgrad[1], cpp_res$da)
  expect_equal(numgrad[2], cpp_res$db)
})
test_that("log Beta(a,b) | hessian", {
  skip_if_not_installed("numDeriv")

  x <- runif(2)
  nummhess <- numDeriv::hessian(RlogB, x)
  cpp_res <- cpp_beta_funs(A = x[1], B = x[2])

  expect_equal(nummhess[1, 1], cpp_res$da2)
  expect_equal(nummhess[2, 2], cpp_res$db2)
  expect_equal(nummhess[2, 1], cpp_res$dadb)
})

#### iBeta(x, a, b) ####
RlogiB <- function(PARS, X) {
  pbeta(X, PARS[1], PARS[2]) * beta(PARS[1], PARS[2])
}


for (seed in 1:3) {
  test_that(paste("iBeta(x, a, b) | gradient | seed ", seed), {
    skip_if_not_installed("numDeriv")

    set.seed(seed)
    ub <- runif(1)
    x <- runif(2)
    numgrad <- numDeriv::grad(RlogiB, x, X = ub)
    cpp_res <- cpp_ibeta_funs(X = ub, A = x[1], B = x[2])

    expect_equal(numgrad[1], cpp_res$da)
    expect_equal(numgrad[2], cpp_res$db)
  })

  test_that(paste("iBeta(x, a, b) | hessian | seed ", seed), {
    skip_if_not_installed("numDeriv")

    set.seed(1)
    ub <- runif(1)
    x <- runif(2)
    numhess <- numDeriv::hessian(RlogiB, x, X = ub)
    cpp_res <- cpp_ibeta_funs(X = ub, A = x[1], B = x[2])

    expect_equal(numhess[1, 1], cpp_res$da2)
    expect_equal(numhess[2, 2], cpp_res$db2)
    expect_equal(numhess[2, 1], cpp_res$dadb)
  })
}

#### CDF beta F(x; a, b) ####
RF <- function(PARS, X) {
  pbeta(X, PARS[1], PARS[2])
}

for (seed in 1:3) {
  test_that(paste("CDF(x, a, b) | gradient | seed ", seed), {
    skip_if_not_installed("numDeriv")

    set.seed(seed)
    ub <- runif(1)
    x <- runif(2)
    numgrad <- numDeriv::grad(RF, x, X = ub)
    cpp_res <- cpp_cdfbeta_funs(X = ub, A = x[1], B = x[2])

    expect_equal(numgrad[1], cpp_res$da)
    expect_equal(numgrad[2], cpp_res$db)
  })

  test_that(paste("CDF(x, a, b) | hessian | seed ", seed), {
    skip_if_not_installed("numDeriv")

    set.seed(seed)
    ub <- runif(1)
    x <- runif(2)
    numhess <- numDeriv::hessian(RF, x, X = ub)
    cpp_res <- cpp_cdfbeta_funs(X = ub, A = x[1], B = x[2])

    expect_equal(numhess[1, 1], cpp_res$da2)
    expect_equal(numhess[2, 2], cpp_res$db2)
    expect_equal(numhess[2, 1], cpp_res$dadb)
  })
}


#### CDF beta F(x; mu, phi) ####
RF <- function(PARS, X) {
  a <- PARS[1] * PARS[2]
  b <- (1 - PARS[1]) * PARS[2]
  pbeta(X, a, b)
}
for (seed in 1:3) {
  test_that(paste("CDF(x, mu, phi) | gradient | seed ", seed), {
    skip_if_not_installed("numDeriv")

    set.seed(seed)
    ub <- runif(1)
    x <- runif(2)
    pars <- c(x[1] / (x[2] + x[1]), x[1] + x[2])

    numgrad <- numDeriv::grad(RF, pars, X = ub)
    cpp_res <- cpp_cdfbeta_muphi_funs(X = ub, MU = pars[1], PHI = pars[2])

    expect_equal(numgrad[1], cpp_res$dmu)
  })

  test_that(paste("CDF(x, mu, phi) | hessian | seed ", seed), {
    skip_if_not_installed("numDeriv")

    set.seed(seed)
    ub <- runif(1)
    x <- runif(2)
    pars <- c(x[1] / (x[2] + x[1]), x[1] + x[2])

    numhess <- numDeriv::hessian(RF, pars, X = ub)
    cpp_res <- cpp_cdfbeta_muphi_funs(X = ub, MU = pars[1], PHI = pars[2])

    expect_equal(numhess[1, 1], cpp_res$dmu2)
  })
}
