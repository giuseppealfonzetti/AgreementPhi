#### Ordinal model | single observation ####
Rpr <- function(MU, PHI, Y, K) {
  a <- MU * PHI
  b <- (1 - MU) * PHI
  log(pbeta(Y / K, a, b) - pbeta((Y - 1) / K, a, b))
}

test_that(paste("log p(y=c; mu, phi) | dmu "), {
  skip_if_not_installed("numDeriv")

  set.seed(123)
  x <- runif(2)
  mu <- x[1] / (x[2] + x[1])
  phi <- x[1] + x[2]

  k <- 5
  for (y in 1:k) {
    numdmu <- numDeriv::grad(Rpr, mu, PHI = phi, Y = y, K = k)
    numdmu2 <- numDeriv::hessian(Rpr, mu, PHI = phi, Y = y, K = k)

    cpp_res <- cpp_ordinal_loglik(Y = y, MU = mu, PHI = phi, K = k)
    expect_equal(numdmu, cpp_res$dmu, tolerance = 1e-3)
    expect_equal(as.numeric(numdmu2), cpp_res$dmu2, tolerance = 1e-3)
  }
})

test_that("log p(y; mu, phi) | dmu, dmu2 over grid", {
  skip_if_not_installed("numDeriv")

  k <- 5
  mus <- c(0.1, 0.25, 0.5, 0.75)
  phis <- c(1, 2, 5, 10)

  for (mu in mus) {
    for (phi in phis) {
      for (y in 1:k) {
        numdmu <- numDeriv::grad(Rpr, mu, PHI = phi, Y = y, K = k)
        numdmu2 <- numDeriv::hessian(Rpr, mu, PHI = phi, Y = y, K = k)

        cpp_res <- cpp_ordinal_loglik(Y = y, MU = mu, PHI = phi, K = k)

        if (is.finite(numdmu)) {
          expect_equal(
            object = cpp_res$dmu,
            expected = numdmu,
            tolerance = 2e-2,
            info = paste("mu=", mu, "phi=", phi, "y=", y)
          )
        }

        if (is.finite(numdmu2)) {
          expect_equal(
            object = cpp_res$dmu2,
            expected = as.numeric(numdmu2),
            tolerance = 1e-1,
            info = paste("mu=", mu, "phi=", phi, "y=", y)
          )
        }
      }
    }
  }
})


test_that("probabilities are normalized", {
  # sum_{y=1}^K exp(loglik) ≈ 1
  mus <- c(0.2, 0.5, 0.8)
  phis <- c(1, 10)
  k <- 8
  for (mu in mus) {
    for (phi in phis) {
      probs <- sapply(1:k, function(y) {
        exp(cpp_ordinal_loglik(Y = y, MU = mu, PHI = phi, K = k)$ll)
      })
      expect_equal(
        sum(probs),
        1,
        tolerance = 1e-8,
        info = paste("mu=", mu, "phi=", phi)
      )
    }
  }
})

#### Ordinal model | sample single item ####
set.seed(1)
items <- 5
budget_per_item <- 10
n_obs <- items * budget_per_item
k <- 5
alphas <- runif(items)
agr <- runif(1)
phi <- agr2prec(agr)

dt <- sim_data(
  J = items,
  B = budget_per_item,
  AGREEMENT = agr,
  ALPHA = alphas,
  DATA_TYPE = "ordinal",
  K = k,
  SEED = 123
)

Rf <- function(
  PAR,
  PHI = phi,
  Y = dt$rating,
  ITEM_INDS = dt$id_item,
  K = k,
  J = items,
  ITEM
) {
  obj <- cpp_ordinal_item_loglik(
    Y = Y,
    ITEM_INDS = ITEM_INDS,
    ALPHA = PAR,
    PHI = PHI,
    K = K,
    J = J,
    ITEM = ITEM
  )

  return(obj$ll)
}

Rf(PAR = .1, ITEM = 1)


j <- 1
dt2 <- dt[dt$id_item == j, ]
eval_alpha <- .2
cpp_res_item <- cpp_ordinal_item_loglik(
  Y = dt$rating,
  ITEM_INDS = dt$id_item,
  ALPHA = eval_alpha,
  PHI = phi,
  K = k,
  J = items,
  ITEM = j - 1
)

ll <- 0
for (i in 1:nrow(dt2)) {
  cpp_res <- cpp_ordinal_loglik(
    Y = dt2$rating[i],
    MU = plogis(eval_alpha),
    PHI = phi,
    K = k
  )

  ll <- ll + cpp_res$ll
}
test_that("ll item accumulation", {
  expect_equal(
    obj = cpp_res_item$ll,
    exp = ll
  )
})

#### Ordinal model | log det obs info ####
set.seed(1)
items <- 3
budget_per_item <- 10
n_obs <- items * budget_per_item
k <- 5
alphas <- runif(items)
agr <- runif(1)
phi <- agr2prec(agr)

dt <- sim_data(
  J = items,
  B = budget_per_item,
  AGREEMENT = agr,
  ALPHA = alphas,
  DATA_TYPE = "ordinal",
  K = k,
  SEED = 123
)

Rf <- function(
  PAR,
  PHI = phi,
  Y = dt$rating,
  ITEM_INDS = dt$id_item,
  K = k,
  J = items
) {
  obj <- 0
  for (j in 1:J) {
    jobj <- cpp_ordinal_item_loglik(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      ALPHA = PAR[j],
      PHI = PHI,
      K = K,
      J = J,
      ITEM = j - 1
    )$ll

    obj <- obj + jobj
  }

  return(obj)
}

numhess <- numDeriv::hessian(Rf, alphas)
numlogdet <- log(det(-numhess))
cpp_res <- cpp_log_det_obs_info(
  Y = dt$rating,
  ITEM_INDS = dt$id_item,
  ALPHA = alphas,
  PHI = phi,
  K = k,
  J = items
)
test_that("log det obs info", {
  expect_equal(cpp_res, numlogdet)
})

cpp_res_E <- cpp_log_det_E0d0d1(
  ITEM_INDS = dt$id_item,
  ALPHA0 = alphas,
  PHI0 = phi,
  ALPHA1 = alphas,
  PHI1 = phi,
  K = k,
  J = items
)

test_that("log det obs info equivalent to E0d0d0", {
  expect_equal(cpp_res_E, cpp_res, tolerance = 1e-1)
})
