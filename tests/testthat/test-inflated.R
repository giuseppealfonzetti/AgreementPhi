J <- 20
B <- 8
ALPHA <- rep(0, J)

# sim_data ----------------------------------------------------------------

test_that("sim_data inflated full model output is in [0,1]", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  expect_true(all(dt$rating >= 0 & dt$rating <= 1))
})

test_that("sim_data inflated full model generates both zeros and ones", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  expect_true(any(dt$rating == 0))
  expect_true(any(dt$rating == 1))
})

test_that("sim_data zero-only inflation produces no ones", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = NA,
    SEED = 1
  )
  expect_true(all(dt$rating <= 1))
  expect_false(any(dt$rating == 1))
  expect_true(any(dt$rating == 0))
})

test_that("sim_data one-only inflation produces no zeros", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = NA,
    K1 = 2,
    SEED = 1
  )
  expect_true(all(dt$rating >= 0))
  expect_false(any(dt$rating == 0))
  expect_true(any(dt$rating == 1))
})

# detection and dispatch --------------------------------------------------

test_that("agreement detects inflated data type", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, NUISANCE = "items")
  expect_equal(fit$data_type, "inflated")
})

test_that("agreement rejects inflated data with worker_ids", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  expect_error(agreement(rd), "one-way only")
})

test_that("agreement stores method correctly for inflated data", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit_p <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  fit_m <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  expect_equal(fit_p$method, "profile")
  expect_equal(fit_m$method, "modified")
})

# basic fitting sanity ----------------------------------------------------

test_that("inflated profile fit converges with finite loglik", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(fit$convergence, 0)
  expect_true(is.finite(fit$loglik))
})

test_that("inflated profile fit parameters are in valid range", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_true(fit$phi > 0)
  expect_true(fit$k0 < fit$k1)
})

test_that("inflated mpl loglik is less than or equal to profile loglik", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  pf <- fit_inflated_profile(
    Y = dt$rating,
    ITEM_INDS = dt$id_item,
    J = J,
    COMPUTE_VCOV = FALSE
  )
  mf <- fit_inflated_mpl(
    Y = dt$rating,
    ITEM_INDS = dt$id_item,
    J = J,
    REF_FIT = pf
  )
  expect_true(mf$loglik <= pf$loglik + 1e-6)
})

# one-sided constraints ---------------------------------------------------

test_that("data with no ones pins k1 to boundary and sets fix_k1", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = NA,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_false(fit$fix_k0)
  expect_true(fit$fix_k1)
  expect_equal(fit$k1, fit$boundary)
})

test_that("data with no zeros pins k0 to boundary and sets fix_k0", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = NA,
    K1 = 2,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_true(fit$fix_k0)
  expect_false(fit$fix_k1)
  expect_equal(fit$k0, -fit$boundary)
})

test_that("full inflated data has both fix flags FALSE", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_false(fit$fix_k0)
  expect_false(fit$fix_k1)
})

test_that("fix_k1 fit has zero SE for k1 and positive SE for phi and k0", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = NA,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(fit$se[["k1"]], 0)
  expect_true(fit$se[["phi"]] > 0)
  expect_true(fit$se[["k0"]] > 0)
})

test_that("fix_k0 fit has zero SE for k0 and positive SE for phi and k1", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = NA,
    K1 = 2,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(fit$se[["k0"]], 0)
  expect_true(fit$se[["phi"]] > 0)
  expect_true(fit$se[["k1"]] > 0)
})

# SE and vcov -------------------------------------------------------------

test_that("full inflated fit SEs are finite and non-negative", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_true(all(is.finite(fit$se)))
  expect_true(all(fit$se >= 0))
})

test_that("inflated fit vcov diagonal matches squared SEs", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(sqrt(diag(fit$vcov)), unname(fit$se), tolerance = 1e-8)
})

test_that("fixed cutpoint row and col in vcov are all zero", {
  dt_fk1 <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = NA,
    SEED = 1
  )
  fit_fk1 <- fit_inflated_profile(
    Y = dt_fk1$rating,
    ITEM_INDS = dt_fk1$id_item,
    J = J
  )
  expect_true(all(fit_fk1$vcov[3, ] == 0))
  expect_true(all(fit_fk1$vcov[, 3] == 0))

  dt_fk0 <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = NA,
    K1 = 2,
    SEED = 1
  )
  fit_fk0 <- fit_inflated_profile(
    Y = dt_fk0$rating,
    ITEM_INDS = dt_fk0$id_item,
    J = J
  )
  expect_true(all(fit_fk0$vcov[2, ] == 0))
  expect_true(all(fit_fk0$vcov[, 2] == 0))
})

# get_ci ------------------------------------------------------------------

test_that("get_ci inflated returns correct structure", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  ci <- confint(fit)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.matrix(ci$parameters))
  expect_equal(rownames(ci$parameters), c("phi", "k0", "k1"))
  expect_equal(colnames(ci$parameters), c("Estimate", "Std. Error", "2.5 %", "97.5 %"))
  expect_true(is.matrix(ci$agreement))
  expect_equal(rownames(ci$agreement), "agreement")
  expect_true(all(is.finite(ci$parameters)))
  expect_true(all(is.finite(ci$agreement)))
})

test_that("get_ci inflated CI contains estimate", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  ci <- confint(fit)
  p <- ci$parameters
  expect_true(p["phi", "2.5 %"] <= p["phi", "Estimate"] && p["phi", "Estimate"] <= p["phi", "97.5 %"])
  expect_true(p["k0", "2.5 %"] <= p["k0", "Estimate"] && p["k0", "Estimate"] <= p["k0", "97.5 %"])
  expect_true(p["k1", "2.5 %"] <= p["k1", "Estimate"] && p["k1", "Estimate"] <= p["k1", "97.5 %"])
})

test_that("get_ci one-sided has zero SE for the pinned cutpoint", {
  dt_fk1 <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = NA,
    SEED = 1
  )
  rd_fk1 <- rating_data(dt_fk1$rating, dt_fk1$id_item, VERBOSE = FALSE)
  fit_fk1 <- agreement(rd_fk1, METHOD = "modified", NUISANCE = "items")
  ci_fk1 <- confint(fit_fk1)
  expect_equal(ci_fk1$parameters["k1", "Std. Error"], 0)
  expect_true(ci_fk1$parameters["phi", "Std. Error"] > 0)
  expect_true(ci_fk1$parameters["k0", "Std. Error"] > 0)

  dt_fk0 <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = NA,
    K1 = 2,
    SEED = 1
  )
  rd_fk0 <- rating_data(dt_fk0$rating, dt_fk0$id_item, VERBOSE = FALSE)
  fit_fk0 <- agreement(rd_fk0, METHOD = "modified", NUISANCE = "items")
  ci_fk0 <- confint(fit_fk0)
  expect_equal(ci_fk0$parameters["k0", "Std. Error"], 0)
  expect_true(ci_fk0$parameters["phi", "Std. Error"] > 0)
  expect_true(ci_fk0$parameters["k1", "Std. Error"] > 0)
})

# recovery ----------------------------------------------------------------

test_that("inflated agreement estimate is in the right ballpark", {
  set.seed(7)
  J_rec <- 50
  alpha_true <- runif(J_rec, -1, 1)
  K0_true <- -2
  K1_true <- 2
  agr_phi <- 0.5
  agr_true <- par2agr(
    PHI = agr2prec(agr_phi),
    ALPHA = alpha_true,
    K0 = K0_true,
    K1 = K1_true
  )$agreement
  dt <- sim_data(
    J = J_rec,
    B = 10,
    AGREEMENT = agr_phi,
    ALPHA = alpha_true,
    DATA_TYPE = "inflated",
    K0 = K0_true,
    K1 = K1_true,
    SEED = 7
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  expect_equal(fit$modified$agreement, agr_true, tolerance = 0.15)
})

# degenerate item handling -----------------------------------------------

test_that("rating_data detects degenerate items; agreement() pre-screens before fitting", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = NA,
    SEED = 1
  )
  all_zero <- as.logical(tapply(dt$rating, dt$id_item, function(x) all(x == 0)))
  n_informative <- sum(!all_zero)
  n_degen <- sum(all_zero)
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  expect_equal(rd$n_items, J)
  expect_equal(length(rd$degen_ids), n_degen)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  expect_equal(fit$fit_data$n_items, n_informative)
  expect_equal(length(fit$alpha), n_informative)
})

test_that("agreement over inflated data uses only non-degenerate items", {
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = NA,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  n_degen <- length(rd$degen_ids)
  keep <- !(rd$item_ids %in% rd$degen_ids)
  fit_iids <- match(rd$item_ids[keep], sort(unique(rd$item_ids[keep])))
  fit_J <- rd$n_items - n_degen
  pf <- fit_inflated_profile(
    Y = rd$ratings[keep],
    ITEM_INDS = fit_iids,
    J = fit_J
  )
  raw_agr <- par2agr(pf$phi, ALPHA = pf$alpha, K0 = pf$k0, K1 = pf$k1)$agreement
  expected <- (fit_J * raw_agr + n_degen) / rd$n_items
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  expect_equal(fit$profile$agreement, expected, tolerance = 1e-8)
})

# vcov Hessian ------------------------------------------------------------

test_that("optimHess vcov matches numDeriv hessian", {
  skip_if_not_installed("numDeriv")
  dt <- sim_data(
    J = J,
    B = B,
    AGREEMENT = 0.5,
    ALPHA = ALPHA,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  expect_equal(fit$vcov, fit$vcov, tolerance = 1e-4)

  pf <- AgreementPhi:::fit_inflated_profile(
    Y = rd$ratings,
    ITEM_INDS = rd$item_ids,
    J = rd$n_items,
    COMPUTE_VCOV = FALSE
  )
  enc <- AgreementPhi:::inflated_encoding(
    pf$fix_k0,
    pf$fix_k1,
    -pf$boundary,
    pf$boundary
  )
  alpha_fixed <- pf$alpha
  frozen_obj <- function(par) {
    p <- enc$decode(par)
    -AgreementPhi:::cpp_inflated_profile(
      as.numeric(rd$ratings),
      as.integer(rd$item_ids),
      as.numeric(alpha_fixed),
      p$phi,
      p$k0,
      p$k1,
      rd$n_items
    )$ll
  }

  H_optimHess <- unname(stats::optimHess(pf$par, frozen_obj))
  H_numDeriv <- numDeriv::hessian(frozen_obj, pf$par)
  expect_equal(H_optimHess, H_numDeriv, tolerance = 1e-4)
})
