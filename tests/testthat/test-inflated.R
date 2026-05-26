J     <- 20
B     <- 8
ALPHA <- rep(0, J)

# sim_data ----------------------------------------------------------------

test_that("sim_data inflated full model output is in [0,1]", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  expect_true(all(dt$rating >= 0 & dt$rating <= 1))
})

test_that("sim_data inflated full model generates both zeros and ones", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  expect_true(any(dt$rating == 0))
  expect_true(any(dt$rating == 1))
})

test_that("sim_data zero-only inflation produces no ones", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  expect_true(all(dt$rating <= 1))
  expect_false(any(dt$rating == 1))
  expect_true(any(dt$rating == 0))
})

test_that("sim_data one-only inflation produces no zeros", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = NA, K1 = 2, SEED = 1)
  expect_true(all(dt$rating >= 0))
  expect_false(any(dt$rating == 0))
  expect_true(any(dt$rating == 1))
})

# detection and dispatch --------------------------------------------------

test_that("agreement detects inflated data type", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item)
  expect_equal(fit$data_type, "inflated")
})

test_that("agreement rejects inflated data with WORKER_INDS", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  expect_error(
    agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item,
              WORKER_INDS = dt$id_worker),
    "one-way only"
  )
})

test_that("agreement stores method correctly for inflated data", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit_p <- agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item,
                     METHOD = "profile")
  fit_m <- agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item,
                     METHOD = "modified")
  expect_equal(fit_p$method, "profile")
  expect_equal(fit_m$method, "modified")
})

# basic fitting sanity ----------------------------------------------------

test_that("inflated profile fit converges with finite loglik", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(fit$convergence, 0)
  expect_true(is.finite(fit$loglik))
})

test_that("inflated profile fit parameters are in valid range", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_true(fit$phi > 0)
  expect_true(fit$k0 < fit$k1)
})

test_that("inflated mpl loglik is less than or equal to profile loglik", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  pf <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J,
                              COMPUTE_VCOV = FALSE)
  mf <- fit_inflated_mpl(Y = dt$rating, ITEM_INDS = dt$id_item, J = J,
                          REF_FIT = pf)
  expect_true(mf$loglik <= pf$loglik + 1e-6)
})

# one-sided constraints ---------------------------------------------------

test_that("data with no ones pins k1 to boundary and sets fix_k1", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_false(fit$fix_k0)
  expect_true(fit$fix_k1)
  expect_equal(fit$k1, fit$boundary)
})

test_that("data with no zeros pins k0 to boundary and sets fix_k0", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = NA, K1 = 2, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_true(fit$fix_k0)
  expect_false(fit$fix_k1)
  expect_equal(fit$k0, -fit$boundary)
})

test_that("full inflated data has both fix flags FALSE", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_false(fit$fix_k0)
  expect_false(fit$fix_k1)
})

test_that("fix_k1 fit has zero SE for k1 and positive SE for phi and k0", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(fit$se[["k1"]], 0)
  expect_true(fit$se[["phi"]] > 0)
  expect_true(fit$se[["k0"]] > 0)
})

test_that("fix_k0 fit has zero SE for k0 and positive SE for phi and k1", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = NA, K1 = 2, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(fit$se[["k0"]], 0)
  expect_true(fit$se[["phi"]] > 0)
  expect_true(fit$se[["k1"]] > 0)
})

# SE and vcov -------------------------------------------------------------

test_that("full inflated fit SEs are finite and non-negative", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_true(all(is.finite(fit$se)))
  expect_true(all(fit$se >= 0))
})

test_that("inflated fit vcov diagonal matches squared SEs", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(sqrt(diag(fit$vcov)), unname(fit$se), tolerance = 1e-8)
})

test_that("fixed cutpoint row and col in vcov are all zero", {
  dt_fk1 <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                     DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  fit_fk1 <- fit_inflated_profile(Y = dt_fk1$rating, ITEM_INDS = dt_fk1$id_item,
                                   J = J)
  expect_true(all(fit_fk1$vcov[3, ] == 0))
  expect_true(all(fit_fk1$vcov[, 3] == 0))

  dt_fk0 <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                     DATA_TYPE = "inflated", K0 = NA, K1 = 2, SEED = 1)
  fit_fk0 <- fit_inflated_profile(Y = dt_fk0$rating, ITEM_INDS = dt_fk0$id_item,
                                   J = J)
  expect_true(all(fit_fk0$vcov[2, ] == 0))
  expect_true(all(fit_fk0$vcov[, 2] == 0))
})

# par2agr -----------------------------------------------------------------

test_that("par2agr without K0/K1 equals prec2agr", {
  phi <- 3.5
  expect_equal(par2agr(phi)$agreement, prec2agr(phi))
})

test_that("par2agr result is in [0,1]", {
  alpha <- runif(10, -2, 2)
  result <- par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 2)
  expect_true(result$agreement >= 0 && result$agreement <= 1)
  expect_true(all(result$agreement_by_item >= 0 & result$agreement_by_item <= 1))
})

test_that("par2agr is increasing in PHI", {
  alpha <- rep(0, 10)
  agr_low  <- par2agr(PHI = 1,  ALPHA = alpha, K0 = -2, K1 = 2)$agreement
  agr_high <- par2agr(PHI = 10, ALPHA = alpha, K0 = -2, K1 = 2)$agreement
  expect_true(agr_high > agr_low)
})

test_that("par2agr NA and Inf sentinels give same result as boundary values", {
  alpha <- rep(0, 5)
  expect_equal(
    par2agr(PHI = 2, ALPHA = alpha, K0 = NA,   K1 = 2)$agreement,
    par2agr(PHI = 2, ALPHA = alpha, K0 = -100, K1 = 2)$agreement
  )
  expect_equal(
    par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = NA)$agreement,
    par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 100)$agreement
  )
})

# get_ci ------------------------------------------------------------------

test_that("get_ci inflated returns correct structure", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item,
                   METHOD = "modified")
  ci <- get_ci(fit)
  expected_names <- c("phi_est", "phi_se", "phi_ci",
                      "k0_est", "k0_se", "k0_ci",
                      "k1_est", "k1_se", "k1_ci")
  expect_true(all(expected_names %in% names(ci)))
  expect_true(all(is.finite(unlist(ci))))
})

test_that("get_ci inflated CI contains estimate", {
  dt <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                 DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1)
  fit <- agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item,
                   METHOD = "modified")
  ci <- get_ci(fit)
  expect_true(ci$phi_ci[1] <= ci$phi_est && ci$phi_est <= ci$phi_ci[2])
  expect_true(ci$k0_ci[1]  <= ci$k0_est  && ci$k0_est  <= ci$k0_ci[2])
  expect_true(ci$k1_ci[1]  <= ci$k1_est  && ci$k1_est  <= ci$k1_ci[2])
})

test_that("get_ci one-sided has zero SE for the pinned cutpoint", {
  dt_fk1 <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                     DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  fit_fk1 <- agreement(RATINGS = dt_fk1$rating, ITEM_INDS = dt_fk1$id_item,
                       METHOD = "modified")
  ci_fk1 <- get_ci(fit_fk1)
  expect_equal(ci_fk1$k1_se, 0)
  expect_true(ci_fk1$phi_se > 0)
  expect_true(ci_fk1$k0_se > 0)

  dt_fk0 <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                     DATA_TYPE = "inflated", K0 = NA, K1 = 2, SEED = 1)
  fit_fk0 <- agreement(RATINGS = dt_fk0$rating, ITEM_INDS = dt_fk0$id_item,
                       METHOD = "modified")
  ci_fk0 <- get_ci(fit_fk0)
  expect_equal(ci_fk0$k0_se, 0)
  expect_true(ci_fk0$phi_se > 0)
  expect_true(ci_fk0$k1_se > 0)
})

# recovery ----------------------------------------------------------------

test_that("inflated agreement estimate is in the right ballpark", {
  set.seed(7)
  J_rec      <- 50
  alpha_true <- runif(J_rec, -1, 1)
  K0_true    <- -2
  K1_true    <- 2
  agr_phi    <- 0.5
  agr_true   <- par2agr(
    PHI = agr2prec(agr_phi), ALPHA = alpha_true,
    K0 = K0_true, K1 = K1_true
  )$agreement
  dt <- sim_data(J = J_rec, B = 10, AGREEMENT = agr_phi,
                 ALPHA = alpha_true,
                 DATA_TYPE = "inflated", K0 = K0_true, K1 = K1_true, SEED = 7)
  fit <- agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item,
                   METHOD = "modified")
  expect_equal(fit$modified$agreement, agr_true, tolerance = 0.15)
})

# is_degen and degenerate item handling ----------------------------------

test_that("fit alpha and is_degen have length J including degenerate items", {
  dt  <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                  DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  fit <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expect_equal(length(fit$alpha),    J)
  expect_equal(length(fit$is_degen), J)
})

test_that("is_degen is TRUE exactly for all-zero items", {
  dt       <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                       DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  fit      <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  all_zero <- as.logical(tapply(dt$rating, dt$id_item, function(x) all(x == 0)))
  expect_true(all(fit$is_degen[all_zero]))
  expect_true(!any(fit$is_degen[!all_zero]))
})

test_that("agreement is computed over non-degenerate items only", {
  dt  <- sim_data(J = J, B = B, AGREEMENT = 0.5, ALPHA = ALPHA,
                  DATA_TYPE = "inflated", K0 = -2, K1 = NA, SEED = 1)
  fit <- agreement(RATINGS = dt$rating, ITEM_INDS = dt$id_item,
                   METHOD = "profile")
  pf  <- fit_inflated_profile(Y = dt$rating, ITEM_INDS = dt$id_item, J = J)
  expected <- par2agr(pf$phi,
                      ALPHA = pf$alpha[!pf$is_degen],
                      K0 = pf$k0, K1 = pf$k1)$agreement
  expect_equal(fit$profile$agreement, expected, tolerance = 1e-8)
})
