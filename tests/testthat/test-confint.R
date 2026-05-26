test_that("get_ci returns list with correct structure", {
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = alphas,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  ci <- confint(fit)

  expect_type(ci, "list")
  expect_true(all(
    c("agreement_est", "agreement_se", "agreement_ci") %in% names(ci)
  ))
  expect_length(ci, 3)
  expect_true(is.numeric(ci$agreement_est) && length(ci$agreement_est) == 1)
  expect_true(is.numeric(ci$agreement_se) && length(ci$agreement_se) == 1)
  expect_true(is.numeric(ci$agreement_ci) && length(ci$agreement_ci) == 2)
  expect_true(ci$agreement_se >= 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)
  expect_true(ci$agreement_ci[1] <= ci$agreement_est)
  expect_true(ci$agreement_est <= ci$agreement_ci[2])
})

test_that("get_ci works for continuous one-way model with low agreement", {
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.3,
    ALPHA = alphas,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  ci <- confint(fit, level = 0.95)
  expect_type(ci, "list")
  expect_true(is.finite(ci$agreement_est))
  expect_true(ci$agreement_se > 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)

  fit <- agreement(
    rd,
    METHOD = "profile",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  ci <- confint(fit, level = 0.95)
  expect_type(ci, "list")
  expect_true(is.finite(ci$agreement_est))
  expect_true(ci$agreement_se > 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)
})

test_that("get_ci works for continuous one-way model with high agreement", {
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.8,
    ALPHA = alphas,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  ci <- confint(fit, level = 0.95)
  expect_type(ci, "list")
  expect_true(is.finite(ci$agreement_est))
  expect_true(ci$agreement_se > 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)
})

test_that("get_ci works for continuous two-way model with low agreement", {
  set.seed(321)
  n_items <- 60
  alphas <- runif(n_items, -1.5, 1.5)

  dt <- sim_data(
    J = n_items,
    B = 6,
    AGREEMENT = 0.4,
    ALPHA = alphas,
    DATA_TYPE = "continuous",
    SEED = 456
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = FALSE
  )
  ci <- confint(fit, level = 0.95)
  expect_type(ci, "list")
  expect_true(is.finite(ci$agreement_est))
  expect_true(ci$agreement_se > 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)
})

test_that("get_ci works for continuous two-way model with high agreement", {
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.8,
    ALPHA = alphas,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = FALSE
  )
  ci <- confint(fit, level = 0.95)
  expect_type(ci, "list")
  expect_true(is.finite(ci$agreement_est))
  expect_true(ci$agreement_se > 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)
})

test_that("get_ci works for ordinal one-way model", {
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = alphas,
    DATA_TYPE = "ordinal",
    K = 5,
    SEED = 123
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  ci <- confint(fit, level = 0.95)
  expect_type(ci, "list")
  expect_true(is.finite(ci$agreement_est))
  expect_true(ci$agreement_se > 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)

  fit <- agreement(
    rd,
    METHOD = "profile",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  ci <- confint(fit, level = 0.95)
  expect_type(ci, "list")
  expect_true(is.finite(ci$agreement_est))
  expect_true(ci$agreement_se > 0)
  expect_true(ci$agreement_ci[1] >= 0 && ci$agreement_ci[2] <= 1)
})

test_that("wider confidence intervals for higher confidence levels", {
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = alphas,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  ci_90 <- confint(fit, level = 0.90)
  ci_95 <- confint(fit, level = 0.95)
  ci_99 <- confint(fit, level = 0.99)

  width_90 <- ci_90$agreement_ci[2] - ci_90$agreement_ci[1]
  width_95 <- ci_95$agreement_ci[2] - ci_95$agreement_ci[1]
  width_99 <- ci_99$agreement_ci[2] - ci_99$agreement_ci[1]

  expect_true(width_90 < width_95)
  expect_true(width_95 < width_99)
  expect_equal(ci_90$agreement_est, ci_95$agreement_est)
  expect_equal(ci_95$agreement_est, ci_99$agreement_est)
})

test_that("get_ci rejects invalid CONFIDENCE values", {
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = alphas,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )

  expect_error(confint(fit, level = 0))
  expect_error(confint(fit, level = 1))
  expect_error(confint(fit, level = -0.5))
  expect_error(confint(fit, level = 1.5))
  expect_error(confint(fit, level = "0.95"))
})
