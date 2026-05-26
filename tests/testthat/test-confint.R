test_that("confint returns list with parameters and agreement matrices", {
  skip_if_not_installed("AlgDesign")
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
  expect_named(ci, c("parameters", "agreement"))

  expect_true(is.matrix(ci$parameters))
  expect_equal(rownames(ci$parameters), "phi")
  expect_equal(colnames(ci$parameters), c("Estimate", "Std. Error", "2.5 %", "97.5 %"))
  expect_true(ci$parameters["phi", "Std. Error"] > 0)
  expect_true(ci$parameters["phi", "Estimate"] > 0)
  expect_true(ci$parameters["phi", "2.5 %"] > 0)

  expect_true(is.matrix(ci$agreement))
  expect_equal(rownames(ci$agreement), "agreement")
  expect_equal(colnames(ci$agreement), c("Estimate", "Std. Error", "2.5 %", "97.5 %"))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)
  expect_true(ci$agreement["agreement", "2.5 %"] <= ci$agreement["agreement", "Estimate"])
  expect_true(ci$agreement["agreement", "Estimate"] <= ci$agreement["agreement", "97.5 %"])
})

test_that("confint works for continuous one-way model with low agreement", {
  skip_if_not_installed("AlgDesign")
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

  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"), VERBOSE = FALSE)
  ci <- confint(fit, level = 0.95)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.finite(ci$agreement["agreement", "Estimate"]))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)

  fit <- agreement(rd, METHOD = "profile", NUISANCE = c("items"), VERBOSE = FALSE)
  ci <- confint(fit, level = 0.95)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.finite(ci$agreement["agreement", "Estimate"]))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)
})

test_that("confint works for continuous one-way model with high agreement", {
  skip_if_not_installed("AlgDesign")
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

  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"), VERBOSE = FALSE)
  ci <- confint(fit, level = 0.95)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.finite(ci$agreement["agreement", "Estimate"]))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)
})

test_that("confint works for continuous two-way model with low agreement", {
  skip_if_not_installed("AlgDesign")
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

  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items", "workers"), VERBOSE = FALSE)
  ci <- confint(fit, level = 0.95)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.finite(ci$agreement["agreement", "Estimate"]))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)
})

test_that("confint works for continuous two-way model with high agreement", {
  skip_if_not_installed("AlgDesign")
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

  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items", "workers"), VERBOSE = FALSE)
  ci <- confint(fit, level = 0.95)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.finite(ci$agreement["agreement", "Estimate"]))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)
})

test_that("confint works for ordinal one-way model", {
  skip_if_not_installed("AlgDesign")
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

  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"), VERBOSE = FALSE)
  ci <- confint(fit, level = 0.95)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.finite(ci$agreement["agreement", "Estimate"]))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)

  fit <- agreement(rd, METHOD = "profile", NUISANCE = c("items"), VERBOSE = FALSE)
  ci <- confint(fit, level = 0.95)
  expect_named(ci, c("parameters", "agreement"))
  expect_true(is.finite(ci$agreement["agreement", "Estimate"]))
  expect_true(ci$agreement["agreement", "Std. Error"] > 0)
  expect_true(ci$agreement["agreement", "2.5 %"] >= 0)
  expect_true(ci$agreement["agreement", "97.5 %"] <= 1)
})

test_that("wider confidence intervals for higher confidence levels", {
  skip_if_not_installed("AlgDesign")
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

  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"), VERBOSE = FALSE)
  ci_90 <- confint(fit, level = 0.90)
  ci_95 <- confint(fit, level = 0.95)
  ci_99 <- confint(fit, level = 0.99)

  width <- function(ci) ci$agreement["agreement", 4] - ci$agreement["agreement", 3]
  expect_true(width(ci_90) < width(ci_95))
  expect_true(width(ci_95) < width(ci_99))
  expect_equal(
    ci_90$agreement["agreement", "Estimate"],
    ci_95$agreement["agreement", "Estimate"]
  )
  expect_equal(
    ci_95$agreement["agreement", "Estimate"],
    ci_99$agreement["agreement", "Estimate"]
  )
})

test_that("confint rejects invalid level values", {
  skip_if_not_installed("AlgDesign")
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

  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"), VERBOSE = FALSE)

  expect_error(confint(fit, level = 0))
  expect_error(confint(fit, level = 1))
  expect_error(confint(fit, level = -0.5))
  expect_error(confint(fit, level = 1.5))
  expect_error(confint(fit, level = "0.95"))
})
