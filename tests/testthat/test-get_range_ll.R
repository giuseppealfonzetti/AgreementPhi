test_that("get_range_ll returns data.frame with correct structure", {
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
  result <- get_range_ll(fit)

  expect_s3_class(result, "data.frame")
  expect_true(all(
    c("precision", "agreement", "profile", "modified") %in% names(result)
  ))
  expect_equal(nrow(result), 15)
  expect_true(is.numeric(result$precision))
  expect_true(is.numeric(result$agreement))
  expect_true(is.numeric(result$profile))
  expect_true(is.numeric(result$modified))
  expect_true(all(is.finite(result$precision)))
  expect_true(all(is.finite(result$agreement)))
  expect_true(all(is.finite(result$profile)))
  expect_true(all(is.finite(result$modified)))
  expect_true(all(result$agreement > 0 & result$agreement < 1))
})

test_that("get_range_ll works for continuous one-way model", {
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
  result <- get_range_ll(fit, RANGE = 0.15, GRID_LENGTH = 10)

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 10)
  expect_true(all(is.finite(result$profile)))
  expect_true(all(is.finite(result$modified)))
})

test_that("get_range_ll works for continuous two-way model", {
  skip_if_not_installed("AlgDesign")
  set.seed(321)
  n_items <- 50
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
    NUISANCE = c("items", "workers"),
    VERBOSE = FALSE
  )
  result <- get_range_ll(fit, RANGE = 0.15, GRID_LENGTH = 10)

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 10)
  expect_true(all(is.finite(result$profile)))
  expect_true(all(is.finite(result$modified)))
})

test_that("get_range_ll works for ordinal one-way model", {
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

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )
  result <- get_range_ll(fit, RANGE = 0.15, GRID_LENGTH = 10)

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 10)
  expect_true(all(is.finite(result$profile)))
  expect_true(all(is.finite(result$modified)))
})

test_that("different RANGE values produce different grid spans", {
  skip_if_not_installed("AlgDesign")
  set.seed(321)
  n_items <- 40
  alphas <- runif(n_items, -2, 2)

  dt <- sim_data(
    J = n_items,
    B = 5,
    AGREEMENT = 0.5,
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
  result_small <- get_range_ll(fit, RANGE = 0.1, GRID_LENGTH = 10)
  result_large <- get_range_ll(fit, RANGE = 0.3, GRID_LENGTH = 10)

  span_small <- max(result_small$agreement) - min(result_small$agreement)
  span_large <- max(result_large$agreement) - min(result_large$agreement)
  expect_true(span_small < span_large)
})

test_that("different GRID_LENGTH values produce different row counts", {
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
  result_5 <- get_range_ll(fit, GRID_LENGTH = 5)
  result_20 <- get_range_ll(fit, GRID_LENGTH = 20)

  expect_equal(nrow(result_5), 5)
  expect_equal(nrow(result_20), 20)
})

test_that("get_range_ll rejects invalid RANGE values", {
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

  expect_error(get_range_ll(fit, RANGE = 0))
  expect_error(get_range_ll(fit, RANGE = -0.1))
  expect_error(get_range_ll(fit, RANGE = 1.5))
})

test_that("get_range_ll rejects invalid GRID_LENGTH values", {
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

  expect_error(get_range_ll(fit, GRID_LENGTH = 0))
  expect_error(get_range_ll(fit, GRID_LENGTH = -5))
})
