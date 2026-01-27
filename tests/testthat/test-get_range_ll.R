test_that("get_range_ll returns data.frame with correct structure", {
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )

  result <- get_range_ll(fit)

  expect_s3_class(result, "data.frame")
  expect_true(all(
    c("precision", "agreement", "profile", "modified") %in% names(result)
  ))
  expect_equal(nrow(result), 15) # default GRID_LENGTH

  # Check all columns are numeric

  expect_true(is.numeric(result$precision))
  expect_true(is.numeric(result$agreement))
  expect_true(is.numeric(result$profile))
  expect_true(is.numeric(result$modified))

  # Check values are finite
  expect_true(all(is.finite(result$precision)))
  expect_true(all(is.finite(result$agreement)))
  expect_true(all(is.finite(result$profile)))
  expect_true(all(is.finite(result$modified)))

  # Check agreement is in (0, 1)
  expect_true(all(result$agreement > 0 & result$agreement < 1))
})

test_that("get_range_ll works for continuous one-way model", {
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )

  expect_error(get_range_ll(fit, RANGE = 0))
  expect_error(get_range_ll(fit, RANGE = -0.1))
  expect_error(get_range_ll(fit, RANGE = 1.5))
})

test_that("get_range_ll rejects invalid GRID_LENGTH values", {
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

  fit <- agreement(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = FALSE
  )

  expect_error(get_range_ll(fit, GRID_LENGTH = 0))
  expect_error(get_range_ll(fit, GRID_LENGTH = -5))
})
