test_that("oneway continuous works", {
  skip_if_not_installed("AlgDesign")
  set.seed(321)
  items <- 200
  budget_per_item <- 5
  workers <- 200
  alphas <- runif(items, -2, 2)
  betas <- rep(0, workers)

  agr <- .7
  dt_oneway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(
    dt_oneway$rating,
    dt_oneway$id_item,
    dt_oneway$id_worker,
    VERBOSE = FALSE
  )

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )
  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)

  agr <- .3
  dt_oneway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(
    dt_oneway$rating,
    dt_oneway$id_item,
    dt_oneway$id_worker,
    VERBOSE = FALSE
  )

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )
  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)

  fitp <- agreement(
    rd,
    METHOD = "profile",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )
  expect_equal(fitp$profile$agreement, fit$profile$agreement)
})

test_that("oneway ordinal works", {
  skip_if_not_installed("AlgDesign")
  set.seed(321)
  items <- 50
  budget_per_item <- 5
  workers <- 50
  alphas <- runif(items, -2, 2)
  betas <- rep(0, workers)

  agr <- .7
  dt_oneway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "ordinal",
    K = 6,
    SEED = 123
  )
  rd <- rating_data(
    dt_oneway$rating,
    dt_oneway$id_item,
    dt_oneway$id_worker,
    VERBOSE = FALSE
  )

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )
  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )

  fitp <- agreement(
    rd,
    METHOD = "profile",
    NUISANCE = c("items"),
    VERBOSE = TRUE
  )
  expect_equal(fitp$profile$agreement, fit$profile$agreement)
})

test_that("twoway continuous works", {
  skip_if_not_installed("AlgDesign")
  set.seed(321)
  items <- 100
  budget_per_item <- 10
  workers <- 100
  alphas <- runif(items, -2, 2)
  betas <- c(0, runif(workers - 1, -2, 2))

  agr <- .7
  dt_twoway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(
    dt_twoway$rating,
    dt_twoway$id_item,
    dt_twoway$id_worker,
    VERBOSE = FALSE
  )

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = TRUE
  )
  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)

  agr <- .3
  dt_twoway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "continuous",
    SEED = 123
  )
  rd <- rating_data(
    dt_twoway$rating,
    dt_twoway$id_item,
    dt_twoway$id_worker,
    VERBOSE = FALSE
  )

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = TRUE
  )
  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
  expect_equal(fit$modified$agreement, agr, tolerance = 1e-1)
})

test_that("twoway ordinal works", {
  skip_if_not_installed("AlgDesign")
  set.seed(321)
  items <- 10
  budget_per_item <- 5
  workers <- 20
  alphas <- runif(items, -1, 1)
  betas <- c(0, runif(workers - 1, -1, 1))

  agr <- .4
  dt_twoway <- sim_data(
    J = items,
    B = budget_per_item,
    AGREEMENT = agr,
    ALPHA = alphas,
    BETA = betas,
    W = workers,
    DATA_TYPE = "ordinal",
    K = 6,
    SEED = 123
  )
  rd <- rating_data(
    dt_twoway$rating,
    dt_twoway$id_item,
    dt_twoway$id_worker,
    VERBOSE = FALSE
  )

  fit <- agreement(
    rd,
    METHOD = "modified",
    NUISANCE = c("items", "workers"),
    VERBOSE = TRUE
  )
  expect_true(
    abs(fit$modified$agreement - agr) < abs(fit$profile$agreement - agr)
  )
})

test_that("agreement handles ordinal data with missing boundary categories", {
  skip_if_not_installed("AlgDesign")
  dt <- sim_data(
    J = 30,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = runif(30, -2, 2),
    DATA_TYPE = "ordinal",
    K = 10,
    SEED = 42
  )
  dt_sub <- dt[dt$rating >= 2 & dt$rating <= 9, ]

  rd <- rating_data(
    dt_sub$rating,
    dt_sub$id_item,
    dt_sub$id_worker,
    K = 10,
    VERBOSE = FALSE
  )
  expect_no_error(fit <- agreement(rd, NUISANCE = "items"))
  expect_true(fit$profile$agreement >= 0 && fit$profile$agreement <= 1)
  expect_equal(fit$data$K, 10L)
})

# Fixture: 3 continuous items, item 1 degenerate (all 0.5)
.degen_ratings  <- c(0.5, 0.5, 0.5,  0.2, 0.7, 0.4,  0.3, 0.8, 0.6)
.degen_item_ids <- c(1L,  1L,  1L,    2L,  2L,  2L,   3L,  3L,  3L )
.degen_worker_ids <- c(1L, 2L, 3L,   1L,  2L,  3L,   1L,  2L,  3L )

test_that("one-way: phi unaffected by degenerate items", {
  rd <- rating_data(.degen_ratings, .degen_item_ids, VERBOSE = FALSE)
  rd_clean <- rating_data(
    .degen_ratings[.degen_item_ids != 1],
    .degen_item_ids[.degen_item_ids != 1],
    VERBOSE = FALSE
  )
  fit       <- agreement(rd,       NUISANCE = "items", METHOD = "profile")
  fit_clean <- agreement(rd_clean, NUISANCE = "items", METHOD = "profile")
  expect_equal(fit$profile$precision, fit_clean$profile$precision, tolerance = 1e-6)
})

test_that("one-way: fit$data is original; fit$fit_data is filtered", {
  rd  <- rating_data(.degen_ratings, .degen_item_ids, VERBOSE = FALSE)
  fit <- agreement(rd, NUISANCE = "items", METHOD = "profile")
  expect_equal(fit$data$n_items,         3L)
  expect_equal(length(fit$data$ratings), 9L)
  expect_equal(fit$data$degen_ids,       1L)
  expect_equal(fit$fit_data$n_items,         2L)
  expect_equal(length(fit$fit_data$ratings), 6L)
  expect_equal(fit$fit_data$degen_ids,       integer(0))
})

test_that("two-way: degenerate items are passed to C++ (all items present)", {
  skip_if_not_installed("AlgDesign")
  dt <- sim_data(
    J = 20, B = 5, AGREEMENT = 0.6, ALPHA = rep(0, 20),
    DATA_TYPE = "continuous", SEED = 42
  )
  dt$rating[dt$id_item == 1] <- 0.5
  rd_tw <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  expect_true(1L %in% rd_tw$degen_ids)
  expect_no_error(
    fit_tw <- agreement(rd_tw, NUISANCE = c("items", "workers"), METHOD = "profile")
  )
  expect_equal(fit_tw$data$n_items, 20L)
  expect_equal(length(fit_tw$alpha), 20L)
})

test_that("inflated one-way: agreement adjusted upward by degenerate items", {
  skip_if_not_installed("AlgDesign")
  dt <- sim_data(
    J = 20, B = 5, AGREEMENT = 0.6, ALPHA = rep(0, 20),
    DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1
  )
  dt$rating[dt$id_item == 1] <- 0
  rd_inf <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  expect_equal(rd_inf$degen_ids, 1L)
  n_degen <- length(rd_inf$degen_ids)
  fit_inf <- agreement(rd_inf, METHOD = "profile", NUISANCE = "items")
  fit_clean <- agreement(
    rating_data(dt$rating[dt$id_item != 1], dt$id_item[dt$id_item != 1],
                VERBOSE = FALSE),
    METHOD = "profile", NUISANCE = "items"
  )
  expected <- (fit_clean$data$n_items * fit_clean$profile$agreement + n_degen) /
    rd_inf$n_items
  expect_equal(fit_inf$profile$agreement, expected, tolerance = 1e-6)
})

test_that("one-way continuous: ADJUST=TRUE weights dropped items by unit", {
  rd <- rating_data(.degen_ratings, .degen_item_ids, VERBOSE = FALSE)
  fit_adj <- agreement(rd, NUISANCE = "items", METHOD = "profile")
  fit_raw <- agreement(rd, NUISANCE = "items", METHOD = "profile", ADJUST = FALSE)

  fit_J <- fit_adj$fit_data$n_items
  n_dropped <- rd$n_items - fit_J
  expect_equal(n_dropped, 1L)

  expect_equal(fit_raw$profile$agreement, prec2agr(fit_raw$profile$precision))
  expect_equal(
    fit_adj$profile$agreement,
    (fit_J * fit_raw$profile$agreement + n_dropped) / (fit_J + n_dropped),
    tolerance = 1e-8
  )
  expect_true(fit_adj$profile$agreement > fit_raw$profile$agreement)
})

test_that("two-way: ADJUST has no effect (degenerate items kept in fit)", {
  skip_if_not_installed("AlgDesign")
  dt <- sim_data(
    J = 20, B = 5, AGREEMENT = 0.6, ALPHA = rep(0, 20),
    DATA_TYPE = "continuous", SEED = 42
  )
  dt$rating[dt$id_item == 1] <- 0.5
  rd_tw <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  expect_true(1L %in% rd_tw$degen_ids)

  fit_adj <- agreement(rd_tw, NUISANCE = c("items", "workers"), METHOD = "profile")
  fit_raw <- agreement(rd_tw, NUISANCE = c("items", "workers"), METHOD = "profile",
                       ADJUST = FALSE)
  expect_equal(fit_adj$fit_data$n_items, rd_tw$n_items)
  expect_equal(fit_adj$profile$agreement, fit_raw$profile$agreement)
})

test_that("inflated one-way: ADJUST=FALSE returns unadjusted clean-subset mean", {
  skip_if_not_installed("AlgDesign")
  dt <- sim_data(
    J = 20, B = 5, AGREEMENT = 0.6, ALPHA = rep(0, 20),
    DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 1
  )
  dt$rating[dt$id_item == 1] <- 0
  rd_inf <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit_raw <- agreement(rd_inf, METHOD = "profile", NUISANCE = "items",
                       ADJUST = FALSE)
  fit_clean <- agreement(
    rating_data(dt$rating[dt$id_item != 1], dt$id_item[dt$id_item != 1],
                VERBOSE = FALSE),
    METHOD = "profile", NUISANCE = "items"
  )
  expect_equal(fit_raw$profile$agreement, fit_clean$profile$agreement,
               tolerance = 1e-6)
})
