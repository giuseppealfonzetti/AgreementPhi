test_that("plot.agreement_fit runs without error for modified method", {
  dt <- sim_data(
    J = 30,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 30),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  pdf(NULL)
  on.exit(dev.off(), add = TRUE)
  expect_no_error(plot(fit))
})

test_that("plot.agreement_fit accepts precomputed RANGE_LL", {
  dt <- sim_data(
    J = 30,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 30),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  rl <- get_range_ll(fit)
  pdf(NULL)
  on.exit(dev.off(), add = TRUE)
  expect_no_error(plot(fit, RANGE_LL = rl))
})

test_that("plot.agreement_fit errors for inflated data", {
  dt <- sim_data(
    J = 20,
    B = 8,
    AGREEMENT = 0.5,
    ALPHA = rep(0, 20),
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified")
  expect_error(plot(fit))
})
