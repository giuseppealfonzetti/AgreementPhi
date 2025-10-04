test_that("sim_data generates correct ordinal data type", {
  dt <- sim_data(
    J = 5,
    B = 3,
    W = 10,
    AGREEMENT = 0.7,
    ALPHA = rep(0, 5),
    DATA_TYPE = "ordinal",
    K = 6,
    SEED = 123
  )
  expect_true(all(dt$rating == as.integer(dt$rating)))
  expect_true(all(dt$rating >= 1 & dt$rating <= 6))
})

test_that("sim_data generates correct continuous data type", {
  dt <- sim_data(
    J = 5,
    B = 3,
    W = 10,
    AGREEMENT = 0.7,
    ALPHA = rep(0, 5),
    DATA_TYPE = "continuous",
    SEED = 123
  )
  expect_true(all(dt$rating > 0 & dt$rating < 1))
  expect_false(all(dt$rating == as.integer(dt$rating)))
})

test_that("sim_data produces correct dataset dimensions", {
  J <- 10
  B <- 5
  dt <- sim_data(
    J = J,
    B = B,
    W = 20,
    AGREEMENT = 0.5,
    ALPHA = rep(0, J),
    DATA_TYPE = "ordinal",
    K = 5,
    SEED = 123
  )
  expect_equal(nrow(dt), J * B)
  expect_equal(ncol(dt), 3)
  expect_true(all(c("id_item", "id_worker", "rating") %in% names(dt)))
})

test_that("sim_data allocates items correctly", {
  J <- 7
  B <- 4
  dt <- sim_data(
    J = J,
    B = B,
    W = 15,
    AGREEMENT = 0.6,
    ALPHA = rep(0, J),
    DATA_TYPE = "ordinal",
    K = 5,
    SEED = 123
  )
  expect_true(all(table(dt$id_item) == B))
  expect_equal(length(unique(dt$id_item)), J)
  expect_equal(sort(unique(dt$id_item)), 1:J)
})

test_that("sim_data allocates workers correctly (one-way: BETA = NULL)", {
  J <- 5
  B <- 3
  W <- 10
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = 0.7,
    ALPHA = rep(0, J),
    BETA = NULL,
    DATA_TYPE = "ordinal",
    K = 5,
    SEED = 123
  )
  # Each item should have B unique workers
  for (j in 1:J) {
    workers_per_item <- dt$id_worker[dt$id_item == j]
    expect_equal(length(workers_per_item), B)
    expect_equal(length(unique(workers_per_item)), B)
  }
  expect_true(max(dt$id_worker) <= W)
})

test_that("sim_data allocates workers correctly (two-way: BETA specified)", {
  J <- 5
  B <- 3
  W <- 10
  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = 0.7,
    ALPHA = rep(0, J),
    BETA = rnorm(W, 0, 0.5),
    DATA_TYPE = "ordinal",
    K = 5,
    SEED = 123
  )
  # Each item should have B unique workers
  for (j in 1:J) {
    workers_per_item <- dt$id_worker[dt$id_item == j]
    expect_equal(length(workers_per_item), B)
    expect_equal(length(unique(workers_per_item)), B)
  }
  expect_true(max(dt$id_worker) <= W)
})

test_that("sim_data ALPHA affects mean ratings correctly", {
  J <- 2
  B <- 50
  dt_low <- sim_data(
    J = J,
    B = B,
    W = 100,
    AGREEMENT = 0.8,
    ALPHA = c(-2, -2),
    DATA_TYPE = "continuous",
    SEED = 123
  )
  dt_high <- sim_data(
    J = J,
    B = B,
    W = 100,
    AGREEMENT = 0.8,
    ALPHA = c(2, 2),
    DATA_TYPE = "continuous",
    SEED = 123
  )
  expect_true(mean(dt_high$rating) > mean(dt_low$rating))
})

test_that("sim_data AGREEMENT affects variance correctly", {
  J <- 5
  B <- 50
  dt_low_agr <- sim_data(
    J = J,
    B = B,
    W = 100,
    AGREEMENT = 0.3,
    ALPHA = rep(0, J),
    DATA_TYPE = "continuous",
    SEED = 123
  )
  dt_high_agr <- sim_data(
    J = J,
    B = B,
    W = 100,
    AGREEMENT = 0.9,
    ALPHA = rep(0, J),
    DATA_TYPE = "continuous",
    SEED = 456
  )
  # Within each item, higher agreement should have lower variance
  var_low <- sapply(1:J, function(j) {
    var(dt_low_agr$rating[dt_low_agr$id_item == j])
  })
  var_high <- sapply(1:J, function(j) {
    var(dt_high_agr$rating[dt_high_agr$id_item == j])
  })
  expect_true(mean(var_low) > mean(var_high))
})

test_that("sim_data is reproducible with same SEED", {
  dt1 <- sim_data(
    J = 5,
    B = 3,
    W = 10,
    AGREEMENT = 0.7,
    ALPHA = rep(0, 5),
    DATA_TYPE = "ordinal",
    K = 5,
    SEED = 999
  )
  dt2 <- sim_data(
    J = 5,
    B = 3,
    W = 10,
    AGREEMENT = 0.7,
    ALPHA = rep(0, 5),
    DATA_TYPE = "ordinal",
    K = 5,
    SEED = 999
  )
  expect_equal(dt1$rating, dt2$rating)
  expect_equal(dt1$id_item, dt2$id_item)
  expect_equal(dt1$id_worker, dt2$id_worker)
})

test_that("sim_data respects K parameter for ordinal data", {
  K <- 4
  dt <- sim_data(
    J = 5,
    B = 10,
    W = 20,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 5),
    DATA_TYPE = "ordinal",
    K = K,
    SEED = 123
  )
  expect_true(all(dt$rating >= 1 & dt$rating <= K))
  # Most categories should be present with enough data
  expect_true(length(setdiff(1:K, unique(dt$rating))) < K)
})

test_that("sim_data BETA affects two-way model correctly", {
  J <- 3
  B <- 5
  W <- 10
  # Worker with large positive effect
  beta_vec <- c(2, rep(0, W - 1))

  dt <- sim_data(
    J = J,
    B = B,
    W = W,
    AGREEMENT = 0.7,
    ALPHA = rep(0, J),
    BETA = beta_vec,
    DATA_TYPE = "continuous",
    SEED = 123
  )

  # Ratings from worker 1 should be higher on average
  ratings_worker1 <- dt$rating[dt$id_worker == 1]
  ratings_others <- dt$rating[dt$id_worker != 1]
  expect_true(mean(ratings_worker1) > mean(ratings_others))
})
