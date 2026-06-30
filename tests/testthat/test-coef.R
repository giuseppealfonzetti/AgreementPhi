test_that("coef.agreement_fit one-way continuous: returns named numeric vector", {
  dt <- sim_data(
    J = 20,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 20),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  cf <- coef(fit)

  expect_true(is.numeric(cf) && !is.list(cf))
  expect_equal(names(cf)[1], "phi")
  expect_true(all(grepl("^alpha_", names(cf)[-1])))
  expect_false(any(grepl("^beta_", names(cf))))
  expect_false(any(c("k0", "k1") %in% names(cf)))
  expect_equal(unname(cf["phi"]), unname(fit$modified$precision))
  expect_equal(length(cf), 1 + fit$data$n_items)
})

test_that("coef.agreement_fit two-way continuous: includes beta", {
  dt <- sim_data(
    J = 20,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 20),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items", "workers"))
  cf <- coef(fit)

  expect_true(any(grepl("^beta_", names(cf))))
  expect_equal(length(cf), 1 + fit$data$n_items + fit$data$n_workers)
})

test_that("coef.agreement_fit profile method: phi is profile precision", {
  dt <- sim_data(
    J = 20,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 20),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  expect_equal(unname(coef(fit)["phi"]), unname(fit$profile$precision))
})

test_that("coef.agreement_fit inflated: includes k0 and k1, no beta", {
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
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  cf <- coef(fit)

  expect_equal(names(cf)[1], "phi")
  expect_true("k0" %in% names(cf))
  expect_true("k1" %in% names(cf))
  expect_true(all(grepl(
    "^alpha_",
    names(cf)[!names(cf) %in% c("phi", "k0", "k1")]
  )))
  expect_false(any(grepl("^beta_", names(cf))))
  expect_equal(unname(cf["phi"]), unname(fit$modified$precision))
})

test_that("coef uses item_labels as alpha names when available", {
  dt <- sim_data(
    J = 5,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 5),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  item_labels <- paste0("item_", dt$id_item)
  rd <- rating_data(
    dt$rating,
    dt$id_item,
    dt$id_worker,
    ITEM_LABELS = item_labels,
    VERBOSE = FALSE
  )
  fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
  cf <- coef(fit)
  expect_equal(names(cf)[-1], paste0("alpha_", rd$item_labels))
  expect_true(all(grepl("^alpha_", names(cf)[-1])))
})

test_that("coef uses worker_labels as beta names in two-way model", {
  dt <- sim_data(
    J = 5,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 5),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  item_labels <- paste0("item_", dt$id_item)
  worker_labels <- paste0("worker_", dt$id_worker)
  rd <- rating_data(
    dt$rating,
    dt$id_item,
    dt$id_worker,
    ITEM_LABELS = item_labels,
    WORKER_LABELS = worker_labels,
    VERBOSE = FALSE
  )
  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items", "workers"))
  cf <- coef(fit)
  expect_true(all(paste0("beta_", rd$worker_labels) %in% names(cf)))
  expect_true(all(grepl("^beta_", names(cf)[grepl("^beta_", names(cf))])))
})

test_that("coef falls back to alpha_i / beta_i when labels absent", {
  dt <- sim_data(
    J = 5,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 5),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items", "workers"))
  cf <- coef(fit)
  expect_true(all(grepl("^alpha_", names(cf)[grepl("^alpha_", names(cf))])))
  expect_true(all(grepl("^beta_", names(cf)[grepl("^beta_", names(cf))])))
})

# Ordinal degenerate item back-fill ------------------------------------------

.ord_degen_fit <- local({
  # 3 items, K=3, B=5 raters each; items 2-3 are genuinely non-degenerate
  base_ratings  <- c(
    1L, 3L, 2L, 1L, 3L,   # item 2: varied
    2L, 1L, 3L, 2L, 1L,   # item 3: varied
    NA, NA, NA, NA, NA     # item 1: filled below per test
  )
  item_ids <- c(rep(2L, 5), rep(3L, 5), rep(1L, 5))

  list(
    make = function(degen_val) {
      r <- base_ratings
      r[item_ids == 1L] <- degen_val
      rd <- rating_data(r, item_ids, VERBOSE = FALSE)
      agreement(rd, METHOD = "profile", NUISANCE = "items")
    }
  )
})

test_that("coef ordinal one-way: all-min degenerate item gets alpha = -Inf", {
  fit <- .ord_degen_fit$make(1L)
  expect_equal(length(fit$data$degen_ids), 1L)
  cf <- coef(fit)
  expect_true(is.infinite(cf["alpha_1"]) && cf["alpha_1"] < 0)
})

test_that("coef ordinal one-way: all-max degenerate item gets alpha = +Inf", {
  fit <- .ord_degen_fit$make(3L)
  expect_equal(length(fit$data$degen_ids), 1L)
  cf <- coef(fit)
  expect_true(is.infinite(cf["alpha_1"]) && cf["alpha_1"] > 0)
})

test_that("coef ordinal one-way: interior degenerate item gets finite MLE alpha", {
  K <- 4L
  # Concentrated anchor items give high phi and a meaningful interior optimum
  ratings  <- c(
    2L, 2L, 2L, 2L, 3L, 2L,   # item 2: mostly category 2
    3L, 3L, 2L, 3L, 3L, 3L,   # item 3: mostly category 3
    2L, 2L, 2L, 2L, 2L, 2L    # item 1: all = 2 (interior for K=4)
  )
  item_ids <- c(rep(2L, 6), rep(3L, 6), rep(1L, 6))
  rd  <- rating_data(ratings, item_ids, K = K, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  expect_equal(length(fit$data$degen_ids), 1L)
  cf <- coef(fit)

  alpha_hat <- unname(cf["alpha_1"])
  expect_true(is.finite(alpha_hat))

  # Verify alpha_hat is a local minimum of the neg-log-likelihood for category 2
  phi <- unname(fit$profile$precision)
  tau <- fit$tau
  neg_ll <- function(a) {
    mu <- plogis(a)
    lp <- log(pbeta(tau[3L], mu * phi, (1 - mu) * phi) -
               pbeta(tau[2L], mu * phi, (1 - mu) * phi))
    if (!is.finite(lp)) 1e30 else -lp
  }
  expect_true(neg_ll(alpha_hat) < neg_ll(alpha_hat + 1))
  expect_true(neg_ll(alpha_hat) < neg_ll(alpha_hat - 1))
})
