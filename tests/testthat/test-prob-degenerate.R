test_that("prob_degenerate: continuous fit returns all zeros", {
  dt <- sim_data(
    J = 10,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 10),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  pd <- prob_degenerate(fit)

  expect_true(is.numeric(pd))
  expect_equal(length(pd), fit$data$n_items)
  expect_true(all(pd == 0))
  expect_true(all(grepl("^item_", names(pd))))
})

test_that("prob_degenerate: ordinal clean fit returns values in (0,1)", {
  set.seed(42)
  J <- 15
  K <- 5L
  n_per_item <- 6L
  ratings <- sample(seq_len(K), J * n_per_item, replace = TRUE)
  item_ids <- rep(seq_len(J), each = n_per_item)
  # Force no degenerate items: ensure each item has at least 2 distinct values
  for (j in seq_len(J)) {
    idx <- which(item_ids == j)
    if (length(unique(ratings[idx])) < 2L) {
      ratings[idx[1]] <- (ratings[idx[1]] %% K) + 1L
    }
  }
  rd <- rating_data(ratings, item_ids, K = K, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  expect_equal(length(fit$data$degen_ids), 0L)

  pd <- prob_degenerate(fit)
  expect_equal(length(pd), J)
  expect_true(all(pd > 0 & pd < 1))
  expect_true(all(grepl("^item_", names(pd))))
})

test_that("prob_degenerate: ordinal boundary-degenerate item gets probability 1", {
  K <- 3L
  ratings <- c(
    1L,
    3L,
    2L,
    1L,
    3L, # item 2: non-degenerate
    2L,
    1L,
    3L,
    2L,
    1L, # item 3: non-degenerate
    1L,
    1L,
    1L,
    1L,
    1L # item 1: all-min degenerate
  )
  item_ids <- c(rep(2L, 5), rep(3L, 5), rep(1L, 5))
  rd <- rating_data(ratings, item_ids, K = K, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")

  expect_equal(length(fit$data$degen_ids), 1L)
  pd <- prob_degenerate(fit)
  expect_equal(length(pd), 3L)
  expect_equal(unname(pd["item_1"]), 1)
  expect_true(pd["item_2"] > 0 && pd["item_2"] < 1)
  expect_true(pd["item_3"] > 0 && pd["item_3"] < 1)
})

test_that("prob_degenerate: inflated items with extreme alpha have higher P(degen)", {
  set.seed(7)
  J <- 30
  # Extreme alpha (near K0 or K1) gives high p0 or p1, so high P(degen).
  # Alpha near 0 is interior, so low P(degen).
  alpha_extreme <- c(rep(-5, 5), rep(0, J - 10), rep(5, 5))
  dt <- sim_data(
    J = J,
    B = 8,
    AGREEMENT = 0.5,
    ALPHA = alpha_extreme,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 7
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")

  pd <- prob_degenerate(fit)
  expect_equal(length(pd), fit$data$n_items)
  expect_true(all(pd >= 0 & pd <= 1))

  # Extreme alpha (high p0 or p1) should give higher P(degen) than centre items.
  # Compare only non-degenerate items.
  non_degen <- setdiff(seq_len(fit$data$n_items), fit$data$degen_ids)
  extreme_nd <- intersect(non_degen, c(1:5, (J - 4):J))
  centre_nd <- intersect(non_degen, 11:(J - 10))

  if (length(extreme_nd) > 0 && length(centre_nd) > 0) {
    expect_true(mean(pd[extreme_nd]) > mean(pd[centre_nd]))
  }
})

test_that("prob_degenerate: two-way model raises an error", {
  dt <- sim_data(
    J = 10,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 10),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = c("items", "workers"))
  expect_error(prob_degenerate(fit), "two-way")
})

# confint_prob_degenerate -------------------------------------------------------

test_that("confint_prob_degenerate: continuous -> SE = 0, CI = [0, 0]", {
  dt <- sim_data(
    J = 10,
    B = 5,
    AGREEMENT = 0.6,
    ALPHA = rep(0, 10),
    DATA_TYPE = "continuous",
    SEED = 1
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  ci <- confint_prob_degenerate(fit)

  expect_true(is.matrix(ci))
  expect_equal(nrow(ci), fit$data$n_items)
  expect_equal(ncol(ci), 4L)
  expect_true(all(ci[, "Std. Error"] == 0))
  expect_true(all(ci[, 3L] == 0 & ci[, 4L] == 0))
})

test_that("confint_prob_degenerate: ordinal clean -> valid matrix", {
  set.seed(42)
  J <- 12
  K <- 4L
  n_per <- 6L
  ratings <- sample(seq_len(K), J * n_per, replace = TRUE)
  item_ids <- rep(seq_len(J), each = n_per)
  for (j in seq_len(J)) {
    idx <- which(item_ids == j)
    if (length(unique(ratings[idx])) < 2L) {
      ratings[idx[1L]] <- (ratings[idx[1L]] %% K) + 1L
    }
  }
  rd <- rating_data(ratings, item_ids, K = K, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  ci <- confint_prob_degenerate(fit)

  expect_equal(nrow(ci), J)
  expect_equal(ncol(ci), 4L)
  expect_equal(colnames(ci), c("Estimate", "Std. Error", "2.5%", "97.5%"))
  expect_true(all(ci[, "Std. Error"] > 0))
  expect_true(all(ci[, 3L] > 0 & ci[, 4L] < 1)) # logit: strictly interior
  expect_true(all(ci[, 3L] <= ci[, "Estimate"] & ci[, "Estimate"] <= ci[, 4L]))
})

test_that("confint_prob_degenerate: ordinal degenerate item -> SE = 0, CI = [1, 1]", {
  K <- 3L
  ratings <- c(
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L, # item 2
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L, # item 3
    1L,
    1L,
    1L,
    1L,
    1L,
    1L,
    1L,
    1L,
    1L,
    1L,
    1L
  ) # item 1: degen
  item_ids <- c(rep(2L, 11), rep(3L, 11), rep(1L, 11))
  rd <- rating_data(ratings, item_ids, K = K, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  ci <- confint_prob_degenerate(fit)

  expect_equal(unname(ci["item_1", "Std. Error"]), 0)
  expect_equal(unname(ci["item_1", 3L]), 1)
  expect_equal(unname(ci["item_1", 4L]), 1)
  expect_true(ci["item_2", "Std. Error"] > 0)
  expect_true(ci["item_3", "Std. Error"] > 0)
})

test_that("confint_prob_degenerate: inflated -> valid matrix with positive SEs", {
  set.seed(3)
  dt <- sim_data(
    J = 20,
    B = 8,
    AGREEMENT = 0.5,
    ALPHA = rep(0, 20),
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 3
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")

  ci <- confint_prob_degenerate(fit)
  non_degen <- setdiff(seq_len(fit$data$n_items), fit$data$degen_ids)

  expect_equal(nrow(ci), fit$data$n_items)
  expect_true(all(ci[non_degen, "Std. Error"] > 0))
  expect_true(all(ci[non_degen, 3L] > 0 & ci[non_degen, 4L] < 1)) # logit: strictly interior
})

test_that("confint_prob_degenerate: level parameter controls interval width", {
  set.seed(7)
  J <- 15
  K <- 3L
  n_per <- 8L
  ratings <- sample(seq_len(K), J * n_per, replace = TRUE)
  item_ids <- rep(seq_len(J), each = n_per)
  for (j in seq_len(J)) {
    idx <- which(item_ids == j)
    if (length(unique(ratings[idx])) < 2L) {
      ratings[idx[1L]] <- (ratings[idx[1L]] %% K) + 1L
    }
  }
  rd <- rating_data(ratings, item_ids, K = K, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")

  ci90 <- confint_prob_degenerate(fit, LEVEL = 0.90)
  ci99 <- confint_prob_degenerate(fit, LEVEL = 0.99)
  width90 <- ci90[, 4L] - ci90[, 3L]
  width99 <- ci99[, 4L] - ci99[, 3L]
  expect_true(all(width90 <= width99))
})

test_that("confint_prob_degenerate: logit CI lower bound is strictly > 0 for all non-zero estimates", {
  K <- 3L
  ratings <- c(
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L,
    3L,
    2L,
    1L
  )
  item_ids <- c(rep(2L, 11), rep(3L, 11))
  rd <- rating_data(ratings, item_ids, K = K, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  ci <- confint_prob_degenerate(fit)
  expect_true(all(ci[, 3L] > 0))
  expect_true(all(ci[, 4L] < 1))
})

test_that("confint_prob_degenerate: inflated SE matches full-joint delta method", {
  skip_if_not_installed("numDeriv")
  J <- 15
  alpha <- c(rep(-3, 5), rep(0, 5), rep(3, 5)) # extremes give sizeable P(degen)
  dt <- sim_data(
    J = J,
    B = 6,
    AGREEMENT = 0.6,
    ALPHA = alpha,
    DATA_TYPE = "inflated",
    K0 = -2,
    K1 = 2,
    SEED = 11
  )
  rd <- rating_data(dt$rating, dt$id_item, VERBOSE = FALSE)
  fit <- agreement(rd, METHOD = "profile", NUISANCE = "items")
  ci <- confint_prob_degenerate(fit)

  # Reference SE: invert the full joint information, then delta method per item.
  y <- fit$fit_data$ratings
  ii <- as.integer(fit$fit_data$item_ids)
  Jf <- fit$fit_data$n_items
  Bf <- as.integer(table(fit$fit_data$item_ids))

  joint_ll <- function(par) {
    al <- par[1:Jf]
    k0 <- par[Jf + 1]
    k1 <- par[Jf + 2]
    phi <- par[Jf + 3]
    a_o <- al[ii]
    L0 <- plogis(a_o - k0)
    L1 <- plogis(a_o - k1)
    mu <- plogis(a_o)
    out <- numeric(length(y))
    i0 <- y == 0
    i1 <- y == 1
    ic <- !i0 & !i1
    out[i0] <- log1p(-L0[i0])
    out[i1] <- log(L1[i1])
    a <- mu[ic] * phi
    b <- (1 - mu[ic]) * phi
    out[ic] <- log(L0[ic] - L1[ic]) +
      lgamma(phi) -
      lgamma(a) -
      lgamma(b) +
      (a - 1) * log(y[ic]) +
      (b - 1) * log(1 - y[ic])
    sum(out)
  }
  mle <- c(fit$alpha, fit$k0, fit$k1, fit$profile$precision)
  Vth <- solve(-numDeriv::hessian(joint_ll, mle))
  pd_of <- function(a, k0, k1, m) (1 - plogis(a - k0))^m + plogis(a - k1)^m

  non_degen <- setdiff(seq_len(fit$data$n_items), fit$data$degen_ids)
  se_bf <- vapply(
    seq_along(non_degen),
    function(i) {
      idx <- c(i, Jf + 1, Jf + 2)
      g <- numDeriv::grad(
        function(x) pd_of(x[1], x[2], x[3], Bf[i]),
        c(fit$alpha[i], fit$k0, fit$k1)
      )
      sqrt(drop(g %*% Vth[idx, idx] %*% g))
    },
    numeric(1)
  )

  big <- se_bf > 1e-4
  expect_true(any(big)) # some items have non-trivial SE
  expect_equal(
    unname(ci[non_degen[big], "Std. Error"]),
    se_bf[big],
    tolerance = 1e-3
  )
})
