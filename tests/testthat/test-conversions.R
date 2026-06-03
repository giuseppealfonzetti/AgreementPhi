test_that("prec2agr / agr2prec", {
  set.seed(123)
  a <- runif(0, 1)
  expect_equal(a, prec2agr(agr2prec(a)))

  expect_error(agr2prec(-1))
  expect_error(agr2prec(1.1))
  expect_equal(prec2agr(0), 0)
  expect_error(prec2agr(-1))
})

test_that("par2agr without K0/K1 equals prec2agr", {
  phi <- 3.5
  expect_equal(par2agr(phi)$agreement, prec2agr(phi))
})

test_that("par2agr result is in [0,1]", {
  alpha <- runif(10, -2, 2)
  result <- par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 2)
  expect_true(result$agreement >= 0 && result$agreement <= 1)
  expect_true(all(
    result$agreement_by_item >= 0 & result$agreement_by_item <= 1
  ))
})

test_that("par2agr is increasing in PHI", {
  alpha <- rep(0, 10)
  agr_low <- par2agr(PHI = 1, ALPHA = alpha, K0 = -2, K1 = 2)$agreement
  agr_high <- par2agr(PHI = 10, ALPHA = alpha, K0 = -2, K1 = 2)$agreement
  expect_true(agr_high > agr_low)
})

test_that("par2agr NA and Inf sentinels give same result as boundary values", {
  alpha <- rep(0, 5)
  expect_equal(
    par2agr(PHI = 2, ALPHA = alpha, K0 = NA, K1 = 2)$agreement,
    par2agr(PHI = 2, ALPHA = alpha, K0 = -100, K1 = 2)$agreement
  )
  expect_equal(
    par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = NA)$agreement,
    par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 100)$agreement
  )
})

test_that("par2agr ADJUST=TRUE with N_DEGENERATE=0 is a no-op", {
  alpha <- c(-1, 0, 1)
  r1 <- par2agr(PHI = 3, ALPHA = alpha, K0 = -2, K1 = 2)$agreement
  r2 <- par2agr(PHI = 3, ALPHA = alpha, K0 = -2, K1 = 2,
                ADJUST = TRUE, N_DEGENERATE = 0L)$agreement
  expect_equal(r1, r2)
})

test_that("par2agr ADJUST=TRUE applies formula correctly (inflated)", {
  alpha <- c(-1, 0, 1)
  n_degen <- 2L
  raw  <- par2agr(PHI = 3, ALPHA = alpha, K0 = -2, K1 = 2)
  adj  <- par2agr(PHI = 3, ALPHA = alpha, K0 = -2, K1 = 2,
                  ADJUST = TRUE, N_DEGENERATE = n_degen)
  fit_J <- length(alpha)
  expected <- (fit_J * mean(raw$agreement_by_item) + n_degen) / (fit_J + n_degen)
  expect_equal(adj$agreement, expected)
})

test_that("par2agr ADJUST=TRUE applies formula correctly (no K0/K1)", {
  alpha <- c(-1, 0, 1)
  n_degen <- 2L
  raw <- par2agr(PHI = 3, ALPHA = alpha)
  adj <- par2agr(PHI = 3, ALPHA = alpha, ADJUST = TRUE, N_DEGENERATE = n_degen)
  fit_J <- length(alpha)
  expected <- (fit_J * raw$agreement + n_degen) / (fit_J + n_degen)
  expect_equal(adj$agreement, expected)
})

test_that("par2agr ADJUST=TRUE moves agreement toward 1 when agreement < 1", {
  alpha <- c(-1, 0, 1)
  raw <- par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 2)$agreement
  adj <- par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 2,
                 ADJUST = TRUE, N_DEGENERATE = 3L)$agreement
  expect_true(adj >= raw)
  expect_true(adj <= 1)
})

test_that("par2agr ADJUST=TRUE extreme: N_DEGENERATE equals fit_J", {
  alpha <- c(-1, 0, 1)
  fit_J <- length(alpha)
  raw <- par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 2)
  adj <- par2agr(PHI = 2, ALPHA = alpha, K0 = -2, K1 = 2,
                 ADJUST = TRUE, N_DEGENERATE = fit_J)
  expected <- (fit_J * mean(raw$agreement_by_item) + fit_J) / (2 * fit_J)
  expect_equal(adj$agreement, expected)
})
