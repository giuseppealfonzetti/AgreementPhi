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
