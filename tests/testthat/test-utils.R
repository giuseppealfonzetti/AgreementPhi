test_that("prec2agr / agr2prec", {
  set.seed(123)
  a <- runif(0, 1)
  expect_equal(a, prec2agr(agr2prec(a)))

  expect_error(agr2prec(-1))
  expect_error(agr2prec(1.1))
  expect_error(prec2agr(0))
  expect_error(prec2agr(-1))
})
