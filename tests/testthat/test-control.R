test_that("validate_cpp_control returns a valid list", {
  ctrl <- validate_cpp_control(list())
  expect_type(ctrl, "list")
  expect_true(all(
    c(
      "SEARCH_RANGE",
      "MAX_ITER",
      "PROF_SEARCH_RANGE",
      "PROF_MAX_ITER",
      "ALT_MAX_ITER",
      "ALT_TOL",
      "BOUNDARY"
    ) %in%
      names(ctrl)
  ))
})
