test_that("detect_data_type identifies ordinal data correctly", {
  ratings <- c(1, 2, 3, 4, 5, 6)
  expect_equal(detect_data_type(ratings), "ordinal")
})

test_that("detect_data_type identifies continuous data correctly", {
  ratings <- c(0.1, 0.5, 0.7, 0.9)
  expect_equal(detect_data_type(ratings), "continuous")
})

test_that("detect_data_type rejects ordinal data not starting at 1", {
  ratings <- c(0, 1, 2, 3)
  expect_error(detect_data_type(ratings), "Lowest category different from 1")
})

test_that("detect_data_type warns about missing ordinal categories", {
  ratings <- c(1, 1, 3, 3, 5, 5) # Missing 2 and 4
  expect_warning(detect_data_type(ratings), "Some category is missing")
})

test_that("detect_data_type rejects continuous data with values <= 0", {
  ratings <- c(0, 0.5, 0.7)
  expect_error(
    detect_data_type(ratings),
    "Minimum value lower or equal than zero"
  )
})


test_that("detect_data_type rejects continuous data with values >= 1", {
  ratings <- c(0.5, 0.7, 1.0)
  expect_error(
    detect_data_type(ratings),
    "Maximum value higher or equal than one"
  )
})

# Tests for validate_data
test_that("validate_data returns correct structure for valid data", {
  ratings <- c(1, 2, 3, 4, 5, 6, 1, 2, 3)
  item_inds <- c(1, 1, 1, 2, 2, 2, 3, 3, 3)

  result <- validate_data(ratings, item_inds, VERBOSE = FALSE)

  expect_true(is.list(result))
  expect_true(all(
    c(
      "item_ids",
      "ratings",
      "n_items",
      "data_type",
      "K",
      "ave_ratings_per_item"
    ) %in%
      names(result)
  ))
})

test_that("validate_data detects ordinal data correctly", {
  ratings <- c(1, 2, 3, 4, 5, 6)
  item_inds <- c(1, 1, 2, 2, 3, 3)

  result <- validate_data(ratings, item_inds, VERBOSE = FALSE)

  expect_equal(result$data_type, "ordinal")
  expect_equal(result$K, 6)
})

test_that("validate_data detects continuous data correctly", {
  ratings <- c(0.1, 0.2, 0.5, 0.7, 0.8, 0.9)
  item_inds <- c(1, 1, 2, 2, 3, 3)

  result <- validate_data(ratings, item_inds, VERBOSE = FALSE)

  expect_equal(result$data_type, "continuous")
  expect_equal(result$K, 1)
})

test_that("validate_data removes degenerate items", {
  ratings <- c(1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 3, 4)
  item_inds <- c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4) # item 3 is degenerate

  result <- validate_data(ratings, item_inds, VERBOSE = FALSE)

  expect_equal(result$n_items, 3) # Only item 2 remains
  expect_equal(length(result$ratings), 9)
  expect_true(all(result$item_ids == c(1, 1, 1, 2, 2, 2, 3, 3, 3))) # Recoded
})


test_that("validate_data computes average ratings per item correctly", {
  ratings <- c(1, 2, 3, 4, 5, 6, 1, 2)
  item_inds <- c(1, 1, 1, 2, 2, 2, 3, 3) # 3, 3, 2 ratings per item

  result <- validate_data(ratings, item_inds, VERBOSE = FALSE)

  expect_equal(result$ave_ratings_per_item, mean(c(3, 3, 2)))
})

test_that("validate_data handles all non-degenerate items", {
  ratings <- c(1, 2, 3, 4, 5, 6)
  item_inds <- c(1, 1, 2, 2, 3, 3)

  result <- validate_data(ratings, item_inds, VERBOSE = FALSE)

  expect_equal(result$n_items, 3)
  expect_equal(length(result$ratings), 6)
})

test_that("validate_data requires numeric ratings", {
  ratings <- c("1", "2", "3")
  item_inds <- c(1, 1, 2)

  expect_error(validate_data(ratings, item_inds, VERBOSE = FALSE))
})

test_that("validate_data requires numeric item indices", {
  ratings <- c(1, 2, 3)
  item_inds <- c("a", "a", "b")

  expect_error(validate_data(ratings, item_inds, VERBOSE = FALSE))
})

test_that("validate_data requires equal length inputs", {
  ratings <- c(1, 2, 3, 4)
  item_inds <- c(1, 1, 2)

  expect_error(validate_data(ratings, item_inds, VERBOSE = FALSE))
})

test_that("validate_data VERBOSE produces messages", {
  ratings <- c(1, 2, 3, 4, 5, 6)
  item_inds <- c(1, 1, 2, 2, 3, 3)

  expect_message(validate_data(ratings, item_inds, VERBOSE = TRUE), "Detected")
  expect_message(
    validate_data(ratings, item_inds, VERBOSE = TRUE),
    "non-degenerate items"
  )
})
