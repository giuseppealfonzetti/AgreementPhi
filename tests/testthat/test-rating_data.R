vd <- function(r, inds) rating_data(r, inds, VERBOSE = FALSE)

test_that("rating_data identifies ordinal data correctly", {
  expect_equal(
    vd(c(1, 2, 3, 1, 2, 3), c(1, 1, 2, 2, 3, 3))$data_type,
    "ordinal"
  )
})

test_that("rating_data identifies continuous data correctly", {
  expect_equal(vd(c(0.1, 0.5, 0.5, 0.9), c(1, 1, 2, 2))$data_type, "continuous")
})

test_that("rating_data rejects ordinal data not starting at 1", {
  expect_error(
    vd(c(0, 1, 2, 0, 1, 2), c(1, 1, 2, 2, 3, 3)),
    "Lowest category different from 1"
  )
})

test_that("rating_data warns about missing ordinal categories", {
  expect_warning(
    vd(c(1, 3, 1, 3, 1, 5), c(1, 1, 2, 2, 3, 3)),
    "Some category is missing"
  )
})

test_that("rating_data classifies data with zeros as inflated", {
  expect_equal(vd(c(0, 0.5, 0.5, 0.7), c(1, 1, 2, 2))$data_type, "inflated")
})

test_that("rating_data classifies data with ones as inflated", {
  expect_equal(vd(c(0.5, 0.7, 0.7, 1.0), c(1, 1, 2, 2))$data_type, "inflated")
})

test_that("rating_data returns a rating_data object with expected fields", {
  result <- rating_data(
    c(1, 2, 3, 1, 2, 3, 1, 2, 3),
    c(1, 1, 1, 2, 2, 2, 3, 3, 3),
    VERBOSE = FALSE
  )
  expect_s3_class(result, "rating_data")
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

test_that("rating_data detects ordinal data correctly", {
  ratings <- c(1, 2, 3, 4, 5, 6)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  result <- rating_data(ratings, item_inds, VERBOSE = FALSE)
  expect_equal(result$data_type, "ordinal")
  expect_equal(result$K, 6)
})

test_that("rating_data detects continuous data correctly", {
  ratings <- c(0.1, 0.2, 0.5, 0.7, 0.8, 0.9)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  result <- rating_data(ratings, item_inds, VERBOSE = FALSE)
  expect_equal(result$data_type, "continuous")
  expect_equal(result$K, 1)
})

test_that("rating_data detects degenerate items without removing them", {
  ratings <- c(1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 3, 4)
  item_inds <- c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)
  result <- rating_data(ratings, item_inds, VERBOSE = FALSE)
  expect_equal(result$n_items, 4)
  expect_equal(length(result$ratings), 12)
  expect_equal(result$degen_ids, 3L)
})

test_that("rating_data detects all-zero inflated items without removing them", {
  ratings <- c(0, 0, 0, 0.3, 0.7, 0, 0, 0)
  item_inds <- c(1, 1, 1, 2, 2, 3, 3, 3)
  result <- rating_data(ratings, item_inds, VERBOSE = FALSE)
  expect_equal(result$n_items, 3)
  expect_equal(length(result$ratings), 8)
  expect_equal(sort(result$degen_ids), c(1L, 3L))
})

test_that("rating_data computes average ratings per item correctly", {
  ratings <- c(1, 2, 3, 4, 5, 6, 1, 2)
  item_inds <- c(1, 1, 1, 2, 2, 2, 3, 3)
  result <- rating_data(ratings, item_inds, VERBOSE = FALSE)
  expect_equal(result$ave_ratings_per_item, mean(c(3, 3, 2)))
})

test_that("rating_data handles all non-degenerate items", {
  ratings <- c(1, 2, 3, 4, 5, 6)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  result <- rating_data(ratings, item_inds, VERBOSE = FALSE)
  expect_equal(result$n_items, 3)
  expect_equal(length(result$ratings), 6)
})

test_that("rating_data requires numeric ratings", {
  ratings <- c("1", "2", "3")
  item_inds <- c(1, 1, 2)
  expect_error(rating_data(ratings, item_inds, VERBOSE = FALSE))
})

test_that("rating_data requires numeric item indices", {
  ratings <- c(1, 2, 3)
  item_inds <- c("a", "a", "b")
  expect_error(rating_data(ratings, item_inds, VERBOSE = FALSE))
})

test_that("rating_data requires equal length inputs", {
  ratings <- c(1, 2, 3, 4)
  item_inds <- c(1, 1, 2)
  expect_error(rating_data(ratings, item_inds, VERBOSE = FALSE))
})

test_that("rating_data VERBOSE produces messages", {
  ratings <- c(1, 2, 3, 4, 5, 6)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  expect_message(rating_data(ratings, item_inds, VERBOSE = TRUE), "Detected")
  expect_message(rating_data(ratings, item_inds, VERBOSE = TRUE), "3 items")
})

test_that("rating_data requires numeric worker indices", {
  ratings <- c(1, 2, 2)
  item_inds <- c(1, 1, 2)
  worker_inds <- c("1", "2", "3")
  expect_error(rating_data(ratings, item_inds, worker_inds, VERBOSE = FALSE))
})

test_that("rating_data accepts explicit K when min(ratings) != 1", {
  ratings <- c(2, 3, 4, 5, 2, 3)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  result <- rating_data(ratings, item_inds, K = 6, VERBOSE = FALSE)
  expect_equal(result$K, 6L)
  expect_equal(result$data_type, "ordinal")
})

test_that("rating_data accepts explicit K when max(ratings) < K", {
  ratings <- c(1, 2, 3, 1, 2, 3)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  result <- rating_data(ratings, item_inds, K = 10, VERBOSE = FALSE)
  expect_equal(result$K, 10L)
})

test_that("rating_data errors when a rating exceeds explicit K", {
  ratings <- c(1, 2, 7, 6)
  item_inds <- c(1, 1, 2, 2)
  expect_error(
    rating_data(ratings, item_inds, K = 6, VERBOSE = FALSE),
    "All ratings must be in"
  )
})

test_that("rating_data errors when ratings are non-integer with explicit K", {
  ratings <- c(1.5, 2.5, 3.5)
  item_inds <- c(1, 1, 2)
  expect_error(
    rating_data(ratings, item_inds, K = 6, VERBOSE = FALSE),
    "not integers"
  )
})

test_that("rating_data verbose message contains user-specified when K provided", {
  ratings <- c(2, 3, 4, 2, 3, 4)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  expect_message(
    rating_data(ratings, item_inds, K = 6, VERBOSE = TRUE),
    "user-specified"
  )
})

test_that("rating_data without K still stops when min(ratings) != 1", {
  ratings <- c(2, 3, 4, 5, 6)
  item_inds <- c(1, 1, 2, 2, 3)
  expect_error(
    rating_data(ratings, item_inds, VERBOSE = FALSE),
    "Lowest category different from 1"
  )
})

test_that("rating_data stores item_labels when provided", {
  ratings <- c(0.1, 0.5, 0.5, 0.9, 0.2, 0.8)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  labels <- c("A", "A", "B", "B", "C", "C")
  result <- rating_data(
    ratings,
    item_inds,
    ITEM_LABELS = labels,
    VERBOSE = FALSE
  )
  expect_equal(result$item_labels, c("A", "B", "C"))
})

test_that("rating_data stores worker_labels for two-way data", {
  ratings <- c(0.1, 0.5, 0.5, 0.9, 0.2, 0.8)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  worker_inds <- c(1, 2, 1, 2, 1, 2)
  item_labels <- c("A", "A", "B", "B", "C", "C")
  worker_labels <- c("W1", "W2", "W1", "W2", "W1", "W2")
  result <- rating_data(
    ratings,
    item_inds,
    worker_inds,
    ITEM_LABELS = item_labels,
    WORKER_LABELS = worker_labels,
    VERBOSE = FALSE
  )
  expect_equal(result$item_labels, c("A", "B", "C"))
  expect_equal(result$worker_labels, c("W1", "W2"))
})

test_that("rating_data errors on wrong-length ITEM_LABELS", {
  ratings <- c(0.1, 0.5, 0.5, 0.9)
  item_inds <- c(1, 1, 2, 2)
  expect_error(
    rating_data(ratings, item_inds, ITEM_LABELS = c("A", "B"), VERBOSE = FALSE)
  )
})

test_that("rating_data errors on non-character ITEM_LABELS", {
  ratings <- c(0.1, 0.5, 0.5, 0.9)
  item_inds <- c(1, 1, 2, 2)
  expect_error(
    rating_data(
      ratings,
      item_inds,
      ITEM_LABELS = c(1, 1, 2, 2),
      VERBOSE = FALSE
    )
  )
})

test_that("rating_data errors on inconsistent labels for same item", {
  ratings <- c(0.1, 0.5, 0.5, 0.9)
  item_inds <- c(1, 1, 2, 2)
  labels <- c("A", "B", "C", "C")
  expect_error(
    rating_data(ratings, item_inds, ITEM_LABELS = labels, VERBOSE = FALSE),
    "multiple labels"
  )
})

test_that("rating_data errors on WORKER_LABELS without WORKER_INDS", {
  ratings <- c(0.1, 0.5, 0.5, 0.9)
  item_inds <- c(1, 1, 2, 2)
  expect_error(
    rating_data(
      ratings,
      item_inds,
      WORKER_LABELS = c("W1", "W1", "W2", "W2"),
      VERBOSE = FALSE
    ),
    "requires WORKER_INDS"
  )
})

test_that("rating_data retains labels for degenerate items", {
  ratings <- c(0.5, 0.5, 0.1, 0.9, 0.3, 0.7)
  item_inds <- c(1, 1, 2, 2, 3, 3)
  labels <- c("Degen", "Degen", "B", "B", "C", "C")
  result <- rating_data(
    ratings,
    item_inds,
    ITEM_LABELS = labels,
    VERBOSE = FALSE
  )
  expect_equal(result$n_items, 3)
  expect_equal(result$item_labels, c("Degen", "B", "C"))
  expect_equal(result$degen_ids, 1L)
})
