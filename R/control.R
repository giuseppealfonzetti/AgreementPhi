detect_data_type <- function(RATINGS) {
  integer_data <- all(RATINGS == as.integer(RATINGS))
  min_val <- min(RATINGS)
  max_val <- max(RATINGS)
  unique_vals <- length(unique(RATINGS))
  n_vals <- length(RATINGS)

  if (integer_data) {
    if (min_val != 1) {
      stop("Lowest category different from 1")
    }
    if (unique_vals != max_val) {
      warning("Some category is missing")
    }

    return("ordinal")
  } else {
    if (min_val <= 0) {
      stop(
        "Minimum value lower or equal than zero. Consider using an ordinal scale."
      )
    }
    if (max_val >= 1) {
      stop(
        "Maximum value higher or equal than one. Consider using an ordinal scale."
      )
    }

    return("continuous")
  }
}

#' @importFrom stats setNames
validate_data <- function(RATINGS, ITEM_INDS, VERBOSE = TRUE) {
  stopifnot(is.numeric(RATINGS))
  stopifnot(is.numeric(ITEM_INDS))
  stopifnot(length(RATINGS) == length(ITEM_INDS))

  degen_collect <- as.numeric(which(sapply(
    split(RATINGS, ITEM_INDS),
    function(x) {
      all(x == x[1])
    }
  )))

  out <- list()
  if (length(degen_collect > 0)) {
    # drop degenerate items
    informative_ids <- ITEM_INDS[!(ITEM_INDS %in% degen_collect)]
    informative_rts <- RATINGS[!(ITEM_INDS %in% degen_collect)]

    # recode item indexes
    unique_ids <- sort(unique(informative_ids))
    id_map <- setNames(seq_along(unique_ids), unique_ids)
    informative_ids_recoded <- as.numeric(id_map[as.character(informative_ids)])

    out$item_ids = informative_ids_recoded
    out$ratings = informative_rts
  } else {
    out$item_ids = ITEM_INDS
    out$ratings = RATINGS
  }

  out$n_items <- length(unique(out$item_ids))

  if (VERBOSE) {
    message(paste0("Detected ", out$n_items, " non-degenerate items."))
  }
  out$data_type <- detect_data_type(RATINGS = out$ratings)

  if (out$data_type == "ordinal") {
    out$K <- max(out$ratings)
    if (VERBOSE) {
      message(paste0("Detected ordinal data on a ", out$K, "-points scale."))
    }
  } else {
    out$K <- 1
    if (VERBOSE) {
      message(paste0("Detected continuous data on the (0,1) range."))
    }
  }

  out$ave_ratings_per_item <- mean(table(out$item_ids))

  if (VERBOSE) {
    message(paste0(
      "Average number of observed ratings per item is ",
      out$ave_ratings_per_item,
      "."
    ))
  }
  return(out)
}


validate_cpp_control <- function(LIST) {
  out <- list()

  # search range for precision
  if (is.null(LIST$SEARCH_RANGE)) {
    LIST$SEARCH_RANGE <- 10
  }
  stopifnot(is.numeric(LIST$SEARCH_RANGE))
  stopifnot(LIST$SEARCH_RANGE > 0)
  out$SEARCH_RANGE <- LIST$SEARCH_RANGE

  # max iter for precision
  if (is.null(LIST$MAX_ITER)) {
    LIST$MAX_ITER <- 100
  }
  stopifnot(is.numeric(LIST$MAX_ITER))
  stopifnot(LIST$MAX_ITER > 0)
  out$MAX_ITER <- LIST$MAX_ITER

  # search range for profiling
  if (is.null(LIST$PROF_SEARCH_RANGE)) {
    LIST$PROF_SEARCH_RANGE <- 5
  }
  stopifnot(is.numeric(LIST$PROF_SEARCH_RANGE))
  stopifnot(LIST$PROF_SEARCH_RANGE > 0)
  out$PROF_SEARCH_RANGE <- LIST$PROF_SEARCH_RANGE

  # max iter for profiling
  if (is.null(LIST$PROF_MAX_ITER)) {
    LIST$PROF_MAX_ITER <- 100
  }
  stopifnot(is.numeric(LIST$PROF_MAX_ITER))
  stopifnot(LIST$PROF_MAX_ITER > 0)
  out$PROF_MAX_ITER <- LIST$PROF_MAX_ITER

  # profiling method
  if (is.null(LIST$PROF_METHOD)) {
    LIST$PROF_METHOD <- "brent"
  }
  stopifnot(LIST$PROF_METHOD %in% c("brent", "newton_raphson"))
  if (LIST$PROF_METHOD == "newton_raphson") {
    out$PROF_METHOD <- 1
  } else {
    out$PROF_METHOD <- 0
  }

  return(out)
}
