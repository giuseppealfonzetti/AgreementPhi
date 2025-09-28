detect_data_type <- function(RATINGS, VERBOSE = TRUE) {
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
    if (VERBOSE) {
      message(paste0("Detected ordinal data on a ", max_val, "-points scale."))
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
    if (VERBOSE) {
      message(paste0("Detected continuous data on the (0,1) range."))
    }
    return("continuous")
  }
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
    LIST$PROF_SEARCH_RANGE <- 10
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
