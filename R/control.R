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
validate_data <- function(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  K = NULL,
  VERBOSE = TRUE
) {
  out <- list()

  # Check args
  stopifnot(is.numeric(RATINGS))
  stopifnot(is.numeric(ITEM_INDS))
  stopifnot(all(ITEM_INDS == as.integer(ITEM_INDS)))
  stopifnot(length(RATINGS) == length(ITEM_INDS))

  if (!is.null(WORKER_INDS)) {
    stopifnot(is.numeric(WORKER_INDS))
    stopifnot(all(WORKER_INDS == as.integer(WORKER_INDS)))
    stopifnot(length(RATINGS) == length(WORKER_INDS))
  }
  stopifnot(is.logical(VERBOSE))

  # Check for degenerate items in the one-way case
  if (is.null(WORKER_INDS)) {
    degen_collect <- as.numeric(which(sapply(
      split(RATINGS, ITEM_INDS),
      function(x) all(x == x[1])
    )))

    if (length(degen_collect) > 0) {
      # Drop degenerate items
      informative_ids <- ITEM_INDS[!(ITEM_INDS %in% degen_collect)]
      informative_rts <- RATINGS[!(ITEM_INDS %in% degen_collect)]

      # Recode item indexes
      unique_ids <- sort(unique(informative_ids))
      id_map <- setNames(seq_along(unique_ids), unique_ids)
      informative_ids_recoded <- as.numeric(id_map[as.character(
        informative_ids
      )])

      out$item_ids <- as.integer(informative_ids_recoded)
      out$ratings <- informative_rts
    } else {
      out$ratings <- RATINGS * 1.0

      # Recode items in case of non consecutive indexes
      unique_items <- sort(unique(ITEM_INDS))
      item_map <- setNames(seq_along(unique_items), unique_items)
      out$item_ids <- as.integer(item_map[as.character(ITEM_INDS)])
    }
  } else {
    out$ratings <- RATINGS * 1.0

    # Recode items in case of non consecutive indexes
    unique_items <- sort(unique(ITEM_INDS))
    item_map <- setNames(seq_along(unique_items), unique_items)
    out$item_ids <- as.integer(item_map[as.character(ITEM_INDS)])

    # Recode workers in case of non consecutive indexes
    unique_workers <- sort(unique(WORKER_INDS))
    worker_map <- setNames(seq_along(unique_workers), unique_workers)
    out$worker_ids <- as.integer(worker_map[as.character(WORKER_INDS)])
  }

  out$n_items <- length(unique(out$item_ids))

  if (is.null(WORKER_INDS)) {
    if (VERBOSE) {
      message(paste0(" - Detected ", out$n_items, " non-degenerate items."))
    }
  } else {
    out$n_workers <- length(unique(out$worker_ids))
    if (VERBOSE) {
      message(paste0(
        " - Detected ",
        out$n_items,
        " items and ",
        out$n_workers,
        " workers."
      ))
    }
  }

  if (!is.null(K)) {
    stopifnot(is.numeric(K), K == as.integer(K), K > 1)
    if (!all(out$ratings == as.integer(out$ratings)))
      stop("Explicit K provided but RATINGS are not integers.")
    if (any(out$ratings < 1) || any(out$ratings > K))
      stop(paste0("All ratings must be in {1, ..., K}. K=", K))
    out$data_type <- "ordinal"
    out$K <- as.integer(K)
    if (VERBOSE) {
      message(paste0(" - Ordinal data on a user-specified ", K, "-point scale."))
      n_missing <- K - length(unique(out$ratings))
      if (n_missing > 0)
        message(paste0(
          " - ", n_missing, " categor",
          ifelse(n_missing == 1, "y", "ies"), " not observed in data."
        ))
    }
  } else {
    out$data_type <- detect_data_type(RATINGS = out$ratings)
    if (out$data_type == "ordinal") {
      out$K <- max(out$ratings)
      if (VERBOSE) {
        message(paste0(" - Detected ordinal data on a ", out$K, "-points scale."))
      }
    } else {
      out$K <- 1
      if (VERBOSE) {
        message(paste0(" - Detected continuous data on the (0,1) range."))
      }
    }
  }

  if (is.null(WORKER_INDS)) {
    out$ave_ratings_per_item <- mean(table(out$item_ids))
    if (VERBOSE) {
      message(paste0(
        " - Average number of observed ratings per item is ",
        round(out$ave_ratings_per_item, 2),
        "."
      ))
    }
  } else {
    out$ave_ratings_per_item <- mean(table(out$item_ids))
    out$ave_ratings_per_worker <- mean(table(out$worker_ids))

    if (VERBOSE) {
      message(paste0(
        " - Average number of observed ratings per item is ",
        round(out$ave_ratings_per_item, 2),
        "."
      ))
      message(paste0(
        " - Average number of observed ratings per worker is ",
        round(out$ave_ratings_per_worker, 2),
        "."
      ))
    }
  }

  return(out)
}

validate_nuisance <- function(NUISANCE) {
  stopifnot(is.vector(NUISANCE))
  stopifnot(all(NUISANCE %in% c("items", "workers")))
  return(NUISANCE)
}

validate_params_type <- function(NUISANCE, TARGET, DATA_TYPE) {
  stopifnot(is.vector(NUISANCE))
  stopifnot(all(NUISANCE %in% c("items", "workers")))
  stopifnot(is.vector(TARGET))
  stopifnot(all(TARGET %in% c("phi")))

  params <- c("items", "workers")
  if (DATA_TYPE == "continuous") {
    params <- c("items", "workers")
  }

  constant <- params[!(params %in% NUISANCE) & !(params %in% TARGET)]

  out <- list(
    constant = constant,
    nuisance = NUISANCE,
    target = TARGET
  )

  return(out)
}

validate_cpp_control <- function(LIST = NULL) {
  out <- list()

  # search range for precision
  if (is.null(LIST$SEARCH_RANGE)) {
    LIST$SEARCH_RANGE <- 8
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
    LIST$PROF_MAX_ITER <- 500
  }
  stopifnot(is.numeric(LIST$PROF_MAX_ITER))
  stopifnot(LIST$PROF_MAX_ITER > 0)
  out$PROF_MAX_ITER <- LIST$PROF_MAX_ITER

  if (is.null(LIST$ALT_MAX_ITER)) {
    LIST$ALT_MAX_ITER <- 10
  }
  stopifnot(is.numeric(LIST$ALT_MAX_ITER))
  stopifnot(LIST$ALT_MAX_ITER > 0)
  out$ALT_MAX_ITER <- LIST$ALT_MAX_ITER
  if (is.null(LIST$ALT_TOL)) {
    LIST$ALT_TOL <- 1e-2
  }
  stopifnot(is.numeric(LIST$ALT_TOL))
  stopifnot(LIST$ALT_TOL > 0)
  out$ALT_TOL <- LIST$ALT_TOL

  return(out)
}
