#' Prepare rating data for analysis
#'
#' @description
#' Validates and preprocesses a raw ratings dataset. Returns a `rating_data`
#' S3 object that can be passed to [agreement()], [plot()], and [print()].
#' Degenerate items (all ratings identical) are detected and their recoded
#' indices stored in `$degen_ids`; no observations are removed here.
#' The decision to drop degenerate items before fitting is delegated to [agreement()].
#'
#' @param RATINGS Ratings vector. Ordinal: integers in \{1,...,K\}. Continuous:
#'   reals in `(0,1)`. Inflated interval: reals in `[0,1]` with exact 0s or 1s.
#' @param ITEM_INDS Integer index vector of item allocations (same length as `RATINGS`).
#' @param WORKER_INDS Integer index vector of worker allocations.
#' @param ITEM_LABELS Optional character vector of item labels (same length as `RATINGS`).
#'   Each unique item index must map to exactly one label. When provided, label names
#'   are used for `alpha` coefficients in [coef()].
#' @param WORKER_LABELS Optional character vector of worker labels (same length as `RATINGS`).
#'   Requires `WORKER_INDS`. When provided, label names are used for `beta` coefficients in [coef()].
#' @param K Number of ordinal categories. If `NULL`, inferred as `max(RATINGS)`.
#'   Provide explicitly when boundary categories may be absent from the data.
#' @param VERBOSE Print data diagnostics on construction. Default `TRUE`.
#'
#' @return An S3 object of class `rating_data`.
#'
#' @examples
#' dt <- sim_data(J = 20, B = 5, AGREEMENT = 0.6,
#'                ALPHA = rep(0, 20), DATA_TYPE = "continuous", SEED = 1)
#' rd <- rating_data(dt$rating, dt$id_item, dt$id_worker)
#' print(rd)
#'
#' @export
#' @importFrom stats setNames
rating_data <- function(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  ITEM_LABELS = NULL,
  WORKER_LABELS = NULL,
  K = NULL,
  VERBOSE = FALSE
) {
  # internal function to detect data type (ordinal, interval, interval with inflation)
  detect_type <- function(r) {
    if (all(r == as.integer(r))) {
      if (min(r) != 1) {
        stop("Lowest category different from 1")
      }
      if (length(unique(r)) != max(r)) {
        warning("Some category is missing")
      }
      return("ordinal")
    }
    if (any(r == 0) || any(r == 1)) {
      if (min(r) < 0 || max(r) > 1) {
        stop("Ratings must be in [0,1] for the inflated interval model.")
      }
      return("inflated")
    }
    if (min(r) < 0) {
      stop("Continuous data must lie strictly in [0, 1]: values < 0 detected.")
    }
    if (max(r) > 1) {
      stop("Continuous data must lie strictly in [0, 1]: values > 1 detected.")
    }
    "continuous"
  }

  # check args
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

  if (!is.null(ITEM_LABELS)) {
    stopifnot(is.character(ITEM_LABELS))
    stopifnot(length(ITEM_LABELS) == length(RATINGS))
  }
  if (!is.null(WORKER_LABELS)) {
    stopifnot(is.character(WORKER_LABELS))
    stopifnot(length(WORKER_LABELS) == length(RATINGS))
    if (is.null(WORKER_INDS)) {
      stop("WORKER_LABELS requires WORKER_INDS to be provided.")
    }
  }

  data_type_early <- if (is.null(K)) detect_type(RATINGS) else "ordinal"

  recode <- function(ids) {
    u <- sort(unique(ids))
    as.integer(setNames(seq_along(u), u)[as.character(ids)])
  }

  extract_labels <- function(inds, labels) {
    u <- sort(unique(inds))
    out_labels <- character(length(u))
    for (i in seq_along(u)) {
      lbls <- unique(labels[inds == u[i]])
      if (length(lbls) != 1L) {
        stop(
          "Index ",
          u[i],
          " maps to multiple labels: ",
          paste(lbls, collapse = ", ")
        )
      }
      out_labels[i] <- lbls
    }
    out_labels
  }

  out <- list()
  out$ratings <- RATINGS * 1.0
  out$item_ids <- recode(ITEM_INDS)
  if (!is.null(WORKER_INDS)) {
    out$worker_ids <- recode(WORKER_INDS)
  }

  out$degen_ids <- as.integer(which(sapply(
    split(out$ratings, out$item_ids),
    function(x) all(x == x[1])
  )))

  out$n_items <- length(unique(out$item_ids))
  if (!is.null(WORKER_INDS)) {
    out$n_workers <- length(unique(out$worker_ids))
  }

  if (!is.null(ITEM_LABELS)) {
    out$item_labels <- extract_labels(ITEM_INDS, ITEM_LABELS)
  }
  if (!is.null(WORKER_LABELS)) {
    out$worker_labels <- extract_labels(WORKER_INDS, WORKER_LABELS)
  }

  if (VERBOSE) {
    n_degen <- length(out$degen_ids)
    degen_note <- if (n_degen > 0) paste0(" (", n_degen, " degenerate)") else ""
    if (is.null(WORKER_INDS)) {
      message(" - Detected ", out$n_items, degen_note, " items.")
    } else {
      message(
        " - Detected ",
        out$n_items,
        degen_note,
        " items and ",
        out$n_workers,
        " workers."
      )
    }
  }

  if (!is.null(K)) {
    stopifnot(is.numeric(K), K == as.integer(K), K > 1)
    if (!all(out$ratings == as.integer(out$ratings))) {
      stop("Explicit K provided but RATINGS are not integers.")
    }
    if (any(out$ratings < 1) || any(out$ratings > K)) {
      stop(paste0("All ratings must be in {1, ..., K}. K=", K))
    }
    out$data_type <- "ordinal"
    out$K <- as.integer(K)
    if (VERBOSE) {
      message(" - Ordinal data on a user-specified ", K, "-point scale.")
      n_miss <- K - length(unique(out$ratings))
      if (n_miss > 0) {
        message(
          " - ",
          n_miss,
          " categor",
          ifelse(n_miss == 1, "y", "ies"),
          " not observed in data."
        )
      }
    }
  } else {
    out$data_type <- data_type_early
    out$K <- switch(
      data_type_early,
      ordinal = max(out$ratings),
      inflated = NA,
      continuous = 1L
    )
    if (VERBOSE) {
      message(switch(
        data_type_early,
        ordinal = paste0(
          " - Detected ordinal data on a ",
          out$K,
          "-points scale."
        ),
        inflated = " - Detected inflated interval data on the [0,1] range.",
        continuous = " - Detected continuous data on the (0,1) range."
      ))
    }
  }

  out$ave_ratings_per_item <- mean(table(out$item_ids))
  if (!is.null(out$worker_ids)) {
    out$ave_ratings_per_worker <- mean(table(out$worker_ids))
  }

  if (VERBOSE) {
    message(
      " - Average budget per item is ",
      round(out$ave_ratings_per_item, 2),
      " ratings."
    )
    if (!is.null(out$worker_ids)) {
      message(
        " - Average workload per worker is ",
        round(out$ave_ratings_per_worker, 2),
        " items."
      )
    }
  }

  structure(out, class = "rating_data")
}

validate_params_type <- function(NUISANCE, TARGET) {
  stopifnot(is.vector(NUISANCE))
  stopifnot(all(NUISANCE %in% c("items", "workers")))
  stopifnot(is.vector(TARGET))
  stopifnot(all(TARGET %in% c("phi")))

  params <- c("items", "workers")
  structure(
    list(
      constant = params[!(params %in% NUISANCE) & !(params %in% TARGET)],
      nuisance = NUISANCE,
      target = TARGET
    ),
    class = "params_spec"
  )
}

validate_cpp_control <- function(LIST = NULL) {
  default <- function(x, val) if (is.null(x)) val else x
  chk_pos <- function(x, nm) {
    stopifnot(is.numeric(x))
    stopifnot(x > 0)
    x
  }

  LIST$SEARCH_RANGE <- chk_pos(default(LIST$SEARCH_RANGE, 8), "SEARCH_RANGE")
  LIST$MAX_ITER <- chk_pos(default(LIST$MAX_ITER, 100), "MAX_ITER")
  LIST$PROF_SEARCH_RANGE <- chk_pos(
    default(LIST$PROF_SEARCH_RANGE, 10),
    "PROF_SEARCH_RANGE"
  )
  LIST$PROF_MAX_ITER <- chk_pos(
    default(LIST$PROF_MAX_ITER, 500),
    "PROF_MAX_ITER"
  )
  LIST$ALT_MAX_ITER <- chk_pos(default(LIST$ALT_MAX_ITER, 50), "ALT_MAX_ITER")
  LIST$ALT_TOL <- chk_pos(default(LIST$ALT_TOL, 1e-3), "ALT_TOL")
  LIST$BOUNDARY <- chk_pos(default(LIST$BOUNDARY, 100), "BOUNDARY")

  LIST[c(
    "SEARCH_RANGE",
    "MAX_ITER",
    "PROF_SEARCH_RANGE",
    "PROF_MAX_ITER",
    "ALT_MAX_ITER",
    "ALT_TOL",
    "BOUNDARY"
  )]
}
