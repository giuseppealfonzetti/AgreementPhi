#' Compute agreement via profile likelihood methods
#'
#' @param RATINGS Ratings vector of dimension n. Ordinal data must be coded in \{1, 2, ..., K\}.
#'   Continuous data can take values in (0, 1).
#' @param ITEM_INDS Index vector with items allocations. Same dimension as `RATINGS`.
#' @param WORKER_INDS Index vector with worker allocations. Same dimension as `RATINGS`. Ignored when MODEL == "oneway".
#'   Must be integers in \{1, 2, ..., J\}.
#' @param METHOD Choose between `"modified"` or `"profile"`. Default is `"modified"`.
#'   \itemize{
#'     \item `"modified"`: Uses modified profile likelihood with Barndorff-Nielsen correction
#'     \item `"profile"`: Uses standard profile likelihood
#'   }
#' @param MODEL Choose between `"oneway"` or `"twoway"`. Default is `"oneway"`.
#'   \itemize{
#'     \item `"oneway"`: Item-specific intercepts. Indistiguishable workers.
#'     \item `"twoway"`: Item-specific and worker-specific intercepts.
#'   }
#' @param ALPHA_START Starting values for item-specific intercepts. Vector of length J. Default is `rep(0, J)` where J is the number of items.
#' @param BETA_START Starting values for worker-specific intercepts. Vector of length W-1. Default is `rep(0, W-1)` where W is the number of workers
#' @param PHI_START Starting value for beta precision parameter. Must be positive. Default is `agr2prec(0.5)` (precision corresponding to 50% agreement).
#' @param CONTROL Control options for the optimization:
#' \describe{
#'     \item{`SEARCH_RANGE`}{Search range for precision parameter optimization.
#'       The algorithm searches in `[1e-8, PHI_START + SEARCH_RANGE]`.
#'       Must be positive. Default: `10`.}
#'     \item{`MAX_ITER`}{Maximum number of iterations for precision parameter optimization.
#'       Must be a positive integer. Default: `100`.}
#'     \item{`PROF_SEARCH_RANGE`}{Search range for profiling out nuisance parameters (item intercepts).
#'       The algorithm searches in `[ALPHA_START[j] - PROF_SEARCH_RANGE, ALPHA_START[j] + PROF_SEARCH_RANGE]`
#'       for each item j. Must be positive. Default: `10`.}
#'     \item{`PROF_MAX_ITER`}{Maximum number of iterations for profiling optimization.
#'       Must be a positive integer. Default: `100`.}
#'     \item{`PROF_METHOD`}{Method for profiling optimization. A character string among:
#'       \itemize{
#'         \item `"brent"`: Brent's method (one-way only). Default for one-way.
#'         \item `"newton_raphson"`: Newton-Raphson method (one-way only).
#'         \item `"bfgs"`: L-BFGS method (two-way only). Default for two-way continuous data.
#'         \item `"alt_brent"`: Alternated profiling via univariate Brent's method (two-way only). Default for two-way ordinal data
#'       }}
#'     \item{`ALT_MAX_ITER`}{Maximum iterations for `alt_brent`.
#'       Must be a positive integer. Default: `10`.}
#'     \item{`ALT_TOL`}{Relative convergence tolerance for `alt_brent`.
#'       Must be positive. Default: `1e-2`}
#'  }
#' @param VERBOSE Verbose output.
#'
#' @return Returns a list with maximum likelihood estimates and corresponding negative log-likelihood.
#' @export
agreement2 <- function(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  METHOD = c("modified", "profile"),
  ALPHA_START = NULL,
  BETA_START = NULL,
  TAU_START = NULL,
  PHI_START = NULL,
  NUISANCE = c("items", "workers", "thresholds"),
  CONTROL = list(),
  VERBOSE = FALSE
) {
  METHOD <- match.arg(METHOD)

  if (VERBOSE) {
    message("\nDATA")
  }
  val_data <- validate_data(
    RATINGS = RATINGS,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS = WORKER_INDS,
    VERBOSE = VERBOSE
  )

  if (val_data$ave_ratings_per_item^3 < val_data$n_items) {
    if (VERBOSE) {
      message("Average number of ratings per item is lower than reccomended")
    }
  }

  params <- if (val_data$data_type == "ordinal") {
    c("items", "workers", "thresholds")
  } else {
    c("items", "workers")
  }
  if (VERBOSE) {
    message(paste(
      "\nMODEL PARAMETERS"
    ))
    message(paste(
      " - Nuisance:",
      paste0(paste0(params[params %in% NUISANCE], collapse = ", "), ".")
    ))
    message(paste(
      " - Constant :",
      paste0(paste0(params[!(params %in% NUISANCE)], collapse = ", "), ".\n")
    ))
  }

  if (is.null(ALPHA_START)) {
    ALPHA_START <- rep(0, val_data$n_items)
  }

  if (is.null(BETA_START)) {
    BETA_START <- rep(0, val_data$n_workers - 1)
  }

  if (is.null(PHI_START)) {
    PHI_START <- agr2prec(.5)
  }

  if (is.null(TAU_START)) {
    # init_tau <- function(y, K) {
    #   counts <- tabulate(factor(y, levels = seq_len(K)), nbins = K)
    #   cum_p <- cumsum(counts) / sum(counts)
    #   c(0, cum_p[-K], 1)
    # }
    counts <- tabulate(
      factor(val_data$ratings, levels = seq_len(val_data$K)),
      nbins = val_data$K
    )
    cum_p <- cumsum(counts) / sum(counts)
    TAU_START <- c(0, cum_p[-K], 1)
    # TAU_START <- seq(0, 1, by = 1 / val_data$K)
  }

  CONTROL <- validate_cpp_control2(CONTROL)
  args <- c(
    list(
      Y = val_data$ratings * 1.0,
      ITEM_INDS = val_data$item_ids,
      WORKER_INDS = val_data$worker_ids,
      ALPHA_START = ALPHA_START,
      BETA_START = c(0, BETA_START),
      TAU_START = TAU_START,
      PHI_START = PHI_START,
      K = val_data$K,
      J = val_data$n_items,
      W = val_data$n_workers,
      METHOD = METHOD,
      DATA_TYPE = val_data$data_type,
      WORKER_NUISANCE = "workers" %in% NUISANCE,
      THRESHOLDS_NUISANCE = "thresholds" %in% NUISANCE,
      VERBOSE = VERBOSE
    ),
    CONTROL
  )

  out <- list(
    "cpp_args" = args,
    "data_type" = val_data$data_type,
    "method" = METHOD,
    "nuisance" = NUISANCE
  )

  opt <- do.call(cpp_get_phi, args)
  if (length(opt) == 3) {
    out$profile$precision <- opt[3]
    out$profile$agreement <- prec2agr(opt[3])
    out$modified$precision <- opt[1]
    out$modified$agreement <- prec2agr(opt[1])
  } else {
    out$profile$precision <- opt[1]
    out$profile$agreement <- prec2agr(opt[1])
    out$modified$precision <- NA
    out$modified$agreement <- NA
  }
  out[["loglik"]] <- opt[2]

  # if (out[["pl_precision"]] < out[["mpl_precision"]]) {
  #   warning(
  #     "Possible divergence detected. Modified estimate might be unreliable. Try profiling via bfgs."
  #   )
  # }

  if (VERBOSE) {
    message("Done!\n")
  }

  return(out)
}
