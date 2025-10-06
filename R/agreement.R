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
#'         \item `"brent"`: Brent's method. Default.
#'         \item `"newton_raphson"`: Newton-Raphson method.
#'       }
#'       Note: Newton-Raphson is automatically selected when the average number of ratings
#'       per item cubed is less than the number of items (i.e., when `mean(table(ITEM_INDS))^3 < max(ITEM_INDS)`).}
#'   }
#' @param VERBOSE Verbose output.
#'
#' @return Returns a list with maximum likelihood estimates and corresponding negative log-likelihood.
#' @export
agreement <- function(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  METHOD = c("modified", "profile"),
  MODEL = c("oneway", "twoway"),
  ALPHA_START = NULL,
  BETA_START = NULL,
  PHI_START = NULL,
  CONTROL = list(),
  VERBOSE = FALSE
) {
  METHOD <- match.arg(METHOD)
  MODEL <- match.arg(MODEL)
  if (MODEL == "twoway") {
    stopifnot(!is.null(WORKER_INDS))
  } else {
    WORKER_INDS <- NULL
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

  if (VERBOSE) {
    message(paste(
      "Fitting",
      MODEL,
      "model:"
    ))
  }

  if (is.null(ALPHA_START)) {
    ALPHA_START <- rep(0, val_data$n_items)
  }

  if (is.null(BETA_START) & MODEL == "twoway") {
    BETA_START <- rep(0, val_data$n_workers - 1)
  }

  if (is.null(PHI_START)) {
    PHI_START <- agr2prec(.5)
  }

  continuous <- TRUE
  if (val_data$data_type == "ordinal") {
    continuous <- FALSE
  }

  CONTROL <- validate_cpp_control(CONTROL, MODEL)
  args <- c(
    list(
      Y = val_data$ratings * 1.0,
      ITEM_INDS = val_data$item_ids,
      ALPHA_START = ALPHA_START,
      PHI = PHI_START,
      K = val_data$K,
      J = val_data$n_items,
      VERBOSE = VERBOSE,
      CONTINUOUS = continuous
    ),
    CONTROL
  )

  out <- list(
    "cpp_args" = args,
    "data_type" = val_data$data_type,
    "method" = METHOD,
    "model" = MODEL
  )

  if (MODEL == "oneway") {
    if (METHOD == "modified") {
      opt <- do.call(cpp_get_phi_mp, args)
      out[["pl_precision"]] <- opt[3]
      out[["pl_agreement"]] <- prec2agr(opt[3])
      out[["mpl_precision"]] <- opt[1]
      out[["mpl_agreement"]] <- prec2agr(opt[1])
      out[["loglik"]] <- opt[2]
    } else {
      opt <- do.call(cpp_get_phi_mle, args)
      out[["pl_precision"]] <- opt[1]
      out[["pl_agreement"]] <- prec2agr(opt[1])
      out[["loglik"]] <- opt[2]
    }
  } else {
    if (METHOD == "modified") {
      opt <- get_phi_modified_profile_twoway(
        Y = val_data$ratings,
        ITEM_INDS = val_data$item_ids,
        WORKER_INDS = val_data$worker_ids,
        LAMBDA_START = c(ALPHA_START, BETA_START),
        PHI_START = PHI_START,
        K = val_data$K,
        J = val_data$n_items,
        W = val_data$n_workers,
        DATA_TYPE = val_data$data_type,
        SEARCH_RANGE = args$SEARCH_RANGE,
        MAX_ITER = args$MAX_ITER,
        PROF_MAX_ITER = args$PROF_MAX_ITER,
        VERBOSE = args$VERBOSE
      )
      out[["pl_precision"]] <- opt$pl_precision
      out[["pl_agreement"]] <- prec2agr(opt$pl_precision)
      out[["mpl_precision"]] <- opt$mpl_precision
      out[["mpl_agreement"]] <- prec2agr(opt$mpl_precision)
      out[["loglik"]] <- opt$loglik
      out[["lambda_mle"]] <- opt$lambda_mle
    } else {
      opt <- get_phi_profile_twoway(
        Y = val_data$ratings,
        ITEM_INDS = val_data$item_ids,
        WORKER_INDS = val_data$worker_ids,
        LAMBDA_START = c(ALPHA_START, BETA_START),
        PHI_START = PHI_START,
        K = val_data$K,
        J = val_data$n_items,
        W = val_data$n_workers,
        DATA_TYPE = val_data$data_type,
        SEARCH_RANGE = args$SEARCH_RANGE,
        MAX_ITER = args$MAX_ITER,
        PROF_MAX_ITER = args$PROF_MAX_ITER,
        VERBOSE = args$VERBOSE
      )
      out[["pl_precision"]] <- opt$precision
      out[["pl_agreement"]] <- prec2agr(opt$precision)
      out[["loglik"]] <- opt$loglik
      out[["lambda_mle"]] <- opt$lambda_mle
    }
  }

  return(out)
}
