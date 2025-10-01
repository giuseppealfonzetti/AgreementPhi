#' Compute agreement via profile likelihood methods
#'
#' @param RATINGS Ratings vector of dimension n. Ordinal data must be coded in \{1, 2, ..., K\}.
#'   Continuous data can take values in (0, 1).
#' @param ITEM_INDS Index vector with items allocations. Same dimension as `RATINGS`.
#'   Must be integers in \{1, 2, ..., J\}.
#' @param METHOD Choose between `"modified"` or `"profile"`. Default is `"modified"`.
#'   \itemize{
#'     \item `"modified"`: Uses modified profile likelihood with Barndorff-Nielsen correction
#'     \item `"profile"`: Uses standard profile likelihood
#'   }
#' @param ALPHA_START Starting values for item-specific intercepts. Vector of length J. Default is `rep(0, J)` where J is the number of items.
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
  METHOD = c("modified", "profile"),
  ALPHA_START = NULL,
  PHI_START = NULL,
  CONTROL = list(),
  VERBOSE = FALSE
) {
  val_data <- validate_data(
    RATINGS = RATINGS,
    ITEM_INDS = ITEM_INDS,
    VERBOSE = VERBOSE
  )
  METHOD <- match.arg(METHOD)

  if (val_data$ave_ratings_per_item^3 < val_data$n_items) {
    warning("Average number of ratings per item is lower than reccomended")
    CONTROL$PROF_METHOD <- 1
  }

  if (is.null(ALPHA_START)) {
    ALPHA_START <- rep(0, val_data$n_items)
  }

  if (is.null(PHI_START)) {
    PHI_START <- agr2prec(.5)
  }

  continuous <- TRUE
  if (val_data$data_type == "ordinal") {
    continuous <- FALSE
  }

  CONTROL <- validate_cpp_control(CONTROL)
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
    "data" = val_data,
    "method" = METHOD
  )
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

  return(out)
}
