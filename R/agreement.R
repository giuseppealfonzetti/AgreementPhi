#’ Compute Agreement
#’
#’ @description
#’
#’ Compute the \eqn{\Phi} agreement proposed in Checco et al. (2017) via profile likelihood methods.
#’ Three data types are supported and detected automatically from `RATINGS`:
#’ \itemize{
#’   \item **Ordinal**: integer-valued in \{1, 2, ..., K\}.
#’   \item **Continuous**: real-valued in the open interval `(0, 1)`.
#’   \item **Inflated interval**: real-valued in `[0, 1]` with point masses at 0 and/or 1.
#’     Fitted via the ordered beta mixture model. One-way only (`WORKER_INDS` must be `NULL`).
#’ }
#’
#’ @references
#’
#’ - Checco A., Roitero K., Maddalena E., Mizzaro S., Demartini G., (2017). “Let’s Agree to Disagree: Fixing Agreement Measures for Crowdsourcing.” *Proceedings of the AAAI Conference on Human Computation and Crowdsourcing* **5**: 11–20. [doi](https://doi.org/10.1609/hcomp.v5i1.13306)
#’
#’ @param RATINGS Ratings vector of dimension n. Ordinal data must be coded in \{1, 2, ..., K\}.
#’   Continuous data must lie in `(0, 1)`. Inflated interval data must lie in `[0, 1]` with at
#’   least one exact 0 or 1.
#’ @param ITEM_INDS Index vector with item allocations. Same dimension as `RATINGS`.
#’   Must be integers in \{1, 2, ..., J\}.
#’ @param WORKER_INDS Index vector with worker allocations. Same dimension as `RATINGS`.
#’   Must be integers in \{1, 2, ..., W\}. Not used for the inflated interval model.
#’ @param METHOD Choose between `”modified”` or `”profile”`. Default is `”modified”`.
#’   \itemize{
#’     \item `”modified”`: Uses modified profile likelihood with Barndorff-Nielsen correction.
#’     \item `”profile”`: Uses standard profile likelihood.
#’   }
#’ @param ALPHA_START Starting values for item-specific intercepts. Vector of length J. Default is `rep(0, J)`.
#’   Ignored for the inflated interval model.
#’ @param BETA_START Starting values for worker-specific intercepts. Vector of length W-1. Default is `rep(0, W-1)`.
#’   Ignored for the inflated interval model.
#’ @param TAU Thresholds for discretisation of the underlying beta distribution. Ignored for the inflated interval model.
#’ @param K Number of ordinal categories. If `NULL` (default), inferred from data as `max(RATINGS)`.
#’   Provide explicitly when some boundary categories (e.g. 1 or K) may be absent from the observed data.
#’   Ignored for continuous and inflated interval data.
#’ @param PHI_START Starting value for the beta precision parameter. Must be positive.
#’   Default is `agr2prec(0.5)`. Ignored for the inflated interval model.
#’ @param NUISANCE Vector containing either `”items”`, `”workers”`, or both. Defines which fixed
#’   effects to profile out during estimation. Ignored for the inflated interval model.
#’ @param CONTROL Control options for the optimization. Ignored for the inflated interval model.
#’ \describe{
#’     \item{`SEARCH_RANGE`}{Search range for precision parameter optimization.
#’       The algorithm searches in \[1e-8, PHI_START + SEARCH_RANGE\].
#’       Must be positive. Default: `8`.}
#’     \item{`MAX_ITER`}{Maximum number of iterations for precision parameter optimization.
#’       Must be a positive integer. Default: `100`.}
#’     \item{`PROF_SEARCH_RANGE`}{Search range for profiling out nuisance parameters (item intercepts).
#’       The algorithm searches in \[ALPHA_START\[j\] - PROF_SEARCH_RANGE, ALPHA_START\[j\] + PROF_SEARCH_RANGE\]
#’       for each item j. Must be positive. Default: `4`.}
#’     \item{`PROF_MAX_ITER`}{Maximum number of iterations for profiling optimization.
#’       Must be a positive integer. Default: `10`.}
#’     \item{`ALT_MAX_ITER`}{Maximum iterations for alternating profiling.
#’       Must be a positive integer. Default: `10`.}
#’     \item{`ALT_TOL`}{Relative convergence tolerance for alternating profiling.
#’       Must be positive. Default: `1e-2`.}
#’  }
#’ @param VERBOSE Verbose output.
#’
#’ @return A list with the following components:
#’ \describe{
#’   \item{`data_type`}{Detected data type: `”ordinal”`, `”continuous”`, or `”inflated”`.}
#’   \item{`method`}{Estimation method used: `”profile”` or `”modified”`.}
#’   \item{`alpha`}{Estimated item-specific intercepts (vector of length J).}
#’   \item{`beta`}{Estimated worker-specific intercepts. `NULL` for one-way models.}
#’   \item{`k0`}{Estimated lower cutpoint on the logit scale. Inflated interval model only.}
#’   \item{`k1`}{Estimated upper cutpoint on the logit scale. Inflated interval model only.}
#’   \item{`profile`}{List with `$precision` (profile MLE of \eqn{\phi}) and `$agreement` (corresponding \eqn{\Phi}).}
#’   \item{`modified`}{List with `$precision` (MPL estimate of \eqn{\phi}) and `$agreement` (corresponding \eqn{\Phi}). `NA` when `METHOD = “profile”`.}
#’   \item{`loglik`}{Profile log-likelihood at the MLE.}
#’   \item{`se`}{Named vector of standard errors. For inflated interval data: `phi`, `k0`, `k1`.}
#’   \item{`vcov`}{Variance-covariance matrix of `(phi, k0, k1)`. Inflated interval model only.}
#’   \item{`inflated_fit`}{Raw output from [fit_inflated_profile()] or [fit_inflated_mpl()]. Inflated interval model only.}
#’ }
#’
#’ @examples
#’ set.seed(321)
#’
#’ items <- 50
#’ budget_per_item <- 5
#’ alphas <- runif(items, -2, 2)
#’ agr <- .6
#’
#’ dt_oneway <- sim_data(
#’   J = items,
#’   B = budget_per_item,
#’   AGREEMENT = agr,
#’   ALPHA = alphas,
#’   DATA_TYPE = “continuous”,
#’   SEED = 123
#’ )
#’
#’ fit <- agreement(
#’   RATINGS = dt_oneway$rating,
#’   ITEM_INDS = dt_oneway$id_item,
#’   WORKER_INDS = dt_oneway$id_worker,
#’   METHOD = “modified”,
#’   NUISANCE = c(“items”),
#’   VERBOSE = TRUE
#’ )
#’ ci <- get_ci(fit)
#’ ci
#’
#’ dt_inflated <- sim_data(
#’   J = items,
#’   B = budget_per_item,
#’   AGREEMENT = agr,
#’   ALPHA = alphas,
#’   DATA_TYPE = “inflated”,
#’   K0 = -2,
#’   K1 = 2,
#’   SEED = 123
#’ )
#’
#’ fit_inf <- agreement(
#’   RATINGS = dt_inflated$rating,
#’   ITEM_INDS = dt_inflated$id_item,
#’   METHOD = “modified”
#’ )
#’ ci_inf <- get_ci(fit_inf)
#’ ci_inf
#'
#' @export
agreement <- function(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  METHOD = c("modified", "profile"),
  ALPHA_START = NULL,
  BETA_START = NULL,
  TAU = NULL,
  K = NULL,
  PHI_START = NULL,
  NUISANCE = c("items", "workers"),
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
    K = K,
    VERBOSE = VERBOSE
  )

  params_type <- validate_params_type(NUISANCE, "phi", val_data$n_items)

  if (val_data$data_type == "inflated") {
    if (!is.null(WORKER_INDS)) {
      stop("Inflated interval model is one-way only; WORKER_INDS must be NULL.")
    }

    if (METHOD == "modified") {
      inflated_fit <- fit_inflated_mpl(
        Y         = val_data$ratings,
        ITEM_INDS = val_data$item_ids,
        J         = val_data$n_items
      )
    } else {
      inflated_fit <- fit_inflated_profile(
        Y         = val_data$ratings,
        ITEM_INDS = val_data$item_ids,
        J         = val_data$n_items
      )
    }

    out <- list(
      data_type = "inflated",
      method = METHOD,
      params_type = params_type,
      alpha = inflated_fit$alpha,
      beta = NULL,
      tau = NULL,
      k0 = inflated_fit$k0,
      k1 = inflated_fit$k1,
      profile = list(
        precision = if (METHOD == "modified") {
          inflated_fit$ref_fit$phi
        } else {
          inflated_fit$phi
        },
        agreement = if (METHOD == "modified") {
          par2agr(inflated_fit$ref_fit$phi,
                  ALPHA = inflated_fit$ref_fit$alpha[!inflated_fit$ref_fit$is_degen],
                  K0    = inflated_fit$ref_fit$k0,
                  K1    = inflated_fit$ref_fit$k1)$agreement
        } else {
          par2agr(inflated_fit$phi,
                  ALPHA = inflated_fit$alpha[!inflated_fit$is_degen],
                  K0    = inflated_fit$k0,
                  K1    = inflated_fit$k1)$agreement
        }
      ),
      modified = list(
        precision = if (METHOD == "modified") inflated_fit$phi else NA_real_,
        agreement = if (METHOD == "modified") {
          par2agr(inflated_fit$phi,
                  ALPHA = inflated_fit$alpha[!inflated_fit$is_degen],
                  K0    = inflated_fit$k0,
                  K1    = inflated_fit$k1)$agreement
        } else {
          NA_real_
        }
      ),
      loglik = inflated_fit$loglik,
      se = inflated_fit$se,
      vcov = inflated_fit$vcov,
      inflated_fit = inflated_fit
    )

    if (VERBOSE) {
      message("Done!\n")
    }
    return(out)
  }

  if (val_data$ave_ratings_per_item^3 < val_data$n_items) {
    if (VERBOSE) {
      message("Average number of ratings per item is lower than reccomended")
    }
  }

  if (VERBOSE) {
    message(paste(
      "\nMODEL PARAMETERS"
    ))
    message(paste(
      " - Constant effects:",
      paste0(paste0(params_type$constant, collapse = ", "))
    ))
    message(paste(
      " - Nuisance effects:",
      paste0(paste0(params_type$nuisance, collapse = ", "))
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

  if (is.null(TAU)) {
    TAU <- seq(0, 1, by = 1 / val_data$K)
  }

  CONTROL <- validate_cpp_control(CONTROL)
  args <- c(
    list(
      Y = val_data$ratings * 1.0,
      ITEM_INDS = val_data$item_ids,
      WORKER_INDS = val_data$worker_ids,
      ALPHA_START = ALPHA_START,
      BETA_START = c(0, BETA_START),
      TAU_START = TAU,
      PHI_START = PHI_START,
      K = val_data$K,
      J = val_data$n_items,
      W = val_data$n_workers,
      METHOD = METHOD,
      DATA_TYPE = val_data$data_type,
      ITEMS_NUISANCE = "items" %in% params_type$nuisance,
      WORKER_NUISANCE = "workers" %in% params_type$nuisance,
      VERBOSE = VERBOSE
    ),
    CONTROL
  )

  out <- list(
    "cpp_args" = args,
    "data_type" = val_data$data_type,
    "method" = METHOD,
    "params_type" = params_type
  )

  opt <- do.call(cpp_get_phi, args)

  out$alpha <- opt$alpha
  out$beta <- opt$beta
  out$tau <- opt$tau
  out$profile$precision <- opt$profile_phi
  out$profile$agreement <- par2agr(opt$profile_phi)$agreement
  out$modified$precision <- opt$modified_phi
  out$modified$agreement <- if (!is.na(opt$modified_phi)) {
    par2agr(opt$modified_phi)$agreement
  } else {
    NaN
  }
  out$loglik <- opt$loglik

  if (VERBOSE) {
    message("Done!\n")
  }

  return(out)
}
