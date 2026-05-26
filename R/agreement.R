#' Compute Agreement
#'
#' @description
#'
#' Compute the \eqn{\Phi} agreement proposed in Checco et al. (2017) via profile likelihood methods.
#' Three data types are supported, detected automatically from the supplied `rating_data` object:
#' \itemize{
#'   \item **Ordinal**: integer-valued in \{1, 2, ..., K\}.
#'   \item **Continuous**: real-valued in the open interval `(0, 1)`.
#'   \item **Inflated interval**: real-valued in `[0, 1]` with point masses at 0 and/or 1.
#'     Fitted via the ordered beta mixture model. One-way only (no workers in `DATA`).
#' }
#'
#' @references
#'
#' - Checco A., Roitero K., Maddalena E., Mizzaro S., Demartini G., (2017). "Let's Agree to Disagree: Fixing Agreement Measures for Crowdsourcing." *Proceedings of the AAAI Conference on Human Computation and Crowdsourcing* **5**: 11–20. [doi](https://doi.org/10.1609/hcomp.v5i1.13306)
#'
#' @param DATA A `rating_data` object created by [rating_data()].
#' @param METHOD Choose between `"modified"` or `"profile"`. Default is `"modified"`.
#'   \itemize{
#'     \item `"modified"`: Uses modified profile likelihood with Barndorff-Nielsen correction.
#'     \item `"profile"`: Uses standard profile likelihood.
#'   }
#' @param ALPHA_START Starting values for item-specific intercepts. Vector of length J.
#'   Default is `init_alpha()`. Ignored for the inflated interval model.
#' @param BETA_START Starting values for worker-specific intercepts. Vector of length W-1.
#'   Default is `rep(0, W-1)`. Ignored for the inflated interval model.
#' @param TAU Thresholds for discretisation of the underlying beta distribution. Ignored for the inflated interval model.
#' @param PHI_START Starting value for the beta precision parameter. Must be positive.
#'   Default is `agr2prec(0.5)`. Ignored for the inflated interval model.
#' @param NUISANCE Vector containing either `"items"`, `"workers"`, or both. Defines which fixed
#'   effects to profile out during estimation. Ignored for the inflated interval model.
#' @param CONTROL Control options for the optimization.
#' \describe{
#'     \item{`SEARCH_RANGE`}{Search range for precision parameter optimization.
#'       The algorithm searches in \[1e-8, PHI_START + SEARCH_RANGE\].
#'       Must be positive. Default: `8`.}
#'     \item{`MAX_ITER`}{Maximum number of iterations for precision parameter optimization.
#'       Must be a positive integer. Default: `100`.}
#'     \item{`PROF_SEARCH_RANGE`}{Search range for profiling out item intercepts (alpha).
#'       The algorithm searches in \[alpha_j - PROF_SEARCH_RANGE, alpha_j + PROF_SEARCH_RANGE\]
#'       for each item j. Applies to both continuous/ordinal and inflated interval data.
#'       Must be positive. Default: `10`.}
#'     \item{`PROF_MAX_ITER`}{Maximum number of iterations for profiling optimization.
#'       Must be a positive integer. Default: `500`.}
#'     \item{`ALT_MAX_ITER`}{Maximum iterations for alternating profiling. Non-inflated only.
#'       Must be a positive integer. Default: `50`.}
#'     \item{`ALT_TOL`}{Relative convergence tolerance for alternating profiling. Non-inflated only.
#'       Must be positive. Default: `1e-3`.}
#'     \item{`BOUNDARY`}{Boundary value for cutpoints when one boundary is absent. Inflated interval only.
#'       Must be positive. Default: `100`.}
#'  }
#' @param VERBOSE Print optimization progress. Default `FALSE`.
#'
#' @return An S3 object of class `agreement_fit` with the following components:
#' \describe{
#'   \item{`data_type`}{Detected data type: `"ordinal"`, `"continuous"`, or `"inflated"`.}
#'   \item{`method`}{Estimation method used: `"profile"` or `"modified"`.}
#'   \item{`alpha`}{Estimated item-specific intercepts (vector of length J).}
#'   \item{`beta`}{Estimated worker-specific intercepts. `NULL` for one-way models.}
#'   \item{`k0`}{Estimated lower cutpoint on the logit scale. Inflated interval model only.}
#'   \item{`k1`}{Estimated upper cutpoint on the logit scale. Inflated interval model only.}
#'   \item{`profile`}{List with `$precision` (profile MLE of \eqn{\phi}) and `$agreement` (corresponding \eqn{\Phi}).}
#'   \item{`modified`}{List with `$precision` (MPL estimate of \eqn{\phi}) and `$agreement` (corresponding \eqn{\Phi}). `NA` when `METHOD = "profile"`.}
#'   \item{`loglik`}{Profile log-likelihood at the MLE.}
#'   \item{`se`}{Named vector of standard errors. For inflated interval data: `phi`, `k0`, `k1`.}
#'   \item{`vcov`}{Variance-covariance matrix of `(phi, k0, k1)`. Inflated interval model only.}
#'   \item{`convergence`}{Optimizer convergence code. Inflated interval model only.}
#' }
#'
#' @examples
#' set.seed(321)
#'
#' items <- 50
#' budget_per_item <- 5
#' alphas <- runif(items, -2, 2)
#' agr <- .6
#'
#' dt_oneway <- sim_data(
#'   J = items, B = budget_per_item, AGREEMENT = agr,
#'   ALPHA = alphas, DATA_TYPE = "continuous", SEED = 123
#' )
#' rd <- rating_data(dt_oneway$rating, dt_oneway$id_item, dt_oneway$id_worker)
#' fit <- agreement(rd, METHOD = "modified", NUISANCE = c("items"))
#' confint(fit)
#' plot(fit)
#'
#' dt_inflated <- sim_data(
#'   J = items, B = budget_per_item, AGREEMENT = agr,
#'   ALPHA = alphas, DATA_TYPE = "inflated", K0 = -2, K1 = 2, SEED = 123
#' )
#' rd_inf <- rating_data(dt_inflated$rating, dt_inflated$id_item)
#' fit_inf <- agreement(rd_inf, METHOD = "modified")
#' confint(fit_inf)
#'
#' @export
agreement <- function(
  DATA,
  METHOD = c("modified", "profile"),
  ALPHA_START = NULL,
  BETA_START = NULL,
  TAU = NULL,
  PHI_START = NULL,
  NUISANCE = c("items", "workers"),
  CONTROL = list(),
  VERBOSE = FALSE
) {
  stopifnot(inherits(DATA, "rating_data"))
  METHOD <- match.arg(METHOD)

  params_type <- validate_params_type(NUISANCE, "phi")
  CONTROL <- validate_cpp_control(CONTROL)

  if (DATA$data_type == "inflated") {
    if (!is.null(DATA$worker_ids)) {
      stop(
        "Inflated interval model is one-way only; DATA must have no worker indices."
      )
    }

    if (METHOD == "modified") {
      inflated_fit <- fit_inflated_mpl(
        Y = DATA$ratings,
        ITEM_INDS = DATA$item_ids,
        J = DATA$n_items,
        PROF_SEARCH_RANGE = CONTROL$PROF_SEARCH_RANGE,
        PROF_MAX_ITER = CONTROL$PROF_MAX_ITER,
        BOUNDARY = CONTROL$BOUNDARY
      )
    } else {
      inflated_fit <- fit_inflated_profile(
        Y = DATA$ratings,
        ITEM_INDS = DATA$item_ids,
        J = DATA$n_items,
        PROF_SEARCH_RANGE = CONTROL$PROF_SEARCH_RANGE,
        PROF_MAX_ITER = CONTROL$PROF_MAX_ITER,
        BOUNDARY = CONTROL$BOUNDARY
      )
    }

    ref <- if (METHOD == "modified") inflated_fit$ref_fit else inflated_fit
    n_degen <- if (!is.null(DATA$n_degen)) DATA$n_degen else 0L
    adj_agr <- function(agr) {
      (DATA$n_items * agr + n_degen) / (DATA$n_items + n_degen)
    }

    out <- structure(
      list(
        data = DATA,
        data_type = "inflated",
        method = METHOD,
        params_type = params_type,
        alpha = inflated_fit$alpha,
        beta = NULL,
        tau = NULL,
        k0 = inflated_fit$k0,
        k1 = inflated_fit$k1,
        profile = list(
          precision = ref$phi,
          agreement = adj_agr(
            par2agr(
              ref$phi,
              ALPHA = ref$alpha,
              K0 = ref$k0,
              K1 = ref$k1
            )$agreement
          )
        ),
        modified = list(
          precision = if (METHOD == "modified") inflated_fit$phi else NA_real_,
          agreement = if (METHOD == "modified") {
            adj_agr(
              par2agr(
                inflated_fit$phi,
                ALPHA = inflated_fit$alpha,
                K0 = inflated_fit$k0,
                K1 = inflated_fit$k1
              )$agreement
            )
          } else {
            NA_real_
          }
        ),
        loglik = inflated_fit$loglik,
        se = inflated_fit$se,
        vcov = inflated_fit$vcov,
        convergence = inflated_fit$convergence,
        control = NULL
      ),
      class = "agreement_fit"
    )

    if (VERBOSE) {
      message("Done!\n")
    }
    return(out)
  }

  if (DATA$ave_ratings_per_item^3 < DATA$n_items && VERBOSE) {
    message("Average number of ratings per item is lower than recommended")
  }

  if (VERBOSE) {
    message("\nMODEL PARAMETERS")
    message(
      " - Constant effects: ",
      paste(params_type$constant, collapse = ", ")
    )
    message(
      " - Nuisance effects: ",
      paste(params_type$nuisance, collapse = ", ")
    )
  }

  if (is.null(ALPHA_START)) {
    ALPHA_START <- init_alpha(DATA$ratings, DATA$item_ids, DATA$n_items, -5, 5)
  }
  if (is.null(BETA_START)) {
    BETA_START <- rep(0, DATA$n_workers - 1)
  }
  if (is.null(PHI_START)) {
    PHI_START <- agr2prec(.5)
  }
  if (is.null(TAU)) {
    TAU <- seq(0, 1, by = 1 / DATA$K)
  }

  cpp_control <- CONTROL[c(
    "SEARCH_RANGE",
    "MAX_ITER",
    "PROF_SEARCH_RANGE",
    "PROF_MAX_ITER",
    "ALT_MAX_ITER",
    "ALT_TOL"
  )]
  args <- c(
    list(
      Y = DATA$ratings * 1.0,
      ITEM_INDS = DATA$item_ids,
      WORKER_INDS = DATA$worker_ids,
      ALPHA_START = ALPHA_START,
      BETA_START = c(0, BETA_START),
      TAU_START = TAU,
      PHI_START = PHI_START,
      K = DATA$K,
      J = DATA$n_items,
      W = DATA$n_workers,
      METHOD = METHOD,
      DATA_TYPE = DATA$data_type,
      ITEMS_NUISANCE = "items" %in% params_type$nuisance,
      WORKER_NUISANCE = "workers" %in% params_type$nuisance,
      VERBOSE = VERBOSE
    ),
    cpp_control
  )

  opt <- do.call(cpp_get_phi, args)

  out <- structure(
    list(
      data = DATA,
      data_type = DATA$data_type,
      method = METHOD,
      params_type = params_type,
      alpha = opt$alpha,
      beta = opt$beta,
      tau = opt$tau,
      k0 = NULL,
      k1 = NULL,
      profile = list(
        precision = opt$profile_phi,
        agreement = par2agr(opt$profile_phi)$agreement
      ),
      modified = list(
        precision = opt$modified_phi,
        agreement = if (!is.na(opt$modified_phi)) {
          par2agr(opt$modified_phi)$agreement
        } else {
          NaN
        }
      ),
      loglik = opt$loglik,
      se = NULL,
      vcov = NULL,
      convergence = NULL,
      control = list(
        PROF_SEARCH_RANGE = CONTROL$PROF_SEARCH_RANGE,
        PROF_MAX_ITER = CONTROL$PROF_MAX_ITER,
        ALT_MAX_ITER = CONTROL$ALT_MAX_ITER,
        ALT_TOL = CONTROL$ALT_TOL
      )
    ),
    class = "agreement_fit"
  )

  if (VERBOSE) {
    message("Done!\n")
  }
  return(out)
}
