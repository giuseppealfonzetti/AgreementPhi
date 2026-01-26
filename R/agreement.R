#' Compute Agreement
#'
#' @description
#'
#' Compute the \eqn{\Phi} agreement proposed in Checco et al. (2017) via profile likelihood methods.
#'
#' @references
#'
#' - Checco, Alessandro, Kevin Roitero, Eddy Maddalena, Stefano Mizzaro, and Gianluca Demartini. 2017. “Let’s Agree to Disagree: Fixing Agreement Measures for Crowdsourcing.” *Proceedings of the AAAI Conference on Human Computation and Crowdsourcing* **5**: 11–20. [doi](https://doi.org/10.1609/hcomp.v5i1.13306)
#'
#' @param RATINGS Ratings vector of dimension n. Ordinal data must be coded in \{1, 2, ..., K\}.
#'   Continuous data can take values in `(0, 1)`.
#' @param ITEM_INDS Index vector with items allocations. Same dimension as `RATINGS`.
#' @param WORKER_INDS Index vector with worker allocations. Same dimension as `RATINGS`. Ignored when MODEL == "oneway".
#'   Must be integers in \{1, 2, ..., J\}.
#' @param METHOD Choose between `"modified"` or `"profile"`. Default is `"modified"`.
#'   \itemize{
#'     \item `"modified"`: Uses modified profile likelihood with Barndorff-Nielsen correction
#'     \item `"profile"`: Uses standard profile likelihood
#'   }
#' @param ALPHA_START Starting values for item-specific intercepts. Vector of length J. Default is `rep(0, J)` where J is the number of items.
#' @param BETA_START Starting values for worker-specific intercepts. Vector of length W-1. Default is `rep(0, W-1)` where W is the number of workers
#' @param TAU Thresholds to use for the discretisation of the underlying beta distribution.
#' @param PHI_START Starting value for beta precision parameter. Must be positive. Default is `agr2prec(0.5)` (precision corresponding to 50% agreement).
#' @param NUISANCE Vector containg either `"items"` or `"workers"` or both. Defines which fixed effects to profile out during estimation.
#' @param CONTROL Control options for the optimization:
#' \describe{
#'     \item{`SEARCH_RANGE`}{Search range for precision parameter optimization.
#'       The algorithm searches in \[1e-8, PHI_START + SEARCH_RANGE\].
#'       Must be positive. Default: `8`.}
#'     \item{`MAX_ITER`}{Maximum number of iterations for precision parameter optimization.
#'       Must be a positive integer. Default: `100`.}
#'     \item{`PROF_SEARCH_RANGE`}{Search range for profiling out nuisance parameters (item intercepts).
#'       The algorithm searches in \[ALPHA_START\[j\] - PROF_SEARCH_RANGE, ALPHA_START\[j\] + PROF_SEARCH_RANGE\]
#'       for each item j. Must be positive. Default: `4`.}
#'     \item{`PROF_MAX_ITER`}{Maximum number of iterations for profiling optimization.
#'       Must be a positive integer. Default: `10`.}
#'     \item{`ALT_MAX_ITER`}{Maximum iterations for alternating profiling.
#'       Must be a positive integer. Default: `10`.}
#'     \item{`ALT_TOL`}{Relative convergence tolerance for alternating profiling.
#'       Must be positive. Default: `1e-2`.}
#'  }
#' @param VERBOSE Verbose output.
#'
#' @return Returns a list with maximum likelihood estimates and corresponding negative log-likelihood.
#'
#' @examples
#' set.seed(321)
#'
#' # setting dimension
#' items <- 50
#' budget_per_item <- 5
#' n_obs <- items * budget_per_item
#' workers <- 50
#'
#' # item-specific intercepts to generate the data
#' alphas <- runif(items, -2, 2)
#'
#' # true agreement (between 0 and 1)
#' agr <- .6
#'
#' # generate continuous rating in (0,1)
#' dt_oneway <- sim_data(
#'   J = items,
#'   B = budget_per_item,
#'   AGREEMENT = agr,
#'   ALPHA = alphas,
#'   DATA_TYPE = "continuous",
#'   SEED = 123
#' )
#'
#' # estimation via oneway specification
#' fit <- agreement(
#'   RATINGS = dt_oneway$rating,
#'   ITEM_INDS = dt_oneway$id_item,
#'   WORKER_INDS = dt_oneway$id_worker,
#'   METHOD = "modified",
#'   NUISANCE = c("items"),
#'   VERBOSE = TRUE
#' )
#' # get standard error and confidence interval
#' ci <- get_ci(fit)
#' ci
#'
#' # compute log-likelihood over a grid
#' range_ll <- get_range_ll(fit)
#'
#' # utility plot function for relative log-likelihood
#' plot_rll(
#'   D = range_ll,
#'   M_EST = fit$modified$agreement,
#'   P_EST = fit$profile$agreement,
#'   M_SE = ci$agreement_se,
#'   CONFIDENCE=.95
#' )
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
    VERBOSE = VERBOSE
  )

  params_type <- validate_params_type(NUISANCE, "phi", val_data$n_items)

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
  out$profile$agreement <- prec2agr(opt$profile_phi)
  out$modified$precision <- opt$modified_phi
  out$modified$agreement <- if (!is.na(opt$modified_phi)) {
    prec2agr(opt$modified_phi)
  } else {
    NaN
  }
  out$loglik <- opt$loglik

  if (VERBOSE) {
    message("Done!\n")
  }

  return(out)
}
