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
agreement <- function(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  METHOD = c("modified", "profile"),
  ALPHA_START = NULL,
  BETA_START = NULL,
  TAU_START = NULL,
  PHI_START = NULL,
  NUISANCE = c("items", "workers"),
  TARGET = c("phi", "thresholds"),
  CONTROL = list(),
  VERBOSE = FALSE,
  NCORES = 1
) {
  METHOD <- match.arg(METHOD)

  RcppParallel::setThreadOptions(numThreads = NCORES)

  if (VERBOSE) {
    message("\nDATA")
  }
  val_data <- validate_data(
    RATINGS = RATINGS,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS = WORKER_INDS,
    VERBOSE = VERBOSE
  )

  params_type <- validate_params_type(NUISANCE, TARGET, val_data$n_items)
  # NUISANCE <- validate_nuisance(NUISANCE)

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
      " - Constant:",
      paste0(paste0(params_type$constant, collapse = ", "))
    ))
    message(paste(
      " - Nuisance:",
      paste0(paste0(params_type$nuisance, collapse = ", "), ".")
    ))
    message(paste(
      " - Target:",
      paste0(paste0(params_type$target, collapse = ", "), ".")
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
    TAU_START <- seq(0, 1, by = 1 / val_data$K)
    if ("thresholds" %in% NUISANCE) {
      TAU_START <- init_tau(val_data$ratings, val_data$K)
    }
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
      ITEMS_NUISANCE = "items" %in% params_type$nuisance,
      WORKER_NUISANCE = "workers" %in% params_type$nuisance,
      # THRESHOLDS_NUISANCE = "thresholds" %in% params_type$nuisance,
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

  if ("thresholds" %in% TARGET) {
    if (METHOD == "modified") {
      stop("Modified profile likelihood not implemented for thresholds target")
    } else {
      # Profile likelihood optimization over both phi and tau

      # Transform starting values to raw parameters
      raw_phi_start <- log(args$PHI_START)
      raw_tau_start <- tau2raw(args$TAU_START)
      start_par <- c(raw_phi_start, raw_tau_start)

      # Extract arguments for C++ functions
      cpp_args <- list(
        Y = args$Y,
        ITEM_INDS = args$ITEM_INDS,
        WORKER_INDS = args$WORKER_INDS,
        ALPHA = args$ALPHA_START,
        BETA = args$BETA_START,
        J = args$J,
        W = args$W,
        K = args$K,
        ITEMS_NUISANCE = args$ITEMS_NUISANCE,
        WORKER_NUISANCE = args$WORKER_NUISANCE,
        PROF_UNI_RANGE = as.integer(args$PROF_SEARCH_RANGE),
        PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
        PROF_MAX_ITER = as.integer(args$ALT_MAX_ITER),
        PROF_TOL = args$ALT_TOL
      )

      # Extract lbfgs control options if provided in CONTROL
      lbfgs_control <- list()
      if (!is.null(args$LBFGS_MAX_LINESEARCH)) {
        lbfgs_control$max_linesearch <- args$LBFGS_MAX_LINESEARCH
      }
      if (!is.null(args$LBFGS_MAX_ITERATIONS)) {
        lbfgs_control$max_iterations <- args$LBFGS_MAX_ITERATIONS
      }
      if (!is.null(args$LBFGS_INVISIBLE)) {
        lbfgs_control$invisible <- args$LBFGS_INVISIBLE
      }

      # Run optimization
      opt_result <- get_phi_tau_profile(
        START_PAR = start_par,
        cpp_args = cpp_args,
        lbfgs_control = lbfgs_control
      )

      # Extract optimal parameters
      raw_phi_opt <- opt_result$par[1]
      raw_tau_opt <- opt_result$par[-1]

      # Transform back to natural scale
      phi_opt <- exp(raw_phi_opt)
      tau_opt <- raw2tau(raw_tau_opt)

      # Profile nuisance parameters at optimum
      profiled_final <- cpp_ordinal_get_lambda2(
        Y = args$Y,
        ITEM_INDS = args$ITEM_INDS,
        WORKER_INDS = args$WORKER_INDS,
        ALPHA = args$ALPHA_START,
        BETA = args$BETA_START,
        TAU = tau_opt,
        PHI = phi_opt,
        J = args$J,
        W = args$W,
        K = args$K,
        ITEMS_NUISANCE = args$ITEMS_NUISANCE,
        WORKER_NUISANCE = args$WORKER_NUISANCE,
        THRESHOLDS_NUISANCE = FALSE,
        PROF_UNI_RANGE = args$PROF_SEARCH_RANGE,
        PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
        PROF_MAX_ITER = as.integer(args$ALT_MAX_ITER),
        TOL = args$ALT_TOL
      )

      # Store results
      out$alpha <- profiled_final$alpha
      out$beta <- profiled_final$beta
      out$tau <- tau_opt
      out$profile$precision <- phi_opt
      out$profile$agreement <- prec2agr(phi_opt)
      out$modified$precision <- NA
      out$modified$agreement <- NaN
      out$loglik <- -opt_result$value # lbfgs minimizes, we want max loglik
      out$convergence <- opt_result$convergence
    }
  } else {
    # NEW PATH: Use nested optimization when thresholds are nuisance
    if (
      "thresholds" %in% params_type$nuisance && val_data$data_type == "ordinal"
    ) {
      # Transform TAU_START to raw parameters
      raw_tau_start <- tau2raw(args$TAU_START)

      # Build cpp_args for C++ profiling functions (common for both methods)
      cpp_args <- list(
        Y = args$Y,
        ITEM_INDS = args$ITEM_INDS,
        WORKER_INDS = args$WORKER_INDS,
        ALPHA = args$ALPHA_START,
        BETA = args$BETA_START,
        J = args$J,
        W = args$W,
        K = args$K,
        ITEMS_NUISANCE = args$ITEMS_NUISANCE,
        WORKER_NUISANCE = args$WORKER_NUISANCE,
        PROF_UNI_RANGE = as.integer(args$PROF_SEARCH_RANGE),
        PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
        PROF_MAX_ITER = as.integer(args$ALT_MAX_ITER),
        PROF_TOL = args$ALT_TOL
      )

      # Extract lbfgs control options if provided (common for both methods)
      lbfgs_control <- list()
      if (!is.null(args$LBFGS_MAX_LINESEARCH)) {
        lbfgs_control$max_linesearch <- args$LBFGS_MAX_LINESEARCH
      }
      if (!is.null(args$LBFGS_MAX_ITERATIONS)) {
        lbfgs_control$max_iterations <- args$LBFGS_MAX_ITERATIONS
      }
      if (!is.null(args$LBFGS_INVISIBLE)) {
        lbfgs_control$invisible <- args$LBFGS_INVISIBLE
      }

      if (METHOD == "profile") {
        # PROFILE LIKELIHOOD: Use Brent over phi with nested L-BFGS over gamma (parsimonious)
        # Initialize gamma from TAU_START
        gamma_start <- tau2gamma(args$TAU_START)

        opt_result <- get_phi_profile_nested_gamma(
          PHI_START = args$PHI_START,
          GAMMA_START = gamma_start,
          cpp_args = cpp_args,
          lbfgs_control = lbfgs_control,
          SEARCH_RANGE = args$SEARCH_RANGE,
          brent_tol = 1e-4
        )

        # Extract optimal parameters
        phi_opt <- opt_result$phi
        tau_opt <- opt_result$tau

        # Profile nuisance parameters at optimum
        profiled_final <- cpp_ordinal_get_lambda2(
          Y = args$Y,
          ITEM_INDS = args$ITEM_INDS,
          WORKER_INDS = args$WORKER_INDS,
          ALPHA = args$ALPHA_START,
          BETA = args$BETA_START,
          TAU = tau_opt,
          PHI = phi_opt,
          J = args$J,
          W = args$W,
          K = args$K,
          ITEMS_NUISANCE = args$ITEMS_NUISANCE,
          WORKER_NUISANCE = args$WORKER_NUISANCE,
          THRESHOLDS_NUISANCE = FALSE,
          PROF_UNI_RANGE = as.integer(args$PROF_SEARCH_RANGE),
          PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
          PROF_MAX_ITER = as.integer(args$ALT_MAX_ITER),
          TOL = args$ALT_TOL
        )

        # Store results
        out$alpha <- profiled_final$alpha
        out$beta <- profiled_final$beta
        out$tau <- tau_opt
        out$profile$precision <- phi_opt
        out$profile$agreement <- prec2agr(phi_opt)
        out$modified$precision <- NA
        out$modified$agreement <- NaN
        out$loglik <- opt_result$loglik
        out$convergence <- opt_result$convergence
      } else if (METHOD == "modified") {
        # MODIFIED PROFILE LIKELIHOOD: First compute MLE, then apply correction

        # STEP 1: Compute MLE using nested optimization with gamma parameterization
        gamma_start <- tau2gamma(args$TAU_START)

        mle_result <- get_phi_profile_nested_gamma(
          PHI_START = args$PHI_START,
          GAMMA_START = gamma_start,
          cpp_args = cpp_args,
          lbfgs_control = lbfgs_control,
          SEARCH_RANGE = args$SEARCH_RANGE,
          brent_tol = 1e-6
        )

        # Extract optimal parameters
        phi_mle <- mle_result$phi
        tau_mle <- mle_result$tau

        # # Extract MLE values
        # raw_phi_mle <- mle_result$par[1]
        # raw_tau_mle <- mle_result$par[-1]
        # phi_mle <- exp(raw_phi_mle)
        # tau_mle <- raw2tau(raw_tau_mle)

        # Profile alpha/beta at MLE
        profiled_mle <- cpp_ordinal_get_lambda2(
          Y = args$Y,
          ITEM_INDS = args$ITEM_INDS,
          WORKER_INDS = args$WORKER_INDS,
          ALPHA = args$ALPHA_START,
          BETA = args$BETA_START,
          TAU = tau_mle,
          PHI = phi_mle,
          J = args$J,
          W = args$W,
          K = args$K,
          ITEMS_NUISANCE = args$ITEMS_NUISANCE,
          WORKER_NUISANCE = args$WORKER_NUISANCE,
          THRESHOLDS_NUISANCE = FALSE,
          PROF_UNI_RANGE = as.integer(args$PROF_SEARCH_RANGE),
          PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
          PROF_MAX_ITER = as.integer(args$ALT_MAX_ITER),
          TOL = args$ALT_TOL
        )

        alpha_mle <- profiled_mle$alpha
        beta_mle <- profiled_mle$beta

        # STEP 2: Compute modified profile using Barndorff-Nielsen correction
        # cpp_args$PROF_UNI_RANGE <- 2
        lbfgs_control$max_iterations <- 5

        mod_result <- get_phi_modified_profile_nested_gamma(
          PHI_START = phi_mle,
          GAMMA_START = tau2gamma(tau_mle),
          ALPHA_MLE = alpha_mle,
          BETA_MLE = beta_mle,
          TAU_MLE = tau_mle,
          PHI_MLE = phi_mle,
          cpp_args = cpp_args,
          lbfgs_control = lbfgs_control,
          SEARCH_RANGE = args$SEARCH_RANGE,
          brent_tol = 1e-6
        )

        # Extract modified profile optimal parameters
        phi_mod <- mod_result$phi
        tau_mod <- mod_result$tau

        # Profile alpha/beta at modified optimum
        profiled_final <- cpp_ordinal_get_lambda2(
          Y = args$Y,
          ITEM_INDS = args$ITEM_INDS,
          WORKER_INDS = args$WORKER_INDS,
          ALPHA = alpha_mle,
          BETA = beta_mle,
          TAU = tau_mod,
          PHI = phi_mod,
          J = args$J,
          W = args$W,
          K = args$K,
          ITEMS_NUISANCE = args$ITEMS_NUISANCE,
          WORKER_NUISANCE = args$WORKER_NUISANCE,
          THRESHOLDS_NUISANCE = FALSE,
          PROF_UNI_RANGE = as.integer(args$PROF_SEARCH_RANGE),
          PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
          PROF_MAX_ITER = as.integer(args$ALT_MAX_ITER),
          TOL = args$ALT_TOL
        )

        # Store results
        out$alpha <- profiled_final$alpha
        out$beta <- profiled_final$beta
        out$tau <- tau_mod
        # out$alpha <- alpha_mle
        # out$beta <- beta_mle
        # out$tau <- tau_mle
        out$profile$precision <- phi_mle
        out$profile$agreement <- prec2agr(phi_mle)
        out$modified$precision <- phi_mod
        out$modified$agreement <- prec2agr(phi_mod)
        out$loglik <- mod_result$loglik
        out$convergence <- mod_result$convergence
      }
    } else {
      # EXISTING PATH: Use current cpp_get_phi for all other cases
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
    }
  }

  if (VERBOSE) {
    message("Done!\n")
  }

  return(out)
}
