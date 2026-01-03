# Parsimonious gamma parameterization helper functions (2 parameters instead of K-1)

#' Convert gamma to tau using parsimonious parameterization
#' @param GAMMA Vector of length 2: [gamma_1, gamma_2]
#' @param K Number of ordinal categories
#' @return Vector of length K+1 with thresholds (tau_0=0, tau_K=1)
#' @export
gamma2tau <- function(GAMMA, K) {
  cpp_gamma2tau(GAMMA, K)
}

#' Convert tau to gamma using parsimonious parameterization
#' @param TAU Vector of length K+1 with thresholds
#' @return Vector of length 2: [gamma_1, gamma_2]
#' @export
tau2gamma <- function(TAU) {
  cpp_tau2gamma(TAU)
}

#' Profile likelihood nested optimization using gamma parameterization
#' @param RAW_PHI Log of precision parameter
#' @param GAMMA_START Starting value for gamma (length 2)
#' @param cpp_args List of arguments to pass to C++ functions
#' @param lbfgs_control List of L-BFGS control parameters
#' @return List with optimization results
#' @export
profile_loglik_nested_gamma <- function(
  RAW_PHI,
  GAMMA_START,
  cpp_args = list(),
  lbfgs_control = list()
) {
  # Objective function for L-BFGS over GAMMA (2 params)
  R_profile_gamma <- function(GAMMA_VEC) {
    -do.call(
      cpp_profile_extended_gamma,
      c(
        list(
          GAMMA = GAMMA_VEC,
          RAW_PHI = RAW_PHI
        ),
        cpp_args
      )
    )
  }

  # Gradient function for L-BFGS
  R_profile_gamma_grad <- function(GAMMA_VEC) {
    -do.call(
      cpp_profile_extended_grad_gamma,
      c(
        list(
          GAMMA = GAMMA_VEC,
          RAW_PHI = RAW_PHI
        ),
        cpp_args
      )
    )
  }

  # Set L-BFGS defaults
  if (is.null(lbfgs_control$max_linesearch)) {
    lbfgs_control$max_linesearch <- 20
  }
  if (is.null(lbfgs_control$max_iterations)) {
    lbfgs_control$max_iterations <- 15
  }
  if (is.null(lbfgs_control$invisible)) {
    lbfgs_control$invisible <- 1
  }

  if (
    !is.null(lbfgs_control$invisible) && is.logical(lbfgs_control$invisible)
  ) {
    lbfgs_control$invisible <- as.integer(lbfgs_control$invisible)
  }

  lbfgs_args <- c(
    list(
      vars = GAMMA_START,
      call_eval = R_profile_gamma,
      call_grad = R_profile_gamma_grad
    ),
    lbfgs_control
  )

  # ucminf_args <- c(
  #   list(
  #     par = GAMMA_START,
  #     fn = R_profile_gamma,
  #     gr = R_profile_gamma_grad
  #   )
  # )

  # Run L-BFGS optimization over GAMMA (2 params instead of K-1)
  opt <- do.call(lbfgs::lbfgs, lbfgs_args)
  # opt <- do.call(ucminf::ucminf, ucminf_args)

  opt$loglik <- -opt$value
  opt$raw_phi <- RAW_PHI
  opt$gamma_opt <- opt$par

  return(opt)
}

#' Outer optimization over PHI using nested gamma optimization
#' @param PHI_START Starting value for PHI
#' @param GAMMA_START Starting value for gamma (length 2)
#' @param cpp_args List of arguments to pass to C++ functions
#' @param lbfgs_control List of L-BFGS control parameters
#' @param SEARCH_RANGE Search range for PHI optimization
#' @param brent_tol Tolerance for Brent's method
#' @return List with optimization results
#' @export
get_phi_profile_nested_gamma <- function(
  PHI_START,
  GAMMA_START,
  cpp_args = list(),
  lbfgs_control = list(),
  SEARCH_RANGE = 5,
  brent_tol = 1e-6
) {
  # Track current best gamma estimate for warm starting
  current_gamma <- GAMMA_START

  objective_phi <- function(RAW_PHI) {
    result <- profile_loglik_nested_gamma(
      RAW_PHI = RAW_PHI,
      # RAW_PHI = log(agr2prec(AGR)),
      GAMMA_START = current_gamma,
      cpp_args = cpp_args,
      lbfgs_control = lbfgs_control
    )

    # cat(paste0(
    #   "phi:",
    #   exp(raw_phi),
    #   " | loglik:",
    #   round(result$loglik, 3),
    #   "\n"
    # ))
    # Update for warm start
    # current_gamma <- result$gamma_opt
    return(-result$loglik)
  }

  # Search interval for PHI
  lower <- max(-0.8, log(PHI_START) - 3)
  upper <- min(3, log(PHI_START) + 3)
  # lower <- 1e-2
  # upper <- 1 - 1e-2
  # Brent's method to find optimal PHI
  opt_result <- optimize(
    f = objective_phi,
    interval = c(lower, upper),
    tol = brent_tol
  )

  raw_phi_opt <- opt_result$minimum
  # raw_phi_opt <- log(agr2prec(opt_result$minimum))

  # Final profiling at optimal PHI
  final_profile <- profile_loglik_nested_gamma(
    RAW_PHI = raw_phi_opt,
    GAMMA_START = current_gamma,
    cpp_args = cpp_args,
    lbfgs_control = lbfgs_control
  )

  # Convert gamma back to tau
  tau_opt <- gamma2tau(final_profile$gamma_opt, cpp_args$K)

  out <- list(
    par = c(raw_phi_opt, final_profile$gamma_opt),
    value = -final_profile$loglik,
    convergence = final_profile$convergence,
    loglik = final_profile$loglik,
    raw_phi = (raw_phi_opt),
    gamma = final_profile$gamma_opt,
    phi = exp(raw_phi_opt),
    tau = tau_opt
  )

  return(out)
}

#' Modified profile likelihood using gamma parameterization
#' @param RAW_PHI Log of precision parameter
#' @param GAMMA_START Starting value for gamma (length 2)
#' @param ALPHA_MLE MLE of alpha parameters
#' @param BETA_MLE MLE of beta parameters
#' @param TAU_MLE MLE of tau parameters
#' @param PHI_MLE MLE of phi parameter
#' @param cpp_args List of arguments to pass to C++ functions
#' @param lbfgs_control List of L-BFGS control parameters
#' @return List with optimization results
#' @export
modified_profile_loglik_nested_gamma <- function(
  RAW_PHI,
  GAMMA_START,
  ALPHA_MLE,
  BETA_MLE,
  TAU_MLE,
  PHI_MLE,
  cpp_args = list(),
  lbfgs_control = list()
) {
  # Get profile likelihood and optimal gamma
  profile_result <- profile_loglik_nested_gamma(
    RAW_PHI = RAW_PHI,
    GAMMA_START = GAMMA_START,
    cpp_args = cpp_args,
    lbfgs_control = lbfgs_control
  )

  tau_profiled <- gamma2tau(profile_result$gamma_opt, cpp_args$K)

  # Profile lambda at this (phi, tau)
  profiled_lambda <- cpp_ordinal_get_lambda2(
    Y = cpp_args$Y,
    ITEM_INDS = cpp_args$ITEM_INDS,
    WORKER_INDS = cpp_args$WORKER_INDS,
    ALPHA = cpp_args$ALPHA,
    BETA = cpp_args$BETA,
    TAU = tau_profiled,
    PHI = exp(RAW_PHI),
    J = cpp_args$J,
    W = cpp_args$W,
    K = cpp_args$K,
    ITEMS_NUISANCE = cpp_args$ITEMS_NUISANCE,
    WORKER_NUISANCE = cpp_args$WORKER_NUISANCE,
    THRESHOLDS_NUISANCE = FALSE,
    PROF_UNI_RANGE = cpp_args$PROF_UNI_RANGE,
    PROF_UNI_MAX_ITER = cpp_args$PROF_UNI_MAX_ITER,
    PROF_MAX_ITER = cpp_args$PROF_MAX_ITER,
    TOL = cpp_args$PROF_TOL
  )

  lambda_profiled <- c(profiled_lambda$alpha, profiled_lambda$beta[-1])
  lambda_mle <- c(ALPHA_MLE, BETA_MLE[-1])

  # Compute Barndorff-Nielsen correction terms
  log_det_J <- cpp_ordinal_twoway_log_det_obs_info(
    Y = cpp_args$Y,
    ITEM_INDS = cpp_args$ITEM_INDS,
    WORKER_INDS = cpp_args$WORKER_INDS,
    LAMBDA = lambda_profiled,
    TAU = tau_profiled,
    PHI = exp(RAW_PHI),
    K = cpp_args$K,
    J = cpp_args$J,
    W = cpp_args$W,
    ITEMS_NUISANCE = cpp_args$ITEMS_NUISANCE,
    WORKER_NUISANCE = cpp_args$WORKER_NUISANCE
  )

  log_det_I <- cpp_ordinal_twoway_log_det_E0d0d1_extended(
    ITEM_INDS = cpp_args$ITEM_INDS,
    WORKER_INDS = cpp_args$WORKER_INDS,
    LAMBDA0 = lambda_mle,
    LAMBDA1 = lambda_profiled,
    PHI0 = PHI_MLE,
    PHI1 = exp(RAW_PHI),
    TAU0 = TAU_MLE,
    TAU1 = tau_profiled,
    J = cpp_args$J,
    W = cpp_args$W,
    K = cpp_args$K,
    ITEMS_NUISANCE = cpp_args$ITEMS_NUISANCE,
    WORKER_NUISANCE = cpp_args$WORKER_NUISANCE
  )

  modified_loglik <- profile_result$loglik + 0.5 * log_det_J - log_det_I

  profile_result$loglik <- modified_loglik
  profile_result$modified_loglik <- modified_loglik

  return(profile_result)
}

#' Modified profile with gamma parameterization
#' @param PHI_START Starting value for PHI
#' @param GAMMA_START Starting value for gamma (length 2)
#' @param ALPHA_MLE MLE of alpha parameters
#' @param BETA_MLE MLE of beta parameters
#' @param TAU_MLE MLE of tau parameters
#' @param PHI_MLE MLE of phi parameter
#' @param cpp_args List of arguments to pass to C++ functions
#' @param lbfgs_control List of L-BFGS control parameters
#' @param SEARCH_RANGE Search range for PHI optimization
#' @param brent_tol Tolerance for Brent's method
#' @return List with optimization results
#' @export
get_phi_modified_profile_nested_gamma <- function(
  PHI_START,
  GAMMA_START,
  ALPHA_MLE,
  BETA_MLE,
  TAU_MLE,
  PHI_MLE,
  cpp_args = list(),
  lbfgs_control = list(),
  SEARCH_RANGE = 5,
  brent_tol = 1e-6
) {
  current_gamma <- GAMMA_START
  objective_phi <- function(RAW_PHI) {
    result <- modified_profile_loglik_nested_gamma(
      RAW_PHI = RAW_PHI,
      # RAW_PHI = log(agr2prec(AGR)),
      GAMMA_START = current_gamma,
      ALPHA_MLE = ALPHA_MLE,
      BETA_MLE = BETA_MLE,
      TAU_MLE = TAU_MLE,
      PHI_MLE = PHI_MLE,
      cpp_args = cpp_args,
      lbfgs_control = lbfgs_control
    )
    # current_gamma <- result$gamma_opt
    return(-result$loglik)
  }

  lower <- max(-0.8, log(PHI_MLE) - 3)
  upper <- min(3, log(PHI_MLE) + 3)
  # lower <- max(0, PHI_MLE - SEARCH_RANGE)
  # upper <- min(20, PHI_MLE + SEARCH_RANGE)
  # lower <- 1e-2
  # upper <- min(1 - 1e-2, prec2agr(PHI_MLE) + .1)

  opt_result <- optimize(
    f = objective_phi,
    interval = c(lower, upper),
    tol = brent_tol
  )

  # raw_phi_opt <- log(agr2prec(opt_result$minimum))
  raw_phi_opt <- opt_result$minimum

  final_profile <- modified_profile_loglik_nested_gamma(
    RAW_PHI = raw_phi_opt,
    GAMMA_START = GAMMA_START,
    ALPHA_MLE = ALPHA_MLE,
    BETA_MLE = BETA_MLE,
    TAU_MLE = TAU_MLE,
    PHI_MLE = PHI_MLE,
    cpp_args = cpp_args,
    lbfgs_control = lbfgs_control
  )

  tau_opt <- gamma2tau(final_profile$gamma_opt, cpp_args$K)

  out <- list(
    par = c(raw_phi_opt, final_profile$gamma_opt),
    value = -final_profile$loglik,
    convergence = final_profile$convergence,
    loglik = final_profile$loglik,
    raw_phi = raw_phi_opt,
    gamma = final_profile$gamma_opt,
    phi = exp(raw_phi_opt),
    tau = tau_opt
  )

  return(out)
}

# COMMENTED OUT 2025-12-26: Old optimization helpers using raw_tau (K-1 parameters)
# Replaced by gamma parameterization functions (profile_loglik_nested_gamma, etc.)
# Functions: get_phi_tau_profile, profile_loglik_nested, get_phi_profile_nested,
#            modified_profile_loglik_nested, get_phi_modified_profile_nested
# Retained for reference and potential backward compatibility.

# #' @export
# get_phi_tau_profile <- function(
#   START_PAR,
#   cpp_args = list(),
#   lbfgs_control = list()
# ) {
#   R_profile_extended <- function(PAR) {
#     -do.call(
#       cpp_profile_extended,
#       c(
#         list(
#           RAW_TAU = PAR[2:length(PAR)],
#           RAW_PHI = PAR[1]
#         ),
#         cpp_args
#       )
#     )
#   }
#   R_profile_extended_grad <- function(PAR) {
#     -do.call(
#       cpp_profile_extended_grad,
#       c(
#         list(
#           RAW_TAU = PAR[2:length(PAR)],
#           RAW_PHI = PAR[1]
#         ),
#         cpp_args
#       )
#     )
#   }
#
#   # Set default lbfgs control parameters if not specified
#   if (is.null(lbfgs_control$max_linesearch)) {
#     lbfgs_control$max_linesearch <- 5
#   }
#   if (is.null(lbfgs_control$max_iterations)) {
#     lbfgs_control$max_iterations <- 15
#   }
#   if (is.null(lbfgs_control$invisible)) {
#     lbfgs_control$invisible <- 1
#   }
#
#   # Convert logical to integer for lbfgs
#   if (
#     !is.null(lbfgs_control$invisible) && is.logical(lbfgs_control$invisible)
#   ) {
#     lbfgs_control$invisible <- as.integer(lbfgs_control$invisible)
#   }
#
#   # Build lbfgs call arguments
#   lbfgs_args <- c(
#     list(
#       vars = START_PAR,
#       call_eval = R_profile_extended,
#       call_grad = R_profile_extended_grad
#     ),
#     lbfgs_control
#   )
#
#   # Run optimization
#   opt <- do.call(lbfgs::lbfgs, lbfgs_args)
#
#   return(opt)
# }
#
#
# #' @export
# profile_loglik_nested <- function(
#   RAW_PHI,
#   RAW_TAU_START,
#   cpp_args = list(),
#   lbfgs_control = list()
# ) {
#   R_profile_tau <- function(RAW_TAU_VEC) {
#     -do.call(
#       cpp_profile_extended,
#       c(
#         list(
#           RAW_TAU = RAW_TAU_VEC,
#           RAW_PHI = RAW_PHI
#         ),
#         cpp_args
#       )
#     )
#   }
#
#   R_profile_tau_grad <- function(RAW_TAU_VEC) {
#     -do.call(
#       cpp_profile_extended_grad_raw_tau,
#       c(
#         list(
#           RAW_TAU = RAW_TAU_VEC,
#           RAW_PHI = RAW_PHI
#         ),
#         cpp_args
#       )
#     )
#   }
#
#   if (is.null(lbfgs_control$max_linesearch)) {
#     lbfgs_control$max_linesearch <- 5
#   }
#   if (is.null(lbfgs_control$max_iterations)) {
#     lbfgs_control$max_iterations <- 15
#   }
#   if (is.null(lbfgs_control$invisible)) {
#     lbfgs_control$invisible <- 1
#   }
#
#   if (
#     !is.null(lbfgs_control$invisible) && is.logical(lbfgs_control$invisible)
#   ) {
#     lbfgs_control$invisible <- as.integer(lbfgs_control$invisible)
#   }
#
#   lbfgs_args <- c(
#     list(
#       vars = RAW_TAU_START,
#       call_eval = R_profile_tau,
#       call_grad = R_profile_tau_grad
#     ),
#     lbfgs_control
#   )
#
#   # Run optimization over RAW_TAU
#   opt <- do.call(lbfgs::lbfgs, lbfgs_args)
#
#   opt$loglik <- -opt$value
#   opt$raw_phi <- RAW_PHI
#   opt$raw_tau_opt <- opt$par
#
#   return(opt)
# }
#
#
# #' @export
# get_phi_profile_nested <- function(
#   PHI_START,
#   RAW_TAU_START,
#   cpp_args = list(),
#   lbfgs_control = list(),
#   SEARCH_RANGE = 5,
#   brent_tol = 1e-4
# ) {
#   # Track current best tau estimate for warm starting
#   current_raw_tau <- RAW_TAU_START
#
#   objective_phi <- function(phi) {
#     result <- profile_loglik_nested(
#       RAW_PHI = log(phi),
#       RAW_TAU_START = current_raw_tau, # Use warm start
#       cpp_args = cpp_args,
#       lbfgs_control = lbfgs_control
#     )
#     # Update starting point for next evaluation (warm start)
#     current_raw_tau <<- result$raw_tau_opt
#     # Return negative log-likelihood for minimization
#     return(-result$loglik)
#   }
#
#   # Define search interval around starting value (on phi scale)
#   lower <- max(1e-2, PHI_START - SEARCH_RANGE)
#   upper <- min(10, PHI_START + SEARCH_RANGE)
#
#   # Run Brent's method to find optimal PHI
#   opt_result <- optimize(
#     f = objective_phi,
#     interval = c(lower, upper),
#     tol = brent_tol
#   )
#
#   # Get optimal PHI from Brent's method
#   phi_opt <- opt_result$minimum
#
#   # Re-run profiling at optimal PHI to get optimal RAW_TAU and other details
#   final_profile <- profile_loglik_nested(
#     RAW_PHI = log(phi_opt),
#     RAW_TAU_START = current_raw_tau, # Use warm start
#     cpp_args = cpp_args,
#     lbfgs_control = lbfgs_control
#   )
#
#   # Build output similar to get_phi_tau_profile()
#   out <- list(
#     par = c(log(phi_opt), final_profile$raw_tau_opt),
#     value = -final_profile$loglik, # Negated for consistency with lbfgs output
#     convergence = final_profile$convergence,
#     loglik = final_profile$loglik,
#     raw_phi = log(phi_opt),
#     raw_tau = final_profile$raw_tau_opt,
#     phi = phi_opt,
#     tau = raw2tau(final_profile$raw_tau_opt)
#   )
#
#   return(out)
# }
#
#
# #' @export
# modified_profile_loglik_nested <- function(
#   RAW_PHI,
#   RAW_TAU_START,
#   ALPHA_MLE,
#   BETA_MLE,
#   TAU_MLE,
#   PHI_MLE,
#   cpp_args = list(),
#   lbfgs_control = list()
# ) {
#   # STEP A: Get profile likelihood and optimal tau at this phi
#   profile_result <- profile_loglik_nested(
#     RAW_PHI = RAW_PHI,
#     RAW_TAU_START = RAW_TAU_START,
#     cpp_args = cpp_args,
#     lbfgs_control = lbfgs_control
#   )
#
#   tau_profiled <- raw2tau(profile_result$raw_tau_opt)
#
#   profiled_lambda <- cpp_ordinal_get_lambda2(
#     Y = cpp_args$Y,
#     ITEM_INDS = cpp_args$ITEM_INDS,
#     WORKER_INDS = cpp_args$WORKER_INDS,
#     ALPHA = cpp_args$ALPHA,
#     BETA = cpp_args$BETA,
#     TAU = tau_profiled,
#     PHI = exp(RAW_PHI),
#     J = cpp_args$J,
#     W = cpp_args$W,
#     K = cpp_args$K,
#     ITEMS_NUISANCE = cpp_args$ITEMS_NUISANCE,
#     WORKER_NUISANCE = cpp_args$WORKER_NUISANCE,
#     THRESHOLDS_NUISANCE = FALSE,
#     PROF_UNI_RANGE = cpp_args$PROF_UNI_RANGE,
#     PROF_UNI_MAX_ITER = cpp_args$PROF_UNI_MAX_ITER,
#     PROF_MAX_ITER = cpp_args$PROF_MAX_ITER,
#     TOL = cpp_args$PROF_TOL
#   )
#
#   lambda_profiled <- c(profiled_lambda$alpha, profiled_lambda$beta[-1])
#   lambda_mle <- c(ALPHA_MLE, BETA_MLE[-1])
#
#   log_det_J <- cpp_ordinal_twoway_log_det_obs_info(
#     Y = cpp_args$Y,
#     ITEM_INDS = cpp_args$ITEM_INDS,
#     WORKER_INDS = cpp_args$WORKER_INDS,
#     LAMBDA = lambda_profiled,
#     TAU = tau_profiled,
#     PHI = exp(RAW_PHI),
#     K = cpp_args$K,
#     J = cpp_args$J,
#     W = cpp_args$W,
#     ITEMS_NUISANCE = cpp_args$ITEMS_NUISANCE,
#     WORKER_NUISANCE = cpp_args$WORKER_NUISANCE
#   )
#
#   log_det_I <- cpp_ordinal_twoway_log_det_E0d0d1_extended(
#     ITEM_INDS = cpp_args$ITEM_INDS,
#     WORKER_INDS = cpp_args$WORKER_INDS,
#     LAMBDA0 = lambda_mle,
#     LAMBDA1 = lambda_profiled,
#     PHI0 = PHI_MLE,
#     PHI1 = exp(RAW_PHI),
#     TAU0 = TAU_MLE,
#     TAU1 = tau_profiled,
#     J = cpp_args$J,
#     W = cpp_args$W,
#     K = cpp_args$K,
#     ITEMS_NUISANCE = cpp_args$ITEMS_NUISANCE,
#     WORKER_NUISANCE = cpp_args$WORKER_NUISANCE
#   )
#   # log_det_I <- cpp_ordinal_twoway_log_det_E0d0d1(
#   #   ITEM_INDS = cpp_args$ITEM_INDS,
#   #   WORKER_INDS = cpp_args$WORKER_INDS,
#   #   LAMBDA0 = lambda_mle,
#   #   LAMBDA1 = lambda_profiled,
#   #   PHI0 = PHI_MLE,
#   #   PHI1 = phi,
#   #   TAU = TAU_MLE,
#   #   J = cpp_args$J,
#   #   W = cpp_args$W,
#   #   K = cpp_args$K,
#   #   ITEMS_NUISANCE = cpp_args$ITEMS_NUISANCE,
#   #   WORKER_NUISANCE = cpp_args$WORKER_NUISANCE
#   # )
#
#   modified_loglik <- profile_result$loglik + 0.5 * log_det_J - log_det_I
#
#   profile_result$loglik <- modified_loglik
#   profile_result$modified_loglik <- modified_loglik
#
#   return(profile_result)
# }
#
#
# #' @export
# get_phi_modified_profile_nested <- function(
#   PHI_START,
#   RAW_TAU_START,
#   ALPHA_MLE,
#   BETA_MLE,
#   TAU_MLE,
#   PHI_MLE,
#   cpp_args = list(),
#   lbfgs_control = list(),
#   SEARCH_RANGE = 5,
#   brent_tol = 1e-4
# ) {
#   # Objective function using modified profile likelihood
#   objective_phi <- function(RAW_PHI) {
#     # cat("current tau:", raw2tau(current_raw_tau), "\n")
#     result <- modified_profile_loglik_nested(
#       RAW_PHI = RAW_PHI,
#       RAW_TAU_START = RAW_TAU_START,
#       ALPHA_MLE = ALPHA_MLE,
#       BETA_MLE = BETA_MLE,
#       TAU_MLE = TAU_MLE,
#       PHI_MLE = PHI_MLE,
#       cpp_args = cpp_args,
#       lbfgs_control = lbfgs_control
#     )
#     # Update starting point for next evaluation (warm start)
#     # current_raw_tau <- result$raw_tau_opt
#     # Return negative log-likelihood for minimization
#     return(-result$loglik)
#   }
#
#   # Define search interval around starting value (on phi scale)
#   lower <- max(-0.8, log(PHI_START) - 1.5)
#   upper <- min(1, log(PHI_START) + 1.5)
#
#   # Run Brent's method to find optimal PHI
#   opt_result <- optimize(
#     f = objective_phi,
#     interval = c(lower, upper),
#     tol = brent_tol
#   )
#
#   # Get optimal PHI from Brent's method
#   phi_opt <- exp(opt_result$minimum)
#
#   # Re-run profiling at optimal PHI to get optimal RAW_TAU and other details
#   final_profile <- modified_profile_loglik_nested(
#     RAW_PHI = log(phi_opt),
#     RAW_TAU_START = RAW_TAU_START,
#     ALPHA_MLE = ALPHA_MLE,
#     BETA_MLE = BETA_MLE,
#     TAU_MLE = TAU_MLE,
#     PHI_MLE = PHI_MLE,
#     cpp_args = cpp_args,
#     lbfgs_control = lbfgs_control
#   )
#
#   # Build output similar to get_phi_profile_nested()
#   out <- list(
#     par = c(log(phi_opt), final_profile$raw_tau_opt),
#     value = -final_profile$loglik, # Negated for consistency with lbfgs output
#     convergence = final_profile$convergence,
#     loglik = final_profile$loglik,
#     raw_phi = log(phi_opt),
#     raw_tau = final_profile$raw_tau_opt,
#     phi = phi_opt,
#     tau = raw2tau(final_profile$raw_tau_opt)
#   )
#
#   return(out)
# }

# END COMMENTED OUT old optimization helpers
