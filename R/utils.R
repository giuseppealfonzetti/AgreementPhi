#' From precision to agreement
#'
#' @param X Precision parameter.
#'
#' @return General agreement.
#'
#' @export
prec2agr <- function(X) {
  stopifnot(X > 0)
  out <- 1 - 2^(-X * log(2) / 2)
  return(out)
}

#' From agreement to precision
#'
#' @param X General agreement.
#'
#' @return Precision parameter.
#'
#' @export
agr2prec <- function(X) {
  stopifnot(X >= 0 & X <= 1)
  out <- -2 * logb(1 - X, base = 2) / log(2)
  return(out)
}


#' @export
raw2tau <- function(X) {
  tau <- cumsum(exp(X)) / (sum(exp(X)) + 1)
  out <- c(0, tau, 1)
  return(out)
}

#' @export
tau2raw <- function(X) {
  n <- length(X) - 2
  gaps <- diff(X)
  last_gap <- gaps[n + 1]
  z <- n * gaps[seq_len(n)] / last_gap
  log(z)
}

#' @export
set_tau <- function(X, K) {
  stopifnot(length(X) == 2)
  stopifnot(all(X < 1) & all(X > 0))
  stopifnot(X[1] < X[2])
  out <- c(0, seq(X[1], X[2], length.out = K - 1), 1)
  return(out)
}

#' @export
init_tau <- function(Y, K) {
  counts <- tabulate(factor(Y, levels = seq_len(K)), nbins = K)
  cum_p <- cumsum(counts) / sum(counts)
  c(0, cum_p[-K], 1)
}

#' Discretise continuous data
#'
#' @param X Vector of continuous data in (0,1).
#' @param K Number of ordinal categories.
#' @param TRESHOLDS Threshold vector of length K-1. If null, thresholds are assumed to be equispaced.
#'
#' @return Discretised vector
#'
#' @export
cont2ord <- function(X, K, TRESHOLDS = NULL) {
  stopifnot(is.numeric(X))
  stopifnot(all(X <= 1))
  stopifnot(all(X >= 0))
  stopifnot(K %% 1 == 0)
  stopifnot(K > 1)
  breaks <- (0:K) / K
  if (!is.null(TRESHOLDS)) {
    stopifnot(is.numeric(TRESHOLDS))
    stopifnot(length(TRESHOLDS) == K - 1)
    breaks <- c(0, TRESHOLDS, 1)
  }
  out <- findInterval(X, breaks, left.open = TRUE)
  out[X == 0] <- 1
  return(out)
}

#' Squeeze [0,100] data
#'
#' @param Y Vector of continuous data in [0,1].
#' @param U Squeezing parameter
#'
#' @return Squeezed vector
#'
#' @export
lemon <- function(X, U = NULL) {
  stopifnot(all(X <= 1))
  stopifnot(all(X >= 0))
  n <- length(X)
  if (is.null(U)) {
    U <- 1 / (2 * (n - 1))
  }

  x <- (X + U) / (1 + 2 * U)
  return(x)
}

#' Get relative log-likelihood
#'
#' @param X Object fitted with [agreement()] function.
#' @param RANGE Range around agreement mle.
#' @param GRID_LENGTH Number of points to be evaluated within RANGE.
#' @param PLOT Plot relative log-likelihood.
#'
#' @return Return a data.frame with GRID_LENGTH rows and columns
#' `precision`, `agreement`, `profile_rll` and `modified_rll`.
#'
#' @importFrom graphics abline legend lines plot
#' @importFrom stats plogis rbeta
#' @export
get_rll <- function(X, RANGE = .2, PLOT = TRUE, GRID_LENGTH = 15) {
  stopifnot(is.numeric(RANGE))
  stopifnot(RANGE > 0)
  stopifnot(RANGE <= 1)
  stopifnot(is.logical(PLOT))
  stopifnot(GRID_LENGTH > 0)

  args <- X$cpp_args

  # Create grid around profile likelihood estimate
  agreement_range <- seq(
    max(X$profile$agreement - RANGE, 1e-2),
    min(X$profile$agreement + RANGE, 1 - 1e-2),
    length.out = GRID_LENGTH
  )
  phi_range <- sapply(agreement_range, agr2prec)

  # Extract parameters from cpp_args
  data_type <- if (!is.null(args$DATA_TYPE)) args$DATA_TYPE else X$data_type
  K <- if (!is.null(args$K)) args$K else 1L

  # Compute profile likelihood over grid
  # Use gamma parameterization for ordinal data with threshold nuisance
  if (data_type == "ordinal" && "thresholds" %in% X$params_type$nuisance) {
    # Build cpp_args for nested gamma optimization
    cpp_args <- list(
      Y = args$Y,
      ITEM_INDS = args$ITEM_INDS,
      WORKER_INDS = args$WORKER_INDS,
      ALPHA = args$ALPHA_START,
      BETA = args$BETA_START,
      J = args$J,
      W = args$W,
      K = K,
      ITEMS_NUISANCE = args$ITEMS_NUISANCE,
      WORKER_NUISANCE = args$WORKER_NUISANCE,
      PROF_UNI_RANGE = as.integer(args$PROF_SEARCH_RANGE),
      PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
      PROF_MAX_ITER = as.integer(args$ALT_MAX_ITER),
      PROF_TOL = args$ALT_TOL
    )

    # Build lbfgs_control from args
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

    gamma_start <- tau2gamma(args$TAU_START)

    # Compute profile likelihood with warm-starting
    pl_range <- numeric(length(phi_range))
    for (i in seq_along(phi_range)) {
      result <- profile_loglik_nested_gamma(
        RAW_PHI = log(phi_range[i]),
        GAMMA_START = gamma_start,
        cpp_args = cpp_args,
        lbfgs_control = lbfgs_control
      )
      pl_range[i] <- result$loglik
      # Warm-start next iteration
      gamma_start <- result$gamma_opt
    }
  } else {
    # For continuous data or ordinal with fixed thresholds
    pl_range <- sapply(phi_range, function(phi) {
      cpp_profile_likelihood(
        Y = args$Y,
        ITEM_INDS = args$ITEM_INDS,
        WORKER_INDS = args$WORKER_INDS,
        ALPHA_START = args$ALPHA_START,
        BETA_START = args$BETA_START,
        TAU_START = X$tau,
        PHI = phi,
        J = args$J,
        W = args$W,
        K = K,
        DATA_TYPE = data_type,
        ITEMS_NUISANCE = args$ITEMS_NUISANCE,
        WORKER_NUISANCE = args$WORKER_NUISANCE,
        THRESHOLDS_NUISANCE = FALSE,
        PROF_SEARCH_RANGE = args$PROF_SEARCH_RANGE,
        PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
        ALT_MAX_ITER = as.integer(args$ALT_MAX_ITER),
        ALT_TOL = args$ALT_TOL
      )
    })
  }

  if (PLOT) {
    plot(
      agreement_range,
      pl_range - max(pl_range),
      type = "l",
      xlab = "Agreement",
      ylab = "Relative log-likelihood",
      col = 1
    )

    abline(v = X$profile$agreement, col = 1, lty = 2)
  }

  mpl_range <- rep(NA, length(phi_range))
  if (X$method == "modified") {
    # Compute modified profile likelihood over grid
    if (data_type == "ordinal" && "thresholds" %in% X$params_type$nuisance) {
      # Initialize gamma from fitted tau
      gamma_start <- tau2gamma(args$TAU_START)

      # Extract MLE values
      alpha_mle <- X$alpha
      beta_mle <- X$beta
      tau_mle <- X$tau
      phi_mle <- X$profile$precision

      # Compute modified profile likelihood with warm-starting
      for (i in seq_along(phi_range)) {
        result <- modified_profile_loglik_nested_gamma(
          RAW_PHI = log(phi_range[i]),
          GAMMA_START = gamma_start,
          ALPHA_MLE = alpha_mle,
          BETA_MLE = beta_mle,
          TAU_MLE = tau_mle,
          PHI_MLE = phi_mle,
          cpp_args = cpp_args,
          lbfgs_control = lbfgs_control
        )
        mpl_range[i] <- result$loglik
        # Warm-start next iteration
        # gamma_start <- result$gamma_opt
      }
    } else {
      # For continuous data or ordinal with fixed thresholds
      mpl_range <- sapply(phi_range, function(phi) {
        cpp_modified_profile_likelihood_extended(
          Y = args$Y,
          ITEM_INDS = args$ITEM_INDS,
          WORKER_INDS = args$WORKER_INDS,
          ALPHA_MLE = X$alpha,
          BETA_MLE = X$beta,
          TAU = X$tau,
          TAU_MLE = X$tau,
          PHI = phi,
          PHI_MLE = X$profile$precision,
          J = args$J,
          W = args$W,
          K = K,
          DATA_TYPE = data_type,
          ITEMS_NUISANCE = args$ITEMS_NUISANCE,
          WORKER_NUISANCE = args$WORKER_NUISANCE,
          PROF_SEARCH_RANGE = args$PROF_SEARCH_RANGE,
          PROF_UNI_MAX_ITER = as.integer(args$PROF_MAX_ITER),
          ALT_MAX_ITER = as.integer(args$ALT_MAX_ITER),
          ALT_TOL = args$ALT_TOL
        )
      })
    }

    if (PLOT && !all(is.na(mpl_range))) {
      lines(agreement_range, mpl_range - max(mpl_range, na.rm = TRUE), col = 2)
      abline(v = X$modified$agreement, col = 2, lty = 2)
      legend(
        "bottomleft",
        legend = c("Profile likelihood", "Modified profile likelihood"),
        col = c(1, 2),
        lty = c(1, 1),
        bty = "n"
      )
    }
  }

  out <- data.frame(
    precision = phi_range,
    agreement = agreement_range,
    profile_ll = pl_range,
    modified_ll = mpl_range
  )

  return(out)
}

#' Get Agreement confidence interval
#'
#' @param X Object fitted with [agreement()] function.
#' @param CONFIDENCE Confidence level.
#'
#' @return Returns a list with estimate, standard error and confidence interval
#'
#' @importFrom stats qnorm
#' @export
get_ci <- function(X, CONFIDENCE = 0.95) {
  stopifnot(is.numeric(CONFIDENCE))
  stopifnot(CONFIDENCE < 1)
  stopifnot(CONFIDENCE > 0)

  mle <- X$profile$precision

  agr_se <- NA
  est <- NA
  if (X$method == "modified") {
    agr_se <- cpp_get_se(
      Y = X$cpp_args$Y,
      ITEM_INDS = X$cpp_args$ITEM_INDS,
      ALPHA_START = X$cpp_args$ALPHA_START,
      PHI_MLE = mle,
      PHI_EVAL = X$modified$precision,
      K = X$cpp_args$K,
      J = X$cpp_args$J,
      SEARCH_RANGE = X$cpp_args$SEARCH_RANGE,
      MAX_ITER = X$cpp_args$MAX_ITER,
      PROF_SEARCH_RANGE = X$cpp_args$PROF_SEARCH_RANGE,
      PROF_MAX_ITER = X$cpp_args$PROF_MAX_ITER,
      PROF_METHOD = X$cpp_args$PROF_METHOD,
      CONTINUOUS = X$cpp_args$CONTINUOUS,
      MODIFIED = TRUE
    )
    est <- X$modified$agreement
  } else {
    agr_se <- cpp_get_se(
      Y = X$cpp_args$Y,
      ITEM_INDS = X$cpp_args$ITEM_INDS,
      ALPHA_START = X$cpp_args$ALPHA_START,
      PHI_MLE = mle,
      PHI_EVAL = mle,
      K = X$cpp_args$K,
      J = X$cpp_args$J,
      SEARCH_RANGE = X$cpp_args$SEARCH_RANGE,
      MAX_ITER = X$cpp_args$MAX_ITER,
      PROF_SEARCH_RANGE = X$cpp_args$PROF_SEARCH_RANGE,
      PROF_MAX_ITER = X$cpp_args$PROF_MAX_ITER,
      PROF_METHOD = X$cpp_args$PROF_METHOD,
      CONTINUOUS = X$cpp_args$CONTINUOUS,
      MODIFIED = FALSE
    )

    est <- X$profile$agreement
  }

  alpha <- 1 - CONFIDENCE
  z <- qnorm(1 - alpha / 2)

  return(list(
    agreement_est = est,
    agreement_se = agr_se,
    agreement_ci = c(est - z * agr_se, est + z * agr_se)
  ))
}
