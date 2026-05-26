#' From precision to agreement
#'
#' @param X Precision parameter.
#'
#' @return General agreement.
#'
#' @export
prec2agr <- function(X) {
  stopifnot(X >= 0)
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


#' Discretise continuous data
#'
#' @param X Vector of continuous data in (0,1).
#' @param K Number of ordinal categories.
#' @param TRESHOLDS Threshold vector of length K-1. If null, thresholds are assumed to be equispaced.
#'
#' @return Discretised vector
#'
#' @examples
#' x <- c(0,runif(5,0,1),1)
#' x
#' cont2ord(x, K=3)
#'
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

#' Squeeze \[0,1\] data
#'
#' @param X Vector of continuous data in \[0,1\].
#' @param U Squeezing parameter. If NULL, default chosen as per Smithson et al. (2006).
#'
#' @return Squeezed vector
#'
#' @references
#'
#' - Smithson, Michael, and Jay Verkuilen. 2006. “A Better Lemon Squeezer? Maximum-Likelihood Regression with Beta-Distributed Dependent Variables.” *Psychological Methods* **11(1)**: 54–71. [doi](https://psycnet.apa.org/doi/10.1037/1082-989X.11.1.54)
#'
#' @examples
#' x <- c(0,runif(5,0,1),1)
#' x
#' lemon(x)
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

#' Get log-likelihood range
#'
#' @param X Object fitted with [agreement()] function.
#' @param RANGE Range around agreement mle.
#' @param GRID_LENGTH Number of points to be evaluated within RANGE.
#'
#' @return Return a data.frame with GRID_LENGTH rows and columns
#' `precision`, `agreement`, `profile` and `modified`.
#'
#' @importFrom stats plogis rbeta
#' @export
get_range_ll <- function(X, RANGE = .2, GRID_LENGTH = 15) {
  stopifnot(is.numeric(RANGE))
  stopifnot(RANGE > 0)
  stopifnot(RANGE <= 1)
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
  data_type <- args$DATA_TYPE
  K <- args$K

  # Compute profile likelihood over grid
  pl_range <- sapply(phi_range, function(phi) {
    cpp_profile_likelihood(
      Y = args$Y,
      ITEM_INDS = as.integer(args$ITEM_INDS),
      WORKER_INDS = if (!is.null(args$WORKER_INDS)) {
        as.integer(args$WORKER_INDS)
      } else {
        integer(0)
      },
      ALPHA_START = args$ALPHA_START,
      BETA_START = args$BETA_START,
      TAU_START = X$tau,
      PHI = phi,
      J = args$J,
      W = if (!is.null(args$W)) args$W else 1L,
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

  mpl_range <- rep(NA, length(phi_range))
  if (X$method == "modified") {
    mpl_range <- sapply(phi_range, function(phi) {
      cpp_modified_profile_likelihood(
        Y = args$Y,
        ITEM_INDS = as.integer(args$ITEM_INDS),
        WORKER_INDS = if (!is.null(args$WORKER_INDS)) {
          as.integer(args$WORKER_INDS)
        } else {
          integer(0)
        },
        ALPHA_MLE = X$alpha,
        BETA_MLE = X$beta,
        TAU = X$tau,
        PHI = phi,
        PHI_MLE = X$profile$precision,
        J = args$J,
        W = if (!is.null(args$W)) args$W else 1L,
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

  out <- data.frame(
    precision = phi_range,
    agreement = agreement_range,
    profile = pl_range,
    modified = mpl_range
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
  stopifnot(is.numeric(CONFIDENCE), CONFIDENCE < 1, CONFIDENCE > 0)

  if (isTRUE(X$data_type == "inflated")) {
    phi_est <- if (X$method == "modified") {
      X$modified$precision
    } else {
      X$profile$precision
    }
    z <- stats::qnorm(1 - (1 - CONFIDENCE) / 2)
    phi_ci <- function(est, se) c(max(0, est - z * se), est + z * se)
    sym_ci <- function(est, se) c(est - z * se, est + z * se)
    return(list(
      phi_est = phi_est,
      phi_se = X$se[["phi"]],
      phi_ci = phi_ci(phi_est, X$se[["phi"]]),
      k0_est = X$k0,
      k0_se = X$se[["k0"]],
      k0_ci = sym_ci(X$k0, X$se[["k0"]]),
      k1_est = X$k1,
      k1_se = X$se[["k1"]],
      k1_ci = sym_ci(X$k1, X$se[["k1"]])
    ))
  }

  args <- X$cpp_args

  # MLE values from fitted object
  alpha_mle <- X$alpha
  beta_mle <- X$beta
  tau_mle <- if (!is.null(X$tau)) X$tau else args$TAU_START
  phi_mle <- X$profile$precision

  # Determine evaluation point
  if (X$method == "modified") {
    phi_eval <- X$modified$precision
    est <- X$modified$agreement
  } else {
    phi_eval <- phi_mle
    est <- X$profile$agreement
  }

  agr_se <- cpp_get_se(
    Y = args$Y,
    ITEM_INDS = as.integer(args$ITEM_INDS),
    WORKER_INDS = if (!is.null(args$WORKER_INDS)) {
      as.integer(args$WORKER_INDS)
    } else {
      integer(0)
    },
    ALPHA_MLE = alpha_mle,
    BETA_MLE = beta_mle,
    TAU_MLE = tau_mle,
    PHI_EVAL = phi_eval,
    PHI_MLE = phi_mle,
    J = args$J,
    W = if (!is.null(args$W)) args$W else 1L,
    K = args$K,
    METHOD = args$METHOD,
    DATA_TYPE = args$DATA_TYPE,
    ITEMS_NUISANCE = args$ITEMS_NUISANCE,
    WORKER_NUISANCE = args$WORKER_NUISANCE,
    PROF_SEARCH_RANGE = as.integer(args$PROF_SEARCH_RANGE),
    PROF_MAX_ITER = as.integer(args$PROF_MAX_ITER),
    ALT_MAX_ITER = as.integer(args$ALT_MAX_ITER),
    ALT_TOL = args$ALT_TOL
  )

  alpha <- 1 - CONFIDENCE
  z <- qnorm(1 - alpha / 2)

  return(list(
    agreement_est = est,
    agreement_se = agr_se,
    agreement_ci = c(max(0, est - z * agr_se), min(1, est + z * agr_se))
  ))
}


#' From model parameters to agreement
#'
#' @param PHI dispersion parameter
#' @param ALPHA item-specific intercepts
#' @param BETA worker-specific intercepts
#' @param K0 zero-inflation threshold
#' @param K1 one-inflation threshold
#'
#' @return return agreement measure according to the estimated parameters
#'
#' @export
par2agr <- function(PHI, ALPHA = NULL, BETA = NULL, K0 = NULL, K1 = NULL) {
  out <- list()
  if (is.null(K0) & is.null(K1)) {
    out$agreement <- prec2agr(PHI)
    return(out)
  }

  stopifnot(!is.null(ALPHA))
  eps    <- .Machine$double.eps^0.5
  K0_eff <- if (!is.finite(K0)) -100 else K0
  K1_eff <- if (!is.finite(K1)) 100  else K1
  L0_i <- plogis(ALPHA - K0_eff)
  L1_i <- plogis(ALPHA - K1_eff)
  p0_i <- 1 - L0_i
  p1_i <- L1_i
  pc_i <- L0_i - L1_i
  mu_i <- plogis(ALPHA)
  m_i <- p1_i + pc_i * mu_i
  vb_i <- mu_i * (1 - mu_i) / (PHI + 1)
  V_i <- pc_i * vb_i + p0_i * m_i^2 + p1_i * (1 - m_i)^2 + pc_i * (mu_i - m_i)^2
  pe_i <- ifelse(V_i <= eps, Inf, m_i * (1 - m_i) / V_i - 1)
  agr_i <- prec2agr(pmax(0, pe_i))
  out$agreement_by_item <- agr_i
  out$agreement <- mean(agr_i)
  return(out)
}
