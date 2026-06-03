#' @noRd
init_alpha <- function(Y, ITEM_INDS, J, LO, HI) {
  ids <- seq_len(J)
  means <- as.numeric(tapply(Y, ITEM_INDS, mean)[ids])
  means <- pmax(pmin(means, 1 - 1e-6), 1e-6)
  alpha <- stats::qlogis(means)
  pmax(pmin(alpha, HI), LO)
}

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
  if (isTRUE(X$data_type == "inflated")) {
    stop("get_range_ll() is not available for inflated interval data.")
  }
  stopifnot(is.numeric(RANGE))
  stopifnot(RANGE > 0)
  stopifnot(RANGE <= 1)
  stopifnot(GRID_LENGTH > 0)

  agreement_range <- seq(
    max(X$profile$agreement - RANGE, 1e-2),
    min(X$profile$agreement + RANGE, 1 - 1e-2),
    length.out = GRID_LENGTH
  )
  phi_range <- sapply(agreement_range, agr2prec)

  d <- X$fit_data
  ctl <- X$control
  worker_inds <- if (!is.null(d$worker_ids)) {
    as.integer(d$worker_ids)
  } else {
    integer(0)
  }
  items_nuisance <- "items" %in% X$params_type$nuisance
  worker_nuisance <- "workers" %in% X$params_type$nuisance
  W <- if (!is.null(d$n_workers)) d$n_workers else 1L

  pl_range <- sapply(phi_range, function(phi) {
    cpp_profile_likelihood(
      Y = d$ratings,
      ITEM_INDS = as.integer(d$item_ids),
      WORKER_INDS = worker_inds,
      ALPHA_START = X$alpha,
      BETA_START = X$beta,
      TAU_START = X$tau,
      PHI = phi,
      J = d$n_items,
      W = W,
      K = d$K,
      DATA_TYPE = X$data_type,
      ITEMS_NUISANCE = items_nuisance,
      WORKER_NUISANCE = worker_nuisance,
      PROF_SEARCH_RANGE = ctl$PROF_SEARCH_RANGE,
      PROF_UNI_MAX_ITER = as.integer(ctl$PROF_MAX_ITER),
      ALT_MAX_ITER = as.integer(ctl$ALT_MAX_ITER),
      ALT_TOL = ctl$ALT_TOL
    )
  })

  mpl_range <- rep(NA, length(phi_range))
  if (X$method == "modified") {
    mpl_range <- sapply(phi_range, function(phi) {
      cpp_modified_profile_likelihood(
        Y = d$ratings,
        ITEM_INDS = as.integer(d$item_ids),
        WORKER_INDS = worker_inds,
        ALPHA_MLE = X$alpha,
        BETA_MLE = X$beta,
        TAU = X$tau,
        PHI = phi,
        PHI_MLE = X$profile$precision,
        J = d$n_items,
        W = W,
        K = d$K,
        DATA_TYPE = X$data_type,
        ITEMS_NUISANCE = items_nuisance,
        WORKER_NUISANCE = worker_nuisance,
        PROF_SEARCH_RANGE = ctl$PROF_SEARCH_RANGE,
        PROF_UNI_MAX_ITER = as.integer(ctl$PROF_MAX_ITER),
        ALT_MAX_ITER = as.integer(ctl$ALT_MAX_ITER),
        ALT_TOL = ctl$ALT_TOL
      )
    })
  }

  data.frame(
    precision = phi_range,
    agreement = agreement_range,
    profile = pl_range,
    modified = mpl_range
  )
}

#' Confidence intervals for an agreement fit
#'
#' @param object Object fitted with [agreement()].
#' @param parm Ignored (included for S3 compatibility).
#' @param level Confidence level. Default `0.95`.
#' @param ... Ignored.
#'
#' @return A named list with two elements, each a numeric matrix with columns
#'   `Estimate`, `Std. Error`, and the lower/upper confidence bounds:
#'   \describe{
#'     \item{`parameters`}{Parameter-scale estimates. One row (`phi`) for
#'       non-inflated data; three rows (`phi`, `k0`, `k1`) for inflated data.}
#'     \item{`agreement`}{Agreement-scale estimate. Always one row (`agreement`).}
#'   }
#'
#' @importFrom stats confint qnorm
#' @export
confint.agreement_fit <- function(object, parm = NULL, level = 0.95, ...) {
  stopifnot(is.numeric(level), level > 0, level < 1)
  z <- stats::qnorm(1 - (1 - level) / 2)
  pct <- paste0(
    format(100 * c((1 - level) / 2, 1 - (1 - level) / 2), trim = TRUE),
    " %"
  )

  make_mat <- function(est, se, lower, upper, row_nms) {
    m <- cbind(est, se, lower, upper)
    colnames(m) <- c("Estimate", "Std. Error", pct)
    rownames(m) <- row_nms
    m
  }

  if (isTRUE(object$data_type == "inflated")) {
    phi_est <- if (object$method == "modified") {
      object$modified$precision
    } else {
      object$profile$precision
    }
    agr_est <- if (object$method == "modified") {
      object$modified$agreement
    } else {
      object$profile$agreement
    }
    se <- object$se

    n_dropped <- object$data$n_items - object$fit_data$n_items

    h <- sqrt(.Machine$double.eps)
    par2agr_agr <- function(phi, k0, k1) {
      par2agr(
        phi,
        ALPHA = object$alpha,
        K0 = k0,
        K1 = k1,
        ADJUST = isTRUE(object$adjust),
        N_DEGENERATE = n_dropped
      )$agreement
    }
    grad <- c(
      (par2agr_agr(phi_est + h * abs(phi_est), object$k0, object$k1) -
        par2agr_agr(phi_est - h * abs(phi_est), object$k0, object$k1)) /
        (2 * h * abs(phi_est)),
      (par2agr_agr(phi_est, object$k0 + h, object$k1) -
        par2agr_agr(phi_est, object$k0 - h, object$k1)) /
        (2 * h),
      (par2agr_agr(phi_est, object$k0, object$k1 + h) -
        par2agr_agr(phi_est, object$k0, object$k1 - h)) /
        (2 * h)
    )
    agr_se <- sqrt(drop(grad %*% object$vcov %*% grad))

    return(list(
      parameters = make_mat(
        est = c(phi_est, object$k0, object$k1),
        se = c(se[["phi"]], se[["k0"]], se[["k1"]]),
        lower = c(
          max(0, phi_est - z * se[["phi"]]),
          object$k0 - z * se[["k0"]],
          object$k1 - z * se[["k1"]]
        ),
        upper = c(
          phi_est + z * se[["phi"]],
          object$k0 + z * se[["k0"]],
          object$k1 + z * se[["k1"]]
        ),
        row_nms = c("phi", "k0", "k1")
      ),
      agreement = make_mat(
        est = agr_est,
        se = agr_se,
        lower = max(0, agr_est - z * agr_se),
        upper = min(1, agr_est + z * agr_se),
        row_nms = "agreement"
      )
    ))
  }

  d <- object$fit_data
  ctl <- object$control
  phi_mle <- object$profile$precision

  if (object$method == "modified") {
    phi_eval <- object$modified$precision
    est <- object$modified$agreement
  } else {
    phi_eval <- phi_mle
    est <- object$profile$agreement
  }

  agr_se <- cpp_get_se(
    Y = d$ratings,
    ITEM_INDS = as.integer(d$item_ids),
    WORKER_INDS = if (!is.null(d$worker_ids)) {
      as.integer(d$worker_ids)
    } else {
      rep(1L, length(d$ratings))
    },
    ALPHA_MLE = object$alpha,
    BETA_MLE = object$beta,
    TAU_MLE = object$tau,
    PHI_EVAL = phi_eval,
    PHI_MLE = phi_mle,
    J = d$n_items,
    W = if (!is.null(d$n_workers)) d$n_workers else 1L,
    K = d$K,
    METHOD = object$method,
    DATA_TYPE = object$data_type,
    ITEMS_NUISANCE = "items" %in% object$params_type$nuisance,
    WORKER_NUISANCE = "workers" %in% object$params_type$nuisance,
    PROF_SEARCH_RANGE = as.integer(ctl$PROF_SEARCH_RANGE),
    PROF_MAX_ITER = as.integer(ctl$PROF_MAX_ITER),
    ALT_MAX_ITER = as.integer(ctl$ALT_MAX_ITER),
    ALT_TOL = ctl$ALT_TOL
  )

  h <- sqrt(.Machine$double.eps) * phi_eval
  dagr_dphi <- (prec2agr(phi_eval + h) - prec2agr(phi_eval - h)) / (2 * h)
  phi_se <- agr_se / abs(dagr_dphi)

  fit_J <- object$fit_data$n_items
  n_dropped <- object$data$n_items - fit_J
  if (isTRUE(object$adjust) && n_dropped > 0) {
    agr_se <- agr_se * fit_J / (fit_J + n_dropped)
  }

  list(
    parameters = make_mat(
      est = phi_eval,
      se = phi_se,
      lower = max(0, phi_eval - z * phi_se),
      upper = phi_eval + z * phi_se,
      row_nms = "phi"
    ),
    agreement = make_mat(
      est = est,
      se = agr_se,
      lower = max(0, est - z * agr_se),
      upper = min(1, est + z * agr_se),
      row_nms = "agreement"
    )
  )
}


#' From model parameters to agreement
#'
#' @param PHI dispersion parameter
#' @param ALPHA item-specific intercepts
#' @param BETA worker-specific intercepts
#' @param K0 zero-inflation threshold
#' @param K1 one-inflation threshold
#' @param ADJUST logical; if `TRUE`, degenerate items (dropped from estimation,
#'   i.e. not in `ALPHA`) are included in the overall mean with a unit
#'   contribution. Requires `ALPHA`.
#' @param N_DEGENERATE number of degenerate items dropped before estimation.
#'   Used only when `ADJUST = TRUE`.
#'
#' @return return agreement measure according to the estimated parameters
#'
#' @export
par2agr <- function(
  PHI,
  ALPHA = NULL,
  BETA = NULL,
  K0 = NULL,
  K1 = NULL,
  ADJUST = FALSE,
  N_DEGENERATE = 0
) {
  out <- list()

  if (is.null(ALPHA)) {
    out$agreement <- prec2agr(PHI)
    return(out)
  }

  fit_J <- length(ALPHA)

  if (is.null(K0) & is.null(K1)) {
    agr_i <- rep(prec2agr(PHI), fit_J)
  } else {
    eps <- .Machine$double.eps^0.5
    K0_eff <- if (!is.finite(K0)) -100 else K0
    K1_eff <- if (!is.finite(K1)) 100 else K1
    L0_i <- plogis(ALPHA - K0_eff)
    L1_i <- plogis(ALPHA - K1_eff)
    p0_i <- 1 - L0_i
    p1_i <- L1_i
    pc_i <- L0_i - L1_i
    mu_i <- plogis(ALPHA)
    m_i <- p1_i + pc_i * mu_i
    vb_i <- mu_i * (1 - mu_i) / (PHI + 1)
    V_i <- pc_i *
      vb_i +
      p0_i * m_i^2 +
      p1_i * (1 - m_i)^2 +
      pc_i * (mu_i - m_i)^2
    pe_i <- ifelse(V_i <= eps, Inf, m_i * (1 - m_i) / V_i - 1)
    agr_i <- prec2agr(pmax(0, pe_i))
  }

  out$agreement_by_item <- agr_i
  out$agreement <- if (ADJUST && N_DEGENERATE > 0) {
    (fit_J * mean(agr_i) + N_DEGENERATE) / (fit_J + N_DEGENERATE)
  } else {
    mean(agr_i)
  }
  return(out)
}
