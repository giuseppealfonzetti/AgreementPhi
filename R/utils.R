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

  agreement_range <- seq(
    max(X$pl_agreement - RANGE, 1e-2),
    min(X$pl_agreement + RANGE, 1 - 1e-2),
    length.out = GRID_LENGTH
  )
  phi_range <- sapply(agreement_range, agr2prec)

  pl_range <- sapply(phi_range, function(x) {
    cpp_profile_likelihood(
      Y = X$cpp_args$Y,
      ITEM_INDS = X$cpp_args$ITEM_INDS,
      ALPHA_START = X$cpp_args$ALPHA_START,
      PHI = x,
      K = X$cpp_args$K,
      J = X$cpp_args$J,
      PROF_SEARCH_RANGE = X$cpp_args$PROF_SEARCH_RANGE,
      PROF_MAX_ITER = X$cpp_args$PROF_MAX_ITER,
      PROF_METHOD = X$cpp_args$PROF_METHOD,
      CONTINUOUS = X$cpp_args$CONTINUOUS
    )
  })

  if (PLOT) {
    plot(
      agreement_range,
      pl_range - max(pl_range),
      type = "l",
      xlab = "Agreement",
      ylab = "Relative log-likelihood",
      col = 1
    )

    abline(v = X$pl_agreement, col = 1, lty = 2)
  }

  mpl_range <- NA
  if (X$method == "modified") {
    mpl_range <- sapply(phi_range, function(x) {
      cpp_modified_profile_likelihood(
        Y = X$cpp_args$Y,
        ITEM_INDS = X$cpp_args$ITEM_INDS,
        ALPHA_START = X$cpp_args$ALPHA_START,
        PHI = x,
        PHI_MLE = X$pl_precision,
        K = X$cpp_args$K,
        J = X$cpp_args$J,
        SEARCH_RANGE = X$cpp_args$SEARCH_RANGE,
        MAX_ITER = X$cpp_args$MAX_ITER,
        PROF_SEARCH_RANGE = X$cpp_args$PROF_SEARCH_RANGE,
        PROF_MAX_ITER = X$cpp_args$PROF_MAX_ITER,
        PROF_METHOD = X$cpp_args$PROF_METHOD,
        CONTINUOUS = X$cpp_args$CONTINUOUS
      )
    })

    if (PLOT) {
      lines(agreement_range, mpl_range - max(mpl_range), col = 2)
      abline(v = X$mpl_agreement, col = 2, lty = 2)
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
    profile_rll = pl_range,
    modified_rll = mpl_range
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

  mle <- X$pl_precision

  agr_se <- NA
  est <- NA
  if (X$method == "modified") {
    agr_se <- cpp_get_se(
      Y = X$cpp_args$Y,
      ITEM_INDS = X$cpp_args$ITEM_INDS,
      ALPHA_START = X$cpp_args$ALPHA_START,
      PHI_MLE = mle,
      PHI_EVAL = X$mpl_precision,
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
    est <- X$mpl_agreement
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

    est <- X$pl_agreement
  }

  alpha <- 1 - CONFIDENCE
  z <- qnorm(1 - alpha / 2)

  return(list(
    agreement_est = est,
    agreement_se = agr_se,
    agreement_ci = c(est - z * agr_se, est + z * agr_se)
  ))
}
