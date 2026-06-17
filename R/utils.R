#' @noRd
init_alpha <- function(Y, ITEM_INDS, J, LO, HI, K = NULL) {
  ids <- seq_len(J)
  means <- as.numeric(tapply(Y, ITEM_INDS, mean)[ids])
  if (!is.null(K)) {
    means <- (means - 0.5) / K   # maps ordinal mean from [1, K] to (0.5/K, (K-0.5)/K)
  }
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

    h <- sqrt(.Machine$double.eps) * abs(phi_est)
    agr_se <- abs(
      (prec2agr(phi_est + h) - prec2agr(phi_est - h)) / (2 * h)
    ) * object$se[["phi"]]

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


#' Model-based probability of item degeneracy
#'
#' For each item, computes the probability that all raters give the same rating
#' according to the fitted model. Returns 0 for continuous fits (exact ties
#' have zero probability under a continuous distribution). Not defined for
#' two-way models (raises an error).
#'
#' @param object An `agreement_fit` object from [agreement()].
#'
#' @return A named numeric vector of length J (all items, including any
#'   degenerate items detected before fitting). Degenerate items always get
#'   probability 1. Names are `item_1, ..., item_J` or `item_<label>` when
#'   item labels are available.
#'
#' @importFrom stats pbeta plogis
#' @export
prob_degenerate <- function(object) {
  stopifnot(inherits(object, "agreement_fit"))

  if ("workers" %in% object$params_type$nuisance) {
    stop("prob_degenerate() is not defined for two-way models.")
  }

  n_total   <- object$data$n_items
  degen_ids <- object$data$degen_ids
  non_degen <- setdiff(seq_len(n_total), degen_ids)
  B         <- as.integer(table(object$data$item_ids))

  item_nms <- if (!is.null(object$data$item_labels)) {
    paste0("item_", object$data$item_labels)
  } else {
    paste0("item_", seq_len(n_total))
  }

  out <- setNames(rep(0, n_total), item_nms)

  if (object$data_type == "continuous") return(out)

  out[degen_ids] <- 1

  phi <- unname(
    if (object$method == "modified") object$modified$precision
    else object$profile$precision
  )
  alpha_j <- object$alpha

  if (object$data_type == "ordinal") {
    tau <- object$tau
    for (i in seq_along(non_degen)) {
      j   <- non_degen[i]
      mu  <- plogis(alpha_j[i])
      p_c <- diff(pbeta(tau, mu * phi, (1 - mu) * phi))
      out[j] <- sum(p_c ^ B[j])
    }
  } else {
    k0 <- unname(object$k0)
    k1 <- unname(object$k1)
    for (i in seq_along(non_degen)) {
      j  <- non_degen[i]
      a  <- alpha_j[i]
      p0 <- 1 - plogis(a - k0)
      p1 <- plogis(a - k1)
      out[j] <- p0 ^ B[j] + p1 ^ B[j]
    }
  }
  out
}


#' Confidence intervals for model-based probability of item degeneracy
#'
#' Applies the delta method to `prob_degenerate()`, propagating parameter
#' uncertainty to per-item probabilities. Item intercepts α_j are treated as
#' fixed at their MLE (plug-in). Not defined for two-way models.
#'
#' @param object An `agreement_fit` object from [agreement()].
#' @param level Confidence level. Default `0.95`.
#'
#' @return A named matrix with one row per item and columns
#'   `Estimate`, `Std. Error`, and the two percentile bounds.
#'
#' @importFrom stats pbeta plogis qlogis optimize qnorm
#' @export
confint_prob_degenerate <- function(object, level = 0.95) {
  stopifnot(inherits(object, "agreement_fit"))

  if ("workers" %in% object$params_type$nuisance) {
    stop("confint_prob_degenerate() is not defined for two-way models.")
  }

  z   <- qnorm(1 - (1 - level) / 2)
  pct <- paste0(formatC(c((1 - level) / 2, 1 - (1 - level) / 2) * 100,
                        format = "g"), "%")

  pd        <- prob_degenerate(object)
  n_total   <- object$data$n_items
  degen_ids <- object$data$degen_ids
  non_degen <- setdiff(seq_len(n_total), degen_ids)
  B         <- as.integer(table(object$data$item_ids))

  se_vec            <- rep(NA_real_, n_total)
  se_vec[degen_ids] <- 0

  phi <- unname(
    if (object$method == "modified") object$modified$precision
    else object$profile$precision
  )

  if (object$data_type == "continuous") {
    se_vec[] <- 0

  } else if (object$data_type == "ordinal") {
    d   <- object$fit_data
    ctl <- object$control
    agr_se <- cpp_get_se(
      Y          = d$ratings,
      ITEM_INDS  = as.integer(d$item_ids),
      WORKER_INDS = if (!is.null(d$worker_ids)) as.integer(d$worker_ids)
                    else rep(1L, length(d$ratings)),
      ALPHA_MLE  = object$alpha,
      BETA_MLE   = object$beta,
      TAU_MLE    = object$tau,
      PHI_EVAL   = phi,
      PHI_MLE    = unname(object$profile$precision),
      J          = d$n_items,
      W          = if (!is.null(d$n_workers)) d$n_workers else 1L,
      K          = d$K,
      METHOD     = object$method,
      DATA_TYPE  = object$data_type,
      ITEMS_NUISANCE  = "items" %in% object$params_type$nuisance,
      WORKER_NUISANCE = "workers" %in% object$params_type$nuisance,
      PROF_SEARCH_RANGE = as.integer(ctl$PROF_SEARCH_RANGE),
      PROF_MAX_ITER     = as.integer(ctl$PROF_MAX_ITER),
      ALT_MAX_ITER      = as.integer(ctl$ALT_MAX_ITER),
      ALT_TOL           = ctl$ALT_TOL
    )
    h_phi  <- sqrt(.Machine$double.eps) * phi
    phi_se <- agr_se /
      abs((prec2agr(phi + h_phi) - prec2agr(phi - h_phi)) / (2 * h_phi))

    tau     <- object$tau
    alpha_j <- object$alpha
    h       <- sqrt(.Machine$double.eps) * abs(phi)

    for (i in seq_along(non_degen)) {
      j    <- non_degen[i]
      mu   <- plogis(alpha_j[i])
      P_ord <- function(pv) sum(diff(pbeta(tau, mu * pv, (1 - mu) * pv)) ^ B[j])
      se_vec[j] <- abs((P_ord(phi + h) - P_ord(phi - h)) / (2 * h)) * phi_se
    }

  } else {
    k0      <- unname(object$k0)
    k1      <- unname(object$k1)
    vc      <- object$vcov
    alpha_j <- object$alpha
    h       <- sqrt(.Machine$double.eps)

    for (i in seq_along(non_degen)) {
      j    <- non_degen[i]
      a    <- alpha_j[i]
      Bj   <- B[j]
      P_inf <- function(k0v, k1v) (1 - plogis(a - k0v)) ^ Bj + plogis(a - k1v) ^ Bj
      g  <- c(
        0,
        (P_inf(k0 + h, k1) - P_inf(k0 - h, k1)) / (2 * h),
        (P_inf(k0, k1 + h) - P_inf(k0, k1 - h)) / (2 * h)
      )
      se_vec[j] <- sqrt(drop(g %*% vc %*% g))
    }
  }

  eps_p <- .Machine$double.eps
  lower <- ifelse(
    se_vec == 0, pd,
    plogis(qlogis(pmax(pd, eps_p)) - z * se_vec / pmax(pd * (1 - pd), eps_p))
  )
  upper <- ifelse(
    se_vec == 0, pd,
    plogis(qlogis(pmin(pd, 1 - eps_p)) + z * se_vec / pmax(pd * (1 - pd), eps_p))
  )
  out <- cbind(pd, se_vec, lower, upper)
  colnames(out) <- c("Estimate", "Std. Error", pct)
  out
}
