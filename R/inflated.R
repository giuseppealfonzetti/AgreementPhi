#' @noRd
init_alpha <- function(Y, ITEM_INDS, J, LO, HI) {
  ids <- seq_len(J)
  means <- as.numeric(tapply(Y, ITEM_INDS, mean)[ids])
  means <- pmax(pmin(means, 1 - 1e-6), 1e-6)
  alpha <- stats::qlogis(means)
  pmax(pmin(alpha, HI), LO)
}

#' @noRd
init_cutpoints <- function(Y) {
  p0 <- mean(Y == 0)
  p1 <- mean(Y == 1)
  k0 <- if (p0 > 0) stats::qlogis(pmax(pmin(p0, 1 - 1e-4), 1e-4)) else -1
  k1 <- if (p1 > 0) -stats::qlogis(pmax(pmin(p1, 1 - 1e-4), 1e-4)) else 1
  c(k0 = k0, k1 = k1)
}

#' @noRd
inflated_profile_cpp <- function(
  Y,
  ITEM_INDS,
  J,
  ALPHA_START,
  PHI,
  K0,
  K1,
  LOWER_ALPHA = -5,
  UPPER_ALPHA = 5,
  PROF_MAX_ITER = 500L
) {
  cpp_inflated_profile(
    as.numeric(Y),
    as.integer(ITEM_INDS),
    as.numeric(ALPHA_START),
    PHI,
    K0,
    K1,
    J,
    LOWER_ALPHA,
    UPPER_ALPHA,
    as.integer(PROF_MAX_ITER)
  )
}

#' @noRd
inflated_mpl_cpp <- function(
  Y,
  ITEM_INDS,
  J,
  ALPHA_START,
  ALPHA_MLE,
  PHI,
  K0,
  K1,
  PHI_MLE,
  K0_MLE,
  K1_MLE,
  LOWER_ALPHA = -5,
  UPPER_ALPHA = 5,
  PROF_MAX_ITER = 500L
) {
  cpp_inflated_mpl(
    as.numeric(Y),
    as.integer(ITEM_INDS),
    as.numeric(ALPHA_START),
    as.numeric(ALPHA_MLE),
    PHI,
    K0,
    K1,
    PHI_MLE,
    K0_MLE,
    K1_MLE,
    J,
    LOWER_ALPHA,
    UPPER_ALPHA,
    as.integer(PROF_MAX_ITER)
  )
}

#' @noRd
inflated_fd_hessian <- function(FN, PAR, STEP = 1e-4) {
  p <- length(PAR)
  H <- matrix(NA_real_, p, p)
  f0 <- FN(PAR)
  h <- STEP * (abs(PAR) + 1)
  for (i in seq_len(p)) {
    ei <- replace(numeric(p), i, h[i])
    H[i, i] <- (FN(PAR + ei) - 2 * f0 + FN(PAR - ei)) / h[i]^2
    if (i < p) {
      for (j in (i + 1):p) {
        ej <- replace(numeric(p), j, h[j])
        H[i, j] <- H[j, i] <- (FN(PAR + ei + ej) -
          FN(PAR + ei - ej) -
          FN(PAR - ei + ej) +
          FN(PAR - ei - ej)) /
          (4 * h[i] * h[j])
      }
    }
  }
  0.5 * (H + t(H))
}

#' @noRd
inflated_add_vcov <- function(
  FIT,
  OBJ,
  STEP = 1e-4,
  FIX_K0 = FALSE,
  FIX_K1 = FALSE
) {
  par <- FIT$par
  p <- length(par)
  safe_obj <- function(x) {
    val <- tryCatch(OBJ(x), error = function(e) NA_real_)
    if (is.finite(val)) val else NA_real_
  }
  H <- if (!FIX_K0 && !FIX_K1 && requireNamespace("numDeriv", quietly = TRUE)) {
    numDeriv::hessian(safe_obj, par)
  } else {
    inflated_fd_hessian(safe_obj, par, STEP)
  }
  V_un <- tryCatch(
    solve(H),
    error = function(e) {
      ee <- eigen(0.5 * (H + t(H)), symmetric = TRUE)
      if (any(abs(ee$values) < 1e-10)) {
        return(matrix(NA_real_, p, p))
      }
      ee$vectors %*% diag(1 / ee$values, p) %*% t(ee$vectors)
    }
  )

  if (FIX_K0) {
    gap <- exp(par[2])
    J2 <- matrix(c(FIT$phi, 0, 0, gap), 2, 2, byrow = TRUE)
    V2 <- J2 %*% V_un %*% t(J2)
    V_nat <- matrix(0, 3, 3)
    V_nat[c(1, 3), c(1, 3)] <- V2
    se <- c(phi = sqrt(V_nat[1, 1]), k0 = 0, k1 = sqrt(V_nat[3, 3]))
  } else if (FIX_K1) {
    J2 <- matrix(c(FIT$phi, 0, 0, 1), 2, 2, byrow = TRUE)
    V2 <- J2 %*% V_un %*% t(J2)
    V_nat <- matrix(0, 3, 3)
    V_nat[c(1, 2), c(1, 2)] <- V2
    se <- c(phi = sqrt(V_nat[1, 1]), k0 = sqrt(V_nat[2, 2]), k1 = 0)
  } else {
    gap <- exp(par[3])
    J_mat <- matrix(
      c(FIT$phi, 0, 0, 0, 1, 0, 0, 1, gap),
      3,
      3,
      byrow = TRUE
    )
    V_nat <- J_mat %*% V_un %*% t(J_mat)
    se <- sqrt(diag(V_nat))
    names(se) <- c("phi", "k0", "k1")
  }

  FIT$vcov_unconstrained <- V_un
  FIT$vcov <- V_nat
  FIT$se <- se
  FIT
}

#' Fit inflated interval model by profile likelihood
#'
#' @description
#' Estimates (phi, k0, k1) for data in \[0,1\] with possible point masses at 0
#' and 1 using profile likelihood. Item intercepts are profiled out via
#' Brent's method at each outer function evaluation.
#'
#' @param Y Numeric vector of responses in \[0,1\].
#' @param ITEM_INDS Integer vector of item indices (1-indexed, length equal to Y).
#' @param J Number of items.
#' @param START Named numeric vector of starting values on the unconstrained scale:
#'   `log_phi` (log of precision), `k0` (lower cutpoint), `log_delta` (log of k1 - k0).
#' @param METHOD Optimization method passed to [optim()]. Default `"BFGS"`.
#' @param CONTROL List of control parameters passed to [optim()].
#' @param COMPUTE_VCOV Logical. If `TRUE`, compute vcov and standard errors.
#' @param HESSIAN_STEP Step size for numerical Hessian when numDeriv is unavailable.
#' @param LOWER_ALPHA Lower bound for per-item alpha search. Default -5.
#' @param UPPER_ALPHA Upper bound for per-item alpha search. Default 5.
#' @param PROF_MAX_ITER Maximum Brent iterations per item. Default 500.
#' @param BOUNDARY Boundary value used to pin the fixed cutpoint. Default 100.
#'
#' @return A list of class `"inflated_fit"` with components `phi`, `k0`, `k1`,
#'   `alpha`, `loglik`, `vcov`, `se`, `convergence`, and others.
#'
#' @importFrom stats optim qlogis
#' @export
fit_inflated_profile <- function(
  Y,
  ITEM_INDS,
  J,
  START = NULL,
  METHOD = "BFGS",
  CONTROL = list(),
  COMPUTE_VCOV = TRUE,
  HESSIAN_STEP = 1e-4,
  LOWER_ALPHA = -5,
  UPPER_ALPHA = 5,
  PROF_MAX_ITER = 500L,
  BOUNDARY = 100
) {
  fix_k0 <- !any(Y == 0)
  fix_k1 <- !any(Y == 1)
  BK0 <- -BOUNDARY
  BK1 <- BOUNDARY

  alpha_warm <- init_alpha(Y, ITEM_INDS, J, LOWER_ALPHA + 1, UPPER_ALPHA - 1)

  if (fix_k0) {
    if (is.null(START)) {
      cuts <- init_cutpoints(Y)
      START <- c(log_phi = log(2), log_delta = log(pmax(cuts["k1"] - BK0, 0.1)))
    }
    obj <- function(par) {
      phi <- exp(par[1])
      k1 <- BK0 + exp(par[2])
      res <- inflated_profile_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_warm,
        phi,
        BK0,
        k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      alpha_warm <<- res$alpha
      -res$ll
    }
    opt <- stats::optim(START, obj, method = METHOD, control = CONTROL)
    phi <- unname(exp(opt$par[1]))
    k0 <- BK0
    k1 <- k0 + unname(exp(opt$par[2]))
  } else if (fix_k1) {
    if (is.null(START)) {
      cuts <- init_cutpoints(Y)
      START <- c(log_phi = log(2), k0 = cuts["k0"])
    }
    obj <- function(par) {
      phi <- exp(par[1])
      k0 <- par[2]
      res <- inflated_profile_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_warm,
        phi,
        k0,
        BK1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      alpha_warm <<- res$alpha
      -res$ll
    }
    opt <- stats::optim(START, obj, method = METHOD, control = CONTROL)
    phi <- unname(exp(opt$par[1]))
    k0 <- unname(opt$par[2])
    k1 <- BK1
  } else {
    if (is.null(START)) {
      cuts <- init_cutpoints(Y)
      START <- c(
        log_phi = log(2),
        k0 = cuts["k0"],
        log_delta = log(pmax(cuts["k1"] - cuts["k0"], 0.5))
      )
    }
    obj <- function(par) {
      phi <- exp(par[1])
      k0 <- par[2]
      k1 <- k0 + exp(par[3])
      res <- inflated_profile_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_warm,
        phi,
        k0,
        k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      alpha_warm <<- res$alpha
      -res$ll
    }
    opt <- stats::optim(START, obj, method = METHOD, control = CONTROL)
    phi <- unname(exp(opt$par[1]))
    k0 <- unname(opt$par[2])
    k1 <- k0 + unname(exp(opt$par[3]))
  }

  final <- inflated_profile_cpp(
    Y,
    ITEM_INDS,
    J,
    alpha_warm,
    phi,
    k0,
    k1,
    LOWER_ALPHA,
    UPPER_ALPHA,
    PROF_MAX_ITER
  )

  alpha_fixed <- final$alpha
  if (fix_k0) {
    frozen_obj <- function(par) {
      p <- exp(par[1])
      k1_ <- BK0 + exp(par[2])
      -inflated_profile_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_fixed,
        p,
        BK0,
        k1_,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )$ll
    }
  } else if (fix_k1) {
    frozen_obj <- function(par) {
      p <- exp(par[1])
      k0_ <- par[2]
      -inflated_profile_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_fixed,
        p,
        k0_,
        BK1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )$ll
    }
  } else {
    frozen_obj <- function(par) {
      p <- exp(par[1])
      k0_ <- par[2]
      k1_ <- k0_ + exp(par[3])
      -inflated_profile_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_fixed,
        p,
        k0_,
        k1_,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )$ll
    }
  }

  is_degen <- as.logical(tapply(Y, ITEM_INDS, function(x) {
    all(x == 0) | all(x == 1)
  }))

  fit <- structure(
    list(
      type = "inflated",
      estimator = "profile",
      phi = phi,
      k0 = k0,
      k1 = k1,
      fix_k0 = fix_k0,
      fix_k1 = fix_k1,
      boundary = BOUNDARY,
      par = opt$par,
      loglik = final$ll,
      alpha = final$alpha,
      is_degen = is_degen,
      optim = opt,
      convergence = opt$convergence
    ),
    class = "inflated_fit"
  )

  if (COMPUTE_VCOV) {
    fit <- inflated_add_vcov(fit, frozen_obj, HESSIAN_STEP, fix_k0, fix_k1)
  }
  fit
}

#' Fit inflated interval model by modified profile likelihood
#'
#' @description
#' Estimates (phi, k0, k1) using the Barndorff-Nielsen modified profile
#' likelihood. A profile likelihood fit is first obtained (or supplied via
#' `REF_FIT`) and used as the MLE reference for the BN correction.
#'
#' @param Y Numeric vector of responses in \[0,1\].
#' @param ITEM_INDS Integer vector of item indices (1-indexed, length equal to Y).
#' @param J Number of items.
#' @param REF_FIT Optional profile likelihood fit from [fit_inflated_profile()].
#'   If `NULL`, it is computed automatically.
#' @param START Named starting values on the unconstrained scale. If `NULL`,
#'   initialized from `REF_FIT`.
#' @param METHOD Optimization method passed to [optim()]. Default `"BFGS"`.
#' @param CONTROL List of control parameters passed to [optim()].
#' @param COMPUTE_VCOV Logical. If `TRUE`, compute vcov and standard errors.
#' @param HESSIAN_STEP Step size for numerical Hessian when numDeriv is unavailable.
#' @param LOWER_ALPHA Lower bound for per-item alpha search. Default -5.
#' @param UPPER_ALPHA Upper bound for per-item alpha search. Default 5.
#' @param PROF_MAX_ITER Maximum Brent iterations per item. Default 500.
#' @param BOUNDARY Boundary value used to pin the fixed cutpoint. Default 100.
#'
#' @return A list of class `"inflated_fit"` with components `phi`, `k0`, `k1`,
#'   `alpha`, `loglik`, `profile_loglik`, `correction`, `vcov`, `se`,
#'   `convergence`, and others.
#'
#' @importFrom stats optim qlogis
#' @export
fit_inflated_mpl <- function(
  Y,
  ITEM_INDS,
  J,
  REF_FIT = NULL,
  START = NULL,
  METHOD = "BFGS",
  CONTROL = list(),
  COMPUTE_VCOV = TRUE,
  HESSIAN_STEP = 1e-4,
  LOWER_ALPHA = -5,
  UPPER_ALPHA = 5,
  PROF_MAX_ITER = 500L,
  BOUNDARY = 100
) {
  if (is.null(REF_FIT)) {
    REF_FIT <- fit_inflated_profile(
      Y,
      ITEM_INDS,
      J,
      COMPUTE_VCOV = FALSE,
      LOWER_ALPHA = LOWER_ALPHA,
      UPPER_ALPHA = UPPER_ALPHA,
      PROF_MAX_ITER = PROF_MAX_ITER,
      BOUNDARY = BOUNDARY
    )
  }

  fix_k0 <- isTRUE(REF_FIT$fix_k0)
  fix_k1 <- isTRUE(REF_FIT$fix_k1)
  bnd <- if (!is.null(REF_FIT$boundary)) REF_FIT$boundary else BOUNDARY
  BK0 <- -bnd
  BK1 <- bnd

  if (is.null(START)) {
    START <- if (fix_k0) {
      c(log_phi = log(REF_FIT$phi), log_delta = log(REF_FIT$k1 - REF_FIT$k0))
    } else if (fix_k1) {
      c(log_phi = log(REF_FIT$phi), k0 = REF_FIT$k0)
    } else {
      c(
        log_phi = log(REF_FIT$phi),
        k0 = REF_FIT$k0,
        log_delta = log(REF_FIT$k1 - REF_FIT$k0)
      )
    }
  }

  alpha_warm <- REF_FIT$alpha

  if (fix_k0) {
    obj <- function(par) {
      phi <- exp(par[1])
      k1 <- BK0 + exp(par[2])
      res <- inflated_mpl_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_warm,
        REF_FIT$alpha,
        phi,
        BK0,
        k1,
        REF_FIT$phi,
        REF_FIT$k0,
        REF_FIT$k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      if (is.finite(res$ll)) {
        alpha_warm <<- res$alpha
      }
      -res$ll
    }
    opt <- stats::optim(START, obj, method = METHOD, control = CONTROL)
    phi <- unname(exp(opt$par[1]))
    k0 <- BK0
    k1 <- k0 + unname(exp(opt$par[2]))
  } else if (fix_k1) {
    obj <- function(par) {
      phi <- exp(par[1])
      k0 <- par[2]
      res <- inflated_mpl_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_warm,
        REF_FIT$alpha,
        phi,
        k0,
        BK1,
        REF_FIT$phi,
        REF_FIT$k0,
        REF_FIT$k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      if (is.finite(res$ll)) {
        alpha_warm <<- res$alpha
      }
      -res$ll
    }
    opt <- stats::optim(START, obj, method = METHOD, control = CONTROL)
    phi <- unname(exp(opt$par[1]))
    k0 <- unname(opt$par[2])
    k1 <- BK1
  } else {
    obj <- function(par) {
      phi <- exp(par[1])
      k0 <- par[2]
      k1 <- k0 + exp(par[3])
      res <- inflated_mpl_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_warm,
        REF_FIT$alpha,
        phi,
        k0,
        k1,
        REF_FIT$phi,
        REF_FIT$k0,
        REF_FIT$k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      if (is.finite(res$ll)) {
        alpha_warm <<- res$alpha
      }
      -res$ll
    }
    opt <- stats::optim(START, obj, method = METHOD, control = CONTROL)
    phi <- unname(exp(opt$par[1]))
    k0 <- unname(opt$par[2])
    k1 <- k0 + unname(exp(opt$par[3]))
  }

  final <- inflated_mpl_cpp(
    Y,
    ITEM_INDS,
    J,
    alpha_warm,
    REF_FIT$alpha,
    phi,
    k0,
    k1,
    REF_FIT$phi,
    REF_FIT$k0,
    REF_FIT$k1,
    LOWER_ALPHA,
    UPPER_ALPHA,
    PROF_MAX_ITER
  )

  alpha_fixed <- final$alpha
  if (fix_k0) {
    frozen_obj <- function(par) {
      p <- exp(par[1])
      k1_ <- BK0 + exp(par[2])
      res <- inflated_mpl_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_fixed,
        REF_FIT$alpha,
        p,
        BK0,
        k1_,
        REF_FIT$phi,
        REF_FIT$k0,
        REF_FIT$k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      -res$ll
    }
  } else if (fix_k1) {
    frozen_obj <- function(par) {
      p <- exp(par[1])
      k0_ <- par[2]
      res <- inflated_mpl_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_fixed,
        REF_FIT$alpha,
        p,
        k0_,
        BK1,
        REF_FIT$phi,
        REF_FIT$k0,
        REF_FIT$k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      -res$ll
    }
  } else {
    frozen_obj <- function(par) {
      p <- exp(par[1])
      k0_ <- par[2]
      k1_ <- k0_ + exp(par[3])
      res <- inflated_mpl_cpp(
        Y,
        ITEM_INDS,
        J,
        alpha_fixed,
        REF_FIT$alpha,
        p,
        k0_,
        k1_,
        REF_FIT$phi,
        REF_FIT$k0,
        REF_FIT$k1,
        LOWER_ALPHA,
        UPPER_ALPHA,
        PROF_MAX_ITER
      )
      -res$ll
    }
  }

  fit <- structure(
    list(
      type = "inflated",
      estimator = "modified_profile",
      phi = phi,
      k0 = k0,
      k1 = k1,
      fix_k0 = fix_k0,
      fix_k1 = fix_k1,
      boundary = bnd,
      par = opt$par,
      loglik = final$ll,
      profile_loglik = final$profile_ll,
      correction = final$correction,
      alpha = final$alpha,
      is_degen = REF_FIT$is_degen,
      ref_fit = REF_FIT,
      optim = opt,
      convergence = opt$convergence
    ),
    class = "inflated_fit"
  )

  if (COMPUTE_VCOV) {
    fit <- inflated_add_vcov(fit, frozen_obj, HESSIAN_STEP, fix_k0, fix_k1)
  }
  fit
}

#' @export
print.inflated_fit <- function(X, ...) {
  cat("Inflated interval fit\n")
  cat("  estimator:", X$estimator, "\n")
  cat("  phi:      ", signif(X$phi, 6), "\n")
  cat("  k0:       ", signif(X$k0, 6), "\n")
  cat("  k1:       ", signif(X$k1, 6), "\n")
  cat("  loglik:   ", signif(X$loglik, 8), "\n")
  if (!is.null(X$correction)) {
    cat("  correction:", signif(X$correction, 6), "\n")
  }
  if (!is.null(X$se) && all(is.finite(X$se))) {
    cat(
      "  se(phi):",
      signif(X$se["phi"], 4),
      " se(k0):",
      signif(X$se["k0"], 4),
      " se(k1):",
      signif(X$se["k1"], 4),
      "\n"
    )
  }
  cat("  convergence:", X$convergence, "\n")
  invisible(X)
}
