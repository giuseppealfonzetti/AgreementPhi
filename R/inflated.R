#' @noRd
init_cutpoints <- function(Y) {
  p0 <- mean(Y == 0)
  p1 <- mean(Y == 1)
  k0 <- if (p0 > 0) stats::qlogis(pmax(pmin(p0, 1 - 1e-4), 1e-4)) else -1
  k1 <- if (p1 > 0) -stats::qlogis(pmax(pmin(p1, 1 - 1e-4), 1e-4)) else 1
  c(k0 = k0, k1 = k1)
}

#' @noRd
inflated_encoding <- function(FIX_K0, FIX_K1, BK0, BK1) {
  if (FIX_K0) {
    list(
      make_start = function(PHI0, K0_INIT, K1_INIT) {
        c(log_phi = log(PHI0), log_delta = log(pmax(K1_INIT - BK0, 0.1)))
      },
      decode = function(PAR) {
        list(phi = exp(PAR[1]), k0 = BK0, k1 = BK0 + exp(PAR[2]))
      }
    )
  } else if (FIX_K1) {
    list(
      make_start = function(PHI0, K0_INIT, K1_INIT) {
        c(log_phi = log(PHI0), k0 = K0_INIT)
      },
      decode = function(PAR) {
        list(phi = exp(PAR[1]), k0 = PAR[2], k1 = BK1)
      }
    )
  } else {
    list(
      make_start = function(PHI0, K0_INIT, K1_INIT) {
        c(
          log_phi = log(PHI0),
          k0 = K0_INIT,
          log_delta = log(pmax(K1_INIT - K0_INIT, 0.5))
        )
      },
      decode = function(PAR) {
        list(phi = exp(PAR[1]), k0 = PAR[2], k1 = PAR[2] + exp(PAR[3]))
      }
    )
  }
}

#' @noRd
inflated_add_vcov <- function(
  FIT,
  OBJ,
  FIX_K0 = FALSE,
  FIX_K1 = FALSE
) {
  par <- FIT$par
  p <- length(par)
  safe_obj <- function(X) {
    val <- tryCatch(OBJ(X), error = function(E) NA_real_)
    if (is.finite(val)) val else NA_real_
  }
  H <- tryCatch(
    stats::optimHess(par, safe_obj),
    error = function(E) tryCatch(
      stats::optimHess(par, safe_obj, control = list(ndeps = rep(1e-5, p))),
      error = function(E2) matrix(NA_real_, p, p)
    )
  )
  V_un <- tryCatch(
    solve(H),
    error = function(E) {
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

#' @noRd
fit_inflated_profile <- function(
  Y,
  ITEM_INDS,
  J,
  START = NULL,
  METHOD = "BFGS",
  OPTIM_CONTROL = list(),
  COMPUTE_VCOV = TRUE,
  PROF_SEARCH_RANGE = 10,
  PROF_MAX_ITER = 500L,
  BOUNDARY = 100
) {
  fix_k0 <- !any(Y == 0)
  fix_k1 <- !any(Y == 1)
  BK0 <- -BOUNDARY
  BK1 <- BOUNDARY
  enc <- inflated_encoding(fix_k0, fix_k1, BK0, BK1)

  alpha_warm <- init_alpha(
    Y,
    ITEM_INDS,
    J,
    -PROF_SEARCH_RANGE + 1,
    PROF_SEARCH_RANGE - 1
  )

  if (is.null(START)) {
    cuts <- init_cutpoints(Y)
    START <- enc$make_start(2, cuts["k0"], cuts["k1"])
  }

  obj <- function(PAR) {
    p <- enc$decode(PAR)
    res <- cpp_inflated_profile(
      as.numeric(Y),
      as.integer(ITEM_INDS),
      as.numeric(alpha_warm),
      p$phi,
      p$k0,
      p$k1,
      J,
      PROF_SEARCH_RANGE,
      as.integer(PROF_MAX_ITER)
    )
    if (is.finite(res$ll)) {
      alpha_warm <<- res$alpha
    }
    -res$ll
  }

  opt <- stats::optim(START, obj, method = METHOD, control = OPTIM_CONTROL)
  p <- enc$decode(opt$par)
  phi <- p$phi
  k0 <- p$k0
  k1 <- p$k1

  final <- cpp_inflated_profile(
    as.numeric(Y),
    as.integer(ITEM_INDS),
    as.numeric(alpha_warm),
    phi,
    k0,
    k1,
    J,
    PROF_SEARCH_RANGE,
    as.integer(PROF_MAX_ITER)
  )
  alpha_fixed <- final$alpha

  frozen_obj <- function(PAR) {
    p <- enc$decode(PAR)
    -cpp_inflated_profile(
      as.numeric(Y),
      as.integer(ITEM_INDS),
      as.numeric(alpha_fixed),
      p$phi,
      p$k0,
      p$k1,
      J,
      PROF_SEARCH_RANGE,
      as.integer(PROF_MAX_ITER)
    )$ll
  }

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
      optim = opt,
      convergence = opt$convergence
    ),
    class = "inflated_fit"
  )

  if (COMPUTE_VCOV) {
    fit <- inflated_add_vcov(fit, frozen_obj, fix_k0, fix_k1)
  }
  fit
}

#' @noRd
fit_inflated_mpl <- function(
  Y,
  ITEM_INDS,
  J,
  REF_FIT = NULL,
  START = NULL,
  METHOD = "BFGS",
  OPTIM_CONTROL = list(),
  COMPUTE_VCOV = TRUE,
  PROF_SEARCH_RANGE = 10,
  PROF_MAX_ITER = 500L,
  BOUNDARY = 100
) {
  if (is.null(REF_FIT)) {
    REF_FIT <- fit_inflated_profile(
      Y,
      ITEM_INDS,
      J,
      COMPUTE_VCOV = FALSE,
      PROF_SEARCH_RANGE = PROF_SEARCH_RANGE,
      PROF_MAX_ITER = PROF_MAX_ITER,
      BOUNDARY = BOUNDARY
    )
  }

  fix_k0 <- isTRUE(REF_FIT$fix_k0)
  fix_k1 <- isTRUE(REF_FIT$fix_k1)
  bnd <- if (!is.null(REF_FIT$boundary)) REF_FIT$boundary else BOUNDARY
  BK0 <- -bnd
  BK1 <- bnd
  enc <- inflated_encoding(fix_k0, fix_k1, BK0, BK1)

  if (is.null(START)) {
    START <- enc$make_start(REF_FIT$phi, REF_FIT$k0, REF_FIT$k1)
  }

  alpha_warm <- REF_FIT$alpha

  obj <- function(PAR) {
    p <- enc$decode(PAR)
    res <- cpp_inflated_mpl(
      as.numeric(Y),
      as.integer(ITEM_INDS),
      as.numeric(alpha_warm),
      as.numeric(REF_FIT$alpha),
      p$phi,
      p$k0,
      p$k1,
      REF_FIT$phi,
      REF_FIT$k0,
      REF_FIT$k1,
      J,
      PROF_SEARCH_RANGE,
      as.integer(PROF_MAX_ITER)
    )
    if (is.finite(res$ll)) {
      alpha_warm <<- res$alpha
    }
    -res$ll
  }

  opt <- stats::optim(START, obj, method = METHOD, control = OPTIM_CONTROL)
  p <- enc$decode(opt$par)
  phi <- p$phi
  k0 <- p$k0
  k1 <- p$k1

  alpha_synced <- cpp_inflated_profile(
    as.numeric(Y),
    as.integer(ITEM_INDS),
    as.numeric(alpha_warm),
    phi,
    k0,
    k1,
    J,
    PROF_SEARCH_RANGE,
    as.integer(PROF_MAX_ITER)
  )$alpha

  final <- cpp_inflated_mpl(
    as.numeric(Y),
    as.integer(ITEM_INDS),
    as.numeric(alpha_synced),
    as.numeric(REF_FIT$alpha),
    phi,
    k0,
    k1,
    REF_FIT$phi,
    REF_FIT$k0,
    REF_FIT$k1,
    J,
    PROF_SEARCH_RANGE,
    as.integer(PROF_MAX_ITER)
  )
  alpha_fixed <- final$alpha

  mpl_ok <- is.finite(final$ll)
  if (!mpl_ok) {
    warning(
      "Modified profile likelihood is degenerate; ",
      "falling back to profile likelihood estimates."
    )
    phi         <- REF_FIT$phi
    k0          <- REF_FIT$k0
    k1          <- REF_FIT$k1
    alpha_fixed <- REF_FIT$alpha
    final       <- list(
      ll         = REF_FIT$loglik,
      alpha      = REF_FIT$alpha,
      profile_ll = REF_FIT$loglik,
      correction = NA_real_
    )
    opt$par         <- REF_FIT$par
    opt$convergence <- 1L
  }

  frozen_obj <- if (mpl_ok) {
    function(PAR) {
      p <- enc$decode(PAR)
      -cpp_inflated_mpl(
        as.numeric(Y),
        as.integer(ITEM_INDS),
        as.numeric(alpha_fixed),
        as.numeric(REF_FIT$alpha),
        p$phi,
        p$k0,
        p$k1,
        REF_FIT$phi,
        REF_FIT$k0,
        REF_FIT$k1,
        J,
        PROF_SEARCH_RANGE,
        as.integer(PROF_MAX_ITER)
      )$ll
    }
  } else {
    function(PAR) {
      p <- enc$decode(PAR)
      -cpp_inflated_profile(
        as.numeric(Y),
        as.integer(ITEM_INDS),
        as.numeric(alpha_fixed),
        p$phi,
        p$k0,
        p$k1,
        J,
        PROF_SEARCH_RANGE,
        as.integer(PROF_MAX_ITER)
      )$ll
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
      ref_fit = REF_FIT,
      optim = opt,
      convergence = opt$convergence
    ),
    class = "inflated_fit"
  )

  if (COMPUTE_VCOV) {
    fit <- inflated_add_vcov(fit, frozen_obj, fix_k0, fix_k1)
  }
  fit
}
