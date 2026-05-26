#' @export
print.rating_data <- function(x, ...) {
  cat("Input data:", x$data_type, "\n")
  cat("  items:  ", x$n_items, "\n")
  if (!is.null(x$n_workers)) {
    cat("  workers:", x$n_workers, "\n")
  }
  cat("  n:      ", length(x$ratings), "\n")
  invisible(x)
}

#' @export
print.params_spec <- function(x, ...) {
  cat("Params spec\n")
  if (length(x$constant)) {
    cat("  constant:", paste(x$constant, collapse = ", "), "\n")
  }
  if (length(x$nuisance)) {
    cat("  nuisance:", paste(x$nuisance, collapse = ", "), "\n")
  }
  if (length(x$target)) {
    cat("  target:  ", paste(x$target, collapse = ", "), "\n")
  }
  invisible(x)
}

#' @export
print.agreement_fit <- function(x, ...) {
  cat("Agreement fit\n")
  cat("  data type:", x$data_type, "\n")
  cat("  method:   ", x$method, "\n")

  if (x$data_type == "inflated") {
    phi_est <- if (x$method == "modified") {
      x$modified$precision
    } else {
      x$profile$precision
    }
    agr_est <- if (x$method == "modified") {
      x$modified$agreement
    } else {
      x$profile$agreement
    }
    cat("  phi:      ", signif(phi_est, 6), "\n")
    cat("  k0:       ", signif(x$k0, 6), "\n")
    cat("  k1:       ", signif(x$k1, 6), "\n")
    cat("  agreement:", signif(agr_est, 6), "\n")
    if (!is.null(x$se) && all(is.finite(x$se))) {
      cat(
        "  se(phi):",
        signif(x$se["phi"], 4),
        " se(k0):",
        signif(x$se["k0"], 4),
        " se(k1):",
        signif(x$se["k1"], 4),
        "\n"
      )
    }
    cat("  convergence:", x$convergence, "\n")
  } else {
    cat("  profile agreement: ", signif(x$profile$agreement, 6), "\n")
    if (!is.na(x$modified$agreement)) {
      cat("  modified agreement:", signif(x$modified$agreement, 6), "\n")
    }
    cat("  loglik:   ", signif(x$loglik, 8), "\n")
  }

  invisible(x)
}

#' Extract coefficients from an agreement fit
#'
#' @param object An `agreement_fit` object from [agreement()].
#' @param ... Ignored.
#'
#' @return A named numeric vector. Always contains `phi` and `alpha_1...alpha_J`.
#'   For inflated data: also `k0` and `k1`. For two-way models (workers profiled
#'   as nuisance): also `beta_1...beta_W`.
#'
#' @examples
#' set.seed(1)
#' dt <- sim_data(J = 20, B = 5, AGREEMENT = 0.6,
#'                ALPHA = rep(0, 20), DATA_TYPE = "continuous", SEED = 1)
#' rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
#' fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
#' coef(fit)
#'
#' @importFrom stats coef
#' @export
coef.agreement_fit <- function(object, ...) {
  phi <- if (object$method == "modified") {
    unname(object$modified$precision)
  } else {
    unname(object$profile$precision)
  }
  out <- c(phi = phi)

  if (isTRUE(object$data_type == "inflated")) {
    out <- c(out, k0 = unname(object$k0), k1 = unname(object$k1))
  }

  alpha_names <- if (!is.null(object$data$item_labels)) {
    paste0("alpha_", object$data$item_labels)
  } else {
    paste0("alpha_", seq_along(object$alpha))
  }
  out <- c(out, setNames(object$alpha, alpha_names))

  if ("workers" %in% object$params_type$nuisance && !is.null(object$beta)) {
    beta_names <- if (!is.null(object$data$worker_labels)) {
      paste0("beta_", object$data$worker_labels)
    } else {
      paste0("beta_", seq_along(object$beta))
    }
    out <- c(out, setNames(object$beta, beta_names))
  }

  out
}
