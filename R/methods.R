#' @export
print.rating_data <- function(x, ...) {
  cat("- Data type:", x$data_type, "\n")
  if (identical(x$data_type, "inflated")) {
    cat(
      "- Inflation: zeros =",
      paste0(round(100 * mean(x$ratings == 0), 1), "%"),
      "/ ones =",
      paste0(round(100 * mean(x$ratings == 1), 1), "%"),
      "\n"
    )
  }
  n_degen <- length(x$degen_ids)
  if (n_degen > 0) {
    cat("- Items:", x$n_items, "(", n_degen, "degenerate )\n")
  } else {
    cat("- Items:", x$n_items, "\n")
  }
  if (!is.null(x$n_workers)) {
    cat("- Workers:", x$n_workers, "\n")
  }
  cat("- Average budget per item:", round(x$ave_ratings_per_item, 2), "\n")
  if (!is.null(x$ave_ratings_per_worker)) {
    cat("- Average load per worker:", round(x$ave_ratings_per_worker, 2), "\n")
  }
  cat("- n:", length(x$ratings), "\n")
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

  cat("  profile agreement: ", signif(x$profile$agreement, 6), "\n")
  if (!is.na(x$modified$agreement)) {
    cat("  modified agreement:", signif(x$modified$agreement, 6), "\n")
  }
  cat("  loglik:   ", signif(x$loglik, 8), "\n")

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

  data_ref <- object$data
  n_total <- data_ref$n_items
  degen_ids <- data_ref$degen_ids

  if (length(object$alpha) < n_total) {
    alpha_full <- rep(NA_real_, n_total)
    alpha_full[setdiff(seq_len(n_total), degen_ids)] <- object$alpha
    if (object$data_type == "inflated") {
      interior_degen <- integer(0)
      for (di in degen_ids) {
        vals <- data_ref$ratings[data_ref$item_ids == di]
        if (all(vals == 0)) {
          alpha_full[di] <- -Inf
        } else if (all(vals == 1)) {
          alpha_full[di] <- Inf
        } else {
          interior_degen <- c(interior_degen, di)
        }
      }
      if (length(interior_degen) > 0) {
        int_ratings <- data_ref$ratings[data_ref$item_ids %in% interior_degen]
        int_item_ids <- match(
          data_ref$item_ids[data_ref$item_ids %in% interior_degen],
          interior_degen
        )
        int_start <- tapply(int_ratings, int_item_ids, function(v) {
          stats::qlogis(mean(v))
        })
        int_alphas <- cpp_inflated_profile(
          as.numeric(int_ratings),
          as.integer(int_item_ids),
          as.numeric(int_start),
          phi,
          object$k0,
          object$k1,
          length(interior_degen)
        )$alpha
        alpha_full[interior_degen] <- int_alphas
      }
    }
  } else {
    alpha_full <- object$alpha
  }

  alpha_names <- if (!is.null(data_ref$item_labels)) {
    paste0("alpha_", data_ref$item_labels)
  } else {
    paste0("alpha_", seq_len(n_total))
  }
  out <- c(out, setNames(alpha_full, alpha_names))

  if ("workers" %in% object$params_type$nuisance && !is.null(object$beta)) {
    beta_names <- if (!is.null(object$fit_data$worker_labels)) {
      paste0("beta_", object$fit_data$worker_labels)
    } else {
      paste0("beta_", seq_along(object$beta))
    }
    out <- c(out, setNames(object$beta, beta_names))
  }

  out
}
