#' @importFrom graphics abline legend lines plot rect axis par
#' @importFrom grDevices adjustcolor
plot_rll <- function(
  D,
  M_EST = NULL,
  P_EST = NULL,
  M_SE = NULL,
  P_SE = NULL,
  CONFIDENCE = 0.95
) {
  alpha <- 1 - CONFIDENCE
  z <- qnorm(1 - alpha / 2)

  plot(
    D$agreement,
    D$profile - max(D$profile),
    type = "l",
    xlab = "Agreement",
    ylab = "Relative log-likelihood",
    col = "#F28E2B",
    bty = "l"
  )
  # axis(1)
  # axis(2)

  if (!is.null(P_SE)) {
    rect(
      xleft = P_EST - z * P_SE,
      xright = P_EST + z * P_SE,
      ybottom = par("usr")[3],
      ytop = par("usr")[4],
      col = adjustcolor("#F28E2B", alpha.f = 0.2),
      border = NA
    )
  }

  abline(v = P_EST, col = "#F28E2B", lty = 2)

  if (!is.null(D$modified)) {
    lines(
      D$agreement,
      D$modified - max(D$modified, na.rm = TRUE),
      col = "#4E79A7"
    )

    if (!is.null(M_SE)) {
      rect(
        xleft = M_EST - z * M_SE,
        xright = M_EST + z * M_SE,
        ybottom = par("usr")[3],
        ytop = par("usr")[4],
        col = adjustcolor("#4E79A7", alpha.f = 0.2),
        border = NA
      )
    }

    abline(v = M_EST, col = "#4E79A7", lty = 2)
    legend(
      "bottomleft",
      legend = c("Profile", "Modified profile"),
      col = c("#F28E2B", "#4E79A7"),
      lty = c(1, 1),
      bty = "n"
    )
  }
}

#' Plot a rating_data object
#'
#' @param x A `rating_data` object from [rating_data()].
#' @param ... Ignored.
#' @returns Invisibly returns `x`. Called for its side effect (a rating matrix plot).
#'
#' @examples
#' \donttest{
#' dt <- sim_data(J = 20, B = 5, AGREEMENT = 0.6,
#'                ALPHA = rep(0, 20), DATA_TYPE = "continuous", SEED = 1)
#' rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
#' plot(rd)
#' }
#' @export
#' @importFrom graphics image layout
#' @importFrom grDevices hcl.colors
plot.rating_data <- function(x, ...) {
  if (is.null(x$worker_ids)) {
    stop(
      "plot() requires a two-way rating_data object (WORKER_INDS must be provided)."
    )
  }

  plot_matrix <- matrix(NA, nrow = x$n_workers, ncol = x$n_items)
  plot_matrix[cbind(x$worker_ids, x$item_ids)] <- x$ratings

  opar <- par(no.readonly = TRUE)
  on.exit(par(opar))

  layout(matrix(1:2, ncol = 2), widths = c(5, 1))
  par(mar = c(5, 4, 4, 1))

  if (x$data_type == "ordinal") {
    K <- x$K
    image(
      t(plot_matrix),
      col = hcl.colors(K, "teal", rev = TRUE),
      zlim = c(1, K),
      axes = FALSE,
      xlab = "Items",
      ylab = "Workers"
    )
    par(mar = c(5, 0.5, 4, 3))
    image(
      1,
      seq_len(K),
      t(as.matrix(K:1)),
      col = hcl.colors(K, "teal", rev = FALSE),
      zlim = c(1, K),
      axes = FALSE,
      xlab = "",
      ylab = ""
    )
    axis(4, at = seq_len(K), labels = seq_len(K), las = 1)
  } else {
    image(
      t(plot_matrix),
      col = hcl.colors(100, "teal", rev = TRUE),
      axes = FALSE,
      xlab = "Items",
      ylab = "Workers"
    )
    par(mar = c(5, 0.5, 4, 3))
    image(
      1,
      seq(0, 1, length.out = 100),
      t(as.matrix(rev(seq(0, 100, length.out = 100)))),
      col = hcl.colors(100, "teal", rev = FALSE),
      axes = FALSE,
      xlab = "",
      ylab = ""
    )
    axis(4, las = 1)
  }
  invisible(x)
}

#' Plot an agreement_fit object
#'
#' @description
#' Plots the relative log-likelihood curve(s) for a fitted agreement model.
#' Calls [get_range_ll()] to evaluate the likelihood over a grid. A precomputed
#' grid can be supplied via `RANGE_LL` to avoid recomputation.
#'
#' @param x An `agreement_fit` object from [agreement()].
#' @param RANGE_LL Optional. A data frame returned by [get_range_ll()]. If
#'   `NULL` (default), the grid is computed internally using `RANGE` and
#'   `GRID_LENGTH`.
#' @param RANGE Range of agreement values around the MLE to evaluate. Default `0.2`.
#' @param GRID_LENGTH Number of grid points. Default `15`.
#' @param CONFIDENCE Confidence level for the shaded interval. Default `0.95`.
#' @param ... Ignored (required for S3 consistency).
#'
#' @return Invisibly returns `x`.
#'
#' @examples
#' \donttest{
#' dt <- sim_data(J = 30, B = 5, AGREEMENT = 0.6,
#'                ALPHA = rep(0, 30), DATA_TYPE = "continuous", SEED = 1)
#' rd <- rating_data(dt$rating, dt$id_item, dt$id_worker, VERBOSE = FALSE)
#' fit <- agreement(rd, METHOD = "modified", NUISANCE = "items")
#' plot(fit)
#' }
#' @export
#' @importFrom graphics abline legend lines plot rect axis par
#' @importFrom grDevices adjustcolor
plot.agreement_fit <- function(
  x,
  RANGE_LL = NULL,
  RANGE = 0.2,
  GRID_LENGTH = 15,
  CONFIDENCE = 0.95,
  ...
) {
  if (isTRUE(x$data_type == "inflated")) {
    stop("plot() is not available for inflated interval fits.")
  }
  if (is.null(RANGE_LL)) {
    RANGE_LL <- get_range_ll(x, RANGE = RANGE, GRID_LENGTH = GRID_LENGTH)
  }
  ci <- confint(x, level = CONFIDENCE)
  plot_rll(
    D = RANGE_LL,
    M_EST = if (x$method == "modified") x$modified$agreement else NULL,
    P_EST = x$profile$agreement,
    M_SE = if (x$method == "modified") {
      ci$agreement["agreement", "Std. Error"]
    } else {
      NULL
    },
    P_SE = if (x$method == "profile") {
      ci$agreement["agreement", "Std. Error"]
    } else {
      NULL
    },
    CONFIDENCE = CONFIDENCE
  )
  invisible(x)
}

#' Forest plot of model-based probability of item degeneracy
#'
#' Plots per-item P(degenerate) estimates and their confidence intervals from
#' [confint_prob_degenerate()]. Items are colour-coded: observed degenerate
#' items (P = 1, CI collapsed) in orange; non-degenerate items in blue.
#'
#' @param x An `agreement_fit` object from [agreement()].
#' @param LEVEL Confidence level passed to [confint_prob_degenerate()].
#'   Default `0.95`.
#' @param SORT Logical; sort items by estimate before plotting. Default `TRUE`.
#' @param ... Ignored (required for S3 consistency).
#'
#' @return Invisibly returns the matrix from [confint_prob_degenerate()],
#'   in the order plotted.
#'
#' @export
#' @importFrom graphics segments points axis abline legend par
plot_prob_degenerate <- function(x, LEVEL = 0.95, SORT = TRUE, ...) {
  ci <- confint_prob_degenerate(x, level = LEVEL)

  if (SORT) {
    ci <- ci[order(ci[, "Estimate"]), , drop = FALSE]
  }

  J <- nrow(ci)
  item_nms <- rownames(ci)

  all_nms <- if (!is.null(x$data$item_labels)) {
    paste0("item_", x$data$item_labels)
  } else {
    paste0("item_", seq_len(x$data$n_items))
  }
  is_degen <- item_nms %in% all_nms[x$data$degen_ids]
  cols <- ifelse(is_degen, "#F28E2B", "#4E79A7")

  opar <- par(no.readonly = TRUE)
  on.exit(par(opar))

  left_mar <- max(4, max(nchar(item_nms)) * 0.55)
  par(mar = c(4, left_mar, 2, 2))

  cex_ax <- max(0.5, min(1, 15 / J))

  plot(
    x = ci[, "Estimate"],
    y = seq_len(J),
    xlim = c(0, 1),
    ylim = c(0.5, J + 0.5),
    xlab = "P(degenerate)",
    ylab = "",
    yaxt = "n",
    pch = 19,
    col = cols,
    bty = "l"
  )

  axis(
    2,
    at = seq_len(J),
    labels = item_nms,
    las = 1,
    tick = FALSE,
    cex.axis = cex_ax
  )

  nt <- ci[, 3L] < ci[, 4L]
  y_nt <- which(nt)
  if (length(y_nt) > 0L) {
    segments(ci[nt, 3L], y_nt, ci[nt, 4L], y_nt, col = cols[nt], lwd = 1.5)
    cap <- 0.18
    segments(ci[nt, 3L], y_nt - cap, ci[nt, 3L], y_nt + cap, col = cols[nt])
    segments(ci[nt, 4L], y_nt - cap, ci[nt, 4L], y_nt + cap, col = cols[nt])
  }

  abline(v = 0, col = "grey80", lty = 2)

  if (any(is_degen)) {
    legend(
      "bottomright",
      legend = c("Non-degenerate", "Observed degenerate"),
      col = c("#4E79A7", "#F28E2B"),
      pch = 19,
      bty = "n"
    )
  }

  invisible(ci)
}
