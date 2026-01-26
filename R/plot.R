#' Plot relative log-likelihood
#'
#' @param D output from [get_range_ll]
#' @param M_EST agreement estimate from modified profile likelihood
#' @param P_EST agreement estimate from profile likelihood
#' @param M_SE standard error for agreement estimate from modified profile likelihood
#' @param P_SE standard error for agreement estimate from profile likelihood
#' @param CONFIDENCE Confidence level to construct confidence intervals
#'
#' @export
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

# lrows <- sort(unique(dt_oneway$id_worker))
# cols <- sort(unique(dt_oneway$id_item))
# plot_matrix <- matrix(NA,
#                       nrow = length(rows),
#                       ncol = length(cols),
#                       dimnames = list(rows, cols))

# row_indices <- match(dt_oneway$id_worker, rows)
# col_indices <- match(dt_oneway$id_item, cols)
# plot_matrix[matrix(c(row_indices, col_indices), ncol = 2)] <- dt_oneway$rating
# image(t(plot_matrix) * 100,
#       axes = FALSE,
#       xlab = "Items",
#       ylab = "Workers",
#       col = hcl.colors(100, "teal", rev=TRUE))
# layout(matrix(1:2, ncol = 2), widths = c(5, 1))
# par(mar = c(5, 4, 4, 1))
# image(t(plot_matrix) * 100, col = hcl.colors(100, "teal", rev=TRUE), axes = FALSE, xlab = "Items", ylab = "Workers")
# par(mar = c(5, 0.5, 4, 3))
# legend_image <- as.matrix(rev(seq(0, 100, length.out = 100)))
# image(1, seq(0, 100, length.out = 100), t(legend_image),
#       col = hcl.colors(100, "teal", rev=FALSE), axes = FALSE, xlab = "", ylab = "")
# axis(4, las = 1)
