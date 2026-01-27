#' Plot relative log-likelihood
#'
#' @param D output from [get_range_ll]
#' @param M_EST agreement estimate from modified profile likelihood
#' @param P_EST agreement estimate from profile likelihood
#' @param M_SE standard error for agreement estimate from modified profile likelihood
#' @param P_SE standard error for agreement estimate from profile likelihood
#' @param CONFIDENCE Confidence level to construct confidence intervals
#'
#' @examples
#' set.seed(321)
#'
#' # setting dimension
#' items <- 50
#' budget_per_item <- 5
#' n_obs <- items * budget_per_item
#' workers <- 50
#'
#' # item-specific intercepts to generate the data
#' alphas <- runif(items, -2, 2)
#'
#' # true agreement (between 0 and 1)
#' agr <- .6
#'
#' # generate continuous rating in (0,1)
#' dt_oneway <- sim_data(
#'   J = items,
#'   B = budget_per_item,
#'   AGREEMENT = agr,
#'   ALPHA = alphas,
#'   DATA_TYPE = "continuous",
#'   SEED = 123
#' )
#'
#' # estimation via oneway specification
#' fit <- agreement(
#'   RATINGS = dt_oneway$rating,
#'   ITEM_INDS = dt_oneway$id_item,
#'   WORKER_INDS = dt_oneway$id_worker,
#'   METHOD = "modified",
#'   NUISANCE = c("items"),
#'   VERBOSE = TRUE
#' )
#' # get standard error and confidence interval
#' ci <- get_ci(fit)
#' ci
#'
#' # compute log-likelihood over a grid
#' range_ll <- get_range_ll(fit)
#'
#' # utility plot function for relative log-likelihood
#' plot_rll(
#'   D = range_ll,
#'   M_EST = fit$modified$agreement,
#'   P_EST = fit$profile$agreement,
#'   M_SE = ci$agreement_se,
#'   CONFIDENCE=.95
#' )
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

#' Plot data
#'
#' @param RATINGS Ratings vector of dimension n. Ordinal data must be coded in \{1, 2, ..., K\}.
#'   Continuous data can take values in `(0, 1)`.
#' @param ITEM_INDS Index vector with items allocations. Same dimension as `RATINGS`.
#' @param WORKER_INDS Index vector with worker allocations. Same dimension as `RATINGS`. Ignored when MODEL == "oneway".
#'   Must be integers in \{1, 2, ..., J\}.
#' @param VERBOSE Verbose output.
#' @returns Plot with the relevance rating matrix
#'
#' @examples
#' set.seed(321)
#'
#' # setting dimension
#' items <- 50
#' budget_per_item <- 5
#' n_obs <- items * budget_per_item
#'
#' # item-specific intercepts to generate the data
#' alphas <- runif(items, -2, 2)
#'
#' # true agreement (between 0 and 1)
#' agr <- .6
#'
#' # generate continuous rating in (0,1)
#' dt <- sim_data(
#'   J = items,
#'   B = budget_per_item,
#'   AGREEMENT = agr,
#'   ALPHA = alphas,
#'   DATA_TYPE = "continuous",
#'   SEED = 123
#' )
#'
#' plot_data(
#'   RATINGS = dt$rating,
#'   ITEM_INDS = dt$id_item,
#'   WORKER_INDS = dt$id_worker
#' )
#'
#' @export
#' @importFrom graphics image layout
#' @importFrom grDevices hcl.colors
plot_data <- function(
  RATINGS,
  ITEM_INDS,
  WORKER_INDS = NULL,
  VERBOSE = FALSE
) {
  val_data <- validate_data(
    RATINGS = RATINGS,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS = WORKER_INDS,
    VERBOSE = VERBOSE
  )

  # setup matrix with correct dimensions
  plot_matrix <- matrix(
    NA,
    nrow = val_data$n_workers,
    ncol = val_data$n_items
  )

  # populate using per-rating indices
  plot_matrix[cbind(val_data$worker_ids, val_data$item_ids)] <- val_data$ratings

  # save and restore graphics state
  opar <- par(no.readonly = TRUE)
  on.exit(par(opar))

  layout(matrix(1:2, ncol = 2), widths = c(5, 1))
  par(mar = c(5, 4, 4, 1))

  if (val_data$data_type == "ordinal") {
    K <- val_data$K
    image(
      t(plot_matrix),
      col = hcl.colors(K, "teal", rev = TRUE),
      zlim = c(1, K),
      axes = FALSE,
      xlab = "Items",
      ylab = "Workers"
    )

    # discrete color legend
    par(mar = c(5, 0.5, 4, 3))
    legend_image <- as.matrix(K:1)
    image(
      1,
      seq_len(K),
      t(legend_image),
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

    # continuous color legend
    par(mar = c(5, 0.5, 4, 3))
    legend_image <- as.matrix(rev(seq(0, 100, length.out = 100)))
    image(
      1,
      seq(0, 1, length.out = 100),
      t(legend_image),
      col = hcl.colors(100, "teal", rev = FALSE),
      axes = FALSE,
      xlab = "",
      ylab = ""
    )
    axis(4, las = 1)
  }
}
