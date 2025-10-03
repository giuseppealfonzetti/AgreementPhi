#' Simulate ordinal or continuous (0,1) ratings
#'
#' @param J Number of items.
#' @param B Budget per item (i.e. number of workers assigned to each item).
#' @param W Maximum number of workers.
#' @param AGREEMENT General agreement.
#' @param ALPHA Item-specific intercepts.
#' @param BETA Worker-specific intercepts.
#' @param DATA_TYPE Choose between `ordinal` or `continuous`.
#' @param K Number of categories in case of ordinal data.
#' @param SEED RNG seed.
#'
#' @return Returns a dataframe with columns id_items, id_worker and rating
#' @export
sim_data <- function(
  J,
  B,
  W = J * 2,
  AGREEMENT,
  ALPHA,
  BETA = NULL,
  DATA_TYPE = c("ordinal", "continuous"),
  K = 6,
  SEED = 123
) {
  set.seed(SEED)
  stopifnot(J > 0)
  stopifnot(AGREEMENT >= 0)
  stopifnot(AGREEMENT <= 1)
  stopifnot(K > 1)
  stopifnot(W > 1)
  stopifnot(B <= W)
  stopifnot(is.numeric(ALPHA))
  stopifnot(length(ALPHA) == J)

  DATA_TYPE <- match.arg(DATA_TYPE)
  stopifnot(is.numeric(SEED))

  n_obs <- J * B
  precision <- agr2prec(AGREEMENT)
  obs_item_ind <- rep(1:J, each = B)
  obs_worker_ind <- as.vector(replicate(J, sample(1:W, B, replace = FALSE)))
  if (is.null(BETA)) {
    BETA <- rep(0, W)
  }
  obs_mu <- plogis(ALPHA[obs_item_ind] + BETA[obs_worker_ind])

  obs_a <- obs_mu * precision
  obs_b <- (1 - obs_mu) * precision

  obs_beta <- apply(cbind(obs_a, obs_b), 1, function(par) {
    rbeta(1, par[1], par[2])
  })

  if (DATA_TYPE == "ordinal") {
    obs_y <- cont2ord(obs_beta, K)
  } else {
    obs_y <- ifelse(
      abs(obs_beta - 1) < .Machine$double.eps / 2,
      obs_beta - 1e-5,
      obs_beta
    )
    obs_y <- ifelse(obs_y < .Machine$double.eps / 2, 1e-5, obs_y)
  }

  dt <- data.frame(
    id_item = obs_item_ind,
    id_worker = obs_worker_ind,
    rating = obs_y * 1.0
  )

  return(dt)
}
