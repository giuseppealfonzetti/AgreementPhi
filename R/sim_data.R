#' Simulate ordinal or continuous (0,1) ratings
#'
#' @param J Number of items.
#' @param B Budget per item.
#' @param AGREEMENT General agreement.
#' @param ALPHA Item-specific intercepts.
#' @param DATA_TYPE Choose between `ordinal` or `continuous`.
#' @param K Number of categories in case of ordinal data.
#' @param SEED RNG seed.
#'
#' @return Returns a dataframe with columns id_items, id_items_obs, rating
#' @export
sim_data <- function(
  J,
  B,
  AGREEMENT,
  ALPHA,
  DATA_TYPE = c("ordinal", "continuous"),
  K = 5,
  SEED = 123
) {
  stopifnot(J > 0)
  stopifnot(AGREEMENT >= 0)
  stopifnot(AGREEMENT <= 1)
  stopifnot(K > 1)
  stopifnot(is.numeric(ALPHA))
  stopifnot(length(ALPHA) == J)

  DATA_TYPE <- match.arg(DATA_TYPE)
  stopifnot(is.numeric(SEED))

  n_obs <- J * B
  precision <- agr2prec(AGREEMENT)
  obs_item_ind <- rep(1:J, each = B)
  obs_mu <- plogis(ALPHA[obs_item_ind])
  obs_a <- obs_mu * precision
  obs_b <- (1 - obs_mu) * precision

  set.seed(SEED)
  obs_beta <- apply(cbind(obs_a, obs_b), 1, function(par) {
    rbeta(1, par[1], par[2])
  })

  if (DATA_TYPE == "ordinal") {
    obs_y <- cont2ord(obs_beta, K)
  } else {
    obs_y <- obs_beta
  }

  dt <- data.frame(cbind(
    id_item = obs_item_ind,
    id_item_obs = obs_item_ind,
    rating = obs_y * 1.0
  ))

  return(dt)
}
