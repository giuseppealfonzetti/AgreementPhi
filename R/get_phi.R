#' @importFrom stats optimize
get_phi_profile_twoway <- function(
  Y,
  ITEM_INDS,
  WORKER_INDS,
  LAMBDA_START,
  PHI_START,
  K,
  J,
  W,
  DATA_TYPE,
  SEARCH_RANGE,
  MAX_ITER,
  PROF_MAX_ITER,
  VERBOSE
) {
  neg_profile_ll <- function(phi) {
    -profile_loglik_twoway(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA_START = LAMBDA_START,
      PHI = phi,
      K = K,
      J = J,
      W = W,
      DATA_TYPE = DATA_TYPE,
      PROF_MAX_ITER = PROF_MAX_ITER
    )$ll
  }

  opt <- optimize(
    f = neg_profile_ll,
    interval = c(1e-8, PHI_START + SEARCH_RANGE),
    maximum = FALSE
  )

  phi_mle <- opt$minimum
  lambda_mle <- twoway_profiling_bfgs(
    Y = Y,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS = WORKER_INDS,
    LAMBDA_START = LAMBDA_START,
    PHI = phi_mle,
    J = J,
    W = W,
    K = K,
    DATA_TYPE = DATA_TYPE,
    MAX_ITER = PROF_MAX_ITER
  )

  if (VERBOSE) {
    message(paste0("Profile agreement: ", round(prec2agr(phi_mle), 4)))
  }

  list(
    precision = phi_mle,
    loglik = -opt$objective,
    lambda_mle = lambda_mle
  )
}

#' @importFrom stats optimize
get_phi_modified_profile_twoway <- function(
  Y,
  ITEM_INDS,
  WORKER_INDS,
  LAMBDA_START,
  PHI_START,
  K,
  J,
  W,
  DATA_TYPE,
  SEARCH_RANGE,
  MAX_ITER,
  PROF_MAX_ITER,
  VERBOSE
) {
  # Get phi via profile likelihood
  opt_pl <- get_phi_profile_twoway(
    Y = Y,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS = WORKER_INDS,
    LAMBDA_START = LAMBDA_START,
    PHI_START = PHI_START,
    K = K,
    J = J,
    W = W,
    DATA_TYPE = DATA_TYPE,
    SEARCH_RANGE = SEARCH_RANGE,
    MAX_ITER = MAX_ITER,
    PROF_MAX_ITER = PROF_MAX_ITER,
    VERBOSE = FALSE
  )

  phi_mle <- opt_pl$precision
  lambda_mle <- opt_pl$lambda_mle

  if (VERBOSE) {
    message(paste0("Non-adjusted agreement: ", round(prec2agr(phi_mle), 4)))
  }

  # Modified profile likelihood
  neg_modified_profile_ll <- function(phi) {
    -modified_profile_loglik_twoway(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA_START = lambda_mle,
      LAMBDA_MLE = lambda_mle,
      PHI = phi,
      PHI_MLE = phi_mle,
      J = J,
      W = W,
      K = K,
      DATA_TYPE = DATA_TYPE,
      PROF_MAX_ITER = PROF_MAX_ITER
    )
  }

  opt <- optimize(
    f = neg_modified_profile_ll,
    interval = c(1e-8, phi_mle + SEARCH_RANGE),
    maximum = FALSE
  )

  if (VERBOSE) {
    message(paste0("Adjusted agreement: ", round(prec2agr(opt$minimum), 4)))
  }

  list(
    mpl_precision = opt$minimum,
    pl_precision = phi_mle,
    loglik = -opt$objective,
    lambda_mle = lambda_mle
  )
}
