profile_loglik_twoway <- function(
  Y,
  ITEM_INDS,
  WORKER_INDS,
  LAMBDA_START,
  PHI,
  K,
  J,
  W,
  DATA_TYPE,
  PROF_MAX_ITER
) {
  lambda_hat <- twoway_profiling_bfgs(
    Y = Y,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS = WORKER_INDS,
    LAMBDA_START = LAMBDA_START,
    PHI = PHI,
    J = J,
    W = W,
    K = K,
    DATA_TYPE = DATA_TYPE,
    MAX_ITER = PROF_MAX_ITER
  )
  if (DATA_TYPE == "continuous") {
    result <- cpp_continuous_twoway_joint_loglik(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA = lambda_hat,
      PHI = PHI,
      J = J,
      W = W,
      GRADFLAG = 0L
    )
  } else {
    result <- cpp_ordinal_twoway_joint_loglik(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA = lambda_hat,
      PHI = PHI,
      J = J,
      W = W,
      K = K,
      GRADFLAG = 0L
    )
  }

  return(
    list(
      ll = result$ll,
      lambda_hat = lambda_hat
    )
  )
}

modified_profile_loglik_twoway <- function(
  Y,
  ITEM_INDS,
  WORKER_INDS,
  LAMBDA_START,
  LAMBDA_MLE,
  PHI,
  PHI_MLE,
  K,
  J,
  W,
  DATA_TYPE,
  PROF_MAX_ITER
) {
  # Initialise by computing profile log-likelihood
  profile_res <- profile_loglik_twoway(
    Y = Y,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS,
    LAMBDA_START = LAMBDA_MLE,
    PHI = PHI,
    K = K,
    J = J,
    W = W,
    DATA_TYPE = DATA_TYPE,
    PROF_MAX_ITER = PROF_MAX_ITER
  )

  ll <- profile_res$ll

  # Evaluate modifier
  if (DATA_TYPE == "continuous") {
    log_det_J <- cpp_continuous_twoway_log_det_obs_info(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA = profile_res$lambda_hat,
      PHI = PHI,
      J = J,
      W = W
    )

    log_det_I <- cpp_continuous_twoway_log_det_E0d0d1(
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA0 = LAMBDA_MLE,
      LAMBDA1 = profile_res$lambda_hat,
      PHI0 = PHI_MLE,
      PHI1 = PHI,
      J = J,
      W = W
    )
  } else {
    log_det_J <- cpp_ordinal_twoway_log_det_obs_info(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA = profile_res$lambda_hat,
      PHI = PHI,
      K = K,
      J = J,
      W = W
    )

    log_det_I <- cpp_ordinal_twoway_log_det_E0d0d1(
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA0 = LAMBDA_MLE,
      LAMBDA1 = profile_res$lambda_hat,
      PHI0 = PHI_MLE,
      PHI1 = PHI,
      J = J,
      W = W,
      K = K
    )
  }

  return(ll + 0.5 * log_det_J - log_det_I)
}
