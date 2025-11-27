profile_likelihood_cpp <- function(
  Y,
  ITEM_INDS,
  WORKER_INDS = NULL,
  ALPHA_START,
  BETA_START = NULL,
  TAU_START = NULL,
  PHI,
  J,
  W = NULL,
  K,
  DATA_TYPE = c("continuous", "ordinal"),
  ITEMS_NUISANCE = TRUE,
  WORKER_NUISANCE = TRUE,
  THRESHOLDS_NUISANCE = FALSE,
  PROF_SEARCH_RANGE,
  PROF_UNI_MAX_ITER,
  ALT_MAX_ITER,
  ALT_TOL
) {
  DATA_TYPE <- match.arg(DATA_TYPE)
  stopifnot(length(Y) == length(ITEM_INDS))

  item_inds <- as.integer(ITEM_INDS)
  worker_inds <- WORKER_INDS
  if (is.null(worker_inds)) {
    worker_inds <- rep(1L, length(Y))
  } else {
    stopifnot(length(worker_inds) == length(Y))
    worker_inds <- as.integer(worker_inds)
  }

  if (is.null(J)) {
    J <- length(unique(item_inds))
  }

  if (is.null(W)) {
    W <- max(worker_inds)
  }

  alpha_start <- ALPHA_START
  stopifnot(length(alpha_start) == J)

  beta_start <- BETA_START
  if (is.null(beta_start)) {
    beta_start <- rep(0, W)
  } else if (length(beta_start) < W) {
    beta_start <- c(beta_start, rep(0, W - length(beta_start)))
  } else if (length(beta_start) > W) {
    beta_start <- beta_start[seq_len(W)]
  }

  if (DATA_TYPE == "continuous") {
    K <- 1L
    THRESHOLDS_NUISANCE <- FALSE
  }

  tau_start <- TAU_START
  if (is.null(tau_start) || length(tau_start) != K + 1) {
    tau_start <- seq(0, 1, length.out = K + 1)
  }

  cpp_profile_likelihood(
    Y = as.numeric(Y),
    ITEM_INDS = item_inds,
    WORKER_INDS = worker_inds,
    ALPHA_START = alpha_start,
    BETA_START = beta_start,
    TAU_START = tau_start,
    PHI = PHI,
    J = as.integer(J),
    W = as.integer(W),
    K = as.integer(K),
    DATA_TYPE = DATA_TYPE,
    ITEMS_NUISANCE = ITEMS_NUISANCE,
    WORKER_NUISANCE = WORKER_NUISANCE,
    THRESHOLDS_NUISANCE = THRESHOLDS_NUISANCE,
    PROF_SEARCH_RANGE = PROF_SEARCH_RANGE,
    PROF_UNI_MAX_ITER = PROF_UNI_MAX_ITER,
    ALT_MAX_ITER = ALT_MAX_ITER,
    ALT_TOL = ALT_TOL
  )
}


modified_profile_likelihood_cpp <- function(
  Y,
  ITEM_INDS,
  WORKER_INDS = NULL,
  ALPHA_START,
  BETA_START = NULL,
  PAR,
  J,
  W = NULL,
  K,
  ITEMS_NUISANCE = TRUE,
  WORKER_NUISANCE = TRUE,
  PROF_SEARCH_RANGE,
  PROF_UNI_MAX_ITER,
  ALT_MAX_ITER,
  ALT_TOL
) {
  R_neg_prof_ext <- function(
    PAR,
    Y,
    ITEM_INDS,
    WORKER_INDS,
    ALPHA_START,
    BETA_START,
    J,
    W,
    K,
    ITEMS_NUISANCE,
    WORKER_NUISANCE,
    PROF_SEARCH_RANGE,
    PROF_UNI_MAX_ITER,
    ALT_MAX_ITER,
    ALT_TOL
  ) {
    raw_phi <- PAR[1]
    raw_tau <- PAR[2:length(PAR)]
    phi <- exp(raw_phi)
    tau <- raw2tau(raw_tau)

    -cpp_profile_likelihood(
      Y = as.numeric(Y),
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      ALPHA_START = ALPHA_START,
      BETA_START = BETA_START,
      TAU_START = tau,
      PHI = phi,
      J = as.integer(J),
      W = as.integer(W),
      K = as.integer(K),
      DATA_TYPE = "ordinal",
      ITEMS_NUISANCE = ITEMS_NUISANCE,
      WORKER_NUISANCE = WORKER_NUISANCE,
      THRESHOLDS_NUISANCE = FALSE,
      PROF_SEARCH_RANGE = PROF_SEARCH_RANGE,
      PROF_UNI_MAX_ITER = PROF_UNI_MAX_ITER,
      ALT_MAX_ITER = ALT_MAX_ITER,
      ALT_TOL = ALT_TOL
    )
  }

  tictoc::tic()
  opt <- optim(
    par = PAR,
    fn = R_neg_prof_ext,
    Y = Y,
    ITEM_INDS = ITEM_INDS,
    WORKER_INDS = WORKER_INDS,
    ALPHA_START = ALPHA_START,
    BETA_START = BETA_START,
    J = J,
    W = W,
    K = K,
    ITEMS_NUISANCE = ITEMS_NUISANCE,
    WORKER_NUISANCE = WORKER_NUISANCE,
    THRESHOLDS_NUISANCE = FALSE,
    PROF_SEARCH_RANGE = .PROF_SEARCH_RANGE,
    PROF_UNI_MAX_ITER = PROF_UNI_MAX_ITER,
    ALT_MAX_ITER = ALT_MAX_ITER,
    ALT_TOL = ALT_TOL
  )
  tictoc::toc()
}


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
  tau_vec <- if (DATA_TYPE == "ordinal") seq(0, 1, length.out = K + 1) else NULL
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
    MAX_ITER = PROF_MAX_ITER,
    WORKER_NUISANCE = TRUE,
    TAU = tau_vec
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
      WORKER_NUISANCE = TRUE,
      GRADFLAG = 0L
    )
  } else {
    result <- cpp_ordinal_twoway_joint_loglik(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA = lambda_hat,
      TAU = tau_vec,
      PHI = PHI,
      J = J,
      W = W,
      K = K,
      WORKER_NUISANCE = TRUE,
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
      W = W,
      WORKER_NUISANCE = TRUE
    )

    log_det_I <- cpp_continuous_twoway_log_det_E0d0d1(
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA0 = LAMBDA_MLE,
      LAMBDA1 = profile_res$lambda_hat,
      PHI0 = PHI_MLE,
      PHI1 = PHI,
      J = J,
      W = W,
      WORKER_NUISANCE = TRUE
    )
  } else {
    tau_vec <- seq(0, 1, length.out = K + 1)
    log_det_J <- cpp_ordinal_twoway_log_det_obs_info(
      Y = Y,
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA = profile_res$lambda_hat,
      TAU = tau_vec,
      PHI = PHI,
      K = K,
      J = J,
      W = W,
      WORKER_NUISANCE = TRUE
    )

    log_det_I <- cpp_ordinal_twoway_log_det_E0d0d1(
      ITEM_INDS = ITEM_INDS,
      WORKER_INDS = WORKER_INDS,
      LAMBDA0 = LAMBDA_MLE,
      LAMBDA1 = profile_res$lambda_hat,
      PHI0 = PHI_MLE,
      PHI1 = PHI,
      TAU = tau_vec,
      J = J,
      W = W,
      K = K,
      WORKER_NUISANCE = TRUE
    )
  }

  return(ll + 0.5 * log_det_J - log_det_I)
}
