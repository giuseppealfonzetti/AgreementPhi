twoway_profiling_bfgs <- function(
  Y,
  ITEM_INDS,
  WORKER_INDS,
  LAMBDA_START,
  PHI,
  J,
  W,
  K,
  DATA_TYPE = c("continuous", "ordinal"),
  MAX_ITER = 100
) {
  if (DATA_TYPE == "continuous") {
    fn <- function(lambda) {
      result <- cpp_continuous_twoway_joint_loglik(
        Y = Y,
        ITEM_INDS = as.integer(ITEM_INDS),
        WORKER_INDS = as.integer(WORKER_INDS),
        LAMBDA = lambda,
        PHI = PHI,
        J = J,
        W = W,
        GRADFLAG = 0L
      )
      -result$ll
    }

    # Gradient: negative gradient
    gr <- function(lambda) {
      result <- cpp_continuous_twoway_joint_loglik(
        Y = Y,
        ITEM_INDS = as.integer(ITEM_INDS),
        WORKER_INDS = as.integer(WORKER_INDS),
        LAMBDA = lambda,
        PHI = PHI,
        J = J,
        W = W,
        GRADFLAG = 1L
      )
      -as.numeric(result$dlambda)
    }

    result <- lbfgs::lbfgs(
      call_eval = fn,
      call_grad = gr,
      vars = LAMBDA_START,
      max_iterations = MAX_ITER,
      invisible = 1
    )
  } else {
    fn <- function(lambda) {
      result <- cpp_ordinal_twoway_joint_loglik(
        Y = Y,
        ITEM_INDS = as.integer(ITEM_INDS),
        WORKER_INDS = as.integer(WORKER_INDS),
        LAMBDA = lambda,
        PHI = PHI,
        J = J,
        W = W,
        K = K,
        GRADFLAG = 0L
      )
      -result$ll
    }

    gr <- function(lambda) {
      result <- cpp_ordinal_twoway_joint_loglik(
        Y = Y,
        ITEM_INDS = as.integer(ITEM_INDS),
        WORKER_INDS = as.integer(WORKER_INDS),
        LAMBDA = lambda,
        PHI = PHI,
        J = J,
        W = W,
        K = K,
        GRADFLAG = 1L
      )
      -as.numeric(result$dlambda)
    }

    result <- lbfgs::lbfgs(
      call_eval = fn,
      call_grad = gr,
      vars = LAMBDA_START,
      max_iterations = MAX_ITER,
      invisible = 1
    )
  }

  return(result$par)
}
