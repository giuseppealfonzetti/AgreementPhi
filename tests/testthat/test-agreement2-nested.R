test_that("nested optimization converges successfully", {
  set.seed(42)

  # Simulation parameters
  J <- 10
  W <- 8
  B <- 5
  K <- 6

  # Generate data
  alphas <- rnorm(J, 0, 0.3)
  betas <- c(0, rnorm(W - 1, 0, 0.2))
  agr_true <- 0.6
  phi_true <- agr2prec(agr_true)
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Simulate ratings
  ratings_list <- list()
  for (j in 1:J) {
    for (b in 1:B) {
      w <- sample(1:W, 1)
      latent <- alphas[j] + rnorm(1, 0, 1/sqrt(phi_true)) + betas[w]
      prob <- plogis(latent)
      rating <- findInterval(prob, tau_true, left.open = TRUE)
      rating <- max(1, min(K, rating))
      ratings_list[[length(ratings_list) + 1]] <- data.frame(
        id_item = j,
        id_worker = w,
        rating = rating
      )
    }
  }
  dt <- do.call(rbind, ratings_list)

  # Test nested profile optimization (thresholds in NUISANCE)
  fit_nested <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = "phi",
    NUISANCE = c("items", "workers", "thresholds"),
    METHOD = "profile",
    PHI_START = phi_true * 0.8,
    TAU_START = tau_true
  )

  # Check convergence
  expect_equal(fit_nested$convergence, 0,
               info = "Nested optimization should converge")

  # Check that estimates are reasonable
  expect_true(fit_nested$pl_precision > 0)
  expect_true(fit_nested$pl_agreement > 0 && fit_nested$pl_agreement < 1)

  # Check that tau is properly ordered
  expect_true(all(diff(fit_nested$tau) > 0))
  expect_equal(fit_nested$tau[1], 0)
  expect_equal(fit_nested$tau[K + 1], 1)
})


test_that("nested modified profile optimization converges", {
  set.seed(123)

  # Simulation parameters
  J <- 10
  W <- 8
  B <- 5
  K <- 6

  # Generate data
  alphas <- rnorm(J, 0, 0.3)
  betas <- c(0, rnorm(W - 1, 0, 0.2))
  agr_true <- 0.6
  phi_true <- agr2prec(agr_true)
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Simulate ratings
  ratings_list <- list()
  for (j in 1:J) {
    for (b in 1:B) {
      w <- sample(1:W, 1)
      latent <- alphas[j] + rnorm(1, 0, 1/sqrt(phi_true)) + betas[w]
      prob <- plogis(latent)
      rating <- findInterval(prob, tau_true, left.open = TRUE)
      rating <- max(1, min(K, rating))
      ratings_list[[length(ratings_list) + 1]] <- data.frame(
        id_item = j,
        id_worker = w,
        rating = rating
      )
    }
  }
  dt <- do.call(rbind, ratings_list)

  # Test nested modified profile optimization
  fit_nested_mod <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = "phi",
    NUISANCE = c("items", "workers", "thresholds"),
    METHOD = "modified",
    PHI_START = phi_true * 0.8,
    TAU_START = tau_true
  )

  # Check convergence
  expect_equal(fit_nested_mod$convergence, 0,
               info = "Nested modified optimization should converge")

  # Check that both profile and modified estimates exist
  expect_true(!is.na(fit_nested_mod$pl_precision))
  expect_true(!is.na(fit_nested_mod$mpl_precision))
  expect_true(!is.na(fit_nested_mod$pl_agreement))
  expect_true(!is.na(fit_nested_mod$mpl_agreement))

  # Check that tau is properly ordered
  expect_true(all(diff(fit_nested_mod$tau) > 0))
})


test_that("joint optimization (thresholds in TARGET) uses get_phi_tau_profile", {
  set.seed(456)

  # Simulation parameters
  J <- 8
  W <- 6
  B <- 4
  K <- 4

  # Generate data
  alphas <- rnorm(J, 0, 0.3)
  betas <- c(0, rnorm(W - 1, 0, 0.2))
  agr_true <- 0.5
  phi_true <- agr2prec(agr_true)
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Simulate ratings
  ratings_list <- list()
  for (j in 1:J) {
    for (b in 1:B) {
      w <- sample(1:W, 1)
      latent <- alphas[j] + rnorm(1, 0, 1/sqrt(phi_true)) + betas[w]
      prob <- plogis(latent)
      rating <- findInterval(prob, tau_true, left.open = TRUE)
      rating <- max(1, min(K, rating))
      ratings_list[[length(ratings_list) + 1]] <- data.frame(
        id_item = j,
        id_worker = w,
        rating = rating
      )
    }
  }
  dt <- do.call(rbind, ratings_list)

  # Test joint optimization (thresholds in TARGET)
  fit_joint <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = c("phi", "thresholds"),
    NUISANCE = c("items", "workers"),
    METHOD = "profile",
    PHI_START = phi_true * 0.8,
    TAU_START = tau_true
  )

  # Check convergence
  expect_equal(fit_joint$convergence, 0,
               info = "Joint optimization should converge")

  # Check that estimates are reasonable
  expect_true(fit_joint$pl_precision > 0)
  expect_true(fit_joint$pl_agreement > 0 && fit_joint$pl_agreement < 1)
  expect_true(all(diff(fit_joint$tau) > 0))
})


test_that("nested vs joint optimization give same MLE", {
  set.seed(789)

  # Simulation parameters
  J <- 10
  W <- 8
  B <- 5
  K <- 5

  # Generate data
  alphas <- rnorm(J, 0, 0.3)
  betas <- c(0, rnorm(W - 1, 0, 0.2))
  agr_true <- 0.6
  phi_true <- agr2prec(agr_true)
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Simulate ratings
  ratings_list <- list()
  for (j in 1:J) {
    for (b in 1:B) {
      w <- sample(1:W, 1)
      latent <- alphas[j] + rnorm(1, 0, 1/sqrt(phi_true)) + betas[w]
      prob <- plogis(latent)
      rating <- findInterval(prob, tau_true, left.open = TRUE)
      rating <- max(1, min(K, rating))
      ratings_list[[length(ratings_list) + 1]] <- data.frame(
        id_item = j,
        id_worker = w,
        rating = rating
      )
    }
  }
  dt <- do.call(rbind, ratings_list)

  # Joint optimization (thresholds in TARGET)
  fit_joint <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = c("phi", "thresholds"),
    NUISANCE = c("items", "workers"),
    METHOD = "profile",
    PHI_START = phi_true,
    TAU_START = tau_true
  )

  # Nested optimization (thresholds in NUISANCE) - should compute same MLE
  fit_nested_mod <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = "phi",
    NUISANCE = c("items", "workers", "thresholds"),
    METHOD = "modified",
    PHI_START = phi_true,
    TAU_START = tau_true
  )

  # Both should converge
  expect_equal(fit_joint$convergence, 0)
  expect_equal(fit_nested_mod$convergence, 0)

  # Profile agreement from joint should match profile agreement from nested modified
  # (both compute the MLE using get_phi_tau_profile)
  expect_equal(fit_joint$pl_agreement, fit_nested_mod$pl_agreement,
               tolerance = 0.01,
               info = "Profile agreement should match between joint and nested modified (both use get_phi_tau_profile for MLE)")

  # Log-likelihoods should be very close (both at MLE)
  expect_equal(fit_joint$loglik, fit_nested_mod$loglik,
               tolerance = 0.1,
               info = "Log-likelihood at MLE should be similar")

  # Tau estimates should be close
  expect_equal(fit_joint$tau, fit_nested_mod$tau,
               tolerance = 0.05,
               info = "Threshold estimates should match")
})


test_that("profile agreement differs from modified agreement when expected", {
  set.seed(999)

  # Simulation parameters
  J <- 12
  W <- 10
  B <- 5
  K <- 6

  # Generate data
  alphas <- rnorm(J, 0, 0.3)
  betas <- c(0, rnorm(W - 1, 0, 0.2))
  agr_true <- 0.55
  phi_true <- agr2prec(agr_true)
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Simulate ratings
  ratings_list <- list()
  for (j in 1:J) {
    for (b in 1:B) {
      w <- sample(1:W, 1)
      latent <- alphas[j] + rnorm(1, 0, 1/sqrt(phi_true)) + betas[w]
      prob <- plogis(latent)
      rating <- findInterval(prob, tau_true, left.open = TRUE)
      rating <- max(1, min(K, rating))
      ratings_list[[length(ratings_list) + 1]] <- data.frame(
        id_item = j,
        id_worker = w,
        rating = rating
      )
    }
  }
  dt <- do.call(rbind, ratings_list)

  # Get modified profile estimate
  fit_mod <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = "phi",
    NUISANCE = c("items", "workers", "thresholds"),
    METHOD = "modified",
    PHI_START = phi_true,
    TAU_START = tau_true
  )

  # Check convergence
  expect_equal(fit_mod$convergence, 0)

  # Profile and modified should differ (Barndorff-Nielsen correction)
  expect_true(abs(fit_mod$pl_agreement - fit_mod$mpl_agreement) > 0.001,
              info = "Profile and modified agreement should differ due to bias correction")

  # Modified agreement should typically be closer to true value
  # (though not guaranteed in small samples)
  expect_true(!is.na(fit_mod$mpl_agreement))
  expect_true(fit_mod$mpl_agreement > 0 && fit_mod$mpl_agreement < 1)
})


test_that("convergence diagnostics work correctly", {
  set.seed(111)

  # Simulation parameters
  J <- 8
  W <- 6
  B <- 4
  K <- 4

  # Generate data
  alphas <- rnorm(J, 0, 0.3)
  betas <- c(0, rnorm(W - 1, 0, 0.2))
  agr_true <- 0.5
  phi_true <- agr2prec(agr_true)
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Simulate ratings
  ratings_list <- list()
  for (j in 1:J) {
    for (b in 1:B) {
      w <- sample(1:W, 1)
      latent <- alphas[j] + rnorm(1, 0, 1/sqrt(phi_true)) + betas[w]
      prob <- plogis(latent)
      rating <- findInterval(prob, tau_true, left.open = TRUE)
      rating <- max(1, min(K, rating))
      ratings_list[[length(ratings_list) + 1]] <- data.frame(
        id_item = j,
        id_worker = w,
        rating = rating
      )
    }
  }
  dt <- do.call(rbind, ratings_list)

  # Test with explicit LBFGS control
  fit <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = "phi",
    NUISANCE = c("items", "workers", "thresholds"),
    METHOD = "profile",
    PHI_START = phi_true,
    TAU_START = tau_true,
    CONTROL = list(
      LBFGS_MAX_ITERATIONS = 20,
      LBFGS_MAX_LINESEARCH = 10,
      LBFGS_INVISIBLE = FALSE
    )
  )

  # Check that convergence code is returned
  expect_true("convergence" %in% names(fit))
  expect_true(fit$convergence %in% c(0, -997, -998, -999, 1, 2))

  # If converged, check estimates are sensible
  if (fit$convergence == 0) {
    expect_true(fit$pl_precision > 0)
    expect_true(fit$pl_agreement > 0 && fit$pl_agreement < 1)
    expect_true(all(diff(fit$tau) > 0))
  }
})


test_that("starting values affect optimization path but not final MLE", {
  set.seed(222)

  # Simulation parameters
  J <- 10
  W <- 8
  B <- 5
  K <- 5

  # Generate data
  alphas <- rnorm(J, 0, 0.3)
  betas <- c(0, rnorm(W - 1, 0, 0.2))
  agr_true <- 0.6
  phi_true <- agr2prec(agr_true)
  thr <- sort(runif(K - 1, 0.2, 0.8))
  tau_true <- c(0, thr, 1)

  # Simulate ratings
  ratings_list <- list()
  for (j in 1:J) {
    for (b in 1:B) {
      w <- sample(1:W, 1)
      latent <- alphas[j] + rnorm(1, 0, 1/sqrt(phi_true)) + betas[w]
      prob <- plogis(latent)
      rating <- findInterval(prob, tau_true, left.open = TRUE)
      rating <- max(1, min(K, rating))
      ratings_list[[length(ratings_list) + 1]] <- data.frame(
        id_item = j,
        id_worker = w,
        rating = rating
      )
    }
  }
  dt <- do.call(rbind, ratings_list)

  # Fit with different starting values
  fit1 <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = c("phi", "thresholds"),
    NUISANCE = c("items", "workers"),
    METHOD = "profile",
    PHI_START = phi_true * 0.5,  # Poor start
    TAU_START = tau_true
  )

  fit2 <- agreement2(
    RATINGS = dt$rating,
    ITEM_INDS = dt$id_item,
    WORKER_INDS = dt$id_worker,
    TARGET = c("phi", "thresholds"),
    NUISANCE = c("items", "workers"),
    METHOD = "profile",
    PHI_START = phi_true * 1.5,  # Different poor start
    TAU_START = tau_true
  )

  # Both should converge
  expect_equal(fit1$convergence, 0)
  expect_equal(fit2$convergence, 0)

  # Final estimates should be very close (same global optimum)
  expect_equal(fit1$pl_agreement, fit2$pl_agreement,
               tolerance = 0.01,
               info = "Different starting values should reach same optimum")

  expect_equal(fit1$tau, fit2$tau,
               tolerance = 0.05,
               info = "Threshold estimates should match despite different starts")
})
