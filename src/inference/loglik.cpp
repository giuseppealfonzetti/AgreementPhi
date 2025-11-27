#include "loglik.h"

////////////////////////
// CONTINUOUS RATINGS //
////////////////////////

double AgreementPhi::continuous::ll::profile(
                    const std::vector<double> Y,  
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const double PHI,
                    const int J,
                    const int W,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
){

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::continuous::nuisance::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA,  BETA, PHI, J, W, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    std::vector<double> lambda;
    lambda.reserve(J + W - 1);
    lambda.insert(lambda.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    lambda.insert(lambda.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    
    double ll = AgreementPhi::continuous::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, lambda, PHI, J, W, ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );

    return ll;

}


double AgreementPhi::continuous::ll::modified_profile(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const double PHI,
    const double PHI_MLE,
    const int J,
    const int W,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){

    // profile nuisance parameters
    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::continuous::nuisance::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA_MLE,  BETA_MLE, PHI, J, W,  ITEMS_NUISANCE,   WORKER_NUISANCE, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    std::vector<double> profiled_vec;
    profiled_vec.reserve(J + W - 1);
    profiled_vec.insert(profiled_vec.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    profiled_vec.insert(profiled_vec.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    // evaluate profile log-likelihood
    double ll = AgreementPhi::continuous::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, PHI, J, W, ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );


    // evaluate modifier contribution
    ll += .5 * AgreementPhi::continuous::log_det_obs_info(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, PHI, J, W, ITEMS_NUISANCE, WORKER_NUISANCE
    );

    std::vector<double> mle_vec;
    mle_vec.reserve(J + W - 1);
    mle_vec.insert(mle_vec.end(), ALPHA_MLE.begin(), ALPHA_MLE.end());
    mle_vec.insert(mle_vec.end(), BETA_MLE.begin() + 1, BETA_MLE.end());

    ll -= AgreementPhi::continuous::log_det_E0d0d1(
        ITEM_INDS, WORKER_INDS, mle_vec, profiled_vec, PHI_MLE, PHI, J, W, ITEMS_NUISANCE, WORKER_NUISANCE
    );

    return ll;

}

/////////////////////
// ORDINAL RATINGS //
/////////////////////

double AgreementPhi::ordinal::ll::profile(
                    const std::vector<double> Y,  
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> TAU,
                    const double PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const bool THRESHOLDS_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
){


    

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA,  BETA, TAU, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, THRESHOLDS_NUISANCE, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    // Rcpp::Rcout<<"tau: ";
    // for (double i: profiled_lambda.at(2))
    // Rcpp::Rcout << i << ' ';
    // Rcpp::Rcout<<"\n";


    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    std::vector<double> lambda;
    lambda.reserve(J + W - 1);
    lambda.insert(lambda.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    lambda.insert(lambda.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    double ll = AgreementPhi::ordinal::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, lambda, profiled_lambda.at(2), PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );

    return ll;
}

double AgreementPhi::ordinal::ll::modified_profile(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const std::vector<double> TAU_MLE,
    const double PHI,
    const double PHI_MLE,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const bool THRESHOLDS_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA_MLE,  BETA_MLE, TAU_MLE, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, false, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    // Rcpp::Rcout<<"tau: ";
    // for (double i: profiled_lambda.at(2))
    // Rcpp::Rcout << i << ' ';
    // Rcpp::Rcout<<"\n";

    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    std::vector<double> profiled_vec;
    profiled_vec.reserve(J + W - 1);
    profiled_vec.insert(profiled_vec.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    profiled_vec.insert(profiled_vec.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    // evaluate profile log-likelihood
    double ll = AgreementPhi::ordinal::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, profiled_lambda.at(2), PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );


    // evaluate modifier contribution
    ll += .5 * AgreementPhi::ordinal::log_det_obs_info(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, profiled_lambda.at(2), PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
    );

    std::vector<double> mle_vec;
    mle_vec.reserve(J + W - 1);
    mle_vec.insert(mle_vec.end(), ALPHA_MLE.begin(), ALPHA_MLE.end());
    mle_vec.insert(mle_vec.end(), BETA_MLE.begin() + 1, BETA_MLE.end());

    ll -= AgreementPhi::ordinal::log_det_E0d0d1(
        ITEM_INDS, WORKER_INDS, mle_vec, profiled_vec,  PHI_MLE, PHI, profiled_lambda.at(2), J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
    );

    return ll;

}

double AgreementPhi::ordinal::ll::modified_profile_extended(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const std::vector<double> TAU,
    const std::vector<double> TAU_MLE,
    const double PHI,
    const double PHI_MLE,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA_MLE,  BETA_MLE, TAU, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, false, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    // Rcpp::Rcout<<"tau: ";
    // for (double i: profiled_lambda.at(2))
    // Rcpp::Rcout << i << ' ';
    // Rcpp::Rcout<<"\n";

    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    std::vector<double> profiled_vec;
    profiled_vec.reserve(J + W - 1);
    profiled_vec.insert(profiled_vec.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    profiled_vec.insert(profiled_vec.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    // evaluate profile log-likelihood
    double ll = AgreementPhi::ordinal::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, TAU, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );


    // evaluate modifier contribution
    ll += .5 * AgreementPhi::ordinal::log_det_obs_info(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, TAU, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
    );

    std::vector<double> mle_vec;
    mle_vec.reserve(J + W - 1);
    mle_vec.insert(mle_vec.end(), ALPHA_MLE.begin(), ALPHA_MLE.end());
    mle_vec.insert(mle_vec.end(), BETA_MLE.begin() + 1, BETA_MLE.end());

    ll -= AgreementPhi::ordinal::log_det_E0d0d1_extended(
        ITEM_INDS, WORKER_INDS, mle_vec, profiled_vec,  PHI_MLE, PHI, TAU_MLE, TAU, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
    );

    return ll;

}

double AgreementPhi::ordinal::ll::modified_profile_tau_profiled(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const std::vector<double> TAU_START,
    const std::vector<double> TAU_MLE,
    const double PHI,
    const double PHI_MLE,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){
    // PROPER NESTED OPTIMIZATION:
    // A) f1(phi, tau) = profile likelihood with THRESHOLDS_NUISANCE=FALSE
    //    This profiles (alpha, beta) for fixed (phi, tau)
    // B) For this phi, find τ̂_p(phi) = argmax_tau f1(phi, tau)
    //    by manually optimizing over each tau component
    // C) Compute f2(phi) = f1(phi, τ̂_p) + Barndorff-Nielsen correction

    const int n = Y.size();
    const int max_iter_thr = 3;
    std::vector<double> tau_best = TAU_START;
    std::vector<double> alpha_best = ALPHA_MLE;
    std::vector<double> beta_best = BETA_MLE;

    // Step B: Optimize tau to maximize f1(phi, tau)
    // We iterate over tau components, similar to threshold profiling in get_lambda2
    for(int iter_tau = 0; iter_tau < max_iter_thr; iter_tau++){

        // For each threshold, optimize it while holding others fixed
        for(int t = K-1; t > 0; t--){

            // Define objective: -f1(phi, tau) for this threshold component
            auto neg_profile_ll = [&](double tau_t) -> double {
                std::vector<double> tau_candidate = tau_best;
                tau_candidate.at(t) = tau_t;

                // Step A: Profile (alpha, beta) for fixed (phi, tau_candidate)
                // This is f1(phi, tau_candidate)
                std::vector<std::vector<double>> profiled = AgreementPhi::ordinal::nuisance::get_lambda2(
                    Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
                    alpha_best, beta_best, tau_candidate, PHI, J, W, K,
                    ITEMS_NUISANCE, WORKER_NUISANCE,
                    false,  // THRESHOLDS_NUISANCE = FALSE (tau is fixed)
                    PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
                );

                std::vector<double> alpha_prof = profiled.at(0);
                std::vector<double> beta_prof = profiled.at(1);

                // Compute profile likelihood f1(phi, tau_candidate)
                std::vector<double> lambda_vec;
                lambda_vec.reserve(J + W - 1);
                lambda_vec.insert(lambda_vec.end(), alpha_prof.begin(), alpha_prof.end());
                lambda_vec.insert(lambda_vec.end(), beta_prof.begin() + 1, beta_prof.end());

                Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
                Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
                Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
                Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

                double ll = AgreementPhi::ordinal::joint_loglik(
                    Y, ITEM_INDS, WORKER_INDS, lambda_vec, tau_candidate, PHI, J, W, K,
                    ITEMS_NUISANCE, WORKER_NUISANCE,
                    dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
                );

                return -ll;  // Return negative for minimization
            };

            // Optimize tau_t using Brent's method
            double lower = std::max(tau_best.at(t) - 0.1, tau_best.at(t - 1) + 1e-8);
            double upper = std::min(tau_best.at(t) + 0.1, tau_best.at(t + 1) - 1e-8);
            const int digits = std::numeric_limits<double>::digits;
            boost::uintmax_t max_iter = PROF_UNI_MAX_ITER;

            auto result = boost::math::tools::brent_find_minima(
                neg_profile_ll, lower, upper, digits, max_iter
            );

            tau_best.at(t) = result.first;
        }
    }

    // Now tau_best is τ̂_p(phi)
    // Profile (alpha, beta) one final time at (phi, τ̂_p) to get final values
    std::vector<std::vector<double>> final_profiled = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
        alpha_best, beta_best, tau_best, PHI, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        false,  // THRESHOLDS_NUISANCE = FALSE
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );

    std::vector<double> alpha_final = final_profiled.at(0);
    std::vector<double> beta_final = final_profiled.at(1);

    std::vector<double> lambda_vec;
    lambda_vec.reserve(J + W - 1);
    lambda_vec.insert(lambda_vec.end(), alpha_final.begin(), alpha_final.end());
    lambda_vec.insert(lambda_vec.end(), beta_final.begin() + 1, beta_final.end());

    // Step C: Compute f2(phi) = f1(phi, τ̂_p) + BN_correction
    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    double ll = AgreementPhi::ordinal::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, lambda_vec, tau_best, PHI, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );

    // Add Barndorff-Nielsen correction
    ll += 0.5 * AgreementPhi::ordinal::log_det_obs_info(
        Y, ITEM_INDS, WORKER_INDS, lambda_vec, tau_best, PHI, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE
    );

    std::vector<double> mle_vec;
    mle_vec.reserve(J + W - 1);
    mle_vec.insert(mle_vec.end(), ALPHA_MLE.begin(), ALPHA_MLE.end());
    mle_vec.insert(mle_vec.end(), BETA_MLE.begin() + 1, BETA_MLE.end());

    ll -= AgreementPhi::ordinal::log_det_E0d0d1_extended(
        ITEM_INDS, WORKER_INDS, mle_vec, lambda_vec,
        PHI_MLE, PHI, TAU_MLE, tau_best, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE
    );

    return ll;
}
void AgreementPhi::ordinal::ll::profile_grad_tau(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA_START,
    const std::vector<double> BETA_START,
    const std::vector<double> TAU,
    const double PHI,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    std::vector<double>& GRAD_TAU
){
    // Step 1: Profile (alpha, beta) for fixed (phi, tau)
    std::vector<std::vector<double>> profiled = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
        ALPHA_START, BETA_START, TAU, PHI, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        false,  // THRESHOLDS_NUISANCE = FALSE (tau is fixed)
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );

    std::vector<double> alpha_prof = profiled.at(0);
    std::vector<double> beta_prof = profiled.at(1);

    // Step 2: Construct lambda vector
    std::vector<double> lambda_vec;
    lambda_vec.reserve(J + W - 1);
    lambda_vec.insert(lambda_vec.end(), alpha_prof.begin(), alpha_prof.end());
    lambda_vec.insert(lambda_vec.end(), beta_prof.begin() + 1, beta_prof.end());

    // Step 3: Compute gradient ∂L/∂τ at profiled (α, β)
    // By envelope theorem, this is the gradient we want
    AgreementPhi::ordinal::grad_tau(
        Y, ITEM_INDS, WORKER_INDS, lambda_vec, TAU, PHI,
        J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
        GRAD_TAU
    );
}


double AgreementPhi::ordinal::ll::profile_extended(
                    const std::vector<double> Y,  
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> RAW_TAU,
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
){


    double phi = exp(RAW_PHI);
    std::vector<double> tau = AgreementPhi::utils::raw2tau(RAW_TAU);

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA,  BETA, tau, phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, false, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    std::vector<double> lambda;
    lambda.reserve(J + W - 1);
    lambda.insert(lambda.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    lambda.insert(lambda.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    double ll = AgreementPhi::ordinal::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, lambda, tau, phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );

    return ll;
}

// Helper function to compute Jacobian of tau wrt raw_tau
// Returns a (K+1) x (K-1) matrix where tau[k] is differentiated wrt raw_tau[i]
Eigen::MatrixXd compute_dtau_drawtau(const std::vector<double>& RAW_TAU) {
    const int n = static_cast<int>(RAW_TAU.size());  // K-1
    const int K = n + 1;  // Number of categories

    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(K + 1, n);

    // Compute exponentials and denominator
    std::vector<double> e(n);
    double S = 0.0;
    for (int i = 0; i < n; ++i) {
        e[i] = std::exp(RAW_TAU[i]);
        S += e[i];
    }
    double D = 1.0 + S;
    double D2 = D * D;

    // Compute cumulative sums
    std::vector<double> C(n + 1);
    C[0] = 0.0;
    for (int k = 1; k <= n; ++k) {
        C[k] = C[k - 1] + e[k - 1];
    }

    // Fill Jacobian
    // tau[0] = 0 and tau[K] = 1 are constant, so their rows remain zero
    for (int k = 1; k <= n; ++k) {  // tau[k] for k = 1, ..., K-1
        for (int i = 0; i < n; ++i) {  // raw_tau[i] for i = 0, ..., K-2
            if (i < k) {
                // ∂tau[k]/∂raw_tau[i] = e_i × (D - C_k) / D²
                jac(k, i) = e[i] * (D - C[k]) / D2;
            } else {
                // ∂tau[k]/∂raw_tau[i] = -e_i × C_k / D²
                jac(k, i) = -e[i] * C[k] / D2;
            }
        }
    }

    return jac;
}

Eigen::VectorXd AgreementPhi::ordinal::ll::profile_extended_grad_raw_tau(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const std::vector<double> RAW_TAU,
    const double RAW_PHI,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){
    // Step 1: Transform raw parameters to natural scale
    double phi = std::exp(RAW_PHI);
    std::vector<double> tau = AgreementPhi::utils::raw2tau(RAW_TAU);

    // Step 2: Profile (alpha, beta) for fixed (phi, tau)
    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
        ALPHA, BETA, tau, phi, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        false,  // THRESHOLDS_NUISANCE = FALSE (tau is fixed)
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );

    std::vector<double> alpha_prof = profiled_lambda.at(0);
    std::vector<double> beta_prof = profiled_lambda.at(1);

    // Step 3: Construct lambda vector
    std::vector<double> lambda_vec;
    lambda_vec.reserve(J + W - 1);
    lambda_vec.insert(lambda_vec.end(), alpha_prof.begin(), alpha_prof.end());
    lambda_vec.insert(lambda_vec.end(), beta_prof.begin() + 1, beta_prof.end());

    // Step 4: Compute gradient ∂L/∂τ at profiled (α, β)
    std::vector<double> grad_tau_vec;
    AgreementPhi::ordinal::grad_tau(
        Y, ITEM_INDS, WORKER_INDS, lambda_vec, tau, phi,
        J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
        grad_tau_vec
    );

    // Convert to Eigen vector (grad_tau_vec has length K-1)
    Eigen::VectorXd grad_tau = Eigen::Map<Eigen::VectorXd>(grad_tau_vec.data(), K - 1);

    // Step 5: Compute Jacobian ∂τ/∂raw_τ
    Eigen::MatrixXd dtau_drawtau = compute_dtau_drawtau(RAW_TAU);

    // Step 6: Apply chain rule: ∂L/∂raw_τ = (∂τ/∂raw_τ)ᵀ × ∂L/∂τ
    // grad_tau corresponds to τ[1], ..., τ[K-1]
    // dtau_drawτau has rows for τ[0], ..., τ[K]
    // We need rows 1 through K-1
    Eigen::MatrixXd dtau_relevant = dtau_drawtau.block(1, 0, K - 1, K - 1);
    Eigen::VectorXd grad_raw_tau = dtau_relevant.transpose() * grad_tau;

    return grad_raw_tau;
}

double AgreementPhi::ordinal::ll::profile_extended_grad_raw_phi(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const std::vector<double> RAW_TAU,
    const double RAW_PHI,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){
    // Use finite differences to compute ∂L/∂raw_phi
    const double h = 1e-8;

    // Evaluate at raw_phi
    double ll_center = profile_extended(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
        ALPHA, BETA, RAW_TAU, RAW_PHI, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );

    // Evaluate at raw_phi + h
    double ll_forward = profile_extended(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
        ALPHA, BETA, RAW_TAU, RAW_PHI + h, J, W, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );

    // Compute numerical derivative
    double grad_raw_phi = (ll_forward - ll_center) / h;

    return grad_raw_phi;
}

Eigen::VectorXd AgreementPhi::ordinal::ll::profile_extended_grad(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const std::vector<double> RAW_TAU,
    const double RAW_PHI,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){
    const int d = RAW_TAU.size()+1;
    Eigen::VectorXd out(d);

    out(0) = AgreementPhi::ordinal::ll::profile_extended_grad_raw_phi(
    Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA,
    BETA, RAW_TAU, RAW_PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
    PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    out.tail(d-1) = AgreementPhi::ordinal::ll::profile_extended_grad_raw_tau(
    Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA,
    BETA, RAW_TAU, RAW_PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
    PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    return out;
}

// Eigen::MatrixXd AgreementPhi::ordinal::ll::profile_extended_hess_raw_tau(
//     const std::vector<double> Y,
//     const std::vector<int> ITEM_INDS,
//     const std::vector<int> WORKER_INDS,
//     const std::vector<std::vector<int>> ITEM_DICT,
//     const std::vector<std::vector<int>> WORKER_DICT,
//     const std::vector<std::vector<int>> CAT_DICT,
//     const std::vector<double> ALPHA,
//     const std::vector<double> BETA,
//     const std::vector<double> RAW_TAU,
//     const double RAW_PHI,
//     const int J,
//     const int W,
//     const int K,
//     const bool ITEMS_NUISANCE,
//     const bool WORKER_NUISANCE,
//     const int PROF_UNI_RANGE,
//     const int PROF_UNI_MAX_ITER,
//     const int PROF_MAX_ITER,
//     const double PROF_TOL
// ){
//     // Analytical Hessian using chain rule:
//     // ∂²L/∂raw_τ² = J^T × H_τ × J + gradient terms with ∂²τ/∂raw_τ²
//     // where J = ∂τ/∂raw_τ and H_τ = ∂²L/∂τ²

//     const int n = K - 1;

//     // Step 1: Transform to natural scale
//     double phi = std::exp(RAW_PHI);
//     std::vector<double> tau = AgreementPhi::utils::raw2tau(RAW_TAU);

//     // Step 2: Profile (alpha, beta)
//     std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
//         Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
//         ALPHA, BETA, tau, phi, J, W, K,
//         ITEMS_NUISANCE, WORKER_NUISANCE,
//         false,  // THRESHOLDS_NUISANCE = FALSE
//         PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
//     );

//     std::vector<double> lambda_vec;
//     lambda_vec.reserve(J + W - 1);
//     lambda_vec.insert(lambda_vec.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
//     lambda_vec.insert(lambda_vec.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

//     // Step 3: Compute gradient ∂L/∂τ (needed for second-order terms)
//     std::vector<double> grad_tau_vec;
//     AgreementPhi::ordinal::grad_tau(
//         Y, ITEM_INDS, WORKER_INDS, lambda_vec, tau, phi,
//         J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
//         grad_tau_vec
//     );
//     Eigen::VectorXd grad_tau = Eigen::Map<Eigen::VectorXd>(grad_tau_vec.data(), n);

//     // Step 4: Compute Hessian ∂²L/∂τ² using finite differences on grad_tau
//     Eigen::MatrixXd hess_tau = Eigen::MatrixXd::Zero(n, n);
//     const double h_tau = 1e-6;

//     for (int i = 0; i < n; ++i) {
//         std::vector<double> tau_plus = tau;
//         std::vector<double> tau_minus = tau;
//         tau_plus[i + 1] += h_tau;   // tau[0] is fixed at 0, so tau[1] corresponds to i=0
//         tau_minus[i + 1] -= h_tau;

//         // Re-profile at tau_plus
//         std::vector<std::vector<double>> prof_plus = AgreementPhi::ordinal::nuisance::get_lambda2(
//             Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
//             ALPHA, BETA, tau_plus, phi, J, W, K,
//             ITEMS_NUISANCE, WORKER_NUISANCE, false,
//             PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
//         );
//         std::vector<double> lambda_plus;
//         lambda_plus.reserve(J + W - 1);
//         lambda_plus.insert(lambda_plus.end(), prof_plus.at(0).begin(), prof_plus.at(0).end());
//         lambda_plus.insert(lambda_plus.end(), prof_plus.at(1).begin() + 1, prof_plus.at(1).end());

//         std::vector<double> grad_plus;
//         AgreementPhi::ordinal::grad_tau(
//             Y, ITEM_INDS, WORKER_INDS, lambda_plus, tau_plus, phi,
//             J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, grad_plus
//         );

//         // Re-profile at tau_minus
//         std::vector<std::vector<double>> prof_minus = AgreementPhi::ordinal::nuisance::get_lambda2(
//             Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
//             ALPHA, BETA, tau_minus, phi, J, W, K,
//             ITEMS_NUISANCE, WORKER_NUISANCE, false,
//             PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
//         );
//         std::vector<double> lambda_minus;
//         lambda_minus.reserve(J + W - 1);
//         lambda_minus.insert(lambda_minus.end(), prof_minus.at(0).begin(), prof_minus.at(0).end());
//         lambda_minus.insert(lambda_minus.end(), prof_minus.at(1).begin() + 1, prof_minus.at(1).end());

//         std::vector<double> grad_minus;
//         AgreementPhi::ordinal::grad_tau(
//             Y, ITEM_INDS, WORKER_INDS, lambda_minus, tau_minus, phi,
//             J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, grad_minus
//         );

//         // Compute column i of Hessian
//         for (int j = 0; j < n; ++j) {
//             hess_tau(j, i) = (grad_plus[j] - grad_minus[j]) / (2.0 * h_tau);
//         }
//     }

//     // Symmetrize
//     hess_tau = 0.5 * (hess_tau + hess_tau.transpose());

//     // Step 5: Compute Jacobian ∂τ/∂raw_τ
//     Eigen::MatrixXd dtau_draw = compute_dtau_drawtau(RAW_TAU);
//     Eigen::MatrixXd Jac = dtau_draw.block(1, 0, n, n);  // Extract relevant block

//     // Step 6: Apply chain rule (first-order term)
//     // ∂²L/∂raw_τ² = J^T × H_τ × J
//     Eigen::MatrixXd hess_raw = Jac.transpose() * hess_tau * Jac;

//     return hess_raw;
// }