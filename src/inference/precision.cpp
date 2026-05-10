#include "precision.h"
/////////////////////////
// CONTINUOUS RATINGS  //
/////////////////////////

std::vector<double> AgreementPhi::continuous::inference::get_phi_profile(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    std::vector<double>& ALPHA,
    std::vector<double>& BETA,
    const double PHI_START,
    const int J,
    const int W,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    struct BestWarmStart {
        double loglik = -std::numeric_limits<double>::infinity();
        std::vector<double> alpha;
        std::vector<double> beta;
        bool initialized = false;
    } best;

    auto neg_profile_likelihood = [&](double phi){
        std::vector<double> alpha_start = best.initialized ? best.alpha : ALPHA;
        std::vector<double> beta_start  = best.initialized ? best.beta  : BETA;

        std::vector<std::vector<double>> profiled = AgreementPhi::continuous::nuisance::get_lambda(
            Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, alpha_start, beta_start,
            phi, J, W, ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

        std::vector<double> lambda;
        lambda.reserve(J + W - 1);
        lambda.insert(lambda.end(), profiled.at(0).begin(), profiled.at(0).end());
        lambda.insert(lambda.end(), profiled.at(1).begin() + 1, profiled.at(1).end());

        Eigen::VectorXd dlambda    = Eigen::VectorXd::Zero(J + W - 1);
        Eigen::VectorXd jaa        = Eigen::VectorXd::Zero(J);
        Eigen::VectorXd jbb        = Eigen::VectorXd::Zero(W - 1);
        Eigen::MatrixXd jab        = Eigen::MatrixXd::Zero(J, W - 1);

        double ll = AgreementPhi::continuous::joint_loglik(
            Y, ITEM_INDS, WORKER_INDS, lambda, phi, J, W,
            ITEMS_NUISANCE, WORKER_NUISANCE, dlambda, jaa, jbb, jab, 0);

        if (ll > best.loglik) {
            best.loglik     = ll;
            best.alpha      = profiled.at(0);
            best.beta       = profiled.at(1);
            best.initialized = true;
        }

        if (!std::isfinite(ll)) return std::numeric_limits<double>::infinity();
        return -ll;
    };

    double lower = 1e-8;
    double upper = PHI_START + SEARCH_RANGE;

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_profile_likelihood, lower, upper, digits, max_iter
    );

    if (best.initialized) {
        ALPHA = best.alpha;
        BETA  = best.beta;
    }

    std::vector<double> out(2);
    out[0] = result.first;
    out[1] = result.second;
    return out;
}




std::vector<double> AgreementPhi::continuous::inference::get_phi_modified_profile(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    std::vector<double>& ALPHA,
    std::vector<double>& BETA,
    const double PHI_START,
    const int J,
    const int W,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    std::vector<double> phi_mle = AgreementPhi::continuous::inference::get_phi_profile(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA, BETA, PHI_START, J, W,
        ITEMS_NUISANCE, WORKER_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL, VERBOSE);

    if(VERBOSE) Rcpp::Rcout << "Non-adjusted agreement: " << utils::prec2agr(phi_mle.at(0)) << "\n";

    std::vector<std::vector<double>> lambda_mle = AgreementPhi::continuous::nuisance::get_lambda(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA, BETA, phi_mle.at(0), J, W,
        ITEMS_NUISANCE, WORKER_NUISANCE, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    std::vector<double> mle_vec;
    mle_vec.reserve(J + W - 1);
    mle_vec.insert(mle_vec.end(), lambda_mle.at(0).begin(), lambda_mle.at(0).end());
    mle_vec.insert(mle_vec.end(), lambda_mle.at(1).begin() + 1, lambda_mle.at(1).end());

    // Evaluate modified profile LL for a given phi, starting nuisance profiling from
    // alpha_s/beta_s. Returns -ll (for minimization); +Inf when ll is not finite.
    auto eval_mpl = [&](double phi,
                        const std::vector<double>& alpha_s,
                        const std::vector<double>& beta_s) -> double {
        std::vector<std::vector<double>> profiled = AgreementPhi::continuous::nuisance::get_lambda(
            Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, alpha_s, beta_s,
            phi, J, W, ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

        std::vector<double> pv;
        pv.reserve(J + W - 1);
        pv.insert(pv.end(), profiled.at(0).begin(), profiled.at(0).end());
        pv.insert(pv.end(), profiled.at(1).begin() + 1, profiled.at(1).end());

        Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
        Eigen::VectorXd jaa     = Eigen::VectorXd::Zero(J);
        Eigen::VectorXd jbb     = Eigen::VectorXd::Zero(W - 1);
        Eigen::MatrixXd jab     = Eigen::MatrixXd::Zero(J, W - 1);

        double ll = AgreementPhi::continuous::joint_loglik(
            Y, ITEM_INDS, WORKER_INDS, pv, phi, J, W,
            ITEMS_NUISANCE, WORKER_NUISANCE, dlambda, jaa, jbb, jab, 0);
        ll += 0.5 * AgreementPhi::continuous::log_det_obs_info(
            Y, ITEM_INDS, WORKER_INDS, pv, phi, J, W, ITEMS_NUISANCE, WORKER_NUISANCE);
        ll -= AgreementPhi::continuous::log_det_E0d0d1(
            ITEM_INDS, WORKER_INDS, mle_vec, pv, phi_mle.at(0), phi,
            J, W, ITEMS_NUISANCE, WORKER_NUISANCE);

        if (!std::isfinite(ll)) return std::numeric_limits<double>::infinity();
        return -ll;
    };

    double eps = 1e-8;
    double lower = std::max(phi_mle.at(0) - SEARCH_RANGE, eps);
    double upper = std::min(phi_mle.at(0) + SEARCH_RANGE, 15.0);

    // Phase 1: coarse grid on the agreement scale, always starting from lambda_mle.
    // This avoids warm-start path-dependence that can lead to spurious local maxima.
    const int N_GRID = 20;
    double agr_lower = utils::prec2agr(lower);
    double agr_upper = utils::prec2agr(upper);
    double best_grid_phi  = phi_mle.at(0);
    double best_grid_val  = std::numeric_limits<double>::infinity();
    for(int g = 0; g < N_GRID; ++g){
        double agr_g = agr_lower + (agr_upper - agr_lower) * g / (N_GRID - 1);
        double phi_g = utils::agr2prec(agr_g);
        if(phi_g < lower || phi_g > upper) continue;
        double val = eval_mpl(phi_g, lambda_mle.at(0), lambda_mle.at(1));
        if(val < best_grid_val){
            best_grid_val = val;
            best_grid_phi = phi_g;
        }
    }

    // Phase 2: Brent refinement in a narrow window around the grid best,
    // using warm-starting initialized from lambda_mle at best_grid_phi.
    double agr_step = (agr_upper - agr_lower) / (N_GRID - 1);
    double agr_best = utils::prec2agr(best_grid_phi);
    double brent_lower = utils::agr2prec(std::max(agr_best - 2.0 * agr_step, agr_lower));
    double brent_upper = utils::agr2prec(std::min(agr_best + 2.0 * agr_step, agr_upper));
    brent_lower = std::max(brent_lower, lower);
    brent_upper = std::min(brent_upper, upper);

    if(brent_lower >= brent_upper) brent_upper = std::min(brent_lower + 0.1, upper);

    struct BestWarmStart {
        double loglik = -std::numeric_limits<double>::infinity();
        std::vector<double> alpha;
        std::vector<double> beta;
        bool initialized = false;
    } best;

    auto neg_mpl_ws = [&](double phi) -> double {
        const std::vector<double>& alpha_s = best.initialized ? best.alpha : lambda_mle.at(0);
        const std::vector<double>& beta_s  = best.initialized ? best.beta  : lambda_mle.at(1);

        std::vector<std::vector<double>> profiled = AgreementPhi::continuous::nuisance::get_lambda(
            Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, alpha_s, beta_s,
            phi, J, W, ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

        std::vector<double> pv;
        pv.reserve(J + W - 1);
        pv.insert(pv.end(), profiled.at(0).begin(), profiled.at(0).end());
        pv.insert(pv.end(), profiled.at(1).begin() + 1, profiled.at(1).end());

        Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
        Eigen::VectorXd jaa     = Eigen::VectorXd::Zero(J);
        Eigen::VectorXd jbb     = Eigen::VectorXd::Zero(W - 1);
        Eigen::MatrixXd jab     = Eigen::MatrixXd::Zero(J, W - 1);

        double ll = AgreementPhi::continuous::joint_loglik(
            Y, ITEM_INDS, WORKER_INDS, pv, phi, J, W,
            ITEMS_NUISANCE, WORKER_NUISANCE, dlambda, jaa, jbb, jab, 0);
        ll += 0.5 * AgreementPhi::continuous::log_det_obs_info(
            Y, ITEM_INDS, WORKER_INDS, pv, phi, J, W, ITEMS_NUISANCE, WORKER_NUISANCE);
        ll -= AgreementPhi::continuous::log_det_E0d0d1(
            ITEM_INDS, WORKER_INDS, mle_vec, pv, phi_mle.at(0), phi,
            J, W, ITEMS_NUISANCE, WORKER_NUISANCE);

        if(ll > best.loglik){
            best.loglik      = ll;
            best.alpha       = profiled.at(0);
            best.beta        = profiled.at(1);
            best.initialized = true;
        }
        if(!std::isfinite(ll)) return std::numeric_limits<double>::infinity();
        return -ll;
    };

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_mpl_ws, brent_lower, brent_upper, digits, max_iter
    );

    // Use the grid result if brent find is worse
    double final_phi = result.first;
    double final_val = result.second;
    if(best_grid_val < std::numeric_limits<double>::infinity() && best_grid_val < final_val){
        final_phi = best_grid_phi;
        final_val = best_grid_val;
    }

    if(VERBOSE) Rcpp::Rcout << "Adjusted agreement: " << utils::prec2agr(final_phi) << "\n";

    std::vector<double> out(3);
    out[0] = final_phi;
    out[1] = -final_val;
    out[2] = phi_mle.at(0);

    return out;
}


std::vector<double> AgreementPhi::ordinal::inference::get_phi_profile(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    std::vector<double>& ALPHA,
    std::vector<double>& BETA,
    std::vector<double>& TAU,
    const double PHI_START,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    // Track best warm start
    struct BestWarmStart {
        double loglik = -std::numeric_limits<double>::infinity();
        std::vector<double> alpha;
        std::vector<double> beta;
        std::vector<double> tau;
        bool initialized = false;
    } best;
    
    auto neg_profile_likelihood = [&](double phi){
        // Choose starting point: best if available, else input values
        std::vector<double> alpha_start = best.initialized ? best.alpha : ALPHA;
        std::vector<double> beta_start = best.initialized ? best.beta : BETA;
        std::vector<double> tau_start = best.initialized ? best.tau : TAU;

        // Profile with warm start
        std::vector<std::vector<double>> profiled = AgreementPhi::ordinal::nuisance::get_lambda(
            Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
            alpha_start, beta_start, tau_start, phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
        );

        // Compute likelihood
        Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
        Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
        Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
        Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
        
        std::vector<double> lambda;
        lambda.reserve(J + W - 1);
        lambda.insert(lambda.end(), profiled.at(0).begin(), profiled.at(0).end());
        lambda.insert(lambda.end(), profiled.at(1).begin() + 1, profiled.at(1).end());

        double ll = AgreementPhi::ordinal::joint_loglik(
            Y, ITEM_INDS, WORKER_INDS, lambda, profiled.at(2), phi,
            J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
        );

        // Update best warm start if this is better
        if (ll > best.loglik) {
            best.loglik = ll;
            best.alpha = profiled.at(0);
            best.beta = profiled.at(1);
            best.tau = profiled.at(2);
            best.initialized = true;
        }

        return -ll; 
    };

    double lower = 1e-8; 
    double upper = PHI_START + SEARCH_RANGE;

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_profile_likelihood, lower, upper, digits, max_iter
    );

    ALPHA = best.alpha;
    BETA  = best.beta;
    TAU   = best.tau;

    std::vector<double> out(2);
    out[0] = result.first;
    out[1] = result.second;
    return out;
}

std::vector<double> AgreementPhi::ordinal::inference::get_phi_modified_profile(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    std::vector<double>& ALPHA,
    std::vector<double>& BETA,
    std::vector<double>& TAU,
    const double PHI_START,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    std::vector<double> phi_mle = AgreementPhi::ordinal::inference::get_phi_profile(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA, BETA, TAU, PHI_START,
        J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL, VERBOSE
    );
    
    if(VERBOSE) Rcpp::Rcout << "Non-adjusted agreement: " << utils::prec2agr(phi_mle.at(0)) << "\n";   
    
    std::vector<std::vector<double>> lambda_mle = AgreementPhi::ordinal::nuisance::get_lambda(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA, BETA, TAU, phi_mle.at(0),
        J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );



    // Track best warm start for modified profile
    struct BestWarmStart {
        double loglik = -std::numeric_limits<double>::infinity();
        std::vector<double> alpha;
        std::vector<double> beta;
        bool initialized = false;
    } best;

    auto neg_modified_profile_likelihood = [&](double phi){
        // Start from best so far, or lambda_mle if first call
        std::vector<double> alpha_start = best.initialized ? best.alpha : lambda_mle.at(0);
        std::vector<double> beta_start = best.initialized ? best.beta : lambda_mle.at(1);

        // Profile with warm start
        std::vector<std::vector<double>> profiled = AgreementPhi::ordinal::nuisance::get_lambda(
            Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT,
            alpha_start, beta_start, TAU, phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
        );

        // Compute modified profile likelihood
        Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
        Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
        Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
        Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

        std::vector<double> profiled_vec;
        profiled_vec.reserve(J + W - 1);
        profiled_vec.insert(profiled_vec.end(), profiled.at(0).begin(), profiled.at(0).end());
        profiled_vec.insert(profiled_vec.end(), profiled.at(1).begin() + 1, profiled.at(1).end());

        double ll = AgreementPhi::ordinal::joint_loglik(
            Y, ITEM_INDS, WORKER_INDS, profiled_vec, lambda_mle.at(2), phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
            dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
        );

        ll += 0.5 * AgreementPhi::ordinal::log_det_obs_info(
            Y, ITEM_INDS, WORKER_INDS, profiled_vec, lambda_mle.at(2), phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
        );

        std::vector<double> mle_vec;
        mle_vec.reserve(J + W - 1);
        mle_vec.insert(mle_vec.end(), lambda_mle.at(0).begin(), lambda_mle.at(0).end());
        mle_vec.insert(mle_vec.end(), lambda_mle.at(1).begin() + 1, lambda_mle.at(1).end());

        ll -= AgreementPhi::ordinal::log_det_E0d0d1(
            ITEM_INDS, WORKER_INDS, mle_vec, profiled_vec, phi_mle.at(0), phi, lambda_mle.at(2),
            J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
        );
        
        // Update best warm start if this is better
        if (ll > best.loglik) {
            best.loglik = ll;
            best.alpha = profiled.at(0);
            best.beta = profiled.at(1);
            best.initialized = true;
        }

        return -ll; 
    };

    double eps = 1e-5; 
    double lower, upper;
    if(phi_mle.at(0) < 2.8){
        lower = eps;
        upper = phi_mle.at(0) + 1.0;
    }else{
        lower = std::max(phi_mle.at(0) - SEARCH_RANGE, eps);
        upper = std::min(phi_mle.at(0) + SEARCH_RANGE, 15.0);
    }
    
    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_modified_profile_likelihood, lower, upper, digits, max_iter
    );

    if(VERBOSE) Rcpp::Rcout << "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

    
    std::vector<double> out(3); 
    out[0] = result.first;
    out[1] = -result.second;
    out[2] = phi_mle.at(0);

    return out;
}
