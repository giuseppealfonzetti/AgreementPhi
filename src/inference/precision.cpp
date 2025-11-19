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
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const double PHI_START,
    const int J,
    const int W,
    const bool WORKER_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    auto neg_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::continuous::ll::profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA, BETA, phi, J, W, WORKER_NUISANCE, PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
        return -ll; 
    };

    double lower = 1e-8; 
    double upper = PHI_START + SEARCH_RANGE;

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_profile_likelihood, lower, upper, digits, max_iter
    );

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
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const double PHI_START,
    const int J,
    const int W,
    const bool WORKER_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    // get mle for phi via profile likleihood
    std::vector<double> phi_mle = AgreementPhi::continuous::inference::get_phi_profile(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA, BETA, PHI_START, J, W,WORKER_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL, VERBOSE);
    
    if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.at(0)) << "\n";   
    
    // get mle for lambda
    std::vector<std::vector<double>> lambda_mle = AgreementPhi::continuous::nuisance::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA,  BETA, phi_mle.at(0), J, W, WORKER_NUISANCE, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    // negative modified profile log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::continuous::ll::modified_profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, lambda_mle.at(0), lambda_mle.at(1), phi, phi_mle.at(0), J, W, WORKER_NUISANCE, PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
        return -ll; 
    };

    double eps = 1e-8; 
    double lower = std::max(phi_mle.at(0) - SEARCH_RANGE, eps);
    double upper = std::min(phi_mle.at(0) + SEARCH_RANGE, 15.0);

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_modified_profile_likelihood, lower, upper, digits, max_iter
    );


    if(VERBOSE) Rcpp::Rcout<< "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

    std::vector<double> out(3); 
    out[0] = result.first;   // estimate
    out[1] = -result.second; // loglik
    out[2] = phi_mle.at(0);  // non-adjusted

    return out;

    
}

///////////////////////
// ORDINAL RATINGS  //
///////////////////////
std::vector<double> AgreementPhi::ordinal::inference::get_phi_profile(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const std::vector<double> TAU,
    const double PHI_START,
    const int J,
    const int W,
    const int K,
    const bool WORKER_NUISANCE,
    const bool THRESHOLDS_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    auto neg_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::ordinal::ll::profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA, BETA, TAU, phi, J, W, K, WORKER_NUISANCE, THRESHOLDS_NUISANCE, PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
        return -ll; 
    };

    double lower = 1e-8; 
    double upper = PHI_START + SEARCH_RANGE;

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_profile_likelihood, lower, upper, digits, max_iter
    );

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
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const std::vector<double> TAU,
    const double PHI_START,
    const int J,
    const int W,
    const int K,
    const bool WORKER_NUISANCE,
    const bool THRESHOLDS_NUISANCE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    // get mle for phi via profile likleihood
    std::vector<double> phi_mle = AgreementPhi::ordinal::inference::get_phi_profile(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA, BETA, TAU, PHI_START, J, W, K, WORKER_NUISANCE, THRESHOLDS_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL, VERBOSE);
    
    if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.at(0)) << "\n";   
    

    // get mle for lambda
    std::vector<std::vector<double>> lambda_mle = AgreementPhi::ordinal::nuisance::get_lambda2(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA,  BETA, TAU, phi_mle.at(0), J, W, K, WORKER_NUISANCE, THRESHOLDS_NUISANCE, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    
    Rcpp::Rcout<<"tau: ";
    for (double i: lambda_mle.at(2))
    Rcpp::Rcout << i << ' ';
    Rcpp::Rcout<<"\n";

    // negative modified profile log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::ordinal::ll::modified_profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, lambda_mle.at(0), lambda_mle.at(1), lambda_mle.at(2), phi, phi_mle.at(0), J, W, K, WORKER_NUISANCE, THRESHOLDS_NUISANCE, PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
        return -ll; 
    };

    double eps = 1e-5; 
    double lower, upper;
    if(phi_mle.at(0)<2.8){
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


    if(VERBOSE) Rcpp::Rcout<< "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

    std::vector<double> out(3); 
    out[0] = result.first;   // estimate
    out[1] = -result.second; // loglik
    out[2] = phi_mle.at(0);  // non-adjusted

    return out;

    
}

