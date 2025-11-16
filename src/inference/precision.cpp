#include "precision.h"

///////////////////////////////////////
// CONTINUOUS RATINGS | ONEWAY MODEL //
///////////////////////////////////////

std::pair<double, double> AgreementPhi::continuous::oneway::inference::get_phi_profile(
    const std::vector<double> Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double> ALPHA_START,
    const double PHI_START,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD
){
    auto neg_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::continuous::oneway::loglik::profile(
            phi, Y, DICT, ALPHA_START, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
        return -ll; 
    };

    double lower = 1e-8; 
    double upper = PHI_START + SEARCH_RANGE;

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_profile_likelihood, lower, upper, digits, max_iter
    );

    return result;
}

std::vector<double> AgreementPhi::continuous::oneway::inference::get_phi_modified_profile(
    const std::vector<double> Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double> ALPHA_START,
    const double PHI_START,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool VERBOSE
){
    // get mle for phi
    std::pair<double, double> phi_mle = get_phi_profile(
        Y, DICT, ALPHA_START, PHI_START, J,
        SEARCH_RANGE, MAX_ITER,
        PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
    );

    if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.first) << "\n";

    // get mle for alpha
    std::vector<double> alpha_mle(J);
    for(int j=0; j<J; j++){

        if(PROF_METHOD==0){
            double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j) = hatalpha;
        }else{
            double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j) = hatalpha;
        }
    }

    // negative log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::continuous::oneway::loglik::modified_profile(
            phi, Y, DICT, alpha_mle, alpha_mle, phi_mle.first, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
        return -ll; 
    };

    double lower = 1e-8; 
    double upper = phi_mle.first + SEARCH_RANGE;

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_modified_profile_likelihood, lower, upper, digits, max_iter
    );


    if(VERBOSE) Rcpp::Rcout<< "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

    std::vector<double> out(3); 
    out[0] = result.first;   // estimate
    out[1] = -result.second; // loglik
    out[2] = phi_mle.first;  // non-adjusted

    return out;
}

///////////////////////////////////////
// ORDINAL RATINGS | ONEWAY MODEL //
///////////////////////////////////////

std::pair<double, double> AgreementPhi::ordinal::oneway::inference::get_phi_profile(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double>& ALPHA_START,
    const double PHI_START,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD
){
    auto neg_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::ordinal::oneway::loglik::profile(
            phi, Y, DICT, ALPHA_START, K, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
        return -ll; 
    };

    double lower = 1e-8; 
    double upper = PHI_START + SEARCH_RANGE;


    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_profile_likelihood, lower, upper, digits, max_iter
    );

    return result;
}

std::vector<double> AgreementPhi::ordinal::oneway::inference::get_phi_modified_profile(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double>& ALPHA_START,
    const double PHI_START,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool VERBOSE
){

    // get mle for phi
    std::pair<double, double> phi_mle = get_phi_profile(
        Y, DICT, ALPHA_START, PHI_START, K, J,
        SEARCH_RANGE, MAX_ITER,
        PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
    );

    if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.first) << "\n";

    // get mle for alpha
    std::vector<double> alpha_mle(J);
    for(int j=0; j<J; j++){

        if(PROF_METHOD==0){
            double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j)=hatalpha;
        }else{
            double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j)=hatalpha;
        }
    }


    // negative log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::ordinal::oneway::loglik::modified_profile(
            phi, Y, DICT, alpha_mle, alpha_mle, phi_mle.first, K, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
        return -ll; 
    };

    double lower = 1e-8; 
    double upper = phi_mle.first + SEARCH_RANGE;


    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_modified_profile_likelihood, lower, upper, digits, max_iter
    );

    // double opt = result.first; 

    if(VERBOSE) Rcpp::Rcout<< "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

    std::vector<double> out(3); 
    out[0] = result.first;
    out[1] = -result.second;
    out[2] = phi_mle.first;

    return out;


}


///////////////////////////////////////
// CONTINUOUS RATINGS | TWOWAY MODEL //
///////////////////////////////////////
std::pair<double, double> AgreementPhi::continuous::twoway::inference::get_phi_profile(
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
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){
    auto neg_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::continuous::twoway::loglik::profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA, BETA, phi, J, W, PROF_UNI_RANGE,
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

    return result;
}


std::vector<double> AgreementPhi::continuous::twoway::inference::get_phi_modified_profile(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<double> ALPHA_START,
    const std::vector<double> BETA_START,
    const double PHI_START,
    const int J,
    const int W,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    // get mle for phi via profile likleihood
    std::pair<double, double> phi_mle = AgreementPhi::continuous::twoway::inference::get_phi_profile(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA_START, BETA_START, PHI_START, J, W, SEARCH_RANGE, MAX_ITER, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
    
    if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.first) << "\n";   
    
    // get mle for lambda
    std::vector<std::vector<double>> lambda_mle = AgreementPhi::continuous::twoway::inference::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA_START,  BETA_START, phi_mle.first, J, W, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    // negative modified profile log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::continuous::twoway::loglik::modified_profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, lambda_mle.at(0), lambda_mle.at(1), phi, phi_mle.first, J, W, PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
        return -ll; 
    };

    double eps = 1e-8; 
    double lower = std::max(phi_mle.first - SEARCH_RANGE, eps);
    double upper = std::min(phi_mle.first + SEARCH_RANGE, 15.0);

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_modified_profile_likelihood, lower, upper, digits, max_iter
    );


    if(VERBOSE) Rcpp::Rcout<< "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

    std::vector<double> out(3); 
    out[0] = result.first;   // estimate
    out[1] = -result.second; // loglik
    out[2] = phi_mle.first;  // non-adjusted

    return out;

    
}

///////////////////////////////////////
// ORDINAL RATINGS | TWOWAY MODEL //
///////////////////////////////////////
std::pair<double, double> AgreementPhi::ordinal::twoway::inference::get_phi_profile(
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
    const int K,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){
    auto neg_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::ordinal::twoway::loglik::profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA, BETA, phi, J, W, K, PROF_UNI_RANGE,
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

    return result;
}

std::vector<double> AgreementPhi::ordinal::twoway::inference::get_phi_modified_profile(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<double> ALPHA_START,
    const std::vector<double> BETA_START,
    const double PHI_START,
    const int J,
    const int W,
    const int K,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL,
    const bool VERBOSE
){
    // get mle for phi via profile likleihood
    std::pair<double, double> phi_mle = AgreementPhi::ordinal::twoway::inference::get_phi_profile(
        Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA_START, BETA_START, PHI_START, J, W, K, SEARCH_RANGE, MAX_ITER, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
    
    if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.first) << "\n";   
    
    // get mle for lambda
    std::vector<std::vector<double>> lambda_mle = AgreementPhi::ordinal::twoway::inference::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA_START,  BETA_START, phi_mle.first, J, W, K, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    // negative modified profile log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = AgreementPhi::ordinal::twoway::loglik::modified_profile(
                Y, ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, lambda_mle.at(0), lambda_mle.at(1), phi, phi_mle.first, J, W, K, PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);
        return -ll; 
    };

    double eps = 1e-5; 
    double lower, upper;
    if(phi_mle.first<2.8){
        lower = eps;
        upper = phi_mle.first + 1.0;
    }else{
        lower = std::max(phi_mle.first - SEARCH_RANGE, eps);
        upper = std::min(phi_mle.first + SEARCH_RANGE, 15.0);
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
    out[2] = phi_mle.first;  // non-adjusted

    return out;

    
}