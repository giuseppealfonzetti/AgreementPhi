#include "profile.h"

///////////////////////////////////////
// CONTINUOUS RATINGS | ONEWAY MODEL //
///////////////////////////////////////

double AgreementPhi::continuous::oneway::inference::profiling_brent(
    const std::vector<double> Y, 
    const std::vector<std::vector<int>> DICT,
    const int ITEM,
    const double ALPHA_START,
    const double PHI,
    const int RANGE,
    const int MAX_ITER
){
    double grad, grad2;
    auto neg_ll = [&](double alpha){
        double ll = AgreementPhi::continuous::oneway::item_loglik(Y, DICT, ITEM, alpha, PHI, grad, grad2, 0);
        return -ll; 
    };

    double lower = ALPHA_START - RANGE; 
    double upper = ALPHA_START + RANGE;

    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_ll, lower, upper, digits, max_iter
    );

    double opt = result.first; 
    return opt;
}

double AgreementPhi::continuous::oneway::inference::profiling_newtonraphson(
    const std::vector<double> Y, 
    const std::vector<std::vector<int>> DICT,
    const int ITEM,
    const double ALPHA_START,
    const double PHI,
    const int RANGE,
    const int MAX_ITER
){
    auto neg_ll = [&](double alpha){
        double grad = 0;
        double grad2 = 0;
        double ll = AgreementPhi::continuous::oneway::item_loglik(Y, DICT, ITEM, alpha, PHI, grad, grad2, 2);
        return std::make_pair(grad, grad2); 
    };

    double lower = ALPHA_START - RANGE; 
    double upper = ALPHA_START + RANGE;

    const int digits = std::numeric_limits<double>::digits;
    int get_digits = static_cast<int>(digits * 0.4);
    boost::uintmax_t max_iter = MAX_ITER;
    double out = boost::math::tools::newton_raphson_iterate(
        neg_ll,           
        ALPHA_START,     
        lower,           
        upper,           
        get_digits,      
        max_iter);

    return out;
}

double AgreementPhi::continuous::oneway::inference::profile_loglik(
    double PHI,
    const std::vector<double> Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double> ALPHA_START,
    const int J,
    const int RANGE,
    const int MAX_ITER,
    const int METHOD
){
    double grad, grad2;
    double ll = 0;
    std::vector<double> alpha(J);
    for(int j=0; j<J; j++){

        // profiling
        if(METHOD==0){
            double hatalpha = profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, RANGE, MAX_ITER
            );
            alpha.at(j) = hatalpha;
        }else{
            double hatalpha = profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), PHI, RANGE, MAX_ITER
            );
            alpha.at(j) = hatalpha;
        }

        // Evaluate ll contribution
        double llj = AgreementPhi::continuous::oneway::item_loglik(Y, DICT, j, alpha.at(j), PHI, grad, grad2, 0);

        ll += llj;
    }

    return ll;
}

double AgreementPhi::continuous::oneway::inference::modified_profile_loglik(
    double PHI,
    const std::vector<double> Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double> ALPHA_START,
    const std::vector<double> ALPHA_MLE,
    const double PHI_MLE,
    const int J,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD
){
    double grad, grad2;
    double ll = 0;
    std::vector<double> prof_alpha(J);
    for(int j=0; j<J; j++){

        // profiling
        if(PROF_METHOD==0){
            double hatalpha = profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            prof_alpha.at(j) = hatalpha;
        }else{
            double hatalpha = profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), PHI, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            prof_alpha.at(j) = hatalpha;
        }

        // Evaluate ll contribution
        double llj = AgreementPhi::continuous::oneway::item_loglik(Y, DICT, j, prof_alpha.at(j), PHI, grad, grad2, 0);

        ll += llj;
    }

    // add modifier
    ll += .5 * AgreementPhi::continuous::oneway::log_det_obs_info(Y, DICT, prof_alpha, PHI);

    ll -= AgreementPhi::continuous::oneway::log_det_E0d0d1(DICT, ALPHA_MLE, prof_alpha, PHI_MLE, PHI);

    return ll;
}
            

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
        double ll = profile_loglik(
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
            double hatalpha = profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j) = hatalpha;
        }else{
            double hatalpha = profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j) = hatalpha;
        }
    }

    // negative log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = modified_profile_loglik(
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


////////////////////////////////////
// ORDINAL RATINGS | ONEWAY MODEL //
////////////////////////////////////
double AgreementPhi::ordinal::oneway::inference::profiling_brent(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const int ITEM,
    const double ALPHA_START,
    const double PHI,
    const int K,
    const int RANGE,
    const int MAX_ITER
){

    double grad, grad2;
    auto neg_ll = [&](double alpha){
        double ll = AgreementPhi::ordinal::oneway::item_loglik(Y, DICT, ITEM, alpha, PHI, K, grad, grad2, 0);
        return -ll; 
    };

    double lower = ALPHA_START - RANGE; 
    double upper = ALPHA_START + RANGE;


    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_ll, lower, upper, digits, max_iter
    );

    double opt = result.first; 
    return opt;
}

double AgreementPhi::ordinal::oneway::inference::profiling_newtonraphson(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const int ITEM,
    const double ALPHA_START,
    const double PHI,
    const int K,
    const int RANGE,
    const int MAX_ITER
){
    auto neg_ll = [&](double alpha){
        double grad = 0;
        double grad2 = 0;
        double ll = AgreementPhi::ordinal::oneway::item_loglik(Y, DICT, ITEM, alpha, PHI, K, grad, grad2, 2);
        return std::make_pair(grad, grad2); 
    };

    double lower = ALPHA_START - RANGE; 
    double upper = ALPHA_START + RANGE;

    const int digits = std::numeric_limits<double>::digits;
    int get_digits = static_cast<int>(digits * 0.4);
    boost::uintmax_t max_iter = MAX_ITER;
    double out = boost::math::tools::newton_raphson_iterate(
        neg_ll,           
        ALPHA_START,     
        lower,           
        upper,           
        get_digits,      
        max_iter);

    return out;
}

double AgreementPhi::ordinal::oneway::inference::profile_loglik(
    double PHI,
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double>& ALPHA_START,
    const int K,
    const int J,
    const int RANGE,
    const int MAX_ITER,
    const int METHOD
){

    double grad, grad2;
    double ll = 0;
    std::vector<double> alpha(J);
    for(int j=0; j<J; j++){

        // profiling
        if(METHOD==0){
            double hatalpha = profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, K, RANGE, MAX_ITER
            );
            alpha.at(j)=hatalpha;
        }else{
            double hatalpha = profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), PHI, K, RANGE, MAX_ITER
            );
            alpha.at(j)=hatalpha;
        }

        // Evaluate ll contribution
        double llj = AgreementPhi::ordinal::oneway::item_loglik(Y, DICT, j, alpha.at(j), PHI, K, grad, grad2, 0);

        ll+=llj;
    }


    return ll;
}

double AgreementPhi::ordinal::oneway::inference::modified_profile_loglik(
    double PHI,
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double>& ALPHA_START,
    const std::vector<double>& ALPHA_MLE,
    const double PHI_MLE,
    const int K,
    const int J,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD
){

    double grad, grad2;
    double ll = 0;
    std::vector<double> prof_alpha(J);
    for(int j=0; j<J; j++){

        // profiling
        if(PROF_METHOD==0){
            double hatalpha = profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            prof_alpha.at(j)=hatalpha;
        }else{
            double hatalpha = profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), PHI, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            prof_alpha.at(j)=hatalpha;
        }

        // Evaluate ll contribution
        double llj = AgreementPhi::ordinal::oneway::item_loglik(Y, DICT, j, prof_alpha.at(j), PHI, K, grad, grad2, 0);

        ll+=llj;
    }

    // add modifier
    ll += .5 * ordinal::oneway::log_det_obs_info(Y, DICT, prof_alpha, PHI, K);

    ll -= ordinal::oneway::log_det_E0d0d1(DICT, ALPHA_MLE, prof_alpha, PHI_MLE, PHI, K);

    return ll;
}

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
        double ll = profile_loglik(
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
            double hatalpha = profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j)=hatalpha;
        }else{
            double hatalpha = profiling_newtonraphson(
                Y, DICT, j, ALPHA_START.at(j), phi_mle.first, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            alpha_mle.at(j)=hatalpha;
        }
    }


    // negative log-likelihood to minimize
    auto neg_modified_profile_likelihood = [&](double phi){
        double ll = modified_profile_loglik(
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