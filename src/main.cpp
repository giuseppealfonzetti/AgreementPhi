#include <Rcpp.h>
#include <boost/math/tools/minima.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include <functional>
#include <Eigen/Dense>
#include "LBFGS.h"

#include "rcpptests.h"
#include "utilities/beta_functions.h"
#include "utilities/link_functions.h"
#include "utilities/utils_functions.h"
#include "ratings/continuous.h"
#include "ratings/ordinal.h"
#include "models/oneway.h"
#include "inference/profile.h"





// [[Rcpp::export]]
std::vector<double> cpp_get_phi_mle(
    const std::vector<double> Y,
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA_START,
    const double PHI_START,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool VERBOSE = false,
    const bool CONTINUOUS = false)
{
    std::vector<std::vector<int>> dict = AgreementPhi::utils::oneway_items_dict(J, ITEM_INDS);
    
    std::pair<double, double> opt;
    if(CONTINUOUS){
        opt = AgreementPhi::continuous::oneway::inference::get_phi_profile(
            Y, dict, ALPHA_START, PHI_START, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
        );
    }else{
        opt = AgreementPhi::ordinal::oneway::inference::get_phi_profile(
            Y, dict, ALPHA_START, PHI_START, K, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
        );
    }
    
    std::vector<double> out(2); 
    out[0] = opt.first;
    out[1] = -opt.second;

    return out;
}

// [[Rcpp::export]]
std::vector<double> cpp_get_phi_mp(
    const std::vector<double> Y,
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA_START,
    const double PHI_START,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool VERBOSE = false,
    const bool CONTINUOUS = false)
{
    std::vector<std::vector<int>> dict = AgreementPhi::utils::oneway_items_dict(J, ITEM_INDS);
    
    std::vector<double> opt;
    if(CONTINUOUS){
        opt = AgreementPhi::continuous::oneway::inference::get_phi_modified_profile(
            Y, dict, ALPHA_START, PHI_START, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD, VERBOSE
        );
    }else{
        opt = AgreementPhi::ordinal::oneway::inference::get_phi_modified_profile(
            Y, dict, ALPHA_START, PHI_START, K, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD, VERBOSE
        );
    }
    
    

    return opt;
}


// [[Rcpp::export]]
double cpp_profile_likelihood(
    const std::vector<double> Y,
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA_START,
    const double PHI,
    const int K,
    const int J,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool CONTINUOUS
){
    std::vector<std::vector<int>> dict = AgreementPhi::utils::oneway_items_dict(J, ITEM_INDS);

    double out;
    if(CONTINUOUS){
        out = AgreementPhi::continuous::oneway::inference::profile_loglik(
                PHI, Y, dict, ALPHA_START, J,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }else{
        out = AgreementPhi::ordinal::oneway::inference::profile_loglik(
                PHI, Y, dict, ALPHA_START, K, J,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }

    return out;

}

// [[Rcpp::export]]
double cpp_modified_profile_likelihood(
    const std::vector<double> Y,
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA_START,
    const double PHI_MLE,
    const double PHI,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool CONTINUOUS
){
    // get mle for phi
    std::vector<std::vector<int>> dict = AgreementPhi::utils::oneway_items_dict(J, ITEM_INDS);

    
    

    double out;
    if(CONTINUOUS){
        // get mle for alpha
        std::vector<double> alpha_mle(J);
        for(int j=0; j<J; j++){

            if(PROF_METHOD==0){
                double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_brent(
                    Y, dict, j, ALPHA_START.at(j), PHI_MLE, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle.at(j) = hatalpha;
            }else{
                double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_newtonraphson(
                    Y, dict, j, ALPHA_START.at(j), PHI_MLE, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle.at(j) = hatalpha;
            }
        }
        out = AgreementPhi::continuous::oneway::inference::modified_profile_loglik(
                PHI, Y, dict, alpha_mle, alpha_mle, PHI_MLE, J, 
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }else{
        std::vector<double> alpha_mle(J);
        for(int j=0; j<J; j++){

            if(PROF_METHOD==0){
                double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_brent(
                    Y, dict, j, ALPHA_START.at(j), PHI_MLE, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle.at(j) = hatalpha;
            }else{
                double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_newtonraphson(
                    Y, dict, j, ALPHA_START.at(j), PHI_MLE, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle.at(j) = hatalpha;
            }
        }
        
        out = AgreementPhi::ordinal::oneway::inference::modified_profile_loglik(
                PHI, Y, dict, alpha_mle, alpha_mle, PHI_MLE, K, J, 
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }

    return out;

}

// [[Rcpp::export]]
double cpp_get_se(
    const std::vector<double> Y,
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA_START,
    const double PHI_EVAL,
    const double PHI_MLE,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool MODIFIED =true,
    const bool CONTINUOUS = true
) {
        
    double agr = AgreementPhi::utils::prec2agr(PHI_EVAL);
    
    std::function<double(double)> f;
    
    if(MODIFIED){
        f = [&](double agr){
            double phi = AgreementPhi::utils::agr2prec(agr);
            
            double out = cpp_modified_profile_likelihood(
                Y, ITEM_INDS, ALPHA_START, 
                PHI_MLE, phi, K, J,  // Fixed: PHI is phi_mle, phi is test value
                SEARCH_RANGE, MAX_ITER, 
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD,
                CONTINUOUS
            );
            return out;
        };
    }else{
        f = [&](double agr){
            double phi = AgreementPhi::utils::agr2prec(agr);
            
            double out = cpp_profile_likelihood(
                Y, ITEM_INDS, ALPHA_START, 
                phi, K, J,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD,
                CONTINUOUS
            );
            return out;
        };
    }
    
    double d2 = boost::math::differentiation::finite_difference_derivative(
        [&](double x){ 
            return boost::math::differentiation::finite_difference_derivative(f, x); 
        },
        agr
    );
    
    if(-d2 > 0){
        return 1.0 / sqrt(-d2);
    }else{
        return std::numeric_limits<double>::quiet_NaN();
    }

}
