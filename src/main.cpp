#include <Rcpp.h>
#include <boost/math/tools/minima.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include <functional>
#include <Eigen/Dense>
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define EIGEN_DONT_PARALLELIZE
#include "rcpptests.h"
#include "utilities/beta_functions.h"
#include "utilities/link_functions.h"
#include "utilities/utils_functions.h"
#include "ratings/continuous.h"
#include "ratings/ordinal.h"
#include "inference/nuisance.h"
#include "inference/precision.h"
#include <stdexcept>

////////////////////////
// SHARED  INTERFACE //
////////////////////////
// [[Rcpp::export]]
Rcpp::List cpp_get_phi(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<double> ALPHA_START,
    const std::vector<double> BETA_START,
    const std::vector<double> TAU_START,
    const double PHI_START,
    const int J,
    const int W,
    const int K,
    const std::string METHOD,
    const std::string DATA_TYPE,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const bool THRESHOLDS_NUISANCE,
    const bool VERBOSE,
    const double SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int ALT_MAX_ITER,
    const double ALT_TOL)
{
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W, WORKER_INDS);
    std::vector<double> res;
    double ll;
    double profile_phi;
    double modified_phi = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<double> tau;


    if(METHOD == "profile"){
        if(DATA_TYPE=="continuous"){
            res = AgreementPhi::continuous::inference::get_phi_profile(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START, BETA_START, PHI_START, J, W, ITEMS_NUISANCE, WORKER_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            profile_phi = res.at(0);
            ll = res.at(1);
            std::vector<std::vector<double>> lambda= AgreementPhi::continuous::nuisance::get_lambda(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START,  BETA_START, profile_phi, J, W, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_SEARCH_RANGE,
                PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL);
            alpha=lambda.at(0);
            beta=lambda.at(1);
            
        }else if(DATA_TYPE=="ordinal"){

            std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);
            res = AgreementPhi::ordinal::inference::get_phi_profile(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, ALPHA_START, BETA_START, TAU_START, PHI_START, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, THRESHOLDS_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            profile_phi = res.at(0);
            ll = res.at(1);

            std::vector<std::vector<double>> lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, ALPHA_START,  BETA_START, TAU_START, profile_phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, THRESHOLDS_NUISANCE, PROF_SEARCH_RANGE,
                PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL);
            alpha=lambda.at(0);
            beta=lambda.at(1);
            tau=lambda.at(2);
            
        }else{
            throw std::invalid_argument("Invalid DATA_TYPE");
        }

        

        
    }else if(METHOD == "modified"){
        if(DATA_TYPE=="continuous"){
            res = AgreementPhi::continuous::inference::get_phi_modified_profile(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START, BETA_START, PHI_START, J, W, ITEMS_NUISANCE, WORKER_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            modified_phi = res.at(0);
            profile_phi = res.at(2);
            ll = res.at(1);
            std::vector<std::vector<double>> lambda= AgreementPhi::continuous::nuisance::get_lambda(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START,  BETA_START, modified_phi, J, W, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_SEARCH_RANGE,
                PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL);
            alpha=lambda.at(0);
            beta=lambda.at(1);
            
        }else if(DATA_TYPE=="ordinal"){
            std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);
            res = AgreementPhi::ordinal::inference::get_phi_modified_profile(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, ALPHA_START, BETA_START, TAU_START, PHI_START, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, THRESHOLDS_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            modified_phi = res.at(0);
            profile_phi = res.at(2);
            ll = res.at(1);

            std::vector<std::vector<double>> lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, ALPHA_START,  BETA_START, TAU_START, modified_phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, THRESHOLDS_NUISANCE, PROF_SEARCH_RANGE,
                PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL);
            alpha=lambda.at(0);
            beta=lambda.at(1);
            tau=lambda.at(2);

        }else{
            throw std::invalid_argument("Invalid DATA_TYPE");
        }
    }else{
       throw std::invalid_argument("Invalid METHOD");

    }
    


    Rcpp::List output = Rcpp::List::create(
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("beta") = beta,
        Rcpp::Named("tau") = tau,
        Rcpp::Named("loglik") = ll,
        Rcpp::Named("profile_phi") = profile_phi,
        Rcpp::Named("modified_phi") = modified_phi
    );

  return(output);
    
    

}

// ///////////////////////////////
// // ONE-WAY MODELS  INTERFACE //
// //////////////////////////////
// // [[Rcpp::export]]
// std::vector<double> cpp_get_phi_oneway_profile(
//     const std::vector<double> Y,
//     const std::vector<double> ITEM_INDS,
//     const std::vector<double> ALPHA_START,
//     const double PHI_START,
//     const int K,
//     const int J,
//     const int SEARCH_RANGE,
//     const int MAX_ITER,
//     const int PROF_SEARCH_RANGE,
//     const int PROF_MAX_ITER,
//     const int PROF_METHOD,
//     const bool VERBOSE = false,
//     const bool CONTINUOUS = false)
// {
//     std::vector<std::vector<int>> dict = AgreementPhi::utils::oneway_items_dict(J, ITEM_INDS);
    
//     std::pair<double, double> opt;
//     if(CONTINUOUS){
//         opt = AgreementPhi::continuous::oneway::inference::get_phi_profile(
//             Y, dict, ALPHA_START, PHI_START, J,
//             SEARCH_RANGE, MAX_ITER,
//             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
//         );
//     }else{
//         opt = AgreementPhi::ordinal::oneway::inference::get_phi_profile(
//             Y, dict, ALPHA_START, PHI_START, K, J,
//             SEARCH_RANGE, MAX_ITER,
//             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
//         );
//     }
    
//     std::vector<double> out(2); 
//     out[0] = opt.first;
//     out[1] = -opt.second;

//     return out;
// }

// // [[Rcpp::export]]
// std::vector<double> cpp_get_phi_oneway_modified_profile(
//     const std::vector<double> Y,
//     const std::vector<double> ITEM_INDS,
//     const std::vector<double> ALPHA_START,
//     const double PHI_START,
//     const int K,
//     const int J,
//     const int SEARCH_RANGE,
//     const int MAX_ITER,
//     const int PROF_SEARCH_RANGE,
//     const int PROF_MAX_ITER,
//     const int PROF_METHOD,
//     const bool VERBOSE = false,
//     const bool CONTINUOUS = false)
// {
//     std::vector<std::vector<int>> dict = AgreementPhi::utils::oneway_items_dict(J, ITEM_INDS);
    
//     std::vector<double> opt;
//     if(CONTINUOUS){
//         opt = AgreementPhi::continuous::oneway::inference::get_phi_modified_profile(
//             Y, dict, ALPHA_START, PHI_START, J,
//             SEARCH_RANGE, MAX_ITER,
//             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD, VERBOSE
//         );
//     }else{
//         opt = AgreementPhi::ordinal::oneway::inference::get_phi_modified_profile(
//             Y, dict, ALPHA_START, PHI_START, K, J,
//             SEARCH_RANGE, MAX_ITER,
//             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD, VERBOSE
//         );
//     }
    
    

//     return opt;
// }






// ///////////////////////////////
// // TWO-WAY MODELS  INTERFACE //
// //////////////////////////////

// // [[Rcpp::export]]
// std::vector<double> cpp_get_phi_twoway_profile(
//     const std::vector<double> Y,  
//     const std::vector<int> ITEM_INDS,
//     const std::vector<int> WORKER_INDS,
//     const std::vector<double> ALPHA_START,
//     const std::vector<double> BETA_START,
//     const double PHI_START,
//     const int J,
//     const int W,
//     const int K,
//     const double SEARCH_RANGE,
//     const int MAX_ITER,
//     const int PROF_SEARCH_RANGE,
//     const int PROF_MAX_ITER,
//     const int ALT_MAX_ITER,
//     const double ALT_TOL,
//     const bool CONTINUOUS
// ){

//     std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
//     std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W, WORKER_INDS);

//     std::pair<double, double> res;
//     if(CONTINUOUS){
//         res = AgreementPhi::continuous::twoway::inference::get_phi_profile(
//                 Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START, BETA_START, PHI_START, J, W, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE,
//                 PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL);
//     }else{
//         res = AgreementPhi::ordinal::twoway::inference::get_phi_profile(
//                 Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START, BETA_START, PHI_START, J, W, K, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE,
//                 PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL);
//     }

//     std::vector<double> out(2);
//     out[0]=res.first;
//     out[1]=res.second;
//     return out;

// }

// // [[Rcpp::export]]
// std::vector<double> cpp_get_phi_twoway_modified_profile(
//     const std::vector<double> Y,  
//     const std::vector<int> ITEM_INDS,
//     const std::vector<int> WORKER_INDS,
//     const std::vector<double> ALPHA_START,
//     const std::vector<double> BETA_START,
//     const double PHI_START,
//     const int J,
//     const int W,
//     const int K,
//     const double SEARCH_RANGE,
//     const int MAX_ITER,
//     const int PROF_SEARCH_RANGE,
//     const int PROF_MAX_ITER,
//     const int ALT_MAX_ITER,
//     const double ALT_TOL,
//     const bool CONTINUOUS,
//     const bool VERBOSE
// ){

//     std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
//     std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W, WORKER_INDS);

//     std::vector<double> res;
//     if(CONTINUOUS){
//         res = AgreementPhi::continuous::twoway::inference::get_phi_modified_profile(
//                 Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START, BETA_START, PHI_START, J, W, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE,
//                 PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
//     }else{
//         res = AgreementPhi::ordinal::twoway::inference::get_phi_modified_profile(
//                 Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, ALPHA_START, BETA_START, PHI_START, J, W, K, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE,
//                 PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
//     }

//     return res;
// }



///////////////////////////////
// TO BE SORTED //
//////////////////////////////
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
    // if(CONTINUOUS){
    //     out = AgreementPhi::continuous::oneway::loglik::profile(
    //             PHI, Y, dict, ALPHA_START, J,
    //             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    // }else{
    //     out = AgreementPhi::ordinal::oneway::loglik::profile(
    //             PHI, Y, dict, ALPHA_START, K, J,
    //             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    // }

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
    // if(CONTINUOUS){
    //     // get mle for alpha
    //     std::vector<double> alpha_mle(J);
    //     for(int j=0; j<J; j++){

    //         if(PROF_METHOD==0){
    //             double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_brent(
    //                 Y, dict, j, ALPHA_START.at(j), PHI_MLE, PROF_SEARCH_RANGE, PROF_MAX_ITER
    //             );
    //             alpha_mle.at(j) = hatalpha;
    //         }else{
    //             double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_newtonraphson(
    //                 Y, dict, j, ALPHA_START.at(j), PHI_MLE, PROF_SEARCH_RANGE, PROF_MAX_ITER
    //             );
    //             alpha_mle.at(j) = hatalpha;
    //         }
    //     }
    //     out = AgreementPhi::continuous::oneway::loglik::modified_profile(
    //             PHI, Y, dict, alpha_mle, alpha_mle, PHI_MLE, J, 
    //             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    // }else{
    //     std::vector<double> alpha_mle(J);
    //     for(int j=0; j<J; j++){

    //         if(PROF_METHOD==0){
    //             double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_brent(
    //                 Y, dict, j, ALPHA_START.at(j), PHI_MLE, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
    //             );
    //             alpha_mle.at(j) = hatalpha;
    //         }else{
    //             double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_newtonraphson(
    //                 Y, dict, j, ALPHA_START.at(j), PHI_MLE, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
    //             );
    //             alpha_mle.at(j) = hatalpha;
    //         }
    //     }
        
    //     out = AgreementPhi::ordinal::oneway::loglik::modified_profile(
    //             PHI, Y, dict, alpha_mle, alpha_mle, PHI_MLE, K, J, 
    //             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    // }

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
    
    // if(MODIFIED){
    //     f = [&](double agr){
    //         double phi = AgreementPhi::utils::agr2prec(agr);
            
    //         double out = cpp_modified_profile_likelihood(
    //             Y, ITEM_INDS, ALPHA_START, 
    //             PHI_MLE, phi, K, J,  // Fixed: PHI is phi_mle, phi is test value
    //             SEARCH_RANGE, MAX_ITER, 
    //             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD,
    //             CONTINUOUS
    //         );
    //         return out;
    //     };
    // }else{
    //     f = [&](double agr){
    //         double phi = AgreementPhi::utils::agr2prec(agr);
            
    //         double out = cpp_profile_likelihood(
    //             Y, ITEM_INDS, ALPHA_START, 
    //             phi, K, J,
    //             PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD,
    //             CONTINUOUS
    //         );
    //         return out;
    //     };
    // }
    
    // double d2 = boost::math::differentiation::finite_difference_derivative(
    //     [&](double x){ 
    //         return boost::math::differentiation::finite_difference_derivative(f, x); 
    //     },
    //     agr
    // );
    
    // if(-d2 > 0){
    //     return 1.0 / sqrt(-d2);
    // }else{
    //     return std::numeric_limits<double>::quiet_NaN();
    // }

    return agr;

}
