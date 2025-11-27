#include <Rcpp.h>
#include <boost/math/tools/minima.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include <functional>
#include <algorithm>
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

///////////////////////////////////////////////////////////
// SHARED  INTERFACE for continuous and fixed thresholds //
///////////////////////////////////////////////////////////
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
    std::vector<double> alpha = ALPHA_START;
    std::vector<double> beta = BETA_START;
    std::vector<double> tau = TAU_START;


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
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha, beta, tau, PHI_START, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, false, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            profile_phi = res.at(0);
            ll = res.at(1);

            std::vector<std::vector<double>> lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha,  beta, tau, profile_phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, false, PROF_SEARCH_RANGE,
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
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha, beta, tau, PHI_START, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, false, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            modified_phi = res.at(0);
            profile_phi = res.at(2);
            ll = res.at(1);

            std::vector<std::vector<double>> lambda = AgreementPhi::ordinal::nuisance::get_lambda2(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha,  beta, tau, modified_phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, false, PROF_SEARCH_RANGE,
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





/////////////////////////////////////////////////////////////////////
// Single functions to be exported to deal with unknown thresholds //
/////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
double cpp_profile_likelihood(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<double> ALPHA_START,
    const std::vector<double> BETA_START,
    const std::vector<double> TAU_START,
    const double PHI,
    const int J,
    const int W,
    const int K,
    const std::string DATA_TYPE,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const bool THRESHOLDS_NUISANCE,
    const double PROF_SEARCH_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int ALT_MAX_ITER,
    const double ALT_TOL
){
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(worker_inds.empty() ? 1 : *std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);

    std::vector<double> beta = BETA_START;
    if(static_cast<int>(beta.size()) < W_eff){
        beta.resize(W_eff, 0.0);
    }

    if(DATA_TYPE == "continuous"){
        return AgreementPhi::continuous::ll::profile(
            Y, ITEM_INDS, worker_inds, item_dict, worker_dict, ALPHA_START, beta, PHI,
            J, W_eff, ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL
        );
    }else if(DATA_TYPE == "ordinal"){
        std::vector<double> tau = TAU_START;
        if(static_cast<int>(tau.size()) != K + 1){
            tau.assign(K + 1, 0.0);
            for(int i = 0; i <= K; ++i){
                tau.at(i) = static_cast<double>(i) / static_cast<double>(K);
            }
        }
        std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);
        return AgreementPhi::ordinal::ll::profile(
            Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
            ALPHA_START, beta, tau, PHI, J, W_eff, K,
            ITEMS_NUISANCE, WORKER_NUISANCE, THRESHOLDS_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL
        );
    }else{
        throw std::invalid_argument("Invalid DATA_TYPE");
    }
}

// [[Rcpp::export]]
double cpp_modified_profile_likelihood_extended(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const std::vector<double> TAU,
    const std::vector<double> TAU_MLE,
    const double PHI,
    const double PHI_MLE,
    const int J,
    const int W,
    const int K,
    const std::string DATA_TYPE,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double PROF_SEARCH_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int ALT_MAX_ITER,
    const double ALT_TOL
){
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);

    std::vector<double> beta_mle = BETA_MLE;
    if(static_cast<int>(beta_mle.size()) < W_eff){
        beta_mle.resize(W_eff, 0.0);
    }

    if(DATA_TYPE == "continuous"){
        return AgreementPhi::continuous::ll::modified_profile(
            Y, ITEM_INDS, worker_inds, item_dict, worker_dict,
            ALPHA_MLE, beta_mle, PHI, PHI_MLE, J, W_eff,
            ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL
        );
    }else if(DATA_TYPE == "ordinal"){
        std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);
        
        return AgreementPhi::ordinal::ll::modified_profile_extended(
            Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
            ALPHA_MLE, beta_mle, TAU, TAU_MLE, PHI, PHI_MLE, J, W_eff, K,
            ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL
        );
    }else{
        throw std::invalid_argument("Invalid DATA_TYPE");
    }
}

// [[Rcpp::export]]
double cpp_modified_profile_likelihood_tau_profiled(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const std::vector<double> TAU_START,
    const std::vector<double> TAU_MLE,
    const double PHI,
    const double PHI_MLE,
    const int J,
    const int W,
    const int K,
    const std::string DATA_TYPE,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double PROF_SEARCH_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int ALT_MAX_ITER,
    const double ALT_TOL
){
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);

    std::vector<double> beta_mle = BETA_MLE;
    if(static_cast<int>(beta_mle.size()) < W_eff){
        beta_mle.resize(W_eff, 0.0);
    }

    if(DATA_TYPE == "ordinal"){
        std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

        return AgreementPhi::ordinal::ll::modified_profile_tau_profiled(
            Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
            ALPHA_MLE, beta_mle, TAU_START, TAU_MLE, PHI, PHI_MLE, J, W_eff, K,
            ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL
        );
    }else{
        throw std::invalid_argument("This function is only implemented for ordinal data");
    }
}

// [[Rcpp::export]]
std::vector<double> cpp_profile_grad_tau(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<double> ALPHA_START,
    const std::vector<double> BETA_START,
    const std::vector<double> TAU,
    const double PHI,
    const int J,
    const int W,
    const int K,
    const std::string DATA_TYPE,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double PROF_SEARCH_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int ALT_MAX_ITER,
    const double ALT_TOL
){
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);

    std::vector<double> beta_start = BETA_START;
    if(static_cast<int>(beta_start.size()) < W_eff){
        beta_start.resize(W_eff, 0.0);
    }

    std::vector<double> grad_tau;

    if(DATA_TYPE == "ordinal"){
        std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

        AgreementPhi::ordinal::ll::profile_grad_tau(
            Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
            ALPHA_START, beta_start, TAU, PHI, J, W_eff, K,
            ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL,
            grad_tau
        );
    }else{
        throw std::invalid_argument("This function is only implemented for ordinal data");
    }

    return grad_tau;
}

// [[Rcpp::export]]
double cpp_profile_extended(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
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
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);
    std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

    std::vector<double> beta = BETA;
    if(static_cast<int>(beta.size()) < W_eff){
        beta.resize(W_eff, 0.0);
    }

    return AgreementPhi::ordinal::ll::profile_extended(
        Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
        ALPHA, beta, RAW_TAU, RAW_PHI, J, W_eff, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );
}

// [[Rcpp::export]]
Eigen::VectorXd cpp_profile_extended_grad_raw_tau(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
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
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);
    std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

    std::vector<double> beta = BETA;
    if(static_cast<int>(beta.size()) < W_eff){
        beta.resize(W_eff, 0.0);
    }

    return AgreementPhi::ordinal::ll::profile_extended_grad_raw_tau(
        Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
        ALPHA, beta, RAW_TAU, RAW_PHI, J, W_eff, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );
}

// [[Rcpp::export]]
double cpp_profile_extended_grad_raw_phi(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
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
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);
    std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

    std::vector<double> beta = BETA;
    if(static_cast<int>(beta.size()) < W_eff){
        beta.resize(W_eff, 0.0);
    }

    return AgreementPhi::ordinal::ll::profile_extended_grad_raw_phi(
        Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
        ALPHA, beta, RAW_TAU, RAW_PHI, J, W_eff, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );
}

// [[Rcpp::export]]
Eigen::VectorXd cpp_profile_extended_grad(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
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
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<int> worker_inds = WORKER_INDS;
    if(worker_inds.empty()){
        worker_inds.assign(Y.size(), 1);
    }
    int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);
    std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

    std::vector<double> beta = BETA;
    if(static_cast<int>(beta.size()) < W_eff){
        beta.resize(W_eff, 0.0);
    }

    return AgreementPhi::ordinal::ll::profile_extended_grad(
        Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
        ALPHA, beta, RAW_TAU, RAW_PHI, J, W_eff, K,
        ITEMS_NUISANCE, WORKER_NUISANCE,
        PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
    );
}

// // [[Rcpp::export]]
// Eigen::MatrixXd cpp_profile_extended_hess_raw_tau(
//     const std::vector<double> Y,
//     const std::vector<int> ITEM_INDS,
//     const std::vector<int> WORKER_INDS,
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
//     std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
//     std::vector<int> worker_inds = WORKER_INDS;
//     if(worker_inds.empty()){
//         worker_inds.assign(Y.size(), 1);
//     }
//     int W_eff = W > 0 ? W : static_cast<int>(*std::max_element(worker_inds.begin(), worker_inds.end()));
//     std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W_eff, worker_inds);
//     std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

//     std::vector<double> beta = BETA;
//     if(static_cast<int>(beta.size()) < W_eff){
//         beta.resize(W_eff, 0.0);
//     }

//     return AgreementPhi::ordinal::ll::profile_extended_hess_raw_tau(
//         Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
//         ALPHA, beta, RAW_TAU, RAW_PHI, J, W_eff, K,
//         ITEMS_NUISANCE, WORKER_NUISANCE,
//         PROF_UNI_RANGE, PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL
//     );
// }

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
