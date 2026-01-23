#include <Rcpp.h>
#include <boost/math/tools/minima.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include <functional>
#include <algorithm>
#include <Eigen/Dense>
#include <RcppParallel.h>
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
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha, beta, tau, PHI_START, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            profile_phi = res.at(0);
            ll = res.at(1);

            std::vector<std::vector<double>> lambda = AgreementPhi::ordinal::nuisance::get_lambda(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha,  beta, tau, profile_phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_SEARCH_RANGE,
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
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha, beta, tau, PHI_START, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, SEARCH_RANGE, MAX_ITER, PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL, VERBOSE);
            modified_phi = res.at(0);
            profile_phi = res.at(2);
            ll = res.at(1);

            std::vector<std::vector<double>> lambda = AgreementPhi::ordinal::nuisance::get_lambda(
                Y,  ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict, alpha,  beta, tau, modified_phi, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_SEARCH_RANGE,
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
            ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL
        );
    }else{
        throw std::invalid_argument("Invalid DATA_TYPE");
    }
}

// [[Rcpp::export]]
double cpp_modified_profile_likelihood(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const std::vector<double> TAU,
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
        
        return AgreementPhi::ordinal::ll::modified_profile(
            Y, ITEM_INDS, worker_inds, item_dict, worker_dict, cat_dict,
            ALPHA_MLE, beta_mle, TAU, PHI, PHI_MLE, J, W_eff, K,
            ITEMS_NUISANCE, WORKER_NUISANCE,
            PROF_SEARCH_RANGE, PROF_UNI_MAX_ITER, ALT_MAX_ITER, ALT_TOL
        );
    }else{
        throw std::invalid_argument("Invalid DATA_TYPE");
    }
}

// [[Rcpp::export]]
double cpp_get_se(
    const std::vector<double> Y,
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<double> ALPHA_MLE,
    const std::vector<double> BETA_MLE,
    const std::vector<double> TAU_MLE,
    const double PHI_EVAL,
    const double PHI_MLE,
    const int J,
    const int W,
    const int K,
    const std::string METHOD,
    const std::string DATA_TYPE,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int ALT_MAX_ITER,
    const double ALT_TOL
) {
    // Build dictionaries (same as cpp_get_phi)
    std::vector<std::vector<int>> item_dict = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    std::vector<std::vector<int>> worker_dict = AgreementPhi::utils::oneway_dict(W, WORKER_INDS);

    // Convert evaluation point to agreement scale
    double agr_eval = AgreementPhi::utils::prec2agr(PHI_EVAL);

    std::function<double(double)> f;

    if(DATA_TYPE == "continuous") {
        if(METHOD == "modified") {
            // Modified profile likelihood for continuous data
            // (same as precision.cpp lines 87-91)
            f = [&](double agr) -> double {
                double phi = AgreementPhi::utils::agr2prec(agr);
                return AgreementPhi::continuous::ll::modified_profile(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict,
                    ALPHA_MLE, BETA_MLE, phi, PHI_MLE, J, W,
                    ITEMS_NUISANCE, WORKER_NUISANCE,
                    PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL
                );
            };
        } else {
            // Profile likelihood for continuous data
            // (same as precision.cpp lines 28-32)
            f = [&](double agr) -> double {
                double phi = AgreementPhi::utils::agr2prec(agr);
                return AgreementPhi::continuous::ll::profile(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict,
                    ALPHA_MLE, BETA_MLE, phi, J, W,
                    ITEMS_NUISANCE, WORKER_NUISANCE,
                    PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL
                );
            };
        }
    } else if(DATA_TYPE == "ordinal") {
        std::vector<std::vector<int>> cat_dict = AgreementPhi::utils::categories_dict(Y, K);

        // Build MLE lambda vector (alpha + beta[1:])
        std::vector<double> mle_vec;
        mle_vec.reserve(J + W - 1);
        mle_vec.insert(mle_vec.end(), ALPHA_MLE.begin(), ALPHA_MLE.end());
        mle_vec.insert(mle_vec.end(), BETA_MLE.begin() + 1, BETA_MLE.end());

        if(METHOD == "modified") {
            // Modified profile likelihood for ordinal data
            // (same as precision.cpp lines 350-402)
            f = [&, cat_dict, mle_vec](double agr) -> double {
                double phi = AgreementPhi::utils::agr2prec(agr);

                // Profile nuisance at this phi (warm start from MLE)
                std::vector<double> alpha_start = ALPHA_MLE;
                std::vector<double> beta_start = BETA_MLE;
                std::vector<double> tau_start = TAU_MLE;

                std::vector<std::vector<double>> profiled = AgreementPhi::ordinal::nuisance::get_lambda(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict,
                    alpha_start, beta_start, tau_start, phi, J, W, K,
                    ITEMS_NUISANCE, WORKER_NUISANCE,
                    PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL
                );

                // Build profiled lambda vector
                std::vector<double> profiled_vec;
                profiled_vec.reserve(J + W - 1);
                profiled_vec.insert(profiled_vec.end(), profiled.at(0).begin(), profiled.at(0).end());
                profiled_vec.insert(profiled_vec.end(), profiled.at(1).begin() + 1, profiled.at(1).end());

                // Compute joint loglik
                Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
                Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
                Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
                Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

                double ll = AgreementPhi::ordinal::joint_loglik(
                    Y, ITEM_INDS, WORKER_INDS, profiled_vec, TAU_MLE, phi,
                    J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
                    dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
                );

                // Add Barndorff-Nielsen correction
                ll += 0.5 * AgreementPhi::ordinal::log_det_obs_info(
                    Y, ITEM_INDS, WORKER_INDS, profiled_vec, TAU_MLE, phi,
                    J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
                );

                // Subtract cross information term
                ll -= AgreementPhi::ordinal::log_det_E0d0d1(
                    ITEM_INDS, WORKER_INDS, mle_vec, profiled_vec, PHI_MLE, phi, TAU_MLE,
                    J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE
                );

                return ll;
            };
        } else {
            // Profile likelihood for ordinal data
            // (same as precision.cpp lines 152-190)
            f = [&, cat_dict](double agr) -> double {
                double phi = AgreementPhi::utils::agr2prec(agr);

                // Profile nuisance at this phi (warm start from MLE)
                std::vector<double> alpha_start = ALPHA_MLE;
                std::vector<double> beta_start = BETA_MLE;
                std::vector<double> tau_start = TAU_MLE;

                std::vector<std::vector<double>> profiled = AgreementPhi::ordinal::nuisance::get_lambda(
                    Y, ITEM_INDS, WORKER_INDS, item_dict, worker_dict, cat_dict,
                    alpha_start, beta_start, tau_start, phi, J, W, K,
                    ITEMS_NUISANCE, WORKER_NUISANCE,
                    PROF_SEARCH_RANGE, PROF_MAX_ITER, ALT_MAX_ITER, ALT_TOL
                );

                // Build profiled lambda vector
                std::vector<double> lambda;
                lambda.reserve(J + W - 1);
                lambda.insert(lambda.end(), profiled.at(0).begin(), profiled.at(0).end());
                lambda.insert(lambda.end(), profiled.at(1).begin() + 1, profiled.at(1).end());

                // Compute joint loglik
                Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
                Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
                Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
                Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

                return AgreementPhi::ordinal::joint_loglik(
                    Y, ITEM_INDS, WORKER_INDS, lambda, profiled.at(2), phi,
                    J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
                    dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
                );
            };
        }
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Compute second derivative via nested finite differences
    double d2 = boost::math::differentiation::finite_difference_derivative(
        [&](double x){
            return boost::math::differentiation::finite_difference_derivative(f, x);
        },
        agr_eval
    );

    // SE from observed Fisher information: SE = 1/sqrt(-d2)
    if(-d2 > 0){
        return 1.0 / std::sqrt(-d2);
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}
