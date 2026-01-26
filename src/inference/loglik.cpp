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
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
){


    

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA,  BETA, TAU, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_UNI_RANGE,
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
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::nuisance::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, CAT_DICT, ALPHA_MLE,  BETA_MLE, TAU_MLE, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);


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

