#include "loglik.h"

double AgreementPhi::continuous::twoway::loglik::profile(
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
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
){

    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::continuous::twoway::inference::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA,  BETA, PHI, J, W, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    std::vector<double> lambda;
    lambda.reserve(J + W - 1);
    lambda.insert(lambda.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    lambda.insert(lambda.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    
    double ll = AgreementPhi::continuous::twoway::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, lambda, PHI, J, W,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );

    return ll;

}


double AgreementPhi::continuous::twoway::loglik::modified_profile(
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
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){

    // profile nuisance parameters
    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::continuous::twoway::inference::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA_MLE,  BETA_MLE, PHI, J, W, PROF_UNI_RANGE,
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
    double ll = AgreementPhi::continuous::twoway::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, PHI, J, W,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );


    // evaluate modifier contribution
    ll += .5 * AgreementPhi::continuous::twoway::log_det_obs_info(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, PHI, J, W
    );

    std::vector<double> mle_vec;
    mle_vec.reserve(J + W - 1);
    mle_vec.insert(mle_vec.end(), ALPHA_MLE.begin(), ALPHA_MLE.end());
    mle_vec.insert(mle_vec.end(), BETA_MLE.begin() + 1, BETA_MLE.end());

    ll -= AgreementPhi::continuous::twoway::log_det_E0d0d1(
        ITEM_INDS, WORKER_INDS, mle_vec, profiled_vec, PHI_MLE, PHI, J, W
    );

    return ll;

}

double AgreementPhi::ordinal::twoway::loglik::profile(
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
                    const int K,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
){
    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::twoway::inference::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA,  BETA, PHI, J, W, K, PROF_UNI_RANGE,
        PROF_UNI_MAX_ITER, PROF_MAX_ITER, PROF_TOL);

    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    std::vector<double> lambda;
    lambda.reserve(J + W - 1);
    lambda.insert(lambda.end(), profiled_lambda.at(0).begin(), profiled_lambda.at(0).end());
    lambda.insert(lambda.end(), profiled_lambda.at(1).begin() + 1, profiled_lambda.at(1).end());

    double ll = AgreementPhi::ordinal::twoway::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, lambda, PHI, J, W, K, 
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );

    return ll;
}

double AgreementPhi::ordinal::twoway::loglik::modified_profile(
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
    const int K,
    const int PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double PROF_TOL
){

    // profile nuisance parameters
    std::vector<std::vector<double>> profiled_lambda = AgreementPhi::ordinal::twoway::inference::get_lambda(
        Y,  ITEM_INDS, WORKER_INDS, ITEM_DICT, WORKER_DICT, ALPHA_MLE,  BETA_MLE, PHI, J, W, K, PROF_UNI_RANGE,
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
    double ll = AgreementPhi::ordinal::twoway::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, PHI, J, W, K,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 0
    );


    // evaluate modifier contribution
    ll += .5 * AgreementPhi::ordinal::twoway::log_det_obs_info(
        Y, ITEM_INDS, WORKER_INDS, profiled_vec, PHI, J, W, K
    );

    std::vector<double> mle_vec;
    mle_vec.reserve(J + W - 1);
    mle_vec.insert(mle_vec.end(), ALPHA_MLE.begin(), ALPHA_MLE.end());
    mle_vec.insert(mle_vec.end(), BETA_MLE.begin() + 1, BETA_MLE.end());

    ll -= AgreementPhi::ordinal::twoway::log_det_E0d0d1(
        ITEM_INDS, WORKER_INDS, mle_vec, profiled_vec, PHI_MLE, PHI, J, W, K
    );

    return ll;

}