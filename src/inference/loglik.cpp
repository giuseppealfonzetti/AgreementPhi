#include "loglik.h"

///////////////////////////////////////
// CONTINUOUS RATINGS | ONEWAY MODEL //
///////////////////////////////////////


double AgreementPhi::continuous::oneway::loglik::profile(
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
            double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, RANGE, MAX_ITER
            );
            alpha.at(j) = hatalpha;
        }else{
            double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_newtonraphson(
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

double AgreementPhi::continuous::oneway::loglik::modified_profile(
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
            double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            prof_alpha.at(j) = hatalpha;
        }else{
            double hatalpha = AgreementPhi::continuous::oneway::inference::profiling_newtonraphson(
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

///////////////////////////////////////
// ORDINAL RATINGS | ONEWAY MODEL //
///////////////////////////////////////

double AgreementPhi::ordinal::oneway::loglik::profile(
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
            double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, K, RANGE, MAX_ITER
            );
            alpha.at(j)=hatalpha;
        }else{
            double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_newtonraphson(
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

double AgreementPhi::ordinal::oneway::loglik::modified_profile(
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
            double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_brent(
                Y, DICT, j, ALPHA_START.at(j), PHI, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
            );
            prof_alpha.at(j)=hatalpha;
        }else{
            double hatalpha = AgreementPhi::ordinal::oneway::inference::profiling_newtonraphson(
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

///////////////////////////////////////
// CONTINUOUS RATINGS | TWOWAY MODEL //
///////////////////////////////////////
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

///////////////////////////////////////
// ORDINAL RATINGS | TWOWAY MODEL //
///////////////////////////////////////
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