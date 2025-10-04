#include "profiling.h"

///////////////////////////////////////
// CONTINUOUS RATINGS | TWOWAY MODEL //
///////////////////////////////////////

Eigen::VectorXd AgreementPhi::continuous::twoway::inference::profiling_lbfgs(
    const std::vector<double>& Y,
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const Eigen::VectorXd& LAMBDA_START,
    const double PHI,
    const int J,
    const int W,
    const int MAX_ITER
){
    // const int dim = J + W - 1;
    
    // LBFGSpp::LBFGSParam<double> param;
    // param.epsilon = 1e-6;
    // param.max_iterations = MAX_ITER;
    
    // LBFGSpp::LBFGSSolver<double> solver(param);
    
    // Eigen::VectorXd lambda = LAMBDA_START;
    
    // auto neg_ll = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
    //     Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(dim);
    //     Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    //     Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    //     Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
        
    //     double ll = AgreementPhi::continuous::twoway::joint_loglik(
    //         Y, ITEM_INDS, WORKER_INDS, x, PHI, J, W,
    //         dlambda, jalphaalpha, jbetabeta, jalphabeta, 1
    //     );
        
    //     grad = -dlambda;
    //     return -ll;
    // };
    
    // double fx;
    // solver.minimize(neg_ll, lambda, fx);
    Eigen::VectorXd lambda = LAMBDA_START;
    return lambda;
}

////////////////////////////////////
// ORDINAL RATINGS | TWOWAY MODEL //
////////////////////////////////////

Eigen::VectorXd AgreementPhi::ordinal::twoway::inference::profiling_lbfgs(
    const std::vector<double>& Y,
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const Eigen::VectorXd& LAMBDA_START,
    const double PHI,
    const int J,
    const int W,
    const int K,
    const int MAX_ITER
){
    // const int dim = J + W - 1;
    
    // LBFGSpp::LBFGSParam<double> param;
    // param.epsilon = 1e-6;
    // param.max_iterations = MAX_ITER;
    
    // LBFGSpp::LBFGSSolver<double> solver(param);
    
    // Eigen::VectorXd lambda = LAMBDA_START;
    
    // auto neg_ll = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
    //     Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(dim);
    //     Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    //     Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    //     Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
        
    //     double ll = AgreementPhi::ordinal::twoway::joint_loglik(
    //         Y, ITEM_INDS, WORKER_INDS, x, PHI, J, W, K,
    //         dlambda, jalphaalpha, jbetabeta, jalphabeta, 1
    //     );
        
    //     grad = -dlambda;
    //     return -ll;
    // };
    
    // double fx;
    // solver.minimize(neg_ll, lambda, fx);
    
    Eigen::VectorXd lambda = LAMBDA_START;
    return lambda;
}