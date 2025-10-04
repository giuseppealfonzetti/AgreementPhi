#include "twoway.h"

double AgreementPhi::continuous::twoway::joint_loglik(
    const std::vector<double>& Y, 
    const std::vector<int>& ITEM_INDS,   // expected to be coded 1 to J
    const std::vector<int>& WORKER_INDS, // expected to be coded 1 to W
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA, //expected length J+W-1
    const double PHI,
    const int J, 
    const int W,
    Eigen::Ref<Eigen::VectorXd> DLAMBDA,
    Eigen::Ref<Eigen::VectorXd> JALPHAALPHA,
    Eigen::Ref<Eigen::VectorXd> JBETABETA,
    Eigen::Ref<Eigen::MatrixXd> JALPHABETA,
    const int GRADFLAG
){
    const int n = Y.size();

    double ll = 0;

    for(int i = 0; i < n; ++i){
        int j = ITEM_INDS.at(i) - 1;   // pass to 0 to J-1 indexing
        int w = WORKER_INDS.at(i) - 1; // pass to 0 to W-1 indexing
        
        double item_intercept = LAMBDA(j);
        double worker_intercept = (w == 0) ? 0.0 : LAMBDA(J + w - 1);
        double eta = item_intercept + worker_intercept;
        double mu = link::mu(eta);
        double dmu_deta = link::dmu(mu);
        double d2mu_deta2 = link::d2mu(mu);
        
        double dwrtmu = 0;
        double dwrtmu2 = 0;
        ll += continuous::loglik(Y.at(i), mu, PHI, dwrtmu, dwrtmu2, GRADFLAG);
        
        if(GRADFLAG > 0){
            DLAMBDA(j) += dwrtmu * dmu_deta;
            if(w>0) DLAMBDA(J + w - 1) += dwrtmu * dmu_deta;
            
            if(GRADFLAG > 1){
                double d2l = dwrtmu2 * pow(dmu_deta, 2) + dwrtmu * d2mu_deta2;
                
                JALPHAALPHA(j) -= d2l;
                if(w>0) JBETABETA(w - 1) -= d2l;
                if(w>0) JALPHABETA(j, w - 1) -= d2l;
            }
        }

    }

    return ll;
}