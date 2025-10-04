#include "twoway.h"

///////////////////////////////////////
// CONTINUOUS RATINGS | TWOWAY MODEL //
///////////////////////////////////////

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

double AgreementPhi::continuous::twoway::log_det_obs_info(
    const std::vector<double>& Y,
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA,
    const double PHI,
    const int J,
    const int W
){
    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    AgreementPhi::continuous::twoway::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, LAMBDA, PHI, J, W,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 2
    );
    
    double log_det_alpha = jalphaalpha.array().log().sum();
    
    Eigen::VectorXd sqrt_inv_alpha = jalphaalpha.array().pow(-0.5);
    Eigen::MatrixXd Ha = sqrt_inv_alpha.asDiagonal() * jalphabeta;
    Eigen::MatrixXd schur = Eigen::MatrixXd(jbetabeta.asDiagonal()) - Ha.transpose() * Ha;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur);
    double log_det_schur = es.eigenvalues().array().log().sum();
    
    return log_det_alpha + log_det_schur;
}

double AgreementPhi::continuous::twoway::log_det_E0d0d1(
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA0,
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA1,
    const double PHI0,
    const double PHI1,
    const int J,
    const int W
){
    const int n = ITEM_INDS.size();
    
    Eigen::VectorXd Ialphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd Ibetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd Ialphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    for(int i = 0; i < n; ++i){
        int j = ITEM_INDS[i] - 1;
        int w = WORKER_INDS[i] - 1;
        
        double eta0 = LAMBDA0(j) + ((w == 0) ? 0.0 : LAMBDA0(J + w - 1));
        double eta1 = LAMBDA1(j) + ((w == 0) ? 0.0 : LAMBDA1(J + w - 1));
        double mu0 = link::mu(eta0);
        double mu1 = link::mu(eta1);
        
        double e = continuous::E0_dmu0dmu1(mu0, PHI0, mu1, PHI1);
        double dmu0 = link::dmu(mu0);
        double dmu1 = link::dmu(mu1);
        
        double w_val = dmu0 * dmu1 * e;
        
        Ialphaalpha(j) += w_val;
        if(w > 0){
            Ibetabeta(w - 1) += w_val;
            Ialphabeta(j, w - 1) += w_val;
        }
    }
    
    double log_det_alpha = Ialphaalpha.array().log().sum();
    
    Eigen::VectorXd sqrt_inv_alpha = Ialphaalpha.array().pow(-0.5);
    Eigen::MatrixXd Ha = sqrt_inv_alpha.asDiagonal() * Ialphabeta;
    Eigen::MatrixXd schur = Eigen::MatrixXd(Ibetabeta.asDiagonal()) - Ha.transpose() * Ha;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur);
    double log_det_schur = es.eigenvalues().array().log().sum();
    
    return log_det_alpha + log_det_schur;
}

////////////////////////////////////
// ORDINAL RATINGS | TWOWAY MODEL //
////////////////////////////////////

double AgreementPhi::ordinal::twoway::joint_loglik(
    const std::vector<double>& Y, 
    const std::vector<int>& ITEM_INDS,   // expected to be coded 1 to J
    const std::vector<int>& WORKER_INDS, // expected to be coded 1 to W
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA, //expected length J+W-1
    const double PHI,
    const int J, 
    const int W,
    const int K,
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
        ll += ordinal::loglik(Y.at(i), mu, PHI, K, dwrtmu, dwrtmu2, GRADFLAG);
        
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

double AgreementPhi::ordinal::twoway::log_det_obs_info(
    const std::vector<double>& Y,
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA,
    const double PHI,
    const int J,
    const int W,
    const int K
){
    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    AgreementPhi::ordinal::twoway::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, LAMBDA, PHI, J, W, K,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 2
    );
    
    double log_det_alpha = jalphaalpha.array().log().sum();
    
    Eigen::VectorXd sqrt_inv_alpha = jalphaalpha.array().pow(-0.5);
    Eigen::MatrixXd Ha = sqrt_inv_alpha.asDiagonal() * jalphabeta;
    Eigen::MatrixXd schur = Eigen::MatrixXd(jbetabeta.asDiagonal()) - Ha.transpose() * Ha;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur);
    double log_det_schur = es.eigenvalues().array().log().sum();
    
    return log_det_alpha + log_det_schur;
}

double AgreementPhi::ordinal::twoway::log_det_E0d0d1(
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA0,
    const Eigen::Ref<const Eigen::VectorXd> LAMBDA1,
    const double PHI0,
    const double PHI1,
    const int J,
    const int W,
    const int K
){
    const int n = ITEM_INDS.size();
    
    std::vector<double> mu0(n), mu1(n);
    for(int i = 0; i < n; ++i){
        int j = ITEM_INDS[i] - 1;
        int w = WORKER_INDS[i] - 1;
        double eta0 = LAMBDA0(j) + ((w == 0) ? 0.0 : LAMBDA0(J + w - 1));
        double eta1 = LAMBDA1(j) + ((w == 0) ? 0.0 : LAMBDA1(J + w - 1));
        mu0[i] = link::mu(eta0);
        mu1[i] = link::mu(eta1);
    }
    
    Eigen::VectorXd Ialphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd Ibetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd Ialphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    for(int i = 0; i < n; ++i){
        int j = ITEM_INDS[i] - 1;
        int w = WORKER_INDS[i] - 1;
        
        double e = ordinal::E0_dmu0dmu1(mu0[i], PHI0, mu1[i], PHI1, K);
        double dmu0_dalpha = link::dmu(mu0[i]);
        double dmu1_dalpha = link::dmu(mu1[i]);
        double dmu0_dbeta = (w > 0) ? link::dmu(mu0[i]) : 0.0;
        double dmu1_dbeta = (w > 0) ? link::dmu(mu1[i]) : 0.0;
        
        Ialphaalpha(j) += dmu0_dalpha * dmu1_dalpha * e;
        if(w > 0){
            Ibetabeta(w - 1) += dmu0_dbeta * dmu1_dbeta * e;
            Ialphabeta(j, w - 1) += dmu0_dalpha * dmu1_dbeta * e;
        }
    }
    
    
    double log_det_alpha = Ialphaalpha.array().log().sum();
    
    Eigen::VectorXd sqrt_inv_alpha = Ialphaalpha.array().pow(-0.5);
    Eigen::MatrixXd Ha = sqrt_inv_alpha.asDiagonal() * Ialphabeta;
    Eigen::MatrixXd schur = Eigen::MatrixXd(Ibetabeta.asDiagonal()) - Ha.transpose() * Ha;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur);
    double log_det_schur = es.eigenvalues().array().log().sum();
    
    return log_det_alpha + log_det_schur;
}