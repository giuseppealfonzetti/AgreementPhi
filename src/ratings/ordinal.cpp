#include "ordinal.h"
#include <cmath>

double AgreementPhi::ordinal::loglik(
    const double Y, 
    const double MU,
    const double PHI,
    const std::vector<double> TAU,
    double &DMU, 
    double &DMU2, 
    const int GRADFLAG
){
    double a = MU*PHI;
    double b = (1-MU)*PHI;
    double t   = TAU.at(static_cast<int>(Y));       
    double tm1 = TAU.at(static_cast<int>(Y-1));   


    double t_log_cdf   = log(boost::math::ibeta(a,b,t));
    double tm1_log_cdf = log(boost::math::ibeta(a,b,tm1));
    double t_cdf = exp(t_log_cdf);
    double tm1_cdf = exp(tm1_log_cdf);
    double logprob = t_log_cdf + log1p(-exp(tm1_log_cdf - t_log_cdf));
    double prob = exp(logprob);

    
    if(GRADFLAG>0){
        double beta = boost::math::beta(a,b);
        double betainv = 1/beta;
        double dlogBda = betamath::dlogBda(a,b);
        double dlogBdb = betamath::dlogBdb(a,b);
        
        //  F(Y=c, mu, phi)
        double t_diBda = betamath::diBda(t,a,b);
        double t_diBdb = betamath::diBdb(t,a,b);

        double t_da = betamath::dF(t_diBda, dlogBda, betainv, t_cdf);
        double t_db = betamath::dF(t_diBdb, dlogBdb, betainv, t_cdf);
        
        double t_dmu = betamath::dFdmu(PHI, t_da, t_db);
        // double t_dphi = CDFFuns::dFdphi(MU, t_da, t_db);

        // F(Y=c-1; mu, phi)
        double tm1_diBda = betamath::diBda(tm1,a,b);
        double tm1_diBdb = betamath::diBdb(tm1,a,b);

        double tm1_da = betamath::dF(tm1_diBda, dlogBda, betainv, tm1_cdf);
        double tm1_db = betamath::dF(tm1_diBdb, dlogBdb, betainv, tm1_cdf);

        double tm1_dmu = betamath::dFdmu(PHI, tm1_da, tm1_db);
        // double tm1_dphi = CDFFuns::dFdphi(MU, tm1_da, tm1_db);

        // combine
        double dmu = (t_dmu - tm1_dmu) / std::max(prob, 1e-12) ;       
        // double dphi = (t_dphi-tm1_dphi)/(prob+1e-8);

        DMU += dmu;

        if(GRADFLAG>1){
                double d2logBda2 = betamath::d2logBda2(a,b);
                double d2logBdb2 = betamath::d2logBdb2(a,b);
                double d2logBdadb = betamath::d2logBdadb(a,b);

                //  F(Y=c, mu, phi)
                double t_d2iBda2 = betamath::d2iBda2(t,a,b);
                double t_d2iBdb2 = betamath::d2iBdb2(t,a,b);
                double t_d2iBdadb = betamath::d2iBdadb(t,a,b);

                double t_da2 = betamath::d2F(t_diBda, t_diBda, dlogBda, dlogBda, t_d2iBda2, d2logBda2, betainv, t_cdf);
                double t_db2 = betamath::d2F(t_diBdb, t_diBdb, dlogBdb, dlogBdb, t_d2iBdb2, d2logBdb2, betainv, t_cdf);
                double t_dadb = betamath::d2F(t_diBda, t_diBdb, dlogBda, dlogBdb, t_d2iBdadb, d2logBdadb, betainv, t_cdf);

                double t_dmu2 = betamath::d2Fdmu2(PHI, t_da, t_db, t_da2, t_db2, t_dadb);
                // double t_dphi2 = CDFFuns::d2Fdphi2(MU, t_da, t_db, t_da2, t_db2, t_dadb);
                // double t_dmudphi = CDFFuns::d2Fdmudphi(MU, PHI, t_da, t_db, t_da2, t_db2, t_dadb);

                // F(Y=c-1; mu, phi)
                double tm1_d2iBda2 = betamath::d2iBda2(tm1,a,b);
                double tm1_d2iBdb2 = betamath::d2iBdb2(tm1,a,b);
                double tm1_d2iBdadb = betamath::d2iBdadb(tm1,a,b);

                double tm1_da2 = betamath::d2F(tm1_diBda, tm1_diBda, dlogBda, dlogBda, tm1_d2iBda2, d2logBda2, betainv, tm1_cdf);
                double tm1_db2 = betamath::d2F(tm1_diBdb, tm1_diBdb, dlogBdb, dlogBdb, tm1_d2iBdb2, d2logBdb2, betainv, tm1_cdf);
                double tm1_dadb = betamath::d2F(tm1_diBda, tm1_diBdb, dlogBda, dlogBdb, tm1_d2iBdadb, d2logBdadb, betainv, tm1_cdf);

                double tm1_dmu2 = betamath::d2Fdmu2(PHI, tm1_da, tm1_db, tm1_da2, tm1_db2, tm1_dadb);
                // double tm1_dphi2 = CDFFuns::d2Fdphi2(MU, tm1_da, tm1_db, tm1_da2, tm1_db2, tm1_dadb);
                // double tm1_dmudphi = CDFFuns::d2Fdmudphi(MU, PHI, tm1_da, tm1_db, tm1_da2, tm1_db2, tm1_dadb);

                // combine
                double dmu2 = (t_dmu2 - tm1_dmu2) / std::max(prob, 1e-12) - pow(dmu, 2);
                // double dphi2 = (t_dphi2-tm1_dphi2)/(prob+1e-8)-pow(dphi,2);
                // double dmudphi = (t_dmudphi-tm1_dmudphi)/(prob+1e-8)- (dmu*dphi);

                DMU2 += dmu2;
        }

    }
    



    return logprob;
}

double AgreementPhi::ordinal::E0_dmu0dmu1(
    const double MU0,
    const double PHI0,
    const double MU1,
    const double PHI1,
    const std::vector<double> TAU,
    const int K)
{


    double out = 0;
    for(int c=1; c<=K; c++){
        double dmu0 = 0;
        double dmu02 = 0;
        double ll0 = ordinal::loglik(c, MU0, PHI0, TAU, dmu0, dmu02, 1);

        double dmu1 = 0;
        double dmu12 = 0;
        ordinal::loglik(c, MU1, PHI1, TAU, dmu1, dmu12, 1);

        out += dmu0*dmu1*exp(ll0);

    }
    return out;
}

double AgreementPhi::ordinal::joint_loglik(
    const std::vector<double>& Y, 
    const std::vector<int>& ITEM_INDS,   // expected to be coded 1 to J
    const std::vector<int>& WORKER_INDS, // expected to be coded 1 to W
    const std::vector<double>& LAMBDA, //expected length J+W-1
    const std::vector<double>& TAU,
    const double PHI,
    const int J, 
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
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
        
        double item_intercept = LAMBDA.at(j);
        double worker_intercept = (w == 0) ? 0.0 : LAMBDA.at(J + w - 1);
        double eta = item_intercept + worker_intercept;
        double mu = link::mu(eta);
        double dmu_deta = link::dmu(mu);
        double d2mu_deta2 = link::d2mu(mu);
        
        double dwrtmu = 0;
        double dwrtmu2 = 0;
        ll += ordinal::loglik(Y.at(i), mu, PHI, TAU, dwrtmu, dwrtmu2, GRADFLAG);
        
        if(GRADFLAG > 0){
            DLAMBDA(j) += dwrtmu * dmu_deta;
            if(w>0) DLAMBDA(J + w - 1) += dwrtmu * dmu_deta;
            
            if(GRADFLAG > 1){

                if(ITEMS_NUISANCE | WORKER_NUISANCE){
                    double d2l = dwrtmu2 * pow(dmu_deta, 2) + dwrtmu * d2mu_deta2;
                
                    if(ITEMS_NUISANCE){
                        JALPHAALPHA(j) -= d2l;
                    }
                    
                    if(WORKER_NUISANCE){
                        if(w>0) JBETABETA(w - 1) -= d2l;
                    }

                    if(ITEMS_NUISANCE & WORKER_NUISANCE){
                        if(w>0) JALPHABETA(j, w - 1) -= d2l;
                    }
                }
                
            }
        }
    }

    return ll;
}

double AgreementPhi::ordinal::log_det_obs_info(
    const std::vector<double>& Y,
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const std::vector<double>&  LAMBDA,
    const std::vector<double>&  TAU,
    const double PHI,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE
){
    Eigen::VectorXd dlambda = Eigen::VectorXd::Zero(J + W - 1);
    Eigen::VectorXd jalphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd jbetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd jalphabeta = Eigen::MatrixXd::Zero(J, W - 1);
    
    AgreementPhi::ordinal::joint_loglik(
        Y, ITEM_INDS, WORKER_INDS, LAMBDA, TAU, PHI, J, W, K, ITEMS_NUISANCE, WORKER_NUISANCE,
        dlambda, jalphaalpha, jbetabeta, jalphabeta, 2
    );
    
    double log_det_ob_info = 0;
    if(ITEMS_NUISANCE){
        double log_det_alpha = jalphaalpha.array().log().sum();
        log_det_ob_info += log_det_alpha;
    }
    
    
    if(WORKER_NUISANCE){
        Eigen::VectorXd sqrt_inv_alpha = jalphaalpha.array().pow(-0.5);
        Eigen::MatrixXd Ha = sqrt_inv_alpha.asDiagonal() * jalphabeta;
        Eigen::MatrixXd schur = Eigen::MatrixXd(jbetabeta.asDiagonal()) - Ha.transpose() * Ha;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur);
        double log_det_schur = es.eigenvalues().array().log().sum();
        
        log_det_ob_info += log_det_schur;
    }

    return log_det_ob_info;
}

double AgreementPhi::ordinal::log_det_E0d0d1(
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const std::vector<double>&  LAMBDA0,
    const std::vector<double>&  LAMBDA1,
    const double PHI0,
    const double PHI1,
    const std::vector<double>&  TAU,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE
){
    const int n = ITEM_INDS.size();
    
    Eigen::VectorXd Ialphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd Ibetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd Ialphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    std::vector<double> mu0(n), mu1(n);
    for(int i = 0; i < n; ++i){
        int j = ITEM_INDS.at(i) - 1;
        int w = WORKER_INDS.at(i) - 1;
        double eta0 = LAMBDA0.at(j) + ((w == 0) ? 0.0 : LAMBDA0.at(J + w - 1));
        double eta1 = LAMBDA1.at(j) + ((w == 0) ? 0.0 : LAMBDA1.at(J + w - 1));
        double mu0 = link::mu(eta0);
        double mu1 = link::mu(eta1);
        double e = ordinal::E0_dmu0dmu1(mu0, PHI0, mu1, PHI1, TAU, K);
        double dmu0_dalpha = link::dmu(mu0);
        double dmu1_dalpha = link::dmu(mu1);
        double dmu0_dbeta = (w > 0) ? dmu0_dalpha : 0.0;
        double dmu1_dbeta = (w > 0) ? dmu1_dalpha : 0.0;
        
        Ialphaalpha(j) += dmu0_dalpha * dmu1_dalpha * e;
        if(WORKER_NUISANCE){
            if(w > 0){
                Ibetabeta(w - 1) += dmu0_dbeta * dmu1_dbeta * e;
                Ialphabeta(j, w - 1) += dmu0_dalpha * dmu1_dbeta * e;
            }
        }
    }
    
    double log_det_ex_info=0;
    if(ITEMS_NUISANCE){
        double log_det_alpha = Ialphaalpha.array().log().sum();
        log_det_ex_info += log_det_alpha;
    }
    
    if(WORKER_NUISANCE){
        Eigen::VectorXd sqrt_inv_alpha = Ialphaalpha.array().pow(-0.5);
        Eigen::MatrixXd Ha = sqrt_inv_alpha.asDiagonal() * Ialphabeta;
        Eigen::MatrixXd schur = Eigen::MatrixXd(Ibetabeta.asDiagonal()) - Ha.transpose() * Ha;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur);
        double log_det_schur = es.eigenvalues().array().log().sum();
        
        log_det_ex_info += log_det_schur;
    }
    
    return log_det_ex_info;
}






double AgreementPhi::ordinal::E0_dmu0dmu1_extended(
    const double MU0,
    const double PHI0,
    const double MU1,
    const double PHI1,
    const std::vector<double> TAU0,
    const std::vector<double> TAU1,
    const int K)
{


    double out = 0;
    for(int c=1; c<=K; c++){
        double dmu0 = 0;
        double dmu02 = 0;
        double ll0 = ordinal::loglik(c, MU0, PHI0, TAU0, dmu0, dmu02, 1);

        double dmu1 = 0;
        double dmu12 = 0;
        ordinal::loglik(c, MU1, PHI1, TAU1, dmu1, dmu12, 1);

        out += dmu0*dmu1*exp(ll0);

    }
    return out;
}

double AgreementPhi::ordinal::log_det_E0d0d1_extended(
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const std::vector<double>&  LAMBDA0,
    const std::vector<double>&  LAMBDA1,
    const double PHI0,
    const double PHI1,
    const std::vector<double>&  TAU0,
    const std::vector<double>&  TAU1,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE
){
    const int n = ITEM_INDS.size();
    
    Eigen::VectorXd Ialphaalpha = Eigen::VectorXd::Zero(J);
    Eigen::VectorXd Ibetabeta = Eigen::VectorXd::Zero(W - 1);
    Eigen::MatrixXd Ialphabeta = Eigen::MatrixXd::Zero(J, W - 1);

    std::vector<double> mu0(n), mu1(n);
    for(int i = 0; i < n; ++i){
        int j = ITEM_INDS.at(i) - 1;
        int w = WORKER_INDS.at(i) - 1;
        double eta0 = LAMBDA0.at(j) + ((w == 0) ? 0.0 : LAMBDA0.at(J + w - 1));
        double eta1 = LAMBDA1.at(j) + ((w == 0) ? 0.0 : LAMBDA1.at(J + w - 1));
        double mu0 = link::mu(eta0);
        double mu1 = link::mu(eta1);
        double e = ordinal::E0_dmu0dmu1_extended(mu0, PHI0, mu1, PHI1, TAU0, TAU1, K);
        double dmu0_dalpha = link::dmu(mu0);
        double dmu1_dalpha = link::dmu(mu1);
        double dmu0_dbeta = (w > 0) ? dmu0_dalpha : 0.0;
        double dmu1_dbeta = (w > 0) ? dmu1_dalpha : 0.0;
        
        Ialphaalpha(j) += dmu0_dalpha * dmu1_dalpha * e;
        if(WORKER_NUISANCE){
            if(w > 0){
                Ibetabeta(w - 1) += dmu0_dbeta * dmu1_dbeta * e;
                Ialphabeta(j, w - 1) += dmu0_dalpha * dmu1_dbeta * e;
            }
        }
    }
    
    double log_det_ex_info=0;
    if(ITEMS_NUISANCE){
        double log_det_alpha = Ialphaalpha.array().log().sum();
        log_det_ex_info += log_det_alpha;
    }
    
    if(WORKER_NUISANCE){
        Eigen::VectorXd sqrt_inv_alpha = Ialphaalpha.array().pow(-0.5);
        Eigen::MatrixXd Ha = sqrt_inv_alpha.asDiagonal() * Ialphabeta;
        Eigen::MatrixXd schur = Eigen::MatrixXd(Ibetabeta.asDiagonal()) - Ha.transpose() * Ha;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur);
        double log_det_schur = es.eigenvalues().array().log().sum();
        
        log_det_ex_info += log_det_schur;
    }
    
    return log_det_ex_info;
}
void AgreementPhi::ordinal::grad_tau(
    const std::vector<double>& Y,
    const std::vector<int>& ITEM_INDS,
    const std::vector<int>& WORKER_INDS,
    const std::vector<double>& LAMBDA,
    const std::vector<double>& TAU,
    const double PHI,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    std::vector<double>& GRAD_TAU
){
    // Compute gradient of joint log-likelihood w.r.t. thresholds
    // By envelope theorem, this is just ∂L/∂τ evaluated at profiled (α, β)

    const int n = Y.size();
    GRAD_TAU.assign(K - 1, 0.0);  // Gradient for τ_1, ..., τ_{K-1}

    for(int i = 0; i < n; ++i){
        int j = ITEM_INDS.at(i) - 1;
        int w = WORKER_INDS.at(i) - 1;
        int c = static_cast<int>(Y.at(i));  // Category (1 to K)

        // Compute mean
        double eta = LAMBDA.at(j);
        if(WORKER_NUISANCE && w > 0){
            eta += LAMBDA.at(J + w - 1);
        }
        double mu = link::mu(eta);

        // Beta distribution parameters
        double a = mu * PHI;
        double b = (1.0 - mu) * PHI;

        // Thresholds for this observation
        double tau_c = TAU.at(c);
        double tau_cm1 = TAU.at(c - 1);

        // CDFs
        double F_c = boost::math::ibeta(a, b, tau_c);
        double F_cm1 = boost::math::ibeta(a, b, tau_cm1);
        double prob = F_c - F_cm1;

        // Contribution to gradient for τ_{c-1} (lower threshold)
        if(c > 1){  // τ_0 = 0 is fixed
            // PDF at τ_{c-1} using ibeta_derivative
            // Add safeguards to prevent numerical overflow
            double pdf_cm1 = 0.0;

            // Check for safe parameter ranges
            // Skip computation if it would likely overflow
            bool safe_cm1 = true;

            // Check 1: Avoid overflow when tau is very close to 0 and a < 1
            if(tau_cm1 < 1e-10 && a < 1.0) safe_cm1 = false;

            // Check 2: Avoid overflow when tau is very close to 1 and b < 1
            if(tau_cm1 > 1.0 - 1e-10 && b < 1.0) safe_cm1 = false;

            // Check 3: Avoid computation with extreme parameter combinations
            if(a > 1e6 || b > 1e6) safe_cm1 = false;

            // Check 4: Avoid when tau is outside reasonable bounds for the beta distribution
            double mode_approx = (a > 1 && b > 1) ? (a - 1) / (a + b - 2) : 0.5;
            if(std::abs(tau_cm1 - mode_approx) > 0.45 && (a > 100 || b > 100)) safe_cm1 = false;

            if(safe_cm1) {
                try {
                    pdf_cm1 = boost::math::ibeta_derivative(a, b, tau_cm1);
                    // Check for inf/nan
                    if(!std::isfinite(pdf_cm1)) pdf_cm1 = 0.0;
                } catch(...) {
                    pdf_cm1 = 0.0;
                }
            }

            // ∂log L/∂τ_{c-1} = -f(τ_{c-1}) / prob
            GRAD_TAU.at(c - 2) -= pdf_cm1 / std::max(prob, 1e-12);
        }

        // Contribution to gradient for τ_c (upper threshold)
        if(c < K){  // τ_K = 1 is fixed
            // PDF at τ_c using ibeta_derivative
            // Add safeguards to prevent numerical overflow
            double pdf_c = 0.0;

            // Check for safe parameter ranges
            bool safe_c = true;

            // Check 1: Avoid overflow when tau is very close to 0 and a < 1
            if(tau_c < 1e-10 && a < 1.0) safe_c = false;

            // Check 2: Avoid overflow when tau is very close to 1 and b < 1
            if(tau_c > 1.0 - 1e-10 && b < 1.0) safe_c = false;

            // Check 3: Avoid computation with extreme parameter combinations
            if(a > 1e6 || b > 1e6) safe_c = false;

            // Check 4: Avoid when tau is outside reasonable bounds for the beta distribution
            double mode_approx = (a > 1 && b > 1) ? (a - 1) / (a + b - 2) : 0.5;
            if(std::abs(tau_c - mode_approx) > 0.45 && (a > 100 || b > 100)) safe_c = false;

            if(safe_c) {
                try {
                    pdf_c = boost::math::ibeta_derivative(a, b, tau_c);
                    // Check for inf/nan
                    if(!std::isfinite(pdf_c)) pdf_c = 0.0;
                } catch(...) {
                    pdf_c = 0.0;
                }
            }

            // ∂log L/∂τ_c = +f(τ_c) / prob
            GRAD_TAU.at(c - 1) += pdf_c / std::max(prob, 1e-12);
        }
    }
}
