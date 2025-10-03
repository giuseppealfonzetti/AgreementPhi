#include "ordinal.h"

double AgreementPhi::ordinal::loglik(
    const double Y, 
    const double MU,
    const double PHI,
    const int K,
    double &DMU, 
    double &DMU2, 
    const int GRADFLAG
){
    double a = MU*PHI;
    double b = (1-MU)*PHI;
    double t   = Y / static_cast<double>(K);       
    double tm1 = (Y-1) / static_cast<double>(K);   


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
    const int K)
{


    double out = 0;
    for(int c=1; c<=K; c++){
        double dmu0 = 0;
        double dmu02 = 0;
        double ll0 = ordinal::loglik(c, MU0, PHI0, K, dmu0, dmu02, 1);

        double dmu1 = 0;
        double dmu12 = 0;
        double ll1 = ordinal::loglik(c, MU1, PHI1, K, dmu1, dmu12, 1);

        out += dmu0*dmu1*exp(ll0);

    }
    return out;
}