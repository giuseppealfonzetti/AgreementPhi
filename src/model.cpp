#include "model.h"

#define DOUBLE_EPS DBL_EPSILON
static const double THRESH = 30.0;
static const double MTHRESH = -30.0;
static const double INVEPS = 1/DOUBLE_EPS;

double LinkFuns::logit::mu(const double ETA){
        double expmeta;
        if(-ETA < MTHRESH){
            expmeta = DOUBLE_EPS;
        }else if(-ETA > THRESH){
            expmeta = INVEPS;
        }else{
            expmeta = exp(-ETA);
        }

        return 1/(1+expmeta);
}

double LinkFuns::logit::dmu(const double MU){
    return MU*(1-MU);
}
double LinkFuns::logit::d2mu(const double MU){
    return MU*(1-MU)*(1-2*MU);
}

double model::ordinal::loglik(
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
        double dlogBda = BetaFuns::dlogBda(a,b);
        double dlogBdb = BetaFuns::dlogBdb(a,b);
        
        //  F(Y=c, mu, phi)
        double t_diBda = iBetaFuns::diBda(t,a,b);
        double t_diBdb = iBetaFuns::diBdb(t,a,b);

        double t_da = CDFFuns::dF(t_diBda, dlogBda, betainv, t_cdf);
        double t_db = CDFFuns::dF(t_diBdb, dlogBdb, betainv, t_cdf);
        
        double t_dmu = CDFFuns::dFdmu(PHI, t_da, t_db);
        // double t_dphi = CDFFuns::dFdphi(MU, t_da, t_db);

        // F(Y=c-1; mu, phi)
        double tm1_diBda = iBetaFuns::diBda(tm1,a,b);
        double tm1_diBdb = iBetaFuns::diBdb(tm1,a,b);

        double tm1_da = CDFFuns::dF(tm1_diBda, dlogBda, betainv, tm1_cdf);
        double tm1_db = CDFFuns::dF(tm1_diBdb, dlogBdb, betainv, tm1_cdf);

        double tm1_dmu = CDFFuns::dFdmu(PHI, tm1_da, tm1_db);
        // double tm1_dphi = CDFFuns::dFdphi(MU, tm1_da, tm1_db);

        // combine
        double dmu = (t_dmu - tm1_dmu) / std::max(prob, 1e-12) ;       
        // double dphi = (t_dphi-tm1_dphi)/(prob+1e-8);

        DMU += dmu;

        if(GRADFLAG>1){
                double d2logBda2 = BetaFuns::d2logBda2(a,b);
                double d2logBdb2 = BetaFuns::d2logBdb2(a,b);
                double d2logBdadb = BetaFuns::d2logBdadb(a,b);

                //  F(Y=c, mu, phi)
                double t_d2iBda2 = iBetaFuns::d2iBda2(t,a,b);
                double t_d2iBdb2 = iBetaFuns::d2iBdb2(t,a,b);
                double t_d2iBdadb = iBetaFuns::d2iBdadb(t,a,b);

                double t_da2 = CDFFuns::d2F(t_diBda, t_diBda, dlogBda, dlogBda, t_d2iBda2, d2logBda2, betainv, t_cdf);
                double t_db2 = CDFFuns::d2F(t_diBdb, t_diBdb, dlogBdb, dlogBdb, t_d2iBdb2, d2logBdb2, betainv, t_cdf);
                double t_dadb = CDFFuns::d2F(t_diBda, t_diBdb, dlogBda, dlogBdb, t_d2iBdadb, d2logBdadb, betainv, t_cdf);

                double t_dmu2 = CDFFuns::d2Fdmu2(PHI, t_da, t_db, t_da2, t_db2, t_dadb);
                // double t_dphi2 = CDFFuns::d2Fdphi2(MU, t_da, t_db, t_da2, t_db2, t_dadb);
                // double t_dmudphi = CDFFuns::d2Fdmudphi(MU, PHI, t_da, t_db, t_da2, t_db2, t_dadb);

                // F(Y=c-1; mu, phi)
                double tm1_d2iBda2 = iBetaFuns::d2iBda2(tm1,a,b);
                double tm1_d2iBdb2 = iBetaFuns::d2iBdb2(tm1,a,b);
                double tm1_d2iBdadb = iBetaFuns::d2iBdadb(tm1,a,b);

                double tm1_da2 = CDFFuns::d2F(tm1_diBda, tm1_diBda, dlogBda, dlogBda, tm1_d2iBda2, d2logBda2, betainv, tm1_cdf);
                double tm1_db2 = CDFFuns::d2F(tm1_diBdb, tm1_diBdb, dlogBdb, dlogBdb, tm1_d2iBdb2, d2logBdb2, betainv, tm1_cdf);
                double tm1_dadb = CDFFuns::d2F(tm1_diBda, tm1_diBdb, dlogBda, dlogBdb, tm1_d2iBdadb, d2logBdadb, betainv, tm1_cdf);

                double tm1_dmu2 = CDFFuns::d2Fdmu2(PHI, tm1_da, tm1_db, tm1_da2, tm1_db2, tm1_dadb);
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

double model::ordinal::E0_dmu0dmu1(
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
        double ll0 = model::ordinal::loglik(c, MU0, PHI0, K, dmu0, dmu02, 1);

        double dmu1 = 0;
        double dmu12 = 0;
        double ll1 = model::ordinal::loglik(c, MU1, PHI1, K, dmu1, dmu12, 1);

        out += dmu0*dmu1*exp(ll0);

    }
    return out;
}

double model::continuous::loglik(
    const double Y, 
    const double MU,
    const double PHI,
    double &DMU, 
    double &DMU2, 
    const int GRADFLAG
){
    double a = MU*PHI;
    double b = (1-MU)*PHI;
    
    // Observed and expected logits
    double y_star = log(Y/(1-Y));
    double mu_star = boost::math::digamma(a) - boost::math::digamma(b);
    
    // Log-likelihood
    double logprob = boost::math::lgamma(PHI) - boost::math::lgamma(a) - boost::math::lgamma(b) 
                    + (a-1)*log(Y) + (b-1)*log(1-Y);
    
    if(GRADFLAG > 0){
        double dmu = PHI * (y_star - mu_star);
        DMU += dmu;
        
        if(GRADFLAG > 1){
            double trigamma_a = boost::math::trigamma(a);
            double trigamma_b = boost::math::trigamma(b);
            double dmu_star_dmu = PHI * (trigamma_a + trigamma_b);
            double dmu2 = -PHI * dmu_star_dmu;
            
            DMU2 += dmu2;
        }
    }
    
    return logprob;
}

double model::continuous::E0_dmu0dmu1(
    const double MU0,
    const double PHI0,
    const double MU1,
    const double PHI1
){
    
    double a0 = MU0 * PHI0;
    double b0 = (1-MU0) * PHI0;
    
    double trigamma_a0 = boost::math::trigamma(a0);
    double trigamma_b0 = boost::math::trigamma(b0);
    
    return PHI0 * PHI1 * (trigamma_a0 + trigamma_b0);
}

double sample::ordinal::item_loglik(
    const Eigen::Ref<const Eigen::VectorXd> Y, 
    const std::vector<std::vector<int>> DICT,
    const int ITEM,
    const double ALPHA,
    const double PHI,
    const int K,
    double &DALPHA,
    double &DALPHA2,
    const int GRADFLAG
){
    const int n = Y.size();
    double ll = 0;
    double mu = LinkFuns::logit::mu(ALPHA);
    double dmu = LinkFuns::logit::dmu(mu);
    double dmu2 = LinkFuns::logit::d2mu(mu);

    for(int i=0; i<DICT.at(ITEM).size(); i++){
        int obs_id = DICT.at(ITEM).at(i);

        double dwrtmu = 0;
        double dwrtmu2 = 0;
        ll += model::ordinal::loglik(
            Y(obs_id),
            mu,
            PHI,
            K,
            dwrtmu,
            dwrtmu2,
            GRADFLAG
        );


        if(GRADFLAG>0){
            DALPHA+=dwrtmu*dmu;
            if(GRADFLAG>1){
                DALPHA2+=dwrtmu2*pow(dmu, 2)+dwrtmu*dmu2;
            }
        }

    }

    return ll;
}

double sample::ordinal::log_det_obs_info(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA,
            const double PHI,
            const int K)
{

    const int n = Y.size();
    const int J = ALPHA.size();


    Eigen::VectorXd eta = ALPHA;
    Eigen::VectorXd mu = eta.unaryExpr(&LinkFuns::logit::mu);

    double logdetobs=0;


    double out =0;
    for(int j=0; j<J; j++){

        double dalpha  =0;
        double dalpha2 =0;
        double ll=sample::ordinal::item_loglik(
                Y, DICT, j, ALPHA(j), PHI, K, dalpha, dalpha2, 2
            );
        
        out += log(-dalpha2);
    }




    return out;
}

double sample::ordinal::log_det_E0d0d1(
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA0,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA1,
            const double PHI0,
            const double PHI1,
            const int K
){
    const int J = ALPHA0.size();

    Eigen::VectorXd eta0 = ALPHA0;
    Eigen::VectorXd mu0 = eta0.unaryExpr(&LinkFuns::logit::mu);

    Eigen::VectorXd eta1 = ALPHA1;
    Eigen::VectorXd mu1 = eta1.unaryExpr(&LinkFuns::logit::mu);

    double out=0;

    for(int j=0; j<J; j++){
        double e = model::ordinal::E0_dmu0dmu1(mu0(j), PHI0, mu1(j), PHI1, K);
        double dmu0 = LinkFuns::logit::dmu(mu0(j));
        double dmu1 = LinkFuns::logit::dmu(mu1(j));
        double n_obs_per_item = static_cast<double>(DICT.at(j).size());
        double outj = n_obs_per_item*dmu0 * dmu1 * e; 
        out += std::log(outj);
    }

    return(out);

}

double sample::continuous::item_loglik(
    const Eigen::Ref<const Eigen::VectorXd> Y, 
    const std::vector<std::vector<int>> DICT,
    const int ITEM,
    const double ALPHA,
    const double PHI,
    double &DALPHA,
    double &DALPHA2,
    const int GRADFLAG
){
    double ll = 0;
    double mu = LinkFuns::logit::mu(ALPHA);
    double dmu = LinkFuns::logit::dmu(mu);
    double dmu2 = LinkFuns::logit::d2mu(mu);

    for(int i=0; i<DICT.at(ITEM).size(); i++){
        int obs_id = DICT.at(ITEM).at(i);

        double dwrtmu = 0;
        double dwrtmu2 = 0;
        ll += model::continuous::loglik(
            Y(obs_id),
            mu,
            PHI,
            dwrtmu,
            dwrtmu2,
            GRADFLAG
        );

        if(GRADFLAG>0){
            DALPHA += dwrtmu*dmu;
            if(GRADFLAG>1){
                DALPHA2 += dwrtmu2*pow(dmu, 2) + dwrtmu*dmu2;
            }
        }
    }

    return ll;
}

double sample::continuous::log_det_obs_info(
    const Eigen::Ref<const Eigen::VectorXd> Y, 
    const std::vector<std::vector<int>> DICT,
    const Eigen::Ref<const Eigen::VectorXd> ALPHA,
    const double PHI
){
    const int J = ALPHA.size();
    
    Eigen::VectorXd eta = ALPHA;
    Eigen::VectorXd mu = eta.unaryExpr(&LinkFuns::logit::mu);

    double out = 0;
    for(int j=0; j<J; j++){
        double dalpha = 0;
        double dalpha2 = 0;
        double ll = sample::continuous::item_loglik(
            Y, DICT, j, ALPHA(j), PHI, dalpha, dalpha2, 2
        );
        
        out += log(-dalpha2);
    }

    return out;
}

double sample::continuous::log_det_E0d0d1(
    const std::vector<std::vector<int>> DICT,
    const Eigen::Ref<const Eigen::VectorXd> ALPHA0,
    const Eigen::Ref<const Eigen::VectorXd> ALPHA1,
    const double PHI0,
    const double PHI1
){
    const int J = ALPHA0.size();

    Eigen::VectorXd eta0 = ALPHA0;
    Eigen::VectorXd mu0 = eta0.unaryExpr(&LinkFuns::logit::mu);

    Eigen::VectorXd eta1 = ALPHA1;
    Eigen::VectorXd mu1 = eta1.unaryExpr(&LinkFuns::logit::mu);

    double out = 0;

    for(int j=0; j<J; j++){
        double e = model::continuous::E0_dmu0dmu1(mu0(j), PHI0, mu1(j), PHI1);
        double dmu0 = LinkFuns::logit::dmu(mu0(j));
        double dmu1 = LinkFuns::logit::dmu(mu1(j));
        double n_obs_per_item = static_cast<double>(DICT.at(j).size());
        double outj = n_obs_per_item * dmu0 * dmu1 * e; 
        out += std::log(outj);
    }

    return out;
}
