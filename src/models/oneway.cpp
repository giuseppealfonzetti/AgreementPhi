#include "oneway.h"

double AgreementPhi::continuous::oneway::item_loglik(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const int ITEM,
    const double ALPHA,
    const double PHI,
    double &DALPHA,
    double &DALPHA2,
    const int GRADFLAG
){
    double ll = 0;
    double mu = link::mu(ALPHA);
    double dmu = link::dmu(mu);
    double dmu2 = link::d2mu(mu);

    for(int i=0; i<DICT.at(ITEM).size(); i++){
        int obs_id = DICT.at(ITEM).at(i);

        double dwrtmu = 0;
        double dwrtmu2 = 0;
        ll += continuous::loglik(
            Y.at(obs_id),
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

double AgreementPhi::continuous::oneway::log_det_obs_info(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>> DICT,
    const std::vector<double>& ALPHA,
    const double PHI
){
    const int J = ALPHA.size();
    
    std::vector<double> mu(ALPHA.size());
    std::transform(ALPHA.begin(), ALPHA.end(), mu.begin(), link::mu);

    double out = 0;
    for(int j=0; j<J; j++){
        double dalpha = 0;
        double dalpha2 = 0;
        double ll = item_loglik(
            Y, DICT, j, ALPHA.at(j), PHI, dalpha, dalpha2, 2
        );
        
        out += log(-dalpha2);
    }

    return out;
}

double AgreementPhi::continuous::oneway::log_det_E0d0d1(
    const std::vector<std::vector<int>> DICT,
    const std::vector<double>& ALPHA0,
    const std::vector<double>& ALPHA1,
    const double PHI0,
    const double PHI1
){
    const int J = ALPHA0.size();

    std::vector<double> mu0(J);
    std::transform(ALPHA0.begin(), ALPHA0.end(), mu0.begin(), link::mu);


    std::vector<double> mu1(J);
    std::transform(ALPHA1.begin(), ALPHA1.end(), mu1.begin(), link::mu);

    double out = 0;

    for(int j=0; j<J; j++){
        double e = continuous::E0_dmu0dmu1(mu0.at(j), PHI0, mu1.at(j), PHI1);
        double dmu0 = link::dmu(mu0.at(j));
        double dmu1 = link::dmu(mu1.at(j));
        double n_obs_per_item = static_cast<double>(DICT.at(j).size());
        double outj = n_obs_per_item * dmu0 * dmu1 * e; 
        
        out += std::log(outj);
    }

    return out;
}

double AgreementPhi::ordinal::oneway::item_loglik(
    const std::vector<double>& Y, 
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
    double mu = link::mu(ALPHA);
    double dmu = link::dmu(mu);
    double dmu2 = link::d2mu(mu);

    for(int i=0; i<DICT.at(ITEM).size(); i++){
        int obs_id = DICT.at(ITEM).at(i);

        double dwrtmu = 0;
        double dwrtmu2 = 0;
        ll += ordinal::loglik(
            Y.at(obs_id),
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

double AgreementPhi::ordinal::oneway::log_det_obs_info(
            const std::vector<double>& Y, 
            const std::vector<std::vector<int>> DICT,
            const std::vector<double>& ALPHA,
            const double PHI,
            const int K)
{

    const int n = Y.size();
    const int J = ALPHA.size();


    std::vector<double> mu(ALPHA.size());
    std::transform(ALPHA.begin(), ALPHA.end(), mu.begin(), link::mu);
    
    double logdetobs=0;


    double out =0;
    for(int j=0; j<J; j++){

        double dalpha  =0;
        double dalpha2 =0;
        double ll = item_loglik(
                Y, DICT, j, ALPHA.at(j), PHI, K, dalpha, dalpha2, 2
            );
        
        // Rcpp::Rcout <<"Trace from log_det_obs_info | Item "<< j <<" | PHI: "<< PHI<< " | Alpha: "<<ALPHA.at(j)<< " | dalpha: "<< dalpha << " | -dalpha2: "<<-dalpha2<<" | log(-dalpha2): "<<std::log(-dalpha2) <<  " | ll: "<<ll << "\n";
        out += log(-dalpha2);
    }




    return out;
}

double AgreementPhi::ordinal::oneway::log_det_E0d0d1(
            const std::vector<std::vector<int>> DICT,
            const std::vector<double>& ALPHA0,
            const std::vector<double>& ALPHA1,
            const double PHI0,
            const double PHI1,
            const int K
){
    const int J = ALPHA0.size();

    std::vector<double> mu0(ALPHA0.size());
    std::transform(ALPHA0.begin(), ALPHA0.end(), mu0.begin(), link::mu);

    std::vector<double> mu1(ALPHA1.size());
    std::transform(ALPHA1.begin(), ALPHA1.end(), mu1.begin(), link::mu);

    double out=0;

    for(int j=0; j<J; j++){
        double e = E0_dmu0dmu1(mu0.at(j), PHI0, mu1.at(j), PHI1, K);
        double dmu0 = link::dmu(mu0.at(j));
        double dmu1 = link::dmu(mu1.at(j));
        double n_obs_per_item = static_cast<double>(DICT.at(j).size());
        double outj = n_obs_per_item*dmu0 * dmu1 * e; 
        // Rcpp::Rcout <<"Trace from log_det_E0d0d1 | Item "<< j <<" | phi0: "<< PHI0<< " | phi1: "<< PHI1 << " | Alpha0: "<<ALPHA0.at(j)<< " | Alpha1: "<<ALPHA1.at(j)<< " | mu0: "<<mu0.at(j)<<  " | mu1: "<<mu1.at(j) << " | dmu0: "<< dmu0 << " | dmu1: " << dmu1 << " | e: " << e << " | outj: " << outj << " | log(outj): " << std::log(outj) << "\n";
        out += std::log(outj);
    }

    return(out);

}