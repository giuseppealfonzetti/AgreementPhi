#include "continuous.h"

double AgreementPhi::continuous::loglik(
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

double AgreementPhi::continuous::E0_dmu0dmu1(
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