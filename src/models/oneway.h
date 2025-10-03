#ifndef AGREEMENTPHI_MODELS_ONEWAY_H
#define AGREEMENTPHI_MODELS_ONEWAY_H
#include "../ratings/continuous.h"
#include "../ratings/ordinal.h"

namespace AgreementPhi{
    namespace continuous{
        namespace oneway{
            double item_loglik(
                const std::vector<double>& Y, 
                const std::vector<std::vector<int>> DICT,
                const int ITEM,
                const double ALPHA,
                const double PHI,
                double &DALPHA,
                double &DALPHA2,
                const int GRADFLAG
            );

            double log_det_obs_info(
                const std::vector<double>& Y, 
                const std::vector<std::vector<int>> DICT,
                const std::vector<double>& ALPHA,
                const double PHI
            );

            double log_det_E0d0d1(
                const std::vector<std::vector<int>> DICT,
                const std::vector<double>& ALPHA0,
                const std::vector<double>& ALPHA1,
                const double PHI0,
                const double PHI1
            );
        }
    }

    namespace ordinal{
        namespace oneway{
            double item_loglik(
                const std::vector<double>& Y, 
                const std::vector<std::vector<int>> DICT,
                const int ITEM,
                const double ALPHA,
                const double PHI,
                const int K,
                double &DALPHA,
                double &DALPHA2,
                const int GRADFLAG
            );

            double log_det_obs_info(
                const std::vector<double>& Y, 
                const std::vector<std::vector<int>> DICT,
                const std::vector<double>& ALPHA,
                const double PHI,
                const int K
            );

            double log_det_E0d0d1(
                const std::vector<std::vector<int>> DICT,
                const std::vector<double>& ALPHA0,
                const std::vector<double>& ALPHA1,
                const double PHI0,
                const double PHI1,
                const int K
            );
        }
        
    }
}

#endif