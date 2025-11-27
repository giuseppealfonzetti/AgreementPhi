#ifndef AGREEMENTPHI_INFERENCE_PRECISION_H
#define AGREEMENTPHI_INFERENCE_PRECISION_H
#include <boost/math/tools/minima.hpp>
#include "../ratings/continuous.h"
#include "../ratings/ordinal.h"
#include "../inference/loglik.h"
#include <Rcpp.h>
#include <algorithm>

namespace AgreementPhi{
    namespace continuous{
        namespace inference{
            std::vector<double> get_phi_profile(
                const std::vector<double> Y,  
                const std::vector<int> ITEM_INDS,
                const std::vector<int> WORKER_INDS,
                const std::vector<std::vector<int>> ITEM_DICT,
                const std::vector<std::vector<int>> WORKER_DICT,
                const std::vector<double> ALPHA,
                const std::vector<double> BETA,
                const double PHI_START,
                const int J,
                const int W,
                const bool ITEMS_NUISANCE,
                const bool WORKER_NUISANCE,
                const double SEARCH_RANGE,
                const int MAX_ITER,
                const double PROF_UNI_RANGE,
                const int PROF_UNI_MAX_ITER,
                const int PROF_MAX_ITER,
                const double PROF_TOL,
                const bool VERBOSE
            );
            std::vector<double> get_phi_modified_profile(
                const std::vector<double> Y,  
                const std::vector<int> ITEM_INDS,
                const std::vector<int> WORKER_INDS,
                const std::vector<std::vector<int>> ITEM_DICT,
                const std::vector<std::vector<int>> WORKER_DICT,
                const std::vector<double> ALPHA,
                const std::vector<double> BETA,
                const double PHI_START,
                const int J,
                const int W,
                const bool ITEMS_NUISANCE,
                const bool WORKER_NUISANCE,
                const double SEARCH_RANGE,
                const int MAX_ITER,
                const double PROF_UNI_RANGE,
                const int PROF_UNI_MAX_ITER,
                const int PROF_MAX_ITER,
                const double PROF_TOL,
                const bool VERBOSE
            );
        }
        


        
    }

    namespace ordinal{
        namespace inference{
            std::vector<double> get_phi_profile(
                const std::vector<double> Y,  
                const std::vector<int> ITEM_INDS,
                const std::vector<int> WORKER_INDS,
                const std::vector<std::vector<int>> ITEM_DICT,
                const std::vector<std::vector<int>> WORKER_DICT,
                const std::vector<std::vector<int>> CAT_DICT,
                std::vector<double>& ALPHA,
                std::vector<double>& BETA,
                std::vector<double>& TAU,
                const double PHI_START,
                const int J,
                const int W,
                const int K,
                const bool ITEMS_NUISANCE,
                const bool WORKER_NUISANCE,
                const bool THRESHOLDS_NUISANCE,
                const double SEARCH_RANGE,
                const int MAX_ITER,
                const double PROF_UNI_RANGE,
                const int PROF_UNI_MAX_ITER,
                const int PROF_MAX_ITER,
                const double PROF_TOL,
                const bool VERBOSE
            );

            std::vector<double> get_phi_modified_profile(
                const std::vector<double> Y,  
                const std::vector<int> ITEM_INDS,
                const std::vector<int> WORKER_INDS,
                const std::vector<std::vector<int>> ITEM_DICT,
                const std::vector<std::vector<int>> WORKER_DICT,
                const std::vector<std::vector<int>> CAT_DICT,
                std::vector<double>& ALPHA,
                std::vector<double>& BETA,
                std::vector<double>& TAU,
                const double PHI_START,
                const int J,
                const int W,
                const int K,
                const bool ITEMS_NUISANCE,
                const bool WORKER_NUISANCE,
                const bool THRESHOLDS_NUISANCE,
                const double SEARCH_RANGE,
                const int MAX_ITER,
                const double PROF_UNI_RANGE,
                const int PROF_UNI_MAX_ITER,
                const int PROF_MAX_ITER,
                const double PROF_TOL,
                const bool VERBOSE
            );
        }
        


    }
}



#endif