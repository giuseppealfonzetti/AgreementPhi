#ifndef AGREEMENTPHI_INFERENCE_NUISANCE_H
#define AGREEMENTPHI_INFERENCE_NUISANCE_H
#include <boost/math/tools/minima.hpp>
#include "../ratings/continuous.h"
#include "../ratings/ordinal.h"
#include <numeric>
#include <Rcpp.h>

namespace AgreementPhi{
    namespace continuous{
        namespace nuisance{
            double brent_profiling(
                const std::vector<double>& Y, 
                const std::vector<std::vector<int>>& DICT,
                const int IDX,
                const std::vector<int>& CONST_DIM_IDXS,
                const std::vector<double>& CONST_DIM_PARS,
                const double START,
                const double PHI,
                const double RANGE,
                const int MAX_ITER
            );

            std::vector<std::vector<double>> get_lambda(
                    const std::vector<double> Y,  
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const double PHI,
                    const int J,
                    const int W,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const double PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double TOL
                );

        }

       
    }

    namespace ordinal{
        namespace nuisance{
            double brent_profiling(
                const std::vector<double>& Y, 
                const std::vector<std::vector<int>>& DICT,
                const int IDX,
                const std::vector<int>& CONST_DIM_IDXS,
                const std::vector<double>& CONST_DIM_PARS,
                const double START,
                const double PHI,
                const std::vector<double>& TAU,
                const double RANGE,
                const int MAX_ITER,
                const double MEAN
            );

            double brent_profiling_thresholds(
                const std::vector<double>& Y, 
                const std::vector<double>& MU, 
                const std::vector<std::vector<int>> CAT_DICT,
                const int IDX,
                const std::vector<double>& TAU,
                const double PHI,
                const int MAX_ITER
            );

            // std::vector<std::vector<double>> get_lambda(
            //     const std::vector<double> Y,  
            //     const std::vector<int> ITEM_INDS,
            //     const std::vector<int> WORKER_INDS,
            //     const std::vector<std::vector<int>> ITEM_DICT,
            //     const std::vector<std::vector<int>> WORKER_DICT,
            //     const std::vector<double> ALPHA,
            //     const std::vector<double> BETA,
            //     const std::vector<double> TAU,
            //     const double PHI,
            //     const int J,
            //     const int W,
            //     const int K,
            //     const bool WORKER_NUISANCE,
            //     const double PROF_UNI_RANGE,
            //     const int PROF_UNI_MAX_ITER,
            //     const int PROF_MAX_ITER,
            //     const double TOL);
            std::vector<std::vector<double>> get_lambda2(
                const std::vector<double> Y,  
                const std::vector<int> ITEM_INDS,
                const std::vector<int> WORKER_INDS,
                const std::vector<std::vector<int>> ITEM_DICT,
                const std::vector<std::vector<int>> WORKER_DICT,
                const std::vector<std::vector<int>> CAT_DICT,
                const std::vector<double> ALPHA,
                const std::vector<double> BETA,
                const std::vector<double> TAU,
                const double PHI,
                const int J,
                const int W,
                const int K,
                const bool ITEMS_NUISANCE,
                const bool WORKER_NUISANCE,
                const bool THRESHOLDS_NUISANCE,
                const double PROF_UNI_RANGE,
                const int PROF_UNI_MAX_ITER,
                const int PROF_MAX_ITER,
                const double TOL);

        
    }
}
}

#endif