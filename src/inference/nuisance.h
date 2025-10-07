#ifndef AGREEMENTPHI_INFERENCE_NUISANCE_H
#define AGREEMENTPHI_INFERENCE_NUISANCE_H
#include <boost/math/tools/minima.hpp>
#include "../models/oneway.h"
#include "../models/twoway.h"

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

        }

        namespace twoway{
            namespace inference{
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
                    const double PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double TOL
                );
            }
            
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
                const int K,
                const double RANGE,
                const int MAX_ITER
            );
        }

        namespace twoway{
            namespace inference{
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
                const int K,
                const double PROF_UNI_RANGE,
                const int PROF_UNI_MAX_ITER,
                const int PROF_MAX_ITER,
                const double TOL);
            }
        }
    }
}

#endif