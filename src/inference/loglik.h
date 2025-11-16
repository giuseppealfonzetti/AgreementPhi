#ifndef AGREEMENTPHI_INFERENCE_LOGLIK_H
#define AGREEMENTPHI_INFERENCE_LOGLIK_H
#include <boost/math/tools/minima.hpp>
#include "../models/oneway.h"
#include "../models/twoway.h"
#include "../inference/nuisance.h"
#include "../inference/profile.h" // temporary, to be moved in nuisance

namespace AgreementPhi{
    namespace continuous{
        namespace oneway{
            namespace loglik{
                double profile(
                    double PHI,
                    const std::vector<double> Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double> ALPHA_START,
                    const int J,
                    const int RANGE,
                    const int MAX_ITER,
                    const int METHOD
                );

                double modified_profile(
                    double PHI,
                    const std::vector<double> Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double> ALPHA_START,
                    const std::vector<double> ALPHA_MLE,
                    const double PHI_MLE,
                    const int J,
                    const int PROF_SEARCH_RANGE,
                    const int PROF_MAX_ITER,
                    const int PROF_METHOD
                );
            }
        }
        namespace twoway{
            namespace loglik{
                double profile(
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
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                double modified_profile(
                    const std::vector<double> Y,  
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<double> ALPHA_MLE,
                    const std::vector<double> BETA_MLE,
                    const double PHI,
                    const double PHI_MLE,
                    const int J,
                    const int W,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );
            }
        }
    }

    namespace ordinal{
        namespace oneway{
            namespace loglik{
                double profile(
                    double PHI,
                    const std::vector<double>& Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double>& ALPHA_START,
                    const int K,
                    const int J,
                    const int RANGE,
                    const int MAX_ITER,
                    const int METHOD
                );

                double modified_profile(
                    double PHI,
                    const std::vector<double>& Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double>& ALPHA_START,
                    const std::vector<double>& ALPHA_MLE,
                    const double PHI_MLE,
                    const int K,
                    const int J,
                    const int PROF_SEARCH_RANGE,
                    const int PROF_MAX_ITER,
                    const int PROF_METHOD
                );
            }
        }
        namespace twoway{
            namespace loglik{
                double profile(
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
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                double modified_profile(
                    const std::vector<double> Y,  
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<double> ALPHA_MLE,
                    const std::vector<double> BETA_MLE,
                    const double PHI,
                    const double PHI_MLE,
                    const int J,
                    const int W,
                    const int K,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );
            }
        }
    }
}
#endif
