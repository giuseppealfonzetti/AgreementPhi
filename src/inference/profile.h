#ifndef AGREEMENTPHI_INFERENCE_PROFILE_H
#define AGREEMENTPHI_INFERENCE_PROFILE_H
#include <boost/math/tools/minima.hpp>
#include <Rcpp.h>
#include "../models/oneway.h"

namespace AgreementPhi{
    namespace continuous{
        namespace oneway{
            namespace inference{
                // Brent search to profile item related nuisance parameter
                double profiling_brent(
                    const std::vector<double> Y, 
                    const std::vector<std::vector<int>> DICT,
                    const int ITEM,
                    const double ALPHA_START,
                    const double PHI,
                    const int RANGE,
                    const int MAX_ITER
                );

                double profiling_newtonraphson(
                    const std::vector<double> Y, 
                    const std::vector<std::vector<int>> DICT,
                    const int ITEM,
                    const double ALPHA_START,
                    const double PHI,
                    const int RANGE,
                    const int MAX_ITER
                );

                // Modified Profile loglikelihood for phi
                double profile_loglik(
                    double PHI,
                    const std::vector<double> Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double> ALPHA_START,
                    const int J,
                    const int RANGE,
                    const int MAX_ITER,
                    const int METHOD
                );

                // Modified Profile loglikelihood for phi
                double modified_profile_loglik(
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

                // Get precision estimate by maximizing the profile likelihood
                std::pair<double, double> get_phi_profile(
                    const std::vector<double> Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double> ALPHA_START,
                    const double PHI_START,
                    const int J,
                    const int SEARCH_RANGE,
                    const int MAX_ITER,
                    const int PROF_SEARCH_RANGE,
                    const int PROF_MAX_ITER,
                    const int PROF_METHOD);

                // Get precision estimate by maximizing the modified profile likelihood
                std::vector<double> get_phi_modified_profile(
                    const std::vector<double> Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double> ALPHA_START,
                    const double PHI_START,
                    const int J,
                    const int SEARCH_RANGE,
                    const int MAX_ITER,
                    const int PROF_SEARCH_RANGE,
                    const int PROF_MAX_ITER,
                    const int PROF_METHOD,
                    const bool VERBOSE = false);
                




                
            }
        }
    }

    namespace ordinal{
        namespace oneway{
            namespace inference{
                // Brent search to profile item related nuisance parameter
                double profiling_brent(
                    const std::vector<double>& Y, 
                    const std::vector<std::vector<int>> DICT,
                    const int ITEM,
                    const double ALPHA_START,
                    const double PHI,
                    const int K,
                    const int RANGE,
                    const int MAX_ITER
                );

                // Newton-Raphson to profile item related nuisance parameter
                double profiling_newtonraphson(
                    const std::vector<double>& Y, 
                    const std::vector<std::vector<int>> DICT,
                    const int ITEM,
                    const double ALPHA_START,
                    const double PHI,
                    const int K,
                    const int RANGE,
                    const int MAX_ITER
                );

                // Profile loglikelihood for phi
                double profile_loglik(
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

                // Modified Profile likelihood for phi
                double modified_profile_loglik(
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

                // Get precision estimate by maximizing the profile likelihood
                std::pair<double, double> get_phi_profile(
                    const std::vector<double>& Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double>& ALPHA_START,
                    const double PHI_START,
                    const int K,
                    const int J,
                    const int SEARCH_RANGE,
                    const int MAX_ITER,
                    const int PROF_SEARCH_RANGE,
                    const int PROF_MAX_ITER,
                    const int PROF_METHOD
                );

                // Get precision estimate by maximizing the modified profile likelihood
                std::vector<double> get_phi_modified_profile(
                    const std::vector<double>& Y, 
                    const std::vector<std::vector<int>> DICT,
                    const std::vector<double>& ALPHA_START,
                    const double PHI_START,
                    const int K,
                    const int J,
                    const int SEARCH_RANGE,
                    const int MAX_ITER,
                    const int PROF_SEARCH_RANGE,
                    const int PROF_MAX_ITER,
                    const int PROF_METHOD,
                    const bool VERBOSE = false
                );
            }
        }
    }
}


#endif