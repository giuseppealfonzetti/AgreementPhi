#ifndef AGREEMENTPHI_INFERENCE_LOGLIK_H
#define AGREEMENTPHI_INFERENCE_LOGLIK_H
#include <boost/math/tools/minima.hpp>
#include "../ratings/continuous.h"
#include "../ratings/ordinal.h"
#include "../inference/nuisance.h"

namespace AgreementPhi{
    namespace continuous{
        namespace ll{
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
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
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
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );
            }

    }

    namespace ordinal{
        namespace ll{
                double profile(
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
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA_MLE,
                    const std::vector<double> BETA_MLE,
                    const std::vector<double> TAU_MLE,
                    const double PHI,
                    const double PHI_MLE,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const bool THRESHOLDS_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                double modified_profile_extended(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA_MLE,
                    const std::vector<double> BETA_MLE,
                    const std::vector<double> TAU,
                    const std::vector<double> TAU_MLE,
                    const double PHI,
                    const double PHI_MLE,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                double modified_profile_tau_profiled(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA_MLE,
                    const std::vector<double> BETA_MLE,
                    const std::vector<double> TAU_START,
                    const std::vector<double> TAU_MLE,
                    const double PHI,
                    const double PHI_MLE,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Profile likelihood gradient w.r.t. tau
                void profile_grad_tau(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA_START,
                    const std::vector<double> BETA_START,
                    const std::vector<double> TAU,
                    const double PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL,
                    std::vector<double>& GRAD_TAU
                );






                // profile likelihood where phi and tau are explicitely reparametrised to
                // account for positivity, bounds and ordering constraints
                double profile_extended(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> RAW_TAU,
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Gradient of profile_extended wrt RAW_TAU
                Eigen::VectorXd profile_extended_grad_raw_tau(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> RAW_TAU,
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Gradient of profile_extended wrt RAW_PHI
                double profile_extended_grad_raw_phi(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> RAW_TAU,
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Gradient of profile_extended wrt RAW_PHI and RAW_TAU
                Eigen::VectorXd profile_extended_grad(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> RAW_TAU,
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Profile likelihood with parsimonious gamma parameterization (2 params instead of K-1)
                double profile_extended_gamma(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> GAMMA,  // 2 parameters
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Gradient of profile_extended_gamma wrt GAMMA
                Eigen::VectorXd profile_extended_grad_gamma(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> GAMMA,
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Combined gradient for RAW_PHI and GAMMA
                Eigen::VectorXd profile_extended_grad_gamma_phi(
                    const std::vector<double> Y,
                    const std::vector<int> ITEM_INDS,
                    const std::vector<int> WORKER_INDS,
                    const std::vector<std::vector<int>> ITEM_DICT,
                    const std::vector<std::vector<int>> WORKER_DICT,
                    const std::vector<std::vector<int>> CAT_DICT,
                    const std::vector<double> ALPHA,
                    const std::vector<double> BETA,
                    const std::vector<double> GAMMA,
                    const double RAW_PHI,
                    const int J,
                    const int W,
                    const int K,
                    const bool ITEMS_NUISANCE,
                    const bool WORKER_NUISANCE,
                    const int PROF_UNI_RANGE,
                    const int PROF_UNI_MAX_ITER,
                    const int PROF_MAX_ITER,
                    const double PROF_TOL
                );

                // Hessian of profile_extended wrt RAW_TAU
                // Eigen::MatrixXd profile_extended_hess_raw_tau(
                //     const std::vector<double> Y,  
                //     const std::vector<int> ITEM_INDS,
                //     const std::vector<int> WORKER_INDS,
                //     const std::vector<std::vector<int>> ITEM_DICT,
                //     const std::vector<std::vector<int>> WORKER_DICT,
                //     const std::vector<std::vector<int>> CAT_DICT,
                //     const std::vector<double> ALPHA,
                //     const std::vector<double> BETA,
                //     const std::vector<double> RAW_TAU,
                //     const double RAW_PHI,
                //     const int J,
                //     const int W,
                //     const int K,
                //     const bool ITEMS_NUISANCE,
                //     const bool WORKER_NUISANCE,
                //     const int PROF_UNI_RANGE,
                //     const int PROF_UNI_MAX_ITER,
                //     const int PROF_MAX_ITER,
                //     const double PROF_TOL
                // );

                
            }


    }
}
#endif
