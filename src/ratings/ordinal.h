#ifndef AGREEMENTPHI_RATINGS_ORDINAL_H
#define AGREEMENTPHI_RATINGS_ORDINAL_H

#include "../utilities/beta_functions.h"
#include "../utilities/link_functions.h"
#include "../utilities/utils_functions.h"
#include <Eigen/Dense>
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define EIGEN_DONT_PARALLELIZE

namespace AgreementPhi{
    namespace ordinal{
         // compute loglik of the model for a single observation
        // if GRADFLAG>0, it additionally overwrite dmu
        // if GRADFLAG>1, it additionally overwrite dmu2
        double loglik(
            const double Y, 
            const double MU,
            const double PHI,
            const std::vector<double> TAU,
            double &DMU, 
            double &DMU2, 
            const int GRADFLAG
        );

        // Compute E_{\theta_0}{dl_i(\theta_0)/dmu_0 dl_i(\theta_1)/dmu_1}
        double E0_dmu0dmu1(
            const double MU0,
            const double PHI0,
            const double MU1,
            const double PHI1,
            const std::vector<double> TAU,
            const int K
        );

        double joint_loglik(
            const std::vector<double>& Y, 
            const std::vector<int>& ITEM_INDS,   // expected to be coded 1 to J
            const std::vector<int>& WORKER_INDS, // expected to be coded 1 to W
            const std::vector<double>&  LAMBDA,
            const std::vector<double>&  TAU,
            const double PHI,
            const int J, 
            const int W,
            const int K,
            const bool WORKER_NUISANCE,
            Eigen::Ref<Eigen::VectorXd> DLAMBDA,
            Eigen::Ref<Eigen::VectorXd> JALPHAALPHA,
            Eigen::Ref<Eigen::VectorXd> JBETABETA,
            Eigen::Ref<Eigen::MatrixXd> JALPHABETA,
            const int GRADFLAG
        );

        double log_det_obs_info(
            const std::vector<double>& Y,
            const std::vector<int>& ITEM_INDS,
            const std::vector<int>& WORKER_INDS,
            const std::vector<double>&  LAMBDA,
            const std::vector<double>&  TAU,
            const double PHI,
            const int J,
            const int W,
            const int K,
            const bool WORKER_NUISANCE
        );

        double log_det_E0d0d1(
            const std::vector<int>& ITEM_INDS,
            const std::vector<int>& WORKER_INDS,
            const std::vector<double>&  LAMBDA0,
            const std::vector<double>&  LAMBDA1,
            const double PHI0,
            const double PHI1,
            const std::vector<double>&  TAU,
            const int J,
            const int W,
            const int K,
            const bool WORKER_NUISANCE
        );
    }
}

# endif