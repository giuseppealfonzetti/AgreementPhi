#ifndef model_H
#define model_H
#include <boost/math/special_functions/beta.hpp>
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#include <RcppEigen.h>
#include "utils.h"
#include <float.h>

namespace LinkFuns{
    namespace logit{
        double mu(const double ETA);
        double dmu(const double MU);
        double d2mu(const double MU);
    }
}

namespace model{

    namespace ordinal{

        // compute loglik of the model for a single observation
        // if GRADFLAG>0, it additionally overwrite dmu
        // if GRADFLAG>1, it additionally overwrite dmu2
        double loglik(
            const double Y, 
            const double MU,
            const double PHI,
            const int K,
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
            const int K
        );
        
    }

    namespace continuous{

        // compute loglik of the model for a single observation
        // if GRADFLAG>0, it additionally overwrite dmu
        // if GRADFLAG>1, it additionally overwrite dmu2
        double loglik(
            const double Y, 
            const double MU,
            const double PHI,
            double &DMU, 
            double &DMU2, 
            const int GRADFLAG
        );

        // Compute E_{\theta_0}{dl_i(\theta_0)/dmu_0 dl_i(\theta_1)/dmu_1}
        double E0_dmu0dmu1(
            const double MU0,
            const double PHI0,
            const double MU1,
            const double PHI1
        );
        
    }
}

namespace sample{
    namespace ordinal{
        double item_loglik(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
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
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA,
            const double PHI,
            const int K
        );

        double log_det_E0d0d1(
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA0,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA1,
            const double PHI0,
            const double PHI1,
            const int K
        );
    }
    
    namespace continuous{
        double item_loglik(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const int ITEM,
            const double ALPHA,
            const double PHI,
            double &DALPHA,
            double &DALPHA2,
            const int GRADFLAG
        );

        double log_det_obs_info(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA,
            const double PHI
        );

        double log_det_E0d0d1(
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA0,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA1,
            const double PHI0,
            const double PHI1
        );
    }
}
#endif