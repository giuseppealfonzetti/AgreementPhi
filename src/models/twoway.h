#ifndef AGREEMENTPHI_MODELS_TWOWAY_H
#define AGREEMENTPHI_MODELS_TWOWAY_H
#include "../ratings/continuous.h"
#include "../ratings/ordinal.h"
#include <Eigen/Dense>
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define EIGEN_DONT_PARALLELIZE

namespace AgreementPhi{
    namespace continuous{
        namespace twoway{
            double joint_loglik(
                const std::vector<double>& Y, 
                const std::vector<int>& ITEM_INDS,   // expected to be coded 1 to J
                const std::vector<int>& WORKER_INDS, // expected to be coded 1 to W
                const Eigen::Ref<const Eigen::VectorXd> LAMBDA, //expected length J+W-1
                const double PHI,
                const int J, 
                const int W,
                Eigen::Ref<Eigen::VectorXd> DLAMBDA,
                Eigen::Ref<Eigen::VectorXd> JALPHAALPHA,
                Eigen::Ref<Eigen::VectorXd> JBETABETA,
                Eigen::Ref<Eigen::MatrixXd> JALPHABETA,
                const int GRADFLAG
            );
        }
    }

    namespace ordinal{
        namespace twoway{
            double joint_loglik(
                const std::vector<double>& Y, 
                const std::vector<int>& ITEM_INDS,   // expected to be coded 1 to J
                const std::vector<int>& WORKER_INDS, // expected to be coded 1 to W
                const Eigen::Ref<const Eigen::VectorXd> LAMBDA, //expected length J+W-1
                const double PHI,
                const int J, 
                const int W,
                const int K,
                Eigen::Ref<Eigen::VectorXd> DLAMBDA,
                Eigen::Ref<Eigen::VectorXd> JALPHAALPHA,
                Eigen::Ref<Eigen::VectorXd> JBETABETA,
                Eigen::Ref<Eigen::MatrixXd> JALPHABETA,
                const int GRADFLAG
            );

        }
    }
}

#endif