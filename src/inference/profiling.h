#ifndef AGREEMENTPHI_INFERENCE_PROFILING_H
#define AGREEMENTPHI_INFERENCE_PROFILING_H
#include <vector>
#include <Eigen/Dense>
#include "../models/twoway.h"

namespace AgreementPhi{
    namespace continuous{
        namespace twoway{
            namespace inference{
                Eigen::VectorXd profiling_lbfgs(
                    const std::vector<double>& Y,
                    const std::vector<int>& ITEM_INDS,
                    const std::vector<int>& WORKER_INDS,
                    const Eigen::VectorXd& LAMBDA_START,
                    const double PHI,
                    const int J,
                    const int W,
                    const int MAX_ITER
                );
            }
        }
    }
    namespace ordinal{
        namespace twoway{
            namespace inference{
                Eigen::VectorXd profiling_lbfgs(
                    const std::vector<double>& Y,
                    const std::vector<int>& ITEM_INDS,
                    const std::vector<int>& WORKER_INDS,
                    const Eigen::VectorXd& LAMBDA_START,
                    const double PHI,
                    const int J,
                    const int W,
                    const int K,
                    const int MAX_ITER
                );
            }
        }
    }
}
        
        
#endif
