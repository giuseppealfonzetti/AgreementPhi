#ifndef AGREEMENTPHI_RATINGS_CONTINUOUS_H
#define AGREEMENTPHI_RATINGS_CONTINUOUS_H

#include "../utilities/beta_functions.h"
#include "../utilities/link_functions.h"
#include "../utilities/utils_functions.h"

namespace AgreementPhi{
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

#endif