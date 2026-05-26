#ifndef AGREEMENTPHI_RATINGS_INFLATED_H
#define AGREEMENTPHI_RATINGS_INFLATED_H

#include <cfloat>
#include <cmath>
#include "../utilities/link_functions.h"
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/minima.hpp>
#include <vector>
#include <limits>
#include <cmath>

namespace AgreementPhi {
    namespace inflated {

        double obs_loglik(
            double Y,
            double ALPHA,
            double PHI,
            double K0,
            double K1,
            double &DALPHA,
            double &NEG_D2ALPHA,
            int GRADFLAG
        );

        double E0_dalpha0dalpha1(
            double ALPHA0, double PHI0, double K0_0, double K1_0,
            double ALPHA1, double PHI1, double K0_1, double K1_1
        );

        double brent_alpha(
            const std::vector<double>& Y,
            const std::vector<int>& ITEM_OBS,
            double ALPHA_START,
            double PHI,
            double K0,
            double K1,
            double LOWER,
            double UPPER,
            int MAX_ITER
        );

    }
}

#endif
