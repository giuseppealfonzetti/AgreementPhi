#ifndef AGREEMENTPHI_UTILITIES_LINK_FUNCTIONS_H
#define AGREEMENTPHI_UTILITIES_LINK_FUNCTIONS_H
#define DOUBLE_EPS DBL_EPSILON
static const double THRESH = 30.0;
static const double MTHRESH = -30.0;
static const double INVEPS = 1/DOUBLE_EPS;

namespace AgreementPhi{
    namespace link{

        // Evaluate mu given linear predictor
        inline double mu(const double ETA){
                double expmeta;
                if(-ETA < MTHRESH){
                    expmeta = DOUBLE_EPS;
                }else if(-ETA > THRESH){
                    expmeta = INVEPS;
                }else{
                    expmeta = exp(-ETA);
                }

                return 1/(1+expmeta);
        }

        // derivative of mu wrt linear predictor
        inline double dmu(const double MU){
            return MU*(1-MU);
        }

        // second derivative of mu wrt linear predictor
        inline double d2mu(const double MU){
            return MU*(1-MU)*(1-2*MU);
        }
    }
}
#endif