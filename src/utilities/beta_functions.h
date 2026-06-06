#ifndef AGREEMENTPHI_UTILITIES_BETA_FUNCTIONS_H
#define AGREEMENTPHI_UTILITIES_BETA_FUNCTIONS_H
#include <boost/math/special_functions/beta.hpp>
#include <cmath>
#include <limits>

namespace AgreementPhi{
    namespace betamath{

        // derivative log beta function wrt first shape parameter
        inline double dlogBda(const double A, const double B){
            double out = boost::math::digamma(A)-boost::math::digamma(A+B);
            return out;
        }

        // derivative log beta function wrt second shape parameter
        inline double dlogBdb(const double A, const double B){
            double out = boost::math::digamma(B)-boost::math::digamma(A+B);
            return out;
        }

        // second derivative log beta function wrt first shape parameter
        inline double d2logBda2(const double A, const double B){
            double out = boost::math::trigamma(A)-boost::math::trigamma(A+B);
            return out;
        }

        // second derivative log beta function wrt second shape parameter
        inline double d2logBdb2(const double A, const double B){
            double out = boost::math::trigamma(B)-boost::math::trigamma(A+B);
            return out;
        }

        // crossed derivative log beta function 
        inline double d2logBdadb(const double A, const double B){
            double out = -boost::math::trigamma(A+B);
            return out;
        }

        // Unnormalized incomplete beta: B(x;a,b) = ibeta(a,b,x) * beta(a,b)
        // = integral_0^x t^(a-1)(1-t)^(b-1) dt.
        inline double B_inc(const double A, const double B, const double X){
            return boost::math::ibeta(A, B, X) * boost::math::beta(A, B);
        }

        // derivative incomplete beta function wrt first shape parameter
        // = d/dA [B(X;A,B)] via centered finite differences. 
        inline double diBda(const double X, const double A, const double B){
            if (X < 1e-12) return 0.0;
            const double eps = std::numeric_limits<double>::epsilon();
            double h = std::max(std::cbrt(eps) * std::max(A, 0.01), 1e-6);
            h = std::min(h, A * 0.5);
            h = std::max(h, 1e-10);
            return (B_inc(A + h, B, X) - B_inc(A - h, B, X)) / (2.0 * h);
        }

        // derivative incomplete beta function wrt second shape parameter
        inline double diBdb(const double X, const double A, const double B){
            if (X < 1e-12) return 0.0;
            const double eps = std::numeric_limits<double>::epsilon();
            double h = std::max(std::cbrt(eps) * std::max(B, 0.01), 1e-6);
            h = std::min(h, B * 0.5);
            h = std::max(h, 1e-10);
            return (B_inc(A, B + h, X) - B_inc(A, B - h, X)) / (2.0 * h);
        }

        // second derivative incomplete beta function wrt first shape parameter
        inline double d2iBda2(const double X, const double A, const double B){
            if (X < 1e-12) return 0.0;
            const double eps = std::numeric_limits<double>::epsilon();
            double h = std::max(std::pow(eps, 0.25) * std::max(A, 0.01), 1e-4);
            h = std::min(h, A * 0.5);
            h = std::max(h, 1e-8);
            return (B_inc(A + h, B, X) - 2.0 * B_inc(A, B, X) + B_inc(A - h, B, X)) / (h * h);
        }

        // second derivative incomplete beta function wrt second shape parameter
        inline double d2iBdb2(const double X, const double A, const double B){
            if (X < 1e-12) return 0.0;
            const double eps = std::numeric_limits<double>::epsilon();
            double h = std::max(std::pow(eps, 0.25) * std::max(B, 0.01), 1e-4);
            h = std::min(h, B * 0.5);
            h = std::max(h, 1e-8);
            return (B_inc(A, B + h, X) - 2.0 * B_inc(A, B, X) + B_inc(A, B - h, X)) / (h * h);
        }

        // crossed derivative incomplete beta function
        inline double d2iBdadb(const double X, const double A, const double B){
            if (X < 1e-12) return 0.0;
            const double eps = std::numeric_limits<double>::epsilon();
            double ha = std::max(std::cbrt(eps) * std::max(A, 0.01), 1e-6);
            ha = std::min(ha, A * 0.5);
            ha = std::max(ha, 1e-10);
            double hb = std::max(std::cbrt(eps) * std::max(B, 0.01), 1e-6);
            hb = std::min(hb, B * 0.5);
            hb = std::max(hb, 1e-10);
            return (B_inc(A+ha, B+hb, X) - B_inc(A+ha, B-hb, X)
                  - B_inc(A-ha, B+hb, X) + B_inc(A-ha, B-hb, X)) / (4.0 * ha * hb);
        }

        // derivative of beta CDF wrt to shape parameter (first or second depends on DIB and DLOGB)
        inline double dF(const double DIB, const double DLOGB, const double BETAINV, const double F){
            double out = DIB*BETAINV-F*DLOGB;
            return out;
        }

        // second derivative of beta CDF wrt to shape parameter 
        // (first or second depends on DIB1, DIB2 and DLOGB1, DLOGB2)
        inline double d2F(
            const double DIB1,
            const double DIB2,  
            const double DLOGB1, 
            const double DLOGB2,   
            const double D2IB,  
            const double D2LOGB,  
            const double BETAINV, 
            const double F
        ){
            double out = D2IB*BETAINV-(DIB1*DLOGB2+DIB2*DLOGB1)*BETAINV+F*(DLOGB1*DLOGB2-D2LOGB);
            return out;
        }

        // Derivative of beta CDF with respect to mu
        inline double dFdmu(
            const double PHI, 
            const double DFDA, 
            const double DFDB
        ){
            double out = PHI*(DFDA-DFDB);
            return out;
        }

         // Second derivative of beta CDF with respect to mu
        inline double d2Fdmu2(
            const double PHI, 
            const double DFDA, 
            const double DFDB, 
            const double D2FDA2,
            const double D2FDB2,
            const double D2FDADB
        ){
            double out = pow(PHI,2)*(D2FDA2+D2FDB2-2*D2FDADB);
            return out;
        }

        // double dFdphi(
        //     const double MU, 
        //     const double DFDA, const double DFDB
        // ){
        //     double out = MU*DFDA+(1-MU)*DFDB;
        //     return out;
        // }

        // double d2Fdphi2(
        //     const double MU, 
        //     const double DFDA, 
        //     const double DFDB, 
        //     const double D2FDA2,
        //     const double D2FDB2,
        //     const double D2FDADB
        // ){
        //     double out = pow(MU,2)*D2FDA2+pow(1-MU, 2)*D2FDB2+2*MU*(1-MU)*D2FDADB;
        //     return out;
        // }

        // double d2Fdmudphi(
        //     const double MU, 
        //     const double PHI,
        //     const double DFDA, 
        //     const double DFDB, 
        //     const double D2FDA2,
        //     const double D2FDB2,
        //     const double D2FDADB
        // ){
        //     double out = DFDA-DFDB + PHI*(MU*D2FDA2-(1-MU)*D2FDB2+(1-2*MU)*D2FDADB);
        //     return out;
        // }
    }
}

#endif
