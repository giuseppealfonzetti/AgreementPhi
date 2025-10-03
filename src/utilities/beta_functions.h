#ifndef AGREEMENTPHI_UTILITIES_BETA_FUNCTIONS_H
#define AGREEMENTPHI_UTILITIES_BETA_FUNCTIONS_H
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>

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

        // derivative incomplete beta function wrt first shape parameter
        inline double diBda(const double X, const double A, const double B){
            auto f = [A, B](double t) {
                return std::log(t) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
            };

            boost::math::quadrature::tanh_sinh<double> integrator(15);
            double out = integrator.integrate(f, 0.0, X, 1e-15);

            return out;
        }

        // derivative incomplete beta function wrt second shape parameter
        inline double diBdb(const double X, const double A, const double B){
            auto f = [A, B](double t) {
                return std::log(1-t) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
            };
            boost::math::quadrature::tanh_sinh<double> integrator(15);
            double out = integrator.integrate(f, 0.0, X, 1e-15);

            return out;
        }

        // second derivative incomplete beta function wrt first shape parameter
        inline double d2iBda2(const double X, const double A, const double B){
            auto f = [A, B](double t) {
                return pow(std::log(t), 2) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
            };
            boost::math::quadrature::tanh_sinh<double> integrator(15);
            double out = integrator.integrate(f, 0.0, X, 1e-15);

            return out;
        }

        // second derivative incomplete beta function wrt second shape parameter
        inline double d2iBdb2(const double X, const double A, const double B){
            auto f = [A, B](double t) {
                return pow(std::log(1-t), 2) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
            };
            boost::math::quadrature::tanh_sinh<double> integrator(15);
            double out = integrator.integrate(f, 0.0, X, 1e-15);

            return out;
        }

        // crossed derivative incomplete beta function
        inline double d2iBdadb(const double X, const double A, const double B){
            auto f = [A, B](double t) {
                return std::log(1-t)*std::log(t) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
            };
            boost::math::quadrature::tanh_sinh<double> integrator(15);
            double out = integrator.integrate(f, 0.0, X, 1e-15);

            return out;
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
