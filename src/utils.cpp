#include "utils.h"

///////////////////
// General utils //
///////////////////
std::vector<std::vector<int>> utils::items_dicts(
        const int J,
        const std::vector<double>& ITEM_INDS
    ){

    const int n = ITEM_INDS.size();
    std::vector<std::vector<int>> dict(J);

    for(int i = 0; i < n; ++i){
        int item_id = static_cast<int>(ITEM_INDS.at(i)-1);
        dict.at(item_id).push_back(i);
    }

    return dict;
}

double utils::agr2prec(double AGREEMENT){
    const double log2_sq = pow(log(2.0), 2);
    return -2.0 * log(1.0 - AGREEMENT) / log2_sq;
}

double utils::prec2agr(double PRECISION){
    const double log2_half = log(2.0) / 2.0;
    return 1.0 - pow(2.0, -PRECISION * log2_half);
}

///////////////////////////////////
// Beta function and derivatives //
///////////////////////////////////
double BetaFuns::dlogBda(const double A, const double B){
    double out = boost::math::digamma(A)-boost::math::digamma(A+B);
    return out;
}

double BetaFuns::dlogBdb(const double A, const double B){
    double out = boost::math::digamma(B)-boost::math::digamma(A+B);
    return out;
}

double BetaFuns::d2logBda2(const double A, const double B){
    double out = boost::math::trigamma(A)-boost::math::trigamma(A+B);
    return out;
}

double BetaFuns::d2logBdb2(const double A, const double B){
    double out = boost::math::trigamma(B)-boost::math::trigamma(A+B);
    return out;
}

double BetaFuns::d2logBdadb(const double A, const double B){
    double out = -boost::math::trigamma(A+B);
    return out;
}

///////////////////////////////////////////////
// Incomplete Beta function and derivatives //
///////////////////////////////////////////////


double iBetaFuns::diBda(const double X, const double A, const double B){
    auto f = [A, B](double t) {
        return std::log(t) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
    };

    boost::math::quadrature::tanh_sinh<double> integrator(15);
    double out = integrator.integrate(f, 0.0, X, 1e-15);

    // boost::math::quadrature::gauss_kronrod<double, 31> integrator; 
    // double out = integrator.integrate(f, 0.0, X, 8, 1e-12); 

    return out;
}

double iBetaFuns::diBdb(const double X, const double A, const double B){
    auto f = [A, B](double t) {
        return std::log(1-t) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
    };
    boost::math::quadrature::tanh_sinh<double> integrator(15);
    double out = integrator.integrate(f, 0.0, X, 1e-15);

    // boost::math::quadrature::gauss_kronrod<double, 31> integrator; 
    // double out = integrator.integrate(f, 0.0, X, 8, 1e-12); 

    return out;
}

double iBetaFuns::d2iBda2(const double X, const double A, const double B){
    auto f = [A, B](double t) {
        return pow(std::log(t), 2) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
    };
    boost::math::quadrature::tanh_sinh<double> integrator(15);
    double out = integrator.integrate(f, 0.0, X, 1e-15);

    // boost::math::quadrature::gauss_kronrod<double, 31> integrator; 
    // double out = integrator.integrate(f, 0.0, X, 8, 1e-12); 

    return out;
}

double iBetaFuns::d2iBdb2(const double X, const double A, const double B){
    auto f = [A, B](double t) {
        return pow(std::log(1-t), 2) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
    };
    boost::math::quadrature::tanh_sinh<double> integrator(15);
    double out = integrator.integrate(f, 0.0, X, 1e-15);

    // boost::math::quadrature::gauss_kronrod<double, 31> integrator; 
    // double out = integrator.integrate(f, 0.0, X, 8, 1e-12);

    return out;
}

double iBetaFuns::d2iBdadb(const double X, const double A, const double B){
    auto f = [A, B](double t) {
        return std::log(1-t)*std::log(t) * std::pow(t, A - 1) * std::pow(1 - t, B - 1);
    };
    boost::math::quadrature::tanh_sinh<double> integrator(15);
    double out = integrator.integrate(f, 0.0, X, 1e-15);

    // boost::math::quadrature::gauss_kronrod<double, 31> integrator; 
    // double out = integrator.integrate(f, 0.0, X, 8, 1e-12);

    return out;
}


//////////////////////////////
// Beta CDF and derivatives //
//////////////////////////////

double CDFFuns::dF(const double DIB, const double DLOGB, const double BETAINV, const double F){
    double out = DIB*BETAINV-F*DLOGB;
    return out;
}

double CDFFuns::d2F(
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

double CDFFuns::dFdmu(
    const double PHI, 
    const double DFDA, 
    const double DFDB
){
    double out = PHI*(DFDA-DFDB);
    return out;
}

double CDFFuns::dFdphi(
    const double MU, 
    const double DFDA, const double DFDB
){
    double out = MU*DFDA+(1-MU)*DFDB;
    return out;
}

double CDFFuns::d2Fdmu2(
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

double CDFFuns::d2Fdphi2(
    const double MU, 
    const double DFDA, 
    const double DFDB, 
    const double D2FDA2,
    const double D2FDB2,
    const double D2FDADB
){
    double out = pow(MU,2)*D2FDA2+pow(1-MU, 2)*D2FDB2+2*MU*(1-MU)*D2FDADB;
    return out;
}

double CDFFuns::d2Fdmudphi(
    const double MU, 
    const double PHI,
    const double DFDA, 
    const double DFDB, 
    const double D2FDA2,
    const double D2FDB2,
    const double D2FDADB
){
    double out = DFDA-DFDB + PHI*(MU*D2FDA2-(1-MU)*D2FDB2+(1-2*MU)*D2FDADB);
    return out;
}



