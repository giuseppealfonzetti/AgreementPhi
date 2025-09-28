#ifndef utils_H
#define utils_H
#include <unordered_map>    
#include <vector>           
#include <string>  
#include <RcppEigen.h>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>


namespace utils{

    std::vector<std::vector<int>> items_dicts(
        const int J,
        const Eigen::Ref<const Eigen::VectorXd> ITEM_INDS
    );

    double prec2agr(double PRECISION);
    double agr2prec(double AGREEMENT);
}

namespace BetaFuns {
    double dlogBda(const double A, const double B);
    double dlogBdb(const double A, const double B);
    double d2logBda2(const double A, const double B);
    double d2logBdb2(const double A, const double B);
    double d2logBdadb(const double A, const double B);
}

namespace iBetaFuns {
    double diBda(const double X, const double A, const double B);
    double diBdb(const double X, const double A, const double B);
    double d2iBda2(const double X, const double A, const double B);
    double d2iBdb2(const double X, const double A, const double B);
    double d2iBdadb(const double X, const double A, const double B);
}

namespace CDFFuns {
    double dF(const double DIB, const double DLOGB, const double BETAINV, const double F);
    double d2F(
            const double DIB1,
            const double DIB2,  
            const double DLOGB1, 
            const double DLOGB2,   
            const double D2IB,  
            const double D2LOGB,  
            const double BETAINV, 
            const double F);
    double dFdmu(double PHI, double DFDA, double DFDB);
    double dFdphi(double MU, double DFDA, double DFDB);
    double d2Fdmu2(
        const double PHI, 
        const double DFDA, 
        const double DFDB, 
        const double D2FDA2,
        const double D2FDB2,
        const double D2FDADB
    );
    double d2Fdphi2(
        const double MU, 
        const double DFDA, 
        const double DFDB, 
        const double D2FDA2,
        const double D2FDB2,
        const double D2FDADB
    );
    double d2Fdmudphi(
        const double MU,
        const double PHI,  
        const double DFDA, 
        const double DFDB, 
        const double D2FDA2,
        const double D2FDB2,
        const double D2FDADB
    );
}


#endif