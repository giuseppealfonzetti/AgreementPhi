#include "TestFuns.h"

Rcpp::List cpp_beta_funs(const double A, const double B){

    double logbeta = log(boost::math::beta(A,B));
    double da = BetaFuns::dlogBda(A,B);
    double db = BetaFuns::dlogBdb(A,B);
    double da2 = BetaFuns::d2logBda2(A,B);
    double db2 = BetaFuns::d2logBdb2(A,B);
    double dadb = BetaFuns::d2logBdadb(A,B);

    Rcpp::List output = 
        Rcpp::List::create(
            Rcpp::Named("logbeta") = logbeta,
            Rcpp::Named("da") = da,
            Rcpp::Named("db") = db,
            Rcpp::Named("da2") = da2,
            Rcpp::Named("db2") = db2,
            Rcpp::Named("dadb") = dadb
        );
  return(output);
}

Rcpp::List cpp_ibeta_funs(const double X, const double A, const double B){
    
    double ibeta = boost::math::ibeta(A, B, X) * boost::math::beta(A, B);
    double da = iBetaFuns::diBda(X,A,B);
    double db = iBetaFuns::diBdb(X,A,B);
    double da2 = iBetaFuns::d2iBda2(X,A,B);
    double db2 = iBetaFuns::d2iBdb2(X,A,B);
    double dadb = iBetaFuns::d2iBdadb(X,A,B);

    Rcpp::List output = 
        Rcpp::List::create(
            Rcpp::Named("ibeta") = ibeta,
            Rcpp::Named("da") = da,
            Rcpp::Named("db") = db,
            Rcpp::Named("da2") = da2,
            Rcpp::Named("db2") = db2,
            Rcpp::Named("dadb") = dadb
        );
  return(output);
}

Rcpp::List cpp_cdfbeta_funs(const double X, const double A, const double B){
    double cdf = boost::math::ibeta(A, B, X);
    
    double beta = boost::math::beta(A,B);
    double betainv = 1/beta;
    double dlogBda = BetaFuns::dlogBda(A,B);
    double dlogBdb = BetaFuns::dlogBdb(A,B);
    double d2logBda2 = BetaFuns::d2logBda2(A,B);
    double d2logBdb2 = BetaFuns::d2logBdb2(A,B);
    double d2logBdadb = BetaFuns::d2logBdadb(A,B);
    
    double ibeta = boost::math::ibeta(A, B, X) * boost::math::beta(A, B);
    double diBda = iBetaFuns::diBda(X,A,B);
    double diBdb = iBetaFuns::diBdb(X,A,B);
    double d2iBda2 = iBetaFuns::d2iBda2(X,A,B);
    double d2iBdb2 = iBetaFuns::d2iBdb2(X,A,B);
    double d2iBdadb = iBetaFuns::d2iBdadb(X,A,B);

    double da = CDFFuns::dF(diBda, dlogBda, betainv, cdf);
    double db = CDFFuns::dF(diBdb, dlogBdb, betainv, cdf);
    double da2 = CDFFuns::d2F(diBda, diBda, dlogBda, dlogBda, d2iBda2, d2logBda2, betainv, cdf);
    double db2 = CDFFuns::d2F(diBdb, diBdb, dlogBdb, dlogBdb, d2iBdb2, d2logBdb2, betainv, cdf);
    double dadb = CDFFuns::d2F(diBda, diBdb, dlogBda, dlogBdb, d2iBdadb, d2logBdadb, betainv, cdf);

    Rcpp::List output = 
        Rcpp::List::create(
            Rcpp::Named("cdf") = cdf,
            Rcpp::Named("da") = da,
            Rcpp::Named("db") = db,
            Rcpp::Named("da2") = da2,
            Rcpp::Named("db2") = db2,
            Rcpp::Named("dadb") = dadb
        );
  return(output);
}

Rcpp::List cpp_cdfbeta_muphi_funs(const double X, const double MU, const double PHI){
    double a = MU*PHI;
    double b = (1-MU)*PHI;
    double cdf = boost::math::ibeta(a, b, X);
    
    double beta = boost::math::beta(a,b);
    double betainv = 1/beta;
    double dlogBda = BetaFuns::dlogBda(a,b);
    double dlogBdb = BetaFuns::dlogBdb(a,b);
    double d2logBda2 = BetaFuns::d2logBda2(a,b);
    double d2logBdb2 = BetaFuns::d2logBdb2(a,b);
    double d2logBdadb = BetaFuns::d2logBdadb(a,b);
    
    double ibeta = boost::math::ibeta(a,b, X) * boost::math::beta(a,b);
    double diBda = iBetaFuns::diBda(X,a,b);
    double diBdb = iBetaFuns::diBdb(X,a,b);
    double d2iBda2 = iBetaFuns::d2iBda2(X,a,b);
    double d2iBdb2 = iBetaFuns::d2iBdb2(X,a,b);
    double d2iBdadb = iBetaFuns::d2iBdadb(X,a,b);

    double da = CDFFuns::dF(diBda, dlogBda, betainv, cdf);
    double db = CDFFuns::dF(diBdb, dlogBdb, betainv, cdf);
    double da2 = CDFFuns::d2F(diBda, diBda, dlogBda, dlogBda, d2iBda2, d2logBda2, betainv, cdf);
    double db2 = CDFFuns::d2F(diBdb, diBdb, dlogBdb, dlogBdb, d2iBdb2, d2logBdb2, betainv, cdf);
    double dadb = CDFFuns::d2F(diBda, diBdb, dlogBda, dlogBdb, d2iBdadb, d2logBdadb, betainv, cdf);

    double dmu = CDFFuns::dFdmu(PHI, da, db);
    double dphi = CDFFuns::dFdphi(MU, da, db);
    double dmu2 = CDFFuns::d2Fdmu2(PHI, da, db, da2, db2, dadb);
    double dphi2 = CDFFuns::d2Fdphi2(MU, da, db, da2, db2, dadb);
    double dmudphi = CDFFuns::d2Fdmudphi(MU, PHI, da, db, da2, db2, dadb);

    Rcpp::List output = 
        Rcpp::List::create(
            Rcpp::Named("cdf") = cdf,
            Rcpp::Named("da") = da,
            Rcpp::Named("db") = db,
            Rcpp::Named("da2") = da2,
            Rcpp::Named("db2") = db2,
            Rcpp::Named("dadb") = dadb,
            Rcpp::Named("dmu") = dmu,
            Rcpp::Named("dphi") = dphi,
            Rcpp::Named("dmu2") = dmu2,
            Rcpp::Named("dphi2") = dphi2,
            Rcpp::Named("dmudphi") = dmudphi
        );
  return(output);
}

Rcpp::List cpp_items_dict(const int J, const std::vector<double> ITEM_INDS){  


    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);

    Rcpp::List output(J);
    for (int j = 0; j < J; ++j) {
        output[j] = Rcpp::wrap(dict.at(j));  // each std::vector<int> -> IntegerVector
    }
    return output;
}

/////////////
Rcpp::List cpp_ordinal_loglik(const double Y, const double MU, const double PHI, const int K){
    double dmu = 0;
    double dmu2 = 0;
    double ll = model::ordinal::loglik(Y, MU, PHI, K, dmu, dmu2, 2);

    // Eigen::VectorXd grad2=grad;
    Rcpp::List output = 
        Rcpp::List::create(
            Rcpp::Named("ll") = ll,
            Rcpp::Named("dmu") = dmu,
            Rcpp::Named("dmu2") = dmu2
        );
    return(output);
}


Rcpp::List cpp_ordinal_item_loglik(
    const std::vector<double> Y, 
    const std::vector<double> ITEM_INDS,
    const double ALPHA,
    const double PHI,
    const int K,
    const int J, 
    const int ITEM)
{

    double dalpha=0;
    double dalpha2=0;

    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);

    double ll=sample::ordinal::item_loglik(
        Y, dict, ITEM, ALPHA, PHI, K, dalpha, dalpha2, 2
    );

    Rcpp::List output = 
    Rcpp::List::create(
        Rcpp::Named("ll") = ll,
        Rcpp::Named("dalpha") = dalpha,
        Rcpp::Named("dalpha2") = dalpha2
    );
    return(output);

}

double cpp_log_det_obs_info(
    const std::vector<double> Y, 
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA,
    const double PHI,
    const int K,
    const int J)
{
    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);

    double out = sample::ordinal::log_det_obs_info(Y, dict, ALPHA, PHI, K);

    return out;

}


double cpp_log_det_E0d0d1(
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA0,
    const std::vector<double> ALPHA1,
    const double PHI0,
    const double PHI1,
    const int K,
    const int J)
{
    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);


    double out = sample::ordinal::log_det_E0d0d1(
        dict, ALPHA0, ALPHA1, PHI0, PHI1, K
    );

    return out;
}
