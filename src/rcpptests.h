#ifndef AGREEMENTPHI_RCPPTESTS_H
#define AGREEMENTPHI_RCPPTESTS_H
#include <RcppEigen.h>
#include "utilities/beta_functions.h"
#include "utilities/link_functions.h"
#include "utilities/utils_functions.h"
#include "ratings/continuous.h"
#include "ratings/ordinal.h"
#include "inference/nuisance.h"
#include "inference/loglik.h"
#include "inference/precision.h"

// [[Rcpp::export]]
Rcpp::List cpp_beta_funs(const double A, const double B){

    double logbeta = log(boost::math::beta(A,B));
    double da = AgreementPhi::betamath::dlogBda(A,B);
    double db = AgreementPhi::betamath::dlogBdb(A,B);
    double da2 = AgreementPhi::betamath::d2logBda2(A,B);
    double db2 = AgreementPhi::betamath::d2logBdb2(A,B);
    double dadb = AgreementPhi::betamath::d2logBdadb(A,B);

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

// [[Rcpp::export]]
Rcpp::List cpp_ibeta_funs(const double X, const double A, const double B){

    double ibeta = boost::math::ibeta(A, B, X) * boost::math::beta(A, B);
    double da = AgreementPhi::betamath::diBda(X,A,B);
    double db = AgreementPhi::betamath::diBdb(X,A,B);
    double da2 = AgreementPhi::betamath::d2iBda2(X,A,B);
    double db2 = AgreementPhi::betamath::d2iBdb2(X,A,B);
    double dadb = AgreementPhi::betamath::d2iBdadb(X,A,B);

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

// [[Rcpp::export]]
Rcpp::List cpp_cdfbeta_funs(const double X, const double A, const double B){
    double cdf = boost::math::ibeta(A, B, X);

    double beta = boost::math::beta(A,B);
    double betainv = 1/beta;
    double dlogBda = AgreementPhi::betamath::dlogBda(A,B);
    double dlogBdb = AgreementPhi::betamath::dlogBdb(A,B);
    double d2logBda2 = AgreementPhi::betamath::d2logBda2(A,B);
    double d2logBdb2 = AgreementPhi::betamath::d2logBdb2(A,B);
    double d2logBdadb = AgreementPhi::betamath::d2logBdadb(A,B);

    double ibeta = boost::math::ibeta(A, B, X) * boost::math::beta(A, B);
    double diBda = AgreementPhi::betamath::diBda(X,A,B);
    double diBdb = AgreementPhi::betamath::diBdb(X,A,B);
    double d2iBda2 = AgreementPhi::betamath::d2iBda2(X,A,B);
    double d2iBdb2 = AgreementPhi::betamath::d2iBdb2(X,A,B);
    double d2iBdadb = AgreementPhi::betamath::d2iBdadb(X,A,B);

    double da = AgreementPhi::betamath::dF(diBda, dlogBda, betainv, cdf);
    double db = AgreementPhi::betamath::dF(diBdb, dlogBdb, betainv, cdf);
    double da2 = AgreementPhi::betamath::d2F(diBda, diBda, dlogBda, dlogBda, d2iBda2, d2logBda2, betainv, cdf);
    double db2 = AgreementPhi::betamath::d2F(diBdb, diBdb, dlogBdb, dlogBdb, d2iBdb2, d2logBdb2, betainv, cdf);
    double dadb = AgreementPhi::betamath::d2F(diBda, diBdb, dlogBda, dlogBdb, d2iBdadb, d2logBdadb, betainv, cdf);

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

// [[Rcpp::export]]
Rcpp::List cpp_cdfbeta_muphi_funs(const double X, const double MU, const double PHI){
    double a = MU*PHI;
    double b = (1-MU)*PHI;
    double cdf = boost::math::ibeta(a, b, X);

    double beta = boost::math::beta(a,b);
    double betainv = 1/beta;
    double dlogBda = AgreementPhi::betamath::dlogBda(a,b);
    double dlogBdb = AgreementPhi::betamath::dlogBdb(a,b);
    double d2logBda2 = AgreementPhi::betamath::d2logBda2(a,b);
    double d2logBdb2 = AgreementPhi::betamath::d2logBdb2(a,b);
    double d2logBdadb = AgreementPhi::betamath::d2logBdadb(a,b);

    double ibeta = boost::math::ibeta(a,b, X) * boost::math::beta(a,b);
    double diBda = AgreementPhi::betamath::diBda(X,a,b);
    double diBdb = AgreementPhi::betamath::diBdb(X,a,b);
    double d2iBda2 = AgreementPhi::betamath::d2iBda2(X,a,b);
    double d2iBdb2 = AgreementPhi::betamath::d2iBdb2(X,a,b);
    double d2iBdadb = AgreementPhi::betamath::d2iBdadb(X,a,b);

    double da = AgreementPhi::betamath::dF(diBda, dlogBda, betainv, cdf);
    double db = AgreementPhi::betamath::dF(diBdb, dlogBdb, betainv, cdf);
    double da2 = AgreementPhi::betamath::d2F(diBda, diBda, dlogBda, dlogBda, d2iBda2, d2logBda2, betainv, cdf);
    double db2 = AgreementPhi::betamath::d2F(diBdb, diBdb, dlogBdb, dlogBdb, d2iBdb2, d2logBdb2, betainv, cdf);
    double dadb = AgreementPhi::betamath::d2F(diBda, diBdb, dlogBda, dlogBdb, d2iBdadb, d2logBdadb, betainv, cdf);

    double dmu = AgreementPhi::betamath::dFdmu(PHI, da, db);
    double dmu2 = AgreementPhi::betamath::d2Fdmu2(PHI, da, db, da2, db2, dadb);

    Rcpp::List output =
        Rcpp::List::create(
            Rcpp::Named("cdf") = cdf,
            Rcpp::Named("da") = da,
            Rcpp::Named("db") = db,
            Rcpp::Named("da2") = da2,
            Rcpp::Named("db2") = db2,
            Rcpp::Named("dadb") = dadb,
            Rcpp::Named("dmu") = dmu,
            Rcpp::Named("dmu2") = dmu2
        );
  return(output);
}

// [[Rcpp::export]]
Rcpp::List cpp_ordinal_loglik(const double Y, const double MU, const double PHI, const int K){
    double dmu = 0;
    double dmu2 = 0;
    std::vector<double> tau(K + 1);
    for(int i = 0; i <= K; ++i){
        tau.at(i) = static_cast<double>(i) / static_cast<double>(K);
    }
    double ll = AgreementPhi::ordinal::loglik(Y, MU, PHI, tau, dmu, dmu2, 2);

    Rcpp::List output =
        Rcpp::List::create(
            Rcpp::Named("ll") = ll,
            Rcpp::Named("dmu") = dmu,
            Rcpp::Named("dmu2") = dmu2
        );
    return(output);
}

// [[Rcpp::export]]
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

    std::vector<std::vector<int>> dict = AgreementPhi::utils::oneway_items_dict(J, ITEM_INDS);
    std::vector<double> tau(K + 1);
    for(int i = 0; i <= K; ++i){
        tau.at(i) = static_cast<double>(i) / static_cast<double>(K);
    }
    const int item_idx = std::max(0, std::min(J - 1, ITEM));
    const std::vector<int>& obs_vec = dict.at(item_idx);
    double ll = 0.0;
    double mu = AgreementPhi::link::mu(ALPHA);
    double dmu_det = AgreementPhi::link::dmu(mu);
    double d2mu_det = AgreementPhi::link::d2mu(mu);

    for(const int obs_id : obs_vec){
        double dmu = 0;
        double dmu2 = 0;
        ll += AgreementPhi::ordinal::loglik(
            Y.at(obs_id), mu, PHI, tau, dmu, dmu2, 2
        );
        dalpha += dmu * dmu_det;
        dalpha2 += dmu2 * dmu_det * dmu_det + dmu * d2mu_det;
    }

    Rcpp::List output =
    Rcpp::List::create(
        Rcpp::Named("ll") = ll,
        Rcpp::Named("dalpha") = dalpha,
        Rcpp::Named("dalpha2") = dalpha2
    );
    return(output);

}

// [[Rcpp::export]]
double cpp_log_det_obs_info(
    const std::vector<double> Y,
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA,
    const double PHI,
    const int K,
    const int J,
    const bool ITEMS_NUISANCE = false,
    const bool WORKERS_NUISANCE = false)
{
    const int n = Y.size();
    std::vector<int> worker_inds(n, 1);
    std::vector<int> item_inds_int(n);
    for(int i = 0; i < n; ++i){
        item_inds_int.at(i) = static_cast<int>(ITEM_INDS.at(i));
    }
    std::vector<double> tau(K + 1);
    for(int i = 0; i <= K; ++i){
        tau.at(i) = static_cast<double>(i) / static_cast<double>(K);
    }
    return AgreementPhi::ordinal::log_det_obs_info(
        Y, item_inds_int, worker_inds, ALPHA, tau, PHI, J, 1, K, ITEMS_NUISANCE, WORKERS_NUISANCE
    );

}

// [[Rcpp::export]]
double cpp_log_det_E0d0d1(
    const std::vector<double> ITEM_INDS,
    const std::vector<double> ALPHA0,
    const std::vector<double> ALPHA1,
    const double PHI0,
    const double PHI1,
    const int K,
    const int J,
    const bool ITEMS_NUISANCE = false,
    const bool WORKERS_NUISANCE = false)
{
    const int n = ITEM_INDS.size();
    std::vector<int> worker_inds(n, 1);
    std::vector<int> item_inds_int(n);
    for(int i = 0; i < n; ++i){
        item_inds_int.at(i) = static_cast<int>(ITEM_INDS.at(i));
    }
    std::vector<double> tau(K + 1);
    for(int i = 0; i <= K; ++i){
        tau.at(i) = static_cast<double>(i) / static_cast<double>(K);
    }

    return AgreementPhi::ordinal::log_det_E0d0d1(
        item_inds_int, worker_inds, ALPHA0, ALPHA1, PHI0, PHI1, tau, J, 1, K, ITEMS_NUISANCE, WORKERS_NUISANCE
    );
}

#endif
