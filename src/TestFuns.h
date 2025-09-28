#ifndef TestFuns_H
#define TestFuns_H



#include "utils.h"
#include "model.h"

#include <Rcpp.h>
#include <unordered_map>    
#include <vector>           
#include <string>           

// [[Rcpp::export]]
Rcpp::List cpp_beta_funs(const double A, const double B);

// [[Rcpp::export]]
Rcpp::List cpp_ibeta_funs(const double X, const double A, const double B);

// [[Rcpp::export]]
Rcpp::List cpp_cdfbeta_funs(const double X, const double A, const double B);

// [[Rcpp::export]]
Rcpp::List cpp_cdfbeta_muphi_funs(const double X, const double MU, const double PHI);



//////////////////////
// [[Rcpp::export]]
Rcpp::List cpp_ordinal_loglik(const double Y, const double MU, const double PHI, const int K);

// [[Rcpp::export]]
Rcpp::List cpp_ordinal_item_loglik(
    Eigen::Map<Eigen::VectorXd> Y, 
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    const double ALPHA,
    const double PHI,
    const int K,
    const int J, 
    const int ITEM);

// [[Rcpp::export]]
double cpp_log_det_obs_info(
    Eigen::Map<Eigen::VectorXd> Y, 
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    Eigen::Map<Eigen::VectorXd> ALPHA,
    const double PHI,
    const int K,
    const int J);

// [[Rcpp::export]]
double cpp_log_det_E0d0d1(
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    Eigen::Map<Eigen::VectorXd> ALPHA0,
    Eigen::Map<Eigen::VectorXd> ALPHA1,
    const double PHI0,
    const double PHI1,
    const int K,
    const int J);

#endif