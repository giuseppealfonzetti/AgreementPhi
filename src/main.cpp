#include <Rcpp.h>
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define EIGEN_DONT_PARALLELIZE
#include <RcppEigen.h>
#include "TestFuns.h"
#include <boost/math/tools/minima.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include <functional>

namespace profile{
    namespace ordinal{
        // Brent search to profile item related nuisance parameter
        double Brent(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const int ITEM,
            const double ALPHA_START,
            const double PHI,
            const int K,
            const int RANGE,
            const int MAX_ITER
        ){

            double grad, grad2;
            auto neg_ll = [&](double alpha){
                double ll = sample::ordinal::item_loglik(Y, DICT, ITEM, alpha, PHI, K, grad, grad2, 0);
                return -ll; 
            };

            double lower = ALPHA_START - RANGE; 
            double upper = ALPHA_START + RANGE;


            const int digits = std::numeric_limits<double>::digits;
            boost::uintmax_t max_iter = MAX_ITER;
            auto result = boost::math::tools::brent_find_minima(
                neg_ll, lower, upper, digits, max_iter
            );

            double opt = result.first; 
            return opt;
        }
        // Newton-Raphson to profile item related nuisance parameter
        double Newton_Raphson(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const int ITEM,
            const double ALPHA_START,
            const double PHI,
            const int K,
            const int RANGE,
            const int MAX_ITER
        ){
            auto neg_ll = [&](double alpha){
                double grad = 0;
                double grad2 = 0;
                double ll = sample::ordinal::item_loglik(Y, DICT, ITEM, alpha, PHI, K, grad, grad2, 2);
                return std::make_pair(grad, grad2); 
            };

            double lower = ALPHA_START - RANGE; 
            double upper = ALPHA_START + RANGE;

            const int digits = std::numeric_limits<double>::digits;
            int get_digits = static_cast<int>(digits * 0.4);
            boost::uintmax_t max_iter = MAX_ITER;
            double out = boost::math::tools::newton_raphson_iterate(
                neg_ll,           
                ALPHA_START,     
                lower,           
                upper,           
                get_digits,      
                max_iter);

            return out;
        }

        // Profile likelihood for phi
        double loglik(
            double PHI,
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const int K,
            const int J,
            const int RANGE,
            const int MAX_ITER,
            const int METHOD
        ){

            double grad, grad2;
            double ll = 0;
            Eigen::VectorXd alpha=Eigen::VectorXd::Zero(J);
            for(int j=0; j<J; j++){

                // profiling
                if(METHOD==0){
                    double hatalpha = profile::ordinal::Brent(
                        Y, DICT, j, ALPHA_START(j), PHI, K, RANGE, MAX_ITER
                    );
                    alpha(j)=hatalpha;
                }else{
                    double hatalpha = profile::ordinal::Newton_Raphson(
                        Y, DICT, j, ALPHA_START(j), PHI, K, RANGE, MAX_ITER
                    );
                    alpha(j)=hatalpha;
                }

                // Evaluate ll contribution
                double llj = sample::ordinal::item_loglik(Y, DICT, j, alpha(j), PHI, K, grad, grad2, 0);

                ll+=llj;
            }


            return ll;
        }

        // Compute the maximum profile likelihood estimator of phi
        std::pair<double, double> get_phi_mle(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const double PHI_START,
            const int K,
            const int J,
            const int SEARCH_RANGE,
            const int MAX_ITER,
            const int PROF_SEARCH_RANGE,
            const int PROF_MAX_ITER,
            const int PROF_METHOD)
        {
            auto neg_profile_likelihood = [&](double phi){
                double ll = profile::ordinal::loglik(
                    phi, Y, DICT, ALPHA_START, K, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
                return -ll; 
            };

            double lower = 1e-8; 
            double upper = PHI_START + SEARCH_RANGE;


            const int digits = std::numeric_limits<double>::digits;
            boost::uintmax_t max_iter = MAX_ITER;
            auto result = boost::math::tools::brent_find_minima(
                neg_profile_likelihood, lower, upper, digits, max_iter
            );

            // double opt = result.first; 
            return result;
        }

        // Modified Profile likelihood for phi
        double mp_loglik(
            double PHI,
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_MLE,
            const double PHI_MLE,
            const int K,
            const int J,
            const int PROF_SEARCH_RANGE,
            const int PROF_MAX_ITER,
            const int PROF_METHOD
        ){

            double grad, grad2;
            double ll = 0;
            Eigen::VectorXd prof_alpha=Eigen::VectorXd::Zero(J);
            for(int j=0; j<J; j++){

                // profiling
                if(PROF_METHOD==0){
                    double hatalpha = profile::ordinal::Brent(
                        Y, DICT, j, ALPHA_START(j), PHI, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    prof_alpha(j)=hatalpha;
                }else{
                    double hatalpha = profile::ordinal::Newton_Raphson(
                        Y, DICT, j, ALPHA_START(j), PHI, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    prof_alpha(j)=hatalpha;
                }

                // Evaluate ll contribution
                double llj = sample::ordinal::item_loglik(Y, DICT, j, prof_alpha(j), PHI, K, grad, grad2, 0);

                ll+=llj;
            }

            // add modifier
            ll +=.5*sample::ordinal::log_det_obs_info(Y, DICT, prof_alpha, PHI, K);

            ll -= sample::ordinal::log_det_E0d0d1(DICT, ALPHA_MLE, prof_alpha, PHI_MLE, PHI, K);

            return ll;
        }

        // Compute the maximum modified profile likelihood estimator of phi
        std::vector<double> get_phi_mp(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const double PHI_START,
            const int K,
            const int J,
            const int SEARCH_RANGE,
            const int MAX_ITER,
            const int PROF_SEARCH_RANGE,
            const int PROF_MAX_ITER,
            const int PROF_METHOD,
            const bool VERBOSE = false)
        {

            // get mle for phi
            std::pair<double, double> phi_mle = profile::ordinal::get_phi_mle(
                Y, DICT, ALPHA_START, PHI_START, K, J,
                SEARCH_RANGE, MAX_ITER,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
            );

            if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.first) << "\n";

            // get mle for alpha
            Eigen::VectorXd alpha_mle=Eigen::VectorXd::Zero(J);
            for(int j=0; j<J; j++){

                if(PROF_METHOD==0){
                    double hatalpha = profile::ordinal::Brent(
                        Y, DICT, j, ALPHA_START(j), phi_mle.first, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    alpha_mle(j)=hatalpha;
                }else{
                    double hatalpha = profile::ordinal::Newton_Raphson(
                        Y, DICT, j, ALPHA_START(j), phi_mle.first, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    alpha_mle(j)=hatalpha;
                }
            }


            // negative log-likelihood to minimize
            auto neg_modified_profile_likelihood = [&](double phi){
                double ll = profile::ordinal::mp_loglik(
                    phi, Y, DICT, alpha_mle, alpha_mle, phi_mle.first, K, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
                return -ll; 
            };

            double lower = 1e-8; 
            double upper = phi_mle.first + SEARCH_RANGE;


            const int digits = std::numeric_limits<double>::digits;
            boost::uintmax_t max_iter = MAX_ITER;
            auto result = boost::math::tools::brent_find_minima(
                neg_modified_profile_likelihood, lower, upper, digits, max_iter
            );

            // double opt = result.first; 

            if(VERBOSE) Rcpp::Rcout<< "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

            std::vector<double> out(3); 
            out[0] = result.first;
            out[1] = -result.second;
            out[2] = phi_mle.first;

            return out;


        }


    }

    namespace continuous{
        // Brent search to profile item related nuisance parameter
        double Brent(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const int ITEM,
            const double ALPHA_START,
            const double PHI,
            const int RANGE,
            const int MAX_ITER
        ){
            double grad, grad2;
            auto neg_ll = [&](double alpha){
                double ll = sample::continuous::item_loglik(Y, DICT, ITEM, alpha, PHI, grad, grad2, 0);
                return -ll; 
            };

            double lower = ALPHA_START - RANGE; 
            double upper = ALPHA_START + RANGE;

            const int digits = std::numeric_limits<double>::digits;
            boost::uintmax_t max_iter = MAX_ITER;
            auto result = boost::math::tools::brent_find_minima(
                neg_ll, lower, upper, digits, max_iter
            );

            double opt = result.first; 
            return opt;
        }

        // Newton-Raphson to profile item related nuisance parameter
        double Newton_Raphson(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const int ITEM,
            const double ALPHA_START,
            const double PHI,
            const int RANGE,
            const int MAX_ITER
        ){
            auto neg_ll = [&](double alpha){
                double grad = 0;
                double grad2 = 0;
                double ll = sample::continuous::item_loglik(Y, DICT, ITEM, alpha, PHI, grad, grad2, 2);
                return std::make_pair(grad, grad2); 
            };

            double lower = ALPHA_START - RANGE; 
            double upper = ALPHA_START + RANGE;

            const int digits = std::numeric_limits<double>::digits;
            int get_digits = static_cast<int>(digits * 0.4);
            boost::uintmax_t max_iter = MAX_ITER;
            double out = boost::math::tools::newton_raphson_iterate(
                neg_ll,           
                ALPHA_START,     
                lower,           
                upper,           
                get_digits,      
                max_iter);

            return out;
        }

        // Profile likelihood for phi
        double loglik(
            double PHI,
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const int J,
            const int RANGE,
            const int MAX_ITER,
            const int METHOD
        ){
            double grad, grad2;
            double ll = 0;
            Eigen::VectorXd alpha = Eigen::VectorXd::Zero(J);
            for(int j=0; j<J; j++){

                // profiling
                if(METHOD==0){
                    double hatalpha = profile::continuous::Brent(
                        Y, DICT, j, ALPHA_START(j), PHI, RANGE, MAX_ITER
                    );
                    alpha(j) = hatalpha;
                }else{
                    double hatalpha = profile::continuous::Newton_Raphson(
                        Y, DICT, j, ALPHA_START(j), PHI, RANGE, MAX_ITER
                    );
                    alpha(j) = hatalpha;
                }

                // Evaluate ll contribution
                double llj = sample::continuous::item_loglik(Y, DICT, j, alpha(j), PHI, grad, grad2, 0);

                ll += llj;
            }

            return ll;
        }

        // Compute the maximum profile likelihood estimator of phi
        std::pair<double, double> get_phi_mle(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const double PHI_START,
            const int J,
            const int SEARCH_RANGE,
            const int MAX_ITER,
            const int PROF_SEARCH_RANGE,
            const int PROF_MAX_ITER,
            const int PROF_METHOD)
        {
            auto neg_profile_likelihood = [&](double phi){
                double ll = profile::continuous::loglik(
                    phi, Y, DICT, ALPHA_START, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
                return -ll; 
            };

            double lower = 1e-8; 
            double upper = PHI_START + SEARCH_RANGE;

            const int digits = std::numeric_limits<double>::digits;
            boost::uintmax_t max_iter = MAX_ITER;
            auto result = boost::math::tools::brent_find_minima(
                neg_profile_likelihood, lower, upper, digits, max_iter
            );

            // double opt = result.first; 
            // return opt;

            return result;
        }

        // Modified Profile likelihood for phi
        double mp_loglik(
            double PHI,
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_MLE,
            const double PHI_MLE,
            const int J,
            const int PROF_SEARCH_RANGE,
            const int PROF_MAX_ITER,
            const int PROF_METHOD
        ){
            double grad, grad2;
            double ll = 0;
            Eigen::VectorXd prof_alpha = Eigen::VectorXd::Zero(J);
            for(int j=0; j<J; j++){

                // profiling
                if(PROF_METHOD==0){
                    double hatalpha = profile::continuous::Brent(
                        Y, DICT, j, ALPHA_START(j), PHI, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    prof_alpha(j) = hatalpha;
                }else{
                    double hatalpha = profile::continuous::Newton_Raphson(
                        Y, DICT, j, ALPHA_START(j), PHI, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    prof_alpha(j) = hatalpha;
                }

                // Evaluate ll contribution
                double llj = sample::continuous::item_loglik(Y, DICT, j, prof_alpha(j), PHI, grad, grad2, 0);

                ll += llj;
            }

            // add modifier
            ll += .5*sample::continuous::log_det_obs_info(Y, DICT, prof_alpha, PHI);

            ll -= sample::continuous::log_det_E0d0d1(DICT, ALPHA_MLE, prof_alpha, PHI_MLE, PHI);

            return ll;
        }

        // Compute the maximum modified profile likelihood estimator of phi
        std::vector<double> get_phi_mp(
            const Eigen::Ref<const Eigen::VectorXd> Y, 
            const std::vector<std::vector<int>> DICT,
            const Eigen::Ref<const Eigen::VectorXd> ALPHA_START,
            const double PHI_START,
            const int J,
            const int SEARCH_RANGE,
            const int MAX_ITER,
            const int PROF_SEARCH_RANGE,
            const int PROF_MAX_ITER,
            const int PROF_METHOD,
            const bool VERBOSE = false)
        {
            // get mle for phi
            std::pair<double, double> phi_mle = profile::continuous::get_phi_mle(
                Y, DICT, ALPHA_START, PHI_START, J,
                SEARCH_RANGE, MAX_ITER,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
            );

            if(VERBOSE) Rcpp::Rcout<< "Non-adjusted agreement: " << utils::prec2agr(phi_mle.first) << "\n";

            // get mle for alpha
            Eigen::VectorXd alpha_mle = Eigen::VectorXd::Zero(J);
            for(int j=0; j<J; j++){

                if(PROF_METHOD==0){
                    double hatalpha = profile::continuous::Brent(
                        Y, DICT, j, ALPHA_START(j), phi_mle.first, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    alpha_mle(j) = hatalpha;
                }else{
                    double hatalpha = profile::continuous::Newton_Raphson(
                        Y, DICT, j, ALPHA_START(j), phi_mle.first, PROF_SEARCH_RANGE, PROF_MAX_ITER
                    );
                    alpha_mle(j) = hatalpha;
                }
            }

            // negative log-likelihood to minimize
            auto neg_modified_profile_likelihood = [&](double phi){
                double ll = profile::continuous::mp_loglik(
                    phi, Y, DICT, alpha_mle, alpha_mle, phi_mle.first, J, PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
                return -ll; 
            };

            double lower = 1e-8; 
            double upper = phi_mle.first + SEARCH_RANGE;

            const int digits = std::numeric_limits<double>::digits;
            boost::uintmax_t max_iter = MAX_ITER;
            auto result = boost::math::tools::brent_find_minima(
                neg_modified_profile_likelihood, lower, upper, digits, max_iter
            );


            if(VERBOSE) Rcpp::Rcout<< "Adjusted agreement: " << utils::prec2agr(result.first) << "\n";

            std::vector<double> out(3); 
            out[0] = result.first;
            out[1] = -result.second;
            out[2] = phi_mle.first;

            return out;
        }


    }
}



// [[Rcpp::export]]
std::vector<double> cpp_get_phi_mle(
    Eigen::Map<Eigen::VectorXd> Y,
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    Eigen::Map<Eigen::VectorXd> ALPHA_START,
    const double PHI_START,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool VERBOSE = false,
    const bool CONTINUOUS = false)
{
    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);
    
    std::pair<double, double> opt;
    if(CONTINUOUS){
        opt = profile::continuous::get_phi_mle(
            Y, dict, ALPHA_START, PHI_START, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
        );
    }else{
        opt = profile::ordinal::get_phi_mle(
            Y, dict, ALPHA_START, PHI_START, K, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD
        );
    }
    
    std::vector<double> out(2); 
    out[0] = opt.first;
    out[1] = -opt.second;

    return out;
}

// [[Rcpp::export]]
std::vector<double> cpp_get_phi_mp(
    Eigen::Map<Eigen::VectorXd> Y,
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    Eigen::Map<Eigen::VectorXd> ALPHA_START,
    const double PHI_START,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool VERBOSE = false,
    const bool CONTINUOUS = false)
{
    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);
    
    std::vector<double> opt;
    if(CONTINUOUS){
        opt = profile::continuous::get_phi_mp(
            Y, dict, ALPHA_START, PHI_START, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD, VERBOSE
        );
    }else{
        opt = profile::ordinal::get_phi_mp(
            Y, dict, ALPHA_START, PHI_START, K, J,
            SEARCH_RANGE, MAX_ITER,
            PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD, VERBOSE
        );
    }
    
    

    return opt;
}


// [[Rcpp::export]]
double cpp_profile_likelihood(
    Eigen::Map<Eigen::VectorXd> Y,
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    Eigen::Map<Eigen::VectorXd> ALPHA_START,
    const double PHI,
    const int K,
    const int J,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool CONTINUOUS
){
    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);

    double out;
    if(CONTINUOUS){
        out = profile::continuous::loglik(
                PHI, Y, dict, ALPHA_START, J,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }else{
        out = profile::ordinal::loglik(
                PHI, Y, dict, ALPHA_START, K, J,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }

    return out;

}

// [[Rcpp::export]]
double cpp_modified_profile_likelihood(
    Eigen::Map<Eigen::VectorXd> Y,
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    Eigen::Map<Eigen::VectorXd> ALPHA_START,
    const double PHI_MLE,
    const double PHI,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool CONTINUOUS
){
    // get mle for phi
    std::vector<std::vector<int>> dict = utils::items_dicts(J, ITEM_INDS);

    
    

    double out;
    if(CONTINUOUS){
        // get mle for alpha
        Eigen::VectorXd alpha_mle = Eigen::VectorXd::Zero(J);
        for(int j=0; j<J; j++){

            if(PROF_METHOD==0){
                double hatalpha = profile::continuous::Brent(
                    Y, dict, j, ALPHA_START(j), PHI_MLE, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle(j) = hatalpha;
            }else{
                double hatalpha = profile::continuous::Newton_Raphson(
                    Y, dict, j, ALPHA_START(j), PHI_MLE, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle(j) = hatalpha;
            }
        }
        out = profile::continuous::mp_loglik(
                PHI, Y, dict, alpha_mle, alpha_mle, PHI_MLE, J, 
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }else{
        Eigen::VectorXd alpha_mle = Eigen::VectorXd::Zero(J);
        for(int j=0; j<J; j++){

            if(PROF_METHOD==0){
                double hatalpha = profile::ordinal::Brent(
                    Y, dict, j, ALPHA_START(j), PHI_MLE, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle(j) = hatalpha;
            }else{
                double hatalpha = profile::ordinal::Newton_Raphson(
                    Y, dict, j, ALPHA_START(j), PHI_MLE, K, PROF_SEARCH_RANGE, PROF_MAX_ITER
                );
                alpha_mle(j) = hatalpha;
            }
        }
        
        out = profile::ordinal::mp_loglik(
                PHI, Y, dict, alpha_mle, alpha_mle, PHI_MLE, K, J, 
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD);
    }

    return out;

}

// [[Rcpp::export]]
double cpp_get_se(
    Eigen::Map<Eigen::VectorXd> Y,
    Eigen::Map<Eigen::VectorXd> ITEM_INDS,
    Eigen::Map<Eigen::VectorXd> ALPHA_START,
    const double PHI_EVAL,
    const double PHI_MLE,
    const int K,
    const int J,
    const int SEARCH_RANGE,
    const int MAX_ITER,
    const int PROF_SEARCH_RANGE,
    const int PROF_MAX_ITER,
    const int PROF_METHOD,
    const bool MODIFIED =true,
    const bool CONTINUOUS = true
) {
        
    double agr = utils::prec2agr(PHI_EVAL);
    
    std::function<double(double)> f;
    
    if(MODIFIED){
        f = [&](double agr){
            double phi = utils::agr2prec(agr);
            
            double out = cpp_modified_profile_likelihood(
                Y, ITEM_INDS, ALPHA_START, 
                PHI_MLE, phi, K, J,  // Fixed: PHI is phi_mle, phi is test value
                SEARCH_RANGE, MAX_ITER, 
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD,
                CONTINUOUS
            );
            return out;
        };
    }else{
        f = [&](double agr){
            double phi = utils::agr2prec(agr);
            
            double out = cpp_profile_likelihood(
                Y, ITEM_INDS, ALPHA_START, 
                phi, K, J,
                PROF_SEARCH_RANGE, PROF_MAX_ITER, PROF_METHOD,
                CONTINUOUS
            );
            return out;
        };
    }
    
    double d2 = boost::math::differentiation::finite_difference_derivative(
        [&](double x){ 
            return boost::math::differentiation::finite_difference_derivative(f, x); 
        },
        agr
    );
    
    if(-d2 > 0){
        return 1.0 / sqrt(-d2);
    }else{
        return std::numeric_limits<double>::quiet_NaN();
    }

}