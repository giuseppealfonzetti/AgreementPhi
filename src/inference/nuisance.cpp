#include "nuisance.h"

///////////////////////////////////////////////
// CONTINUOUS RATINGS | UNIVARIATE PROFILING //
///////////////////////////////////////////////

//' Profile univariate nuisance parameter
//' @param Y Vector of ratings. Length n.
//' @param DICT List of vector of indexes. Each element of the list relates to a specific item or worker, and collects the indexes of the observations related to that item or worker
//' @param IDX Index of the elemnt of dict we want to profile with respect to.
//' @param CONST_DIM_IDXS Vector of indexes related to the dimension (Item/worker) kept constant. Length n.
//' @param CONST_DIM_PARS Vector of parameters related to the dimension (Item/worker) kept constant
//' @param START Starting value for the optimiser
//' @param PHI Precision parameter
//' @param RANGE Search range for the optimiser (START+-RANGE/2)
//' @param MAX_ITER Maximum number of iterations
//' @returns A double with the profiled nuisance parameter
double AgreementPhi::continuous::nuisance::brent_profiling(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>>& DICT,
    const int IDX,
    const std::vector<int>& CONST_DIM_IDXS,
    const std::vector<double>& CONST_DIM_PARS,
    const double START,
    const double PHI,
    const double RANGE,
    const int MAX_ITER
){
    const std::vector<int>& obs_vec = DICT.at(IDX-1);
    int n_j = obs_vec.size();
    double grad, grad2;
    auto neg_ll = [&](double nuisance){

        double nll=0;
        for(int i=0; i<n_j; i++){
            const int obs_id = obs_vec.at(i);
            const int const_dim_idx = CONST_DIM_IDXS.at(obs_id)-1;
            const double const_dim_par = CONST_DIM_PARS.at(const_dim_idx);
            double dmu, dmu2;
            double mu = link::mu(nuisance+const_dim_par);
            nll -= AgreementPhi::continuous::loglik(Y.at(obs_id), mu, PHI, dmu, dmu2, 0); 
        }

        return nll;
    };

    double lower = START - RANGE; 
    double upper = START + RANGE;
    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;

    auto result = boost::math::tools::brent_find_minima(
        neg_ll, lower, upper, digits, max_iter
    );

    double opt = result.first; 
    return opt;
}

std::vector<std::vector<double>> AgreementPhi::continuous::nuisance::get_lambda(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const double PHI,
    const int J,
    const int W,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double TOL
){
        
    std::vector<double> alphas = ALPHA;
    std::vector<double> betas = BETA;
    betas.at(0) = 0;
    
    if(WORKER_NUISANCE){
        for(int iter = 0; iter < PROF_MAX_ITER; ++iter){
            double max_change = 0;
            
            // Profile items
            for(int j = 0; j < J; ++j){
                double old_alpha = alphas.at(j);
                alphas.at(j) = AgreementPhi::continuous::nuisance::brent_profiling(
                    Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
                    old_alpha, PHI, PROF_UNI_RANGE, PROF_UNI_MAX_ITER
                );
                max_change = std::max(max_change, std::abs(alphas.at(j) - old_alpha));
            }
            
            // Profile workers
            for(int w = 1; w < W; ++w){
                double old_beta = betas.at(w);
                betas.at(w) = AgreementPhi::continuous::nuisance::brent_profiling(
                    Y, WORKER_DICT, w+1, ITEM_INDS, alphas, 
                    old_beta, PHI, PROF_UNI_RANGE, PROF_UNI_MAX_ITER
                );
                max_change = std::max(max_change, std::abs(betas.at(w) - old_beta));
            }
            
            if(max_change < TOL) break;
        }
    }else{
        for(int j = 0; j < J; ++j){
            double old_alpha = alphas.at(j);
            alphas.at(j) = AgreementPhi::continuous::nuisance::brent_profiling(
                Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
                old_alpha, PHI, PROF_UNI_RANGE, PROF_UNI_MAX_ITER
            );
        }
    }
    
    
    std::vector<std::vector<double>> out(2);
    out.at(0) = alphas;
    out.at(1) = betas;
    return out;
}








///////////////////////////////////////////////
// ORDINAL RATINGS | UNIVARIATE PROFILING //
///////////////////////////////////////////////

//' Profile univariate nuisance parameter
//' @param Y Vector of ratings. Length n.
//' @param DICT List of vector of indexes. Each element of the list relates to a specific item or worker, and collects the indexes of the observations related to that item or worker
//' @param IDX Index of the elemnt of dict we want to profile with respect to.
//' @param CONST_DIM_IDXS Vector of indexes related to the dimension (Item/worker) kept constant. Length n.
//' @param CONST_DIM_PARS Vector of parameters related to the dimension (Item/worker) kept constant
//' @param START Starting value for the optimiser
//' @param PHI Precision parameter
//' @param K number of ordinal categories.
//' @param RANGE Search range for the optimiser (START+-RANGE/2)
//' @param MAX_ITER Maximum number of iterations
//' @returns A double with the profiled nuisance parameter
double AgreementPhi::ordinal::nuisance::brent_profiling(
    const std::vector<double>& Y, 
    const std::vector<std::vector<int>>& DICT,
    const int IDX,
    const std::vector<int>& CONST_DIM_IDXS,
    const std::vector<double>& CONST_DIM_PARS,
    const double START,
    const double PHI,
    const std::vector<double>& TAU,
    const double RANGE,
    const int MAX_ITER,
    const double MEAN = 0
){
    const std::vector<int>& obs_vec = DICT.at(IDX-1);
    int n_j = obs_vec.size();


    // Check for degeneracy
    bool all_equal = true;
    if(n_j >= 2) {
        double first_val = Y.at(obs_vec.at(0));
        for(int i = 1; i < n_j; i++){
            if(Y.at(obs_vec.at(i))!=first_val) {
                all_equal = false;
                break;
            }
        }
    }

    if(all_equal || n_j < 2) {
        // double lambda_reg = (all_equal) ? 1.0 : 0.5;
        
        auto neg_ll_regularized = [&](double nuisance){
            double nll = 0;
            
            for(int i=0; i<n_j; i++){
                const int obs_id = obs_vec.at(i);
                const int const_dim_idx = CONST_DIM_IDXS.at(obs_id)-1;
                const double const_dim_par = CONST_DIM_PARS.at(const_dim_idx);
                double dmu, dmu2;
                double mu = link::mu(nuisance + const_dim_par);
                nll -= AgreementPhi::ordinal::loglik(Y.at(obs_id), mu, PHI, TAU, dmu, dmu2, 0);
            }
            
            nll += 0.5 * pow(nuisance - MEAN, 2);
            
            return nll;
        };
        
        double lower = std::max(START - RANGE/2, -10.0);
        double upper = std::min(START + RANGE/2,  10.0);
        
        const int digits = std::numeric_limits<double>::digits;
        boost::uintmax_t max_iter = MAX_ITER;
        
        auto result = boost::math::tools::brent_find_minima(
            neg_ll_regularized, lower, upper, digits, max_iter
        );
        
        return result.first;
    }else{
        double grad, grad2;
        auto neg_ll = [&](double nuisance){

            double nll=0;
            for(int i=0; i<n_j; i++){
                const int obs_id = obs_vec.at(i);
                const int const_dim_idx = CONST_DIM_IDXS.at(obs_id)-1;
                const double const_dim_par = CONST_DIM_PARS.at(const_dim_idx);
                double dmu, dmu2;
                double mu = link::mu(nuisance+const_dim_par);
                nll -= AgreementPhi::ordinal::loglik(Y.at(obs_id), mu, PHI, TAU, dmu, dmu2, 0); 
            }

            return nll;
        };

        double lower = std::max(START - RANGE/2, -10.0);
        double upper = std::min(START + RANGE/2,  10.0);
        const int digits = std::numeric_limits<double>::digits;
        boost::uintmax_t max_iter = MAX_ITER;

        auto result = boost::math::tools::brent_find_minima(
            neg_ll, lower, upper, digits, max_iter
        );

        double opt = result.first; 
        return opt;
    }

    
}


double AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
                const std::vector<double>& Y, 
                const std::vector<double>& MU, 
                const std::vector<std::vector<int>> CAT_DICT,
                const int IDX,
                const std::vector<double>& TAU,
                const double PHI,
                const int MAX_ITER)
{
    
    const std::vector<int>& cat_t = CAT_DICT.at(IDX - 1);
    const std::vector<int>& cat_tp1 = CAT_DICT.at(IDX);

    auto neg_ll = [&](double thr){
        std::vector<double> tau_candidate = TAU;
        tau_candidate.at(IDX) = thr;
        double ll = 0;

        for(const int idx : cat_t){
            double d1 = 0.0, d2 = 0.0;
            ll += AgreementPhi::ordinal::loglik(
                Y.at(idx), MU.at(idx), PHI, tau_candidate, d1, d2, 0
            );
        }

        for(const int idx : cat_tp1){
            double d1 = 0.0, d2 = 0.0;
            ll += AgreementPhi::ordinal::loglik(
                Y.at(idx), MU.at(idx), PHI, tau_candidate, d1, d2, 0
            );
        }

        return -ll;
    };

    double lower = TAU.at(IDX - 1) + 1e-8;
    double upper = TAU.at(IDX + 1) - 1e-8;
    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_ll, lower, upper, digits, max_iter
    );

    double opt = result.first; 
    return opt;
}

// std::vector<std::vector<double>> AgreementPhi::ordinal::nuisance::get_lambda(
//     const std::vector<double> Y,  
//     const std::vector<int> ITEM_INDS,
//     const std::vector<int> WORKER_INDS,
//     const std::vector<std::vector<int>> ITEM_DICT,
//     const std::vector<std::vector<int>> WORKER_DICT,
//     const std::vector<double> ALPHA,
//     const std::vector<double> BETA,
//     const std::vector<double> TAU,
//     const double PHI,
//     const int J,
//     const int W,
//     const int K,
//     const bool WORKER_NUISANCE,
//     const double PROF_UNI_RANGE,
//     const int PROF_UNI_MAX_ITER,
//     const int PROF_MAX_ITER,
//     const double TOL
// ){
    
    
//     std::vector<double> alphas = ALPHA;
//     std::vector<double> betas = BETA;
//     betas.at(0) = 0;
    
//     if(WORKER_NUISANCE){
//         for(int iter = 0; iter < PROF_MAX_ITER; ++iter){
//             double max_change = 0;
            

//             double mean_alpha = std::accumulate(std::begin(alphas), std::end(alphas), 0.0);
//             mean_alpha /= J;

//             double mean_beta = std::accumulate(std::begin(betas), std::end(betas), 0.0);
//             mean_beta /= W;

//             // Profile items
//             for(int j = 0; j < J; ++j){
//                 double old_alpha = alphas.at(j);
//                 alphas.at(j) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                     Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
//                     old_alpha, PHI, TAU, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_alpha
//                 );
//                 max_change = std::max(max_change, std::abs(alphas.at(j) - old_alpha));
//             }
            
            
//             // Profile workers
//             for(int w = 1; w < W; ++w){
//                 double old_beta = betas.at(w);
//                 betas.at(w) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                     Y, WORKER_DICT, w+1, ITEM_INDS, alphas, 
//                     old_beta, PHI, TAU, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_beta
//                 );
//                 max_change = std::max(max_change, std::abs(betas.at(w) - old_beta));
//             }
            
//             if(max_change < TOL) break;
//         }
//     }else{
//         // Profile items
//         double mean_alpha = std::accumulate(std::begin(alphas), std::end(alphas), 0.0);
//         mean_alpha /= J;

//         for(int j = 0; j < J; ++j){
//             double old_alpha = alphas.at(j);
//             alphas.at(j) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                 Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
//                 old_alpha, PHI, TAU, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_alpha
//             );
//         }
//     }

    
//     std::vector<std::vector<double>> out(2);
//     out.at(0) = alphas;
//     out.at(1) = betas;
//     return out;
// }

// std::vector<std::vector<double>> AgreementPhi::ordinal::nuisance::get_lambda2(
//     const std::vector<double> Y,  
//     const std::vector<int> ITEM_INDS,
//     const std::vector<int> WORKER_INDS,
//     const std::vector<std::vector<int>> ITEM_DICT,
//     const std::vector<std::vector<int>> WORKER_DICT,
//     const std::vector<std::vector<int>> CAT_DICT,
//     const std::vector<double> ALPHA,
//     const std::vector<double> BETA,
//     const std::vector<double> TAU,
//     const double PHI,
//     const int J,
//     const int W,
//     const int K,
//     const bool WORKER_NUISANCE,
//     const bool THRESHOLDS_NUISANCE,
//     const double PROF_UNI_RANGE,
//     const int PROF_UNI_MAX_ITER,
//     const int PROF_MAX_ITER,
//     const double TOL
// ){
    
//     const int n = Y.size();
//     std::vector<double> alphas = ALPHA;
//     std::vector<double> betas = BETA;
//     std::vector<double> taus = TAU;
//     betas.at(0) = 0;

//     int prof_max_iter = 1;
//     if(WORKER_NUISANCE | THRESHOLDS_NUISANCE){
//         prof_max_iter = PROF_MAX_ITER;
//     }
    
//     for(int iter = 0; iter < prof_max_iter; ++iter){
//         double max_change = 0;
        

//         double mean_alpha = std::accumulate(std::begin(alphas), std::end(alphas), 0.0);
//         mean_alpha /= J;

//         double mean_beta = std::accumulate(std::begin(betas), std::end(betas), 0.0);
//         mean_beta /= W;

//         // Profile items
//         for(int j = 0; j < J; ++j){
//             double old_alpha = alphas.at(j);
//             alphas.at(j) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                 Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
//                 old_alpha, PHI, taus, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_alpha
//             );
//             max_change = std::max(max_change, std::abs(alphas.at(j) - old_alpha));

//             // Profile thresholds
//             if(THRESHOLDS_NUISANCE){

//                 std::vector<double> mu_vec(n);
//                 for(int i = 0; i < n; ++i){
//                     int item_idx = ITEM_INDS.at(i) - 1;
//                     int worker_idx = WORKER_INDS.at(i) - 1;
//                     double eta = alphas.at(item_idx);
//                     if(worker_idx > 0){
//                         eta += betas.at(worker_idx);
//                     }
//                     mu_vec.at(i) = link::mu(eta);
//                 }

//                 for(int t = K-1; t >0; t--){
//                     // Rcpp::Rcout << "Iter: "<< iter << ", cat="<<t<<", lb ="<<taus.at(t-1)<<" old="<<taus.at(t)<< ", ub="<<taus.at(t+1)<<" ";
//                     double old_tau = taus.at(t);
//                     taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
//                         Y, mu_vec,  CAT_DICT, t, taus, PHI, PROF_UNI_MAX_ITER);
//                     max_change = std::max(max_change, std::abs(taus.at(t) - old_tau));
//                     // Rcpp::Rcout << "-> est: "<< taus.at(t) << "\n";
//                 }

//             }
//         }
        
        

//         // Profile workers
//         if(WORKER_NUISANCE){
//             for(int w = 1; w < W; ++w){
//                 double old_beta = betas.at(w);
//                 betas.at(w) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                     Y, WORKER_DICT, w+1, ITEM_INDS, alphas, 
//                     old_beta, PHI, taus, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_beta
//                 );
//                 max_change = std::max(max_change, std::abs(betas.at(w) - old_beta));

//                 // Profile thresholds
//                 if(THRESHOLDS_NUISANCE){

//                     std::vector<double> mu_vec(n);
//                     for(int i = 0; i < n; ++i){
//                         int item_idx = ITEM_INDS.at(i) - 1;
//                         int worker_idx = WORKER_INDS.at(i) - 1;
//                         double eta = alphas.at(item_idx);
//                         if(worker_idx > 0){
//                             eta += betas.at(worker_idx);
//                         }
//                         mu_vec.at(i) = link::mu(eta);
//                     }

//                     for(int t = K-1; t >0; t--){
//                         // Rcpp::Rcout << "Iter: "<< iter << ", cat="<<t<<", lb ="<<taus.at(t-1)<<" old="<<taus.at(t)<< ", ub="<<taus.at(t+1)<<" ";
//                         double old_tau = taus.at(t);
//                         taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
//                             Y, mu_vec,  CAT_DICT, t, taus, PHI, PROF_UNI_MAX_ITER);
//                         max_change = std::max(max_change, std::abs(taus.at(t) - old_tau));
//                         // Rcpp::Rcout << "-> est: "<< taus.at(t) << "\n";
//                     }

//                 }
//             }


//         }
        

        
        
        
//         if(max_change < TOL) break;
//     }
 
//     std::vector<std::vector<double>> out(3);
//     out.at(0) = alphas;
//     out.at(1) = betas;
//     out.at(2) = taus;
//     return out;
// }

// std::vector<std::vector<double>> AgreementPhi::ordinal::nuisance::get_lambda2(
//     const std::vector<double> Y,  
//     const std::vector<int> ITEM_INDS,
//     const std::vector<int> WORKER_INDS,
//     const std::vector<std::vector<int>> ITEM_DICT,
//     const std::vector<std::vector<int>> WORKER_DICT,
//     const std::vector<std::vector<int>> CAT_DICT,
//     const std::vector<double> ALPHA,
//     const std::vector<double> BETA,
//     const std::vector<double> TAU,
//     const double PHI,
//     const int J,
//     const int W,
//     const int K,
//     const bool WORKER_NUISANCE,
//     const bool THRESHOLDS_NUISANCE,
//     const double PROF_UNI_RANGE,
//     const int PROF_UNI_MAX_ITER,
//     const int PROF_MAX_ITER,
//     const double TOL
// ){
    
//     const int n = Y.size();
//     std::vector<double> alphas = ALPHA;
//     std::vector<double> betas = BETA;
//     std::vector<double> taus = TAU;
//     betas.at(0) = 0;

//     int prof_max_iter = 1;
//     if(WORKER_NUISANCE | THRESHOLDS_NUISANCE){
//         prof_max_iter = PROF_MAX_ITER;
//     }
    
//     // Function to compute log-likelihood
//     auto compute_loglik = [&]() -> double {
//         double ll = 0.0;
//         for(int i = 0; i < n; ++i){
//             int item_idx = ITEM_INDS.at(i) - 1;
//             int worker_idx = WORKER_INDS.at(i) - 1;
//             double eta = alphas.at(item_idx);
//             if(worker_idx > 0 && WORKER_NUISANCE){
//                 eta += betas.at(worker_idx);
//             }
//             double mu = link::mu(eta);
//             double d1 = 0.0, d2 = 0.0;
//             ll += AgreementPhi::ordinal::loglik(Y.at(i), mu, PHI, taus, d1, d2, 0);
//         }
//         return ll;
//     };
    
//     double ll_best = compute_loglik();
//     std::vector<double> alphas_best = alphas;
//     std::vector<double> betas_best = betas;
//     std::vector<double> taus_best = taus;
    
//     int stall_count = 0;
//     const int max_stall = 3;  // Stop if likelihood doesn't improve for 3 iterations
    
//     for(int iter = 0; iter < prof_max_iter; ++iter){
//         double max_change = 0;
//         double ll_iter_start = ll_best;
        
//         // Save state before alpha profiling
//         auto alphas_before = alphas;
        
//         double mean_alpha = std::accumulate(std::begin(alphas), std::end(alphas), 0.0);
//         mean_alpha /= J;

//         // Profile items
//         for(int j = 0; j < J; ++j){
//             double old_alpha = alphas.at(j);
//             alphas.at(j) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                 Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
//                 old_alpha, PHI, taus, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_alpha
//             );
//             max_change = std::max(max_change, std::abs(alphas.at(j) - old_alpha));
//         }
        
//         // Check if alpha update improved likelihood
//         double ll_after_alpha = compute_loglik();
//         if(ll_after_alpha < ll_best - 1e-6){  // Small tolerance for numerical error
//             // Reject alpha update
//             alphas = alphas_before;
//             Rcpp::Rcout << "Iter " << iter << ": Alpha update REJECTED (LL decreased by " 
//                        << (ll_best - ll_after_alpha) << ")\n";
//         } else {
//             ll_best = ll_after_alpha;
//             alphas_best = alphas;
//         }
        
//         // Profile workers if needed
//         if(WORKER_NUISANCE){
//             auto betas_before = betas;
//             double mean_beta = std::accumulate(std::begin(betas), std::end(betas), 0.0);
//             mean_beta /= W;
            
//             for(int w = 1; w < W; ++w){
//                 double old_beta = betas.at(w);
//                 betas.at(w) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                     Y, WORKER_DICT, w+1, ITEM_INDS, alphas, 
//                     old_beta, PHI, taus, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_beta
//                 );
//                 max_change = std::max(max_change, std::abs(betas.at(w) - old_beta));
//             }
            
//             // Check if beta update improved likelihood
//             double ll_after_beta = compute_loglik();
//             if(ll_after_beta < ll_best - 1e-6){
//                 // Reject beta update
//                 betas = betas_before;
//                 Rcpp::Rcout << "Iter " << iter << ": Beta update REJECTED (LL decreased by " 
//                            << (ll_best - ll_after_beta) << ")\n";
//             } else {
//                 ll_best = ll_after_beta;
//                 betas_best = betas;
//             }
//         }

//         // Profile thresholds if needed
//         if(THRESHOLDS_NUISANCE){
//             auto taus_before = taus;
            
//             std::vector<double> mu_vec(n);
//             for(int i = 0; i < n; ++i){
//                 int item_idx = ITEM_INDS.at(i) - 1;
//                 int worker_idx = WORKER_INDS.at(i) - 1;
//                 double eta = alphas.at(item_idx);
//                 if(worker_idx > 0 && WORKER_NUISANCE){
//                     eta += betas.at(worker_idx);
//                 }
//                 mu_vec.at(i) = link::mu(eta);
//             }

//             for(int t = 1; t < K; t++){
//                 double old_tau = taus.at(t);
//                 taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
//                     Y, mu_vec, CAT_DICT, t, taus, PHI, PROF_UNI_MAX_ITER
//                 );
//                 max_change = std::max(max_change, std::abs(taus.at(t) - old_tau));
//             }
            
//             // Check if threshold update improved likelihood
//             double ll_after_tau = compute_loglik();
//             if(ll_after_tau < ll_best - 1e-6){
//                 // Reject threshold update
//                 taus = taus_before;
//                 Rcpp::Rcout << "Iter " << iter << ": Tau update REJECTED (LL decreased by " 
//                            << (ll_best - ll_after_tau) << ")\n";
//             } else {
//                 ll_best = ll_after_tau;
//                 taus_best = taus;
//             }
//         }
        
//         // Check for convergence
//         if(max_change < TOL){
//             Rcpp::Rcout << "Converged at iteration " << iter << " (max change < TOL)\n";
//             break;
//         }
        
//         // Check for stalling
//         if(ll_best - ll_iter_start < 1e-6){
//             stall_count++;
//             if(stall_count >= max_stall){
//                 Rcpp::Rcout << "Stopping at iteration " << iter << " (likelihood not improving)\n";
//                 break;
//             }
//         } else {
//             stall_count = 0;  // Reset if we made progress
//         }
//     }
    
//     // Return best estimates
//     std::vector<std::vector<double>> out(3);
//     out.at(0) = alphas_best;
//     out.at(1) = betas_best;
//     out.at(2) = taus_best;
//     return out;
// }

std::vector<std::vector<double>> AgreementPhi::ordinal::nuisance::get_lambda2(
    const std::vector<double> Y,  
    const std::vector<int> ITEM_INDS,
    const std::vector<int> WORKER_INDS,
    const std::vector<std::vector<int>> ITEM_DICT,
    const std::vector<std::vector<int>> WORKER_DICT,
    const std::vector<std::vector<int>> CAT_DICT,
    const std::vector<double> ALPHA,
    const std::vector<double> BETA,
    const std::vector<double> TAU,
    const double PHI,
    const int J,
    const int W,
    const int K,
    const bool ITEMS_NUISANCE,
    const bool WORKER_NUISANCE,
    const bool THRESHOLDS_NUISANCE,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double TOL
){
    
    const int n = Y.size();
    std::vector<double> alphas = ALPHA;
    std::vector<double> betas = BETA;
    std::vector<double> taus = TAU;
    betas.at(0) = 0;

    int prof_max_iter = 1;
    if(WORKER_NUISANCE + ITEMS_NUISANCE > 1 | THRESHOLDS_NUISANCE ){
        prof_max_iter = PROF_MAX_ITER;
    }
    
    // Function to compute log-likelihood
    auto compute_loglik = [&]() -> double {
        double ll = 0.0;
        for(int i = 0; i < n; ++i){
            int item_idx = ITEM_INDS.at(i) - 1;
            int worker_idx = WORKER_INDS.at(i) - 1;
            double eta = alphas.at(item_idx);
            if(worker_idx > 0 && WORKER_NUISANCE){
                eta += betas.at(worker_idx);
            }
            double mu = link::mu(eta);
            double d1 = 0.0, d2 = 0.0;
            ll += AgreementPhi::ordinal::loglik(Y.at(i), mu, PHI, taus, d1, d2, 0);
        }
        return ll;
    };
    
    double ll_best = compute_loglik();
    std::vector<double> alphas_best = alphas;
    std::vector<double> betas_best = betas;
    std::vector<double> taus_best = taus;
    
    int stall_count = 0;
    const int max_stall = 3;
    
    for(int iter = 0; iter < prof_max_iter; iter++){
        double max_change = 0;
        double ll_iter_start = ll_best;

        double mean_alpha = std::accumulate(std::begin(alphas), std::end(alphas), 0.0);
        mean_alpha /= J;

        double mean_beta = std::accumulate(std::begin(betas), std::end(betas), 0.0);
        mean_beta /= W;

        // Profile items
        if(ITEMS_NUISANCE){
            for(int j = 0; j < J; ++j){
                double ll_before_alpha_j = compute_loglik();
                auto alphas_before = alphas;
                auto taus_before = taus;
                
                
                double old_alpha = alphas.at(j);
                alphas.at(j) = AgreementPhi::ordinal::nuisance::brent_profiling(
                    Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
                    old_alpha, PHI, taus, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_alpha
                );
                max_change = std::max(max_change, std::abs(alphas.at(j) - old_alpha));

                // Profile thresholds after alpha_j update
                if(THRESHOLDS_NUISANCE){
                    std::vector<double> mu_vec(n);
                    for(int i = 0; i < n; ++i){
                        int item_idx = ITEM_INDS.at(i) - 1;
                        int worker_idx = WORKER_INDS.at(i) - 1;
                        double eta = alphas.at(item_idx);
                        if(worker_idx > 0){
                            eta += betas.at(worker_idx);
                        }
                        mu_vec.at(i) = link::mu(eta);
                    }

                    for(int iter_t = 0; iter_t < 3; iter_t++){
                        for(int t = K-1; t > 0; t--){
                            double old_tau = taus.at(t);
                            taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
                                Y, mu_vec, CAT_DICT, t, taus, PHI, PROF_UNI_MAX_ITER);
                            max_change = std::max(max_change, std::abs(taus.at(t) - old_tau));
                        }
                    }
                    
                }
                
                // Check if this alpha_j + tau update improved likelihood
                double ll_after = compute_loglik();
                if(ll_after < ll_before_alpha_j - 1e-6){
                    // Reject both alpha_j and tau updates
                    alphas = alphas_before;
                    taus = taus_before;
                } else if(ll_after > ll_best){
                    // Accept and update best
                    ll_best = ll_after;
                    alphas_best = alphas;
                    taus_best = taus;
                }
            }
        }

        
        
        // Profile workers
        if(WORKER_NUISANCE){
            for(int w = 1; w < W; ++w){
                double ll_before_beta_w = compute_loglik();
                auto betas_before = betas;
                auto taus_before = taus;
                
                double old_beta = betas.at(w);
                betas.at(w) = AgreementPhi::ordinal::nuisance::brent_profiling(
                    Y, WORKER_DICT, w+1, ITEM_INDS, alphas, 
                    old_beta, PHI, taus, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_beta
                );
                max_change = std::max(max_change, std::abs(betas.at(w) - old_beta));

                // Profile thresholds after beta_w update
                if(THRESHOLDS_NUISANCE){
                    std::vector<double> mu_vec(n);
                    for(int i = 0; i < n; ++i){
                        int item_idx = ITEM_INDS.at(i) - 1;
                        int worker_idx = WORKER_INDS.at(i) - 1;
                        double eta = alphas.at(item_idx);
                        if(worker_idx > 0){
                            eta += betas.at(worker_idx);
                        }
                        mu_vec.at(i) = link::mu(eta);
                    }

                    for(int t = K-1; t > 0; t--){
                        double old_tau = taus.at(t);
                        taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
                            Y, mu_vec, CAT_DICT, t, taus, PHI, PROF_UNI_MAX_ITER);
                        max_change = std::max(max_change, std::abs(taus.at(t) - old_tau));
                    }
                }
                
                // Check if this beta_w + tau update improved likelihood
                double ll_after = compute_loglik();
                if(ll_after < ll_before_beta_w - 1e-6){
                    // Reject both beta_w and tau updates
                    betas = betas_before;
                    taus = taus_before;
                } else if(ll_after > ll_best){
                    // Accept and update best
                    ll_best = ll_after;
                    betas_best = betas;
                    taus_best = taus;
                }
            }
        }

        if((ITEMS_NUISANCE+WORKER_NUISANCE==0) & (THRESHOLDS_NUISANCE)){
            std::vector<double> mu_vec(n);
            for(int i = 0; i < n; ++i){
                int item_idx = ITEM_INDS.at(i) - 1;
                int worker_idx = WORKER_INDS.at(i) - 1;
                double eta = alphas.at(item_idx);
                if(worker_idx > 0){
                    eta += betas.at(worker_idx);
                }
                mu_vec.at(i) = link::mu(eta);
            }

            double ll_before_tau = compute_loglik();
            auto taus_before = taus;
            for(int iter_t = 0; iter_t < 3; iter_t++){
                for(int t = K-1; t > 0; t--){
                    double old_tau = taus.at(t);
                    taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
                        Y, mu_vec, CAT_DICT, t, taus, PHI, PROF_UNI_MAX_ITER);
                    max_change = std::max(max_change, std::abs(taus.at(t) - old_tau));
                }
            }

            double ll_after = compute_loglik();
                if(ll_after < ll_before_tau - 1e-6){
                    taus = taus_before;
                } else if(ll_after > ll_best){
                    ll_best = ll_after;
                    taus_best = taus;
                }
        }
        
        // Check convergence
        if(max_change < TOL) break;
        
        // Check for stalling
        if(ll_best - ll_iter_start < 1e-6){
            stall_count++;
            if(stall_count >= max_stall){
                break;
            }
        } else {
            stall_count = 0;
        }
    }
 
    std::vector<std::vector<double>> out(3);
    out.at(0) = alphas_best;
    out.at(1) = betas_best;
    out.at(2) = taus_best;
    return out;
}