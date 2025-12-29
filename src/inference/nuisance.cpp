#include "nuisance.h"
#include "nuisance_parallel.h"

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

    double lower = std::max(TAU.at(IDX) - 0.1, TAU.at(IDX - 1) + 1e-8);
    double upper = std::min(TAU.at(IDX) + 0.1, TAU.at(IDX + 1) - 1e-8);
    // double lower = TAU.at(IDX - 1) + 1e-8;
    // double upper = TAU.at(IDX + 1) - 1e-8;
    const int digits = std::numeric_limits<double>::digits;
    boost::uintmax_t max_iter = MAX_ITER;
    auto result = boost::math::tools::brent_find_minima(
        neg_ll, lower, upper, digits, max_iter
    );

    double opt = result.first; 
    return opt;
}





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
    const int max_iter_thr = 3;
    const int n = Y.size();
    std::vector<double> alphas_best = ALPHA;
    std::vector<double> betas_best = BETA;
    std::vector<double> taus_best = TAU;
    betas_best.at(0) = 0;

    int prof_max_iter = 1;
    if(WORKER_NUISANCE + ITEMS_NUISANCE > 1 | THRESHOLDS_NUISANCE ){
        prof_max_iter = PROF_MAX_ITER;
    }
    
    // Function to compute log-likelihood
    auto compute_loglik = [&](const std::vector<double>& alphas,
                               const std::vector<double>& betas,
                               const std::vector<double>& taus) -> double {
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

    double ll_best = compute_loglik(alphas_best, betas_best, taus_best);


    int stall_count = 0;
    const int max_stall = 3;
    
    for(int iter = 0; iter < prof_max_iter; iter++){
        double max_change = 0;
        double ll_iter_start = ll_best;
        double ll_after = ll_best;

        double mean_alpha = std::accumulate(std::begin(alphas_best), std::end(alphas_best), 0.0);
        mean_alpha /= J;

        double mean_beta = std::accumulate(std::begin(betas_best), std::end(betas_best), 0.0);
        mean_beta /= W;

        // Profile items
        if(ITEMS_NUISANCE){
            // Allocate output vector
            std::vector<double> alphas_new(J);

            // Create worker
            ParallelItemWorker worker(
                Y, ITEM_INDS, WORKER_INDS,
                alphas_best, betas_best, taus_best,
                ITEM_DICT, PHI, PROF_UNI_RANGE, PROF_UNI_MAX_ITER,
                mean_alpha, alphas_new
            );

            // Run with specified number of cores (handles NCORES=1 automatically)
            RcppParallel::parallelFor(0, J, worker, 1);

            // Sequential acceptance phase
            for(int j = 0; j < J; ++j) {
                std::vector<double> working_alphas = alphas_best;
                working_alphas.at(j) = alphas_new[j];

                double ll_after = compute_loglik(working_alphas, betas_best, taus_best);
                if(ll_after > ll_best) {
                    ll_best = ll_after;
                    alphas_best = working_alphas;
                    max_change = std::max(max_change, std::abs(alphas_new[j] - alphas_best.at(j)));
                }
            }
        }

        // Profile thresholds after alpha update
        // if(THRESHOLDS_NUISANCE){
        //     std::vector<double> mu_vec(n);
        //     for(int i = 0; i < n; ++i){
        //         int item_idx = ITEM_INDS.at(i) - 1;
        //         int worker_idx = WORKER_INDS.at(i) - 1;
        //         double eta = alphas_best.at(item_idx);
        //         if(worker_idx > 0){
        //             eta += betas_best.at(worker_idx);
        //         }
        //         mu_vec.at(i) = link::mu(eta);
        //     }

                            
        //     for(int iter_t = 0; iter_t < max_iter_thr; iter_t++){
        //         std::vector<double> working_taus = taus_best;
        //         // Optimize all thresholds
        //         for(int t = K-1; t > 0; t--){
        //             working_taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
        //                 Y, mu_vec, CAT_DICT, t, working_taus, PHI, PROF_UNI_MAX_ITER);
        //             max_change = std::max(max_change, std::abs(working_taus.at(t) - taus_best.at(t)));
        //         }
        //         // Check likelihood after ALL thresholds have been updated
        //         double ll_after = compute_loglik(alphas_best, betas_best, working_taus);
        //         if(ll_after > ll_best){
        //             ll_best = ll_after;
        //             taus_best = working_taus;
        //         }
        //     }
            
        // }
        
        // Profile workers
        if(WORKER_NUISANCE){
            // Allocate output vector
            std::vector<double> betas_new(W);
            betas_new[0] = 0.0;  // First worker constrained to 0

            // Create worker
            ParallelWorkerWorker worker(
                Y, ITEM_INDS, WORKER_INDS,
                alphas_best, betas_best, taus_best,
                WORKER_DICT, PHI, PROF_UNI_RANGE, PROF_UNI_MAX_ITER,
                mean_beta, betas_new
            );

            // Run with specified number of cores (handles NCORES=1 automatically)
            RcppParallel::parallelFor(0, W, worker, 1);

            // Sequential acceptance phase
            for(int w = 1; w < W; ++w) {  // Skip first worker (constrained to 0)
                std::vector<double> working_betas = betas_best;
                working_betas.at(w) = betas_new[w];

                double ll_after = compute_loglik(alphas_best, working_betas, taus_best);
                if(ll_after > ll_best) {
                    ll_best = ll_after;
                    betas_best = working_betas;
                    max_change = std::max(max_change, std::abs(betas_new[w] - betas_best.at(w)));
                }
            }
        }

        // Profile thresholds after alpha update
        // if(THRESHOLDS_NUISANCE){
        //     std::vector<double> mu_vec(n);
        //     for(int i = 0; i < n; ++i){
        //         int item_idx = ITEM_INDS.at(i) - 1;
        //         int worker_idx = WORKER_INDS.at(i) - 1;
        //         double eta = alphas_best.at(item_idx);
        //         if(worker_idx > 0){
        //             eta += betas_best.at(worker_idx);
        //         }
        //         mu_vec.at(i) = link::mu(eta);
        //     }

                            
        //     for(int iter_t = 0; iter_t < max_iter_thr; iter_t++){
        //         std::vector<double> working_taus = taus_best;
        //         // Optimize all thresholds
        //         for(int t = K-1; t > 0; t--){
        //             working_taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
        //                 Y, mu_vec, CAT_DICT, t, working_taus, PHI, PROF_UNI_MAX_ITER);
        //             max_change = std::max(max_change, std::abs(working_taus.at(t) - taus_best.at(t)));
        //         }
        //         // Check likelihood after ALL thresholds have been updated
        //         double ll_after = compute_loglik(alphas_best, betas_best, working_taus);
        //         if(ll_after > ll_best){
        //             ll_best = ll_after;
        //             taus_best = working_taus;
        //         }
        //     }
            
        // }


        
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
//     const bool ITEMS_NUISANCE,
//     const bool WORKER_NUISANCE,
//     const bool THRESHOLDS_NUISANCE,
//     const double PROF_UNI_RANGE,
//     const int PROF_UNI_MAX_ITER,
//     const int PROF_MAX_ITER,
//     const double TOL,
//     const int NCORES
// ){
//     const int max_iter_thr = 3;
//     const int n = Y.size();
//     std::vector<double> alphas_best = ALPHA;
//     std::vector<double> betas_best = BETA;
//     std::vector<double> taus_best = TAU;
//     betas_best.at(0) = 0;

//     int prof_max_iter = 1;
//     if(WORKER_NUISANCE + ITEMS_NUISANCE > 1 | THRESHOLDS_NUISANCE ){
//         prof_max_iter = PROF_MAX_ITER;
//     }
    
//     // Function to compute log-likelihood
//     auto compute_loglik = [&](const std::vector<double>& alphas,
//                                const std::vector<double>& betas,
//                                const std::vector<double>& taus) -> double {
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

//     double ll_best = compute_loglik(alphas_best, betas_best, taus_best);


//     int stall_count = 0;
//     const int max_stall = 3;
    
//     for(int iter = 0; iter < prof_max_iter; iter++){
//         double max_change = 0;
//         double ll_iter_start = ll_best;
//         double ll_after = ll_best;

//         double mean_alpha = std::accumulate(std::begin(alphas_best), std::end(alphas_best), 0.0);
//         mean_alpha /= J;

//         double mean_beta = std::accumulate(std::begin(betas_best), std::end(betas_best), 0.0);
//         mean_beta /= W;

//         // Profile items
//         if(ITEMS_NUISANCE){
//             for(int j = 0; j < J; ++j){
//                 std::vector<double> working_alphas = alphas_best;


//                 working_alphas.at(j) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                     Y, ITEM_DICT, j+1, WORKER_INDS, betas_best,
//                     alphas_best.at(j), PHI, taus_best, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_alpha
//                 );
//                 max_change = std::max(max_change, std::abs(working_alphas.at(j) - alphas_best.at(j)));

//                 double ll_after = compute_loglik(working_alphas, betas_best, taus_best);
//                 if(ll_after > ll_best){
//                     ll_best = ll_after;
//                     alphas_best = working_alphas;
//                 }
//             }
//         }

//         // Profile thresholds after alpha update
//         // if(THRESHOLDS_NUISANCE){
//         //     std::vector<double> mu_vec(n);
//         //     for(int i = 0; i < n; ++i){
//         //         int item_idx = ITEM_INDS.at(i) - 1;
//         //         int worker_idx = WORKER_INDS.at(i) - 1;
//         //         double eta = alphas_best.at(item_idx);
//         //         if(worker_idx > 0){
//         //             eta += betas_best.at(worker_idx);
//         //         }
//         //         mu_vec.at(i) = link::mu(eta);
//         //     }

                            
//         //     for(int iter_t = 0; iter_t < max_iter_thr; iter_t++){
//         //         std::vector<double> working_taus = taus_best;
//         //         // Optimize all thresholds
//         //         for(int t = K-1; t > 0; t--){
//         //             working_taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
//         //                 Y, mu_vec, CAT_DICT, t, working_taus, PHI, PROF_UNI_MAX_ITER);
//         //             max_change = std::max(max_change, std::abs(working_taus.at(t) - taus_best.at(t)));
//         //         }
//         //         // Check likelihood after ALL thresholds have been updated
//         //         double ll_after = compute_loglik(alphas_best, betas_best, working_taus);
//         //         if(ll_after > ll_best){
//         //             ll_best = ll_after;
//         //             taus_best = working_taus;
//         //         }
//         //     }
            
//         // }
        
//         // Profile workers
//         if(WORKER_NUISANCE){
//             for(int w = 1; w < W; ++w){
//                 double ll_before_beta_w = ll_after;
//                 std::vector<double> working_betas = betas_best;

//                 working_betas.at(w) = AgreementPhi::ordinal::nuisance::brent_profiling(
//                     Y, WORKER_DICT, w+1, ITEM_INDS, alphas_best,
//                     betas_best.at(w), PHI, taus_best, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_beta
//                 );
//                 max_change = std::max(max_change, std::abs(working_betas.at(w) - betas_best.at(w)));

//                 // Check if this beta_w + tau update improved likelihood
//                 double ll_after = compute_loglik(alphas_best, working_betas, taus_best);
//                 if(ll_after > ll_best){
//                     ll_best = ll_after;
//                     betas_best = working_betas;
//                 }
//             }
//         }

//         // Profile thresholds after alpha update
//         // if(THRESHOLDS_NUISANCE){
//         //     std::vector<double> mu_vec(n);
//         //     for(int i = 0; i < n; ++i){
//         //         int item_idx = ITEM_INDS.at(i) - 1;
//         //         int worker_idx = WORKER_INDS.at(i) - 1;
//         //         double eta = alphas_best.at(item_idx);
//         //         if(worker_idx > 0){
//         //             eta += betas_best.at(worker_idx);
//         //         }
//         //         mu_vec.at(i) = link::mu(eta);
//         //     }

                            
//         //     for(int iter_t = 0; iter_t < max_iter_thr; iter_t++){
//         //         std::vector<double> working_taus = taus_best;
//         //         // Optimize all thresholds
//         //         for(int t = K-1; t > 0; t--){
//         //             working_taus.at(t) = AgreementPhi::ordinal::nuisance::brent_profiling_thresholds(
//         //                 Y, mu_vec, CAT_DICT, t, working_taus, PHI, PROF_UNI_MAX_ITER);
//         //             max_change = std::max(max_change, std::abs(working_taus.at(t) - taus_best.at(t)));
//         //         }
//         //         // Check likelihood after ALL thresholds have been updated
//         //         double ll_after = compute_loglik(alphas_best, betas_best, working_taus);
//         //         if(ll_after > ll_best){
//         //             ll_best = ll_after;
//         //             taus_best = working_taus;
//         //         }
//         //     }
            
//         // }


        
//         // Check convergence
//         if(max_change < TOL) break;
        
//         // Check for stalling
//         if(ll_best - ll_iter_start < 1e-6){
//             stall_count++;
//             if(stall_count >= max_stall){
//                 break;
//             }
//         } else {
//             stall_count = 0;
//         }
//     }
 
//     std::vector<std::vector<double>> out(3);
//     out.at(0) = alphas_best;
//     out.at(1) = betas_best;
//     out.at(2) = taus_best;
//     return out;
// }