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

std::vector<std::vector<double>> AgreementPhi::continuous::twoway::inference::get_lambda(
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
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double TOL
){
    // std::vector<std::vector<int>> dict_items = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    // std::vector<std::vector<int>> dict_workers = AgreementPhi::utils::oneway_dict(W, WORKER_INDS);
    
    std::vector<double> alphas = ALPHA;
    std::vector<double> betas = BETA;
    betas.at(0) = 0;
    
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
    const int K,
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
        double lambda_reg = (all_equal) ? 1.0 : 0.5;
        
        auto neg_ll_regularized = [&](double nuisance){
            double nll = 0;
            
            for(int i=0; i<n_j; i++){
                const int obs_id = obs_vec.at(i);
                const int const_dim_idx = CONST_DIM_IDXS.at(obs_id)-1;
                const double const_dim_par = CONST_DIM_PARS.at(const_dim_idx);
                double dmu, dmu2;
                double mu = link::mu(nuisance + const_dim_par);
                nll -= AgreementPhi::ordinal::loglik(Y.at(obs_id), mu, PHI, K, dmu, dmu2, 0);
            }
            
            nll += lambda_reg * 0.5 * pow(nuisance - MEAN, 2);
            
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
        // return START;
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
                nll -= AgreementPhi::ordinal::loglik(Y.at(obs_id), mu, PHI, K, dmu, dmu2, 0); 
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



std::vector<std::vector<double>> AgreementPhi::ordinal::twoway::inference::get_lambda(
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
    const int K,
    const double PROF_UNI_RANGE,
    const int PROF_UNI_MAX_ITER,
    const int PROF_MAX_ITER,
    const double TOL
){
    // std::vector<std::vector<int>> dict_items = AgreementPhi::utils::oneway_dict(J, ITEM_INDS);
    // std::vector<std::vector<int>> dict_workers = AgreementPhi::utils::oneway_dict(W, WORKER_INDS);
    
    std::vector<double> alphas = ALPHA;
    std::vector<double> betas = BETA;
    betas.at(0) = 0;
    
    for(int iter = 0; iter < PROF_MAX_ITER; ++iter){
        double max_change = 0;
        

        double mean_alpha = std::accumulate(std::begin(alphas), std::end(alphas), 0.0);
        mean_alpha /= J;

        double mean_beta = std::accumulate(std::begin(betas), std::end(betas), 0.0);
        mean_beta /= W;

        // Profile items
        for(int j = 0; j < J; ++j){
            double old_alpha = alphas.at(j);
            alphas.at(j) = AgreementPhi::ordinal::nuisance::brent_profiling(
                Y, ITEM_DICT, j+1, WORKER_INDS, betas, 
                old_alpha, PHI, K, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_alpha
            );
            max_change = std::max(max_change, std::abs(alphas.at(j) - old_alpha));
        }
        
        // Profile workers
        for(int w = 1; w < W; ++w){
            double old_beta = betas.at(w);
            betas.at(w) = AgreementPhi::ordinal::nuisance::brent_profiling(
                Y, WORKER_DICT, w+1, ITEM_INDS, alphas, 
                old_beta, PHI, K, PROF_UNI_RANGE, PROF_UNI_MAX_ITER, mean_beta
            );
            max_change = std::max(max_change, std::abs(betas.at(w) - old_beta));
        }
        
        if(max_change < TOL) break;
    }
    
    std::vector<std::vector<double>> out(2);
    out.at(0) = alphas;
    out.at(1) = betas;
    return out;
}