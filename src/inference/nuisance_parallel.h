#ifndef AGREEMENTPHI_INFERENCE_NUISANCE_PARALLEL_H
#define AGREEMENTPHI_INFERENCE_NUISANCE_PARALLEL_H

#include <RcppParallel.h>
#include "nuisance.h"

namespace AgreementPhi {
namespace ordinal {
namespace nuisance {

// Worker for parallel item profiling
struct ParallelItemWorker : public RcppParallel::Worker {
    // Read-only inputs (shared across threads)
    const std::vector<double>& Y;
    const std::vector<int>& ITEM_INDS;
    const std::vector<int>& WORKER_INDS;
    const std::vector<double>& alphas_best;
    const std::vector<double>& betas_best;
    const std::vector<double>& taus_best;
    const std::vector<std::vector<int>>& ITEM_DICT;
    const double PHI;
    const double PROF_UNI_RANGE;
    const int PROF_UNI_MAX_ITER;
    const double mean_alpha;

    // Write-once outputs (each thread writes to different index)
    std::vector<double>& alphas_new;

    // Constructor
    ParallelItemWorker(
        const std::vector<double>& Y,
        const std::vector<int>& ITEM_INDS,
        const std::vector<int>& WORKER_INDS,
        const std::vector<double>& alphas_best,
        const std::vector<double>& betas_best,
        const std::vector<double>& taus_best,
        const std::vector<std::vector<int>>& ITEM_DICT,
        const double PHI,
        const double PROF_UNI_RANGE,
        const int PROF_UNI_MAX_ITER,
        const double mean_alpha,
        std::vector<double>& alphas_new
    ) : Y(Y), ITEM_INDS(ITEM_INDS), WORKER_INDS(WORKER_INDS),
        alphas_best(alphas_best), betas_best(betas_best), taus_best(taus_best),
        ITEM_DICT(ITEM_DICT), PHI(PHI), PROF_UNI_RANGE(PROF_UNI_RANGE),
        PROF_UNI_MAX_ITER(PROF_UNI_MAX_ITER), mean_alpha(mean_alpha),
        alphas_new(alphas_new) {}

    // Parallel operator
    void operator()(std::size_t begin, std::size_t end) {
        // Profile items in parallel
        for(std::size_t j = begin; j < end; ++j) {
            alphas_new[j] = AgreementPhi::ordinal::nuisance::brent_profiling(
                Y,
                ITEM_DICT,
                j + 1,  // 1-indexed
                WORKER_INDS,
                betas_best,
                alphas_best[j],  // Starting value
                PHI,
                taus_best,
                PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER,
                mean_alpha
            );
        }
    }
};

// Worker for parallel worker profiling
struct ParallelWorkerWorker : public RcppParallel::Worker {
    // Read-only inputs (shared across threads)
    const std::vector<double>& Y;
    const std::vector<int>& ITEM_INDS;
    const std::vector<int>& WORKER_INDS;
    const std::vector<double>& alphas_best;
    const std::vector<double>& betas_best;
    const std::vector<double>& taus_best;
    const std::vector<std::vector<int>>& WORKER_DICT;
    const double PHI;
    const double PROF_UNI_RANGE;
    const int PROF_UNI_MAX_ITER;
    const double mean_beta;

    // Write-once outputs (each thread writes to different index)
    std::vector<double>& betas_new;

    // Constructor
    ParallelWorkerWorker(
        const std::vector<double>& Y,
        const std::vector<int>& ITEM_INDS,
        const std::vector<int>& WORKER_INDS,
        const std::vector<double>& alphas_best,
        const std::vector<double>& betas_best,
        const std::vector<double>& taus_best,
        const std::vector<std::vector<int>>& WORKER_DICT,
        const double PHI,
        const double PROF_UNI_RANGE,
        const int PROF_UNI_MAX_ITER,
        const double mean_beta,
        std::vector<double>& betas_new
    ) : Y(Y), ITEM_INDS(ITEM_INDS), WORKER_INDS(WORKER_INDS),
        alphas_best(alphas_best), betas_best(betas_best), taus_best(taus_best),
        WORKER_DICT(WORKER_DICT), PHI(PHI), PROF_UNI_RANGE(PROF_UNI_RANGE),
        PROF_UNI_MAX_ITER(PROF_UNI_MAX_ITER), mean_beta(mean_beta),
        betas_new(betas_new) {}

    // Parallel operator
    void operator()(std::size_t begin, std::size_t end) {
        // Profile workers in parallel (skip first worker as it's constrained to 0)
        for(std::size_t w = begin; w < end; ++w) {
            if(w == 0) continue;  // First worker is constrained to 0

            betas_new[w] = AgreementPhi::ordinal::nuisance::brent_profiling(
                Y,
                WORKER_DICT,
                w + 1,  // 1-indexed
                ITEM_INDS,
                alphas_best,
                betas_best[w],  // Starting value
                PHI,
                taus_best,
                PROF_UNI_RANGE,
                PROF_UNI_MAX_ITER,
                mean_beta
            );
        }
    }
};

} // namespace nuisance
} // namespace ordinal
} // namespace AgreementPhi

#endif
