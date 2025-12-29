#ifndef AGREEMENTPHI_UTILITIES_UTILS_FUNCTIONS_H
#define AGREEMENTPHI_UTILITIES_UTILS_FUNCTIONS_H

#include <Eigen/Dense>

namespace AgreementPhi{
    namespace utils{

        // Create dictionary for one-way estimation
        inline std::vector<std::vector<int>> oneway_items_dict(
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

        //' Create dictionary for one-way estimation
        //' @param UNIQUE_IDS Number of unique ids.
        //' @param IDS Vector of ids. Length n. Expected to start from 1.
        //' @returns A list collecting indexes of observations id by id.
        inline std::vector<std::vector<int>> oneway_dict(
            const int UNIQUE_IDS,
            const std::vector<int>& IDS
        ){

            const int n = IDS.size();
            std::vector<std::vector<int>> dict(UNIQUE_IDS);

            for(int i = 0; i < n; ++i){
                int item_id = static_cast<int>(IDS.at(i)-1);
                dict.at(item_id).push_back(i);
            }

            return dict;
        }

        inline std::vector<std::vector<int>> categories_dict(
            const std::vector<double> Y,
            const int K
        ){
            const int n = Y.size();
            std::vector<std::vector<int>> dict(K);
            for(int i = 0; i < n; ++i){
                int cat = static_cast<int>(Y.at(i));
                dict.at(cat-1).push_back(i);
            }

            return dict;
        }

        // Transform agreement into precision
        inline double agr2prec(double AGREEMENT){
            const double log2_sq = pow(log(2.0), 2);
            return -2.0 * log(1.0 - AGREEMENT) / log2_sq;
        }

        // Transform precision into agreement
        inline double prec2agr(double PRECISION){
            const double log2_half = log(2.0) / 2.0;
            return 1.0 - pow(2.0, -PRECISION * log2_half);
        }

        /* COMMENTED OUT 2025-12-26: Old raw_tau parameterization (K-1 parameters)
         * Replaced by parsimonious gamma parameterization (2 parameters) for thresholds as nuisance.
         * See notes.tex lines 354-372 for mathematical details of new parameterization.
         * This code is retained for reference and potential future use.
         */
        /*
        inline std::vector<double> raw2tau(const std::vector<double> RAW_TAU) {
            const int n = static_cast<int>(RAW_TAU.size());
            std::vector<double> out(n + 2);
            out.front() = 0.0;

            double denom = 1.0;
            for (double v : RAW_TAU) {
                denom += std::exp(v);
            }

            double csum = 0.0;
            for (int i = 0; i < n; ++i) {
                csum += std::exp(RAW_TAU[i]);
                out[i + 1] = csum / denom;
            }
            out.back() = 1.0;
            return out;
        }

        inline std::vector<double> tau2raw(const std::vector<double> TAU) {
            const int m = static_cast<int>(TAU.size());
            const int n = m - 2;

            std::vector<double> gaps(m - 1);
            for (int i = 0; i < m - 1; ++i) {
                gaps[i] = TAU[i + 1] - TAU[i];
            }
            const double last_gap = gaps[n];

            std::vector<double> raw(n);
            for (int i = 0; i < n; ++i) {
                raw[i] = std::log(static_cast<double>(n) * gaps[i] / last_gap);
            }
            return raw;
        }
        */
        /* END COMMENTED OUT raw_tau parameterization */

        // Parsimonious threshold parameterization (2 parameters)
        inline std::vector<double> gamma2tau_parsimonious(
            const std::vector<double>& GAMMA,
            const int K
        ) {
            const double e1 = std::exp(GAMMA[0]);
            const double e2 = std::exp(GAMMA[1]);
            const double denom = 1.0 + e1 + e2;

            std::vector<double> tau(K + 1);
            tau[0] = 0.0;
            tau[K] = 1.0;

            // tau_1 = exp(gamma_1) / (1 + exp(gamma_1) + exp(gamma_2))
            const double tau_1 = e1 / denom;
            tau[1] = tau_1;

            // tau_{K-1} = (exp(gamma_1) + exp(gamma_2)) / (1 + exp(gamma_1) + exp(gamma_2))
            const double tau_Km1 = (e1 + e2) / denom;
            tau[K - 1] = tau_Km1;

            // tau_k = tau_1 + (k-1) * (tau_{K-1} - tau_1) / (K-2)
            if (K > 2) {
                const double delta = (tau_Km1 - tau_1) / static_cast<double>(K - 2);
                for (int k = 2; k < K - 1; ++k) {
                    tau[k] = tau_1 + static_cast<double>(k - 1) * delta;
                }
            }

            return tau;
        }

        // Extracts tau_1 and tau_{K-1} and solves for gamma_1, gamma_2
        inline std::vector<double> tau2gamma_parsimonious(
            const std::vector<double>& TAU
        ) {
            const double tau_1 = TAU[1];
            const double tau_Km1 = TAU[TAU.size() - 2];

            // e1 = tau_1 / (1 - tau_{K-1})
            // e2 = tau_{K-1} / (1 - tau_{K-1}) - e1

            const double one_minus_tau_Km1 = std::max(1.0 - tau_Km1, 1e-10);
            const double e1 = tau_1 / one_minus_tau_Km1;
            const double e2 = tau_Km1 / one_minus_tau_Km1 - e1;

            std::vector<double> gamma(2);
            gamma[0] = std::log(std::max(e1, 1e-10));  
            gamma[1] = std::log(std::max(e2, 1e-10));

            return gamma;
        }

        // Compute Jacobian matrix dtau/dgamma for parsimonious parameterization
        // Returns (K+1) x 2 matrix where tau[k] is differentiated wrt gamma[0], gamma[1]
        inline Eigen::MatrixXd compute_dtau_dgamma_parsimonious(
            const std::vector<double>& GAMMA,
            const int K
        ) {
            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(K + 1, 2);

            const double e1 = std::exp(GAMMA[0]);
            const double e2 = std::exp(GAMMA[1]);
            const double denom = 1.0 + e1 + e2;
            const double denom2 = denom * denom;

            // Derivatives for tau_1 (row 1)
            // dtau_1/dgamma_1 = e1 * (1 + e2) / denom^2
            jac(1, 0) = e1 * (1.0 + e2) / denom2;
            // dtau_1/dgamma_2 = -e1 * e2 / denom^2
            jac(1, 1) = -e1 * e2 / denom2;

            // Derivatives for tau_{K-1} (row K-1)
            // dtau_{K-1}/dgamma_1 = e1 / denom^2
            jac(K - 1, 0) = e1 / denom2;
            // dtau_{K-1}/dgamma_2 = e2 / denom^2
            jac(K - 1, 1) = e2 / denom2;

            // Derivatives for intermediate thresholds (rows 2 to K-2)
            // tau_k = tau_1 + (k-1) * (tau_{K-1} - tau_1) / (K-2)
            // dtau_k/dgamma_r = dtau_1/dgamma_r + (k-1)/(K-2) * (dtau_{K-1}/dgamma_r - dtau_1/dgamma_r)
            if (K > 2) {
                const double K_2 = static_cast<double>(K - 2);
                for (int k = 2; k < K - 1; ++k) {
                    const double weight = static_cast<double>(k - 1) / K_2;
                    for (int r = 0; r < 2; ++r) {
                        jac(k, r) = jac(1, r) + weight * (jac(K - 1, r) - jac(1, r));
                    }
                }
            }


            return jac;
        }
    }


}

#endif
