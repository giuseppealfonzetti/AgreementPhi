#ifndef AGREEMENTPHI_UTILITIES_UTILS_FUNCTIONS_H
#define AGREEMENTPHI_UTILITIES_UTILS_FUNCTIONS_H

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
    }
}

#endif